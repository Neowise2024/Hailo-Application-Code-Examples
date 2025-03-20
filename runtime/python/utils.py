from typing import List, Generator, Optional, Tuple, Dict
from pathlib import Path
from functools import partial
import queue
from loguru import logger
import numpy as np
import time
from hailo_platform import (HEF, VDevice,
                            FormatType, HailoSchedulingAlgorithm)
IMAGE_EXTENSIONS: Tuple[str, ...] = ('.jpg', '.png', '.bmp', '.jpeg')


class HailoAsyncInference:
    def __init__(
        self, 
        hef_path: str, 
        input_queue: queue.Queue,
        output_queue: queue.Queue, 
        batch_size: int = 1,
        input_type: Optional[str] = None, 
        output_type: Optional[Dict[str, str]] = None,
        send_original_frame: bool = False,
        net_group: str = 'net_group1') -> None:
        """
        Initialize the HailoAsyncInference class with the provided HEF model 
        file path and input/output queues.

        Args:
            hef_path (str): Path to the HEF model file.
            input_queue (queue.Queue): Queue from which to pull input frames 
                                       for inference.
            output_queue (queue.Queue): Queue to hold the inference results.
            batch_size (int): Batch size for inference. Defaults to 1.
            input_type (Optional[str]): Format type of the input stream. 
                                        Possible values: 'UINT8', 'UINT16'.
            output_type Optional[dict[str, str]] : Format type of the output stream. 
                                         Possible values: 'UINT8', 'UINT16', 'FLOAT32'.
        """
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.batch_size = batch_size
        
        params = VDevice.create_params()
        # Set the scheduling algorithm to round-robin to activate the scheduler
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        params.multi_process_service = True
        params.group_id = net_group

        self.hef = HEF(hef_path)
        self.target = VDevice(params)
        self.infer_model = self.target.create_infer_model(hef_path)
        self.infer_model.set_batch_size(batch_size)      
        if input_type is not None:
            self._set_input_type(input_type)
        if output_type is not None:
            self._set_output_type(output_type)

        self.output_type = output_type
        self.send_original_frame = send_original_frame

    def _set_input_type(self, input_type: Optional[str] = None) -> None:
        """
        Set the input type for the HEF model. If the model has multiple inputs,
        it will set the same type of all of them.

        Args:
            input_type (Optional[str]): Format type of the input stream.
        """
        self.infer_model.input().set_format_type(getattr(FormatType, input_type))
    
    def _set_output_type(self, output_type_dict: Optional[Dict[str, str]] = None) -> None:
        """
        Set the output type for the HEF model. If the model has multiple outputs,
        it will set the same type for all of them.

        Args:
            output_type_dict (Optional[dict[str, str]]): Format type of the output stream.
        """
        for output_name, output_type in output_type_dict.items():
            self.infer_model.output(output_name).set_format_type(
                getattr(FormatType, output_type)
            )

    def callback(
        self, completion_info, bindings_list: list, input_batch: list,
    ) -> None:
        """
        Callback function for handling inference results.

        Args:
            completion_info: Information about the completion of the 
                             inference task.
            bindings_list (list): List of binding objects containing input 
                                  and output buffers.
            processed_batch (list): The processed batch of images.
        """
        if completion_info.exception:
            logger.error(f'Inference error: {completion_info.exception}')
        else:
            for i, bindings in enumerate(bindings_list):
                # If the model has a single output, return the output buffer. 
                # Else, return a dictionary of output buffers, where the keys are the output names.
                if len(bindings._output_names) == 1:
                    result = bindings.output().get_buffer()
                else:
                    result = {
                        name: np.expand_dims(
                            bindings.output(name).get_buffer(), axis=0
                        )
                        for name in bindings._output_names
                    }
                self.output_queue.put((input_batch[i], result))

    def get_vstream_info(self) -> Tuple[list, list]:

        """
        Get information about input and output stream layers.

        Returns:
            Tuple[list, list]: List of input stream layer information, List of 
                               output stream layer information.
        """
        return (
            self.hef.get_input_vstream_infos(), 
            self.hef.get_output_vstream_infos()
        )

    def get_hef(self) -> HEF:
        """
        Get the object's HEF file
        
        Returns:
            HEF: A HEF (Hailo Executable File) containing the model.
        """
        return self.hef

    def get_input_shape(self) -> Tuple[int, ...]:
        """
        Get the shape of the model's input layer.

        Returns:
            Tuple[int, ...]: Shape of the model's input layer.
        """
        return self.hef.get_input_vstream_infos()[0].shape  # Assumes one input

    def run(self) -> None:
        logger.info(f"HailoAsyncInference 실행 시작 (send_original_frame={self.send_original_frame}, batch_size={self.batch_size})")
        with self.infer_model.configure() as configured_infer_model:
            while True:
                batch_data = self.input_queue.get()
                # logger.info(f"HailoAsyncInference batch_data: {batch_data}")
                if batch_data is None:
                    break  # Sentinel value to stop the inference loop

                if self.send_original_frame:
                    original_batch, preprocessed_batch = batch_data
                else:
                    preprocessed_batch = batch_data

                bindings_list = []
                for frame in preprocessed_batch:
                    bindings = self._create_bindings(configured_infer_model)
                    bindings.input().set_buffer(np.array(frame))
                    bindings_list.append(bindings)

                configured_infer_model.wait_for_async_ready(timeout_ms=10000)
                job = configured_infer_model.run_async(
                    bindings_list, partial(
                        self.callback,
                        input_batch=original_batch if self.send_original_frame else preprocessed_batch,
                        bindings_list=bindings_list
                    )
                )
            job.wait(10000)  # Wait for the last job

    def _get_output_type_str(self, output_info) -> str:
        if self.output_type is None:
            return str(output_info.format.type).split(".")[1].lower()
        else:
            self.output_type[output_info.name].lower()

    def _create_bindings(self, configured_infer_model) -> object:
        """
        Create bindings for input and output buffers.

        Args:
            configured_infer_model: The configured inference model.

        Returns:
            object: Bindings object with input and output buffers.
        """
        if self.output_type is None:
            output_buffers = {
                output_info.name: np.empty(
                    self.infer_model.output(output_info.name).shape,
                    dtype=(getattr(np, self._get_output_type_str(output_info)))
                )
            for output_info in self.hef.get_output_vstream_infos()
            }
        else:
            output_buffers = {
                name: np.empty(
                    self.infer_model.output(name).shape, 
                    dtype=(getattr(np, self.output_type[name].lower()))
                )
            for name in self.output_type
            }
        return configured_infer_model.create_bindings(
            output_buffers=output_buffers
        )

    def multiple_run(self, model_input_queues=None, model_output_queues=None, model_stop_events=None) -> None:
        """
        여러 모델의 입력 큐를 처리하는 함수입니다. run() 메서드와 동일한 방식으로 작동합니다.
        
        Args:
            model_input_queues (list, optional): 모델 입력 큐 리스트. 미지정 시 self.input_queue 사용
            model_output_queues (list, optional): 모델 출력 큐 리스트. 미지정 시 self.output_queue 사용
            model_stop_events (list, optional): 모델 중지 이벤트 리스트
        """
        logger.info(f"HailoAsyncInference multiple_run 시작 (send_original_frame={self.send_original_frame}, batch_size={self.batch_size})")
        
        # 기본값 설정
        if model_input_queues is None:
            model_input_queues = [self.input_queue]
        if model_output_queues is None:
            model_output_queues = [self.output_queue]
        
        # 모델 수 확인
        model_count = len(model_input_queues)
        if len(model_output_queues) != model_count:
            raise ValueError(f"입력 큐({model_count}개)와 출력 큐({len(model_output_queues)}개)의 수가 일치하지 않습니다.")
        
        logger.info(f"처리할 모델 수: {model_count}")
        
        # 현재 처리 중인 모델 인덱스를 저장할 변수
        current_model_idx = 0
        
        with self.infer_model.configure() as configured_infer_model:
            # 각 모델별 활성 상태 추적
            active_models = [True] * model_count
            
            while any(active_models):
                # 라운드 로빈 방식으로 모든 모델 큐 순회
                model_idx = current_model_idx
                current_model_idx = (current_model_idx + 1) % model_count
                
                if not active_models[model_idx]:
                    continue
                
                # 모델 중지 이벤트 확인
                if model_stop_events and model_stop_events[model_idx].is_set():
                    active_models[model_idx] = False
                    logger.info(f"모델 {model_idx+1} 중지 이벤트가 설정되어 처리를 중단합니다.")
                    continue
                
                try:
                    # 타임아웃을 짧게 설정하여 다른 모델 큐도 주기적으로 확인
                    batch_data = model_input_queues[model_idx].get(timeout=0.01)
                    
                    # None 값은 종료 신호
                    if batch_data is None:
                        logger.info(f"모델 {model_idx+1} 종료 신호를 받았습니다.")
                        active_models[model_idx] = False
                        continue
                    
                    # run() 메서드와 동일한 방식으로 데이터 처리
                    if self.send_original_frame:
                        # 입력 데이터 형식 처리
                        if isinstance(batch_data, tuple):
                            # 3-요소 튜플인 경우 (model_type, original_frame, processed_frame)
                            if len(batch_data) == 3:
                                _, original_batch, preprocessed_batch = batch_data
                                # 단일 프레임인 경우 리스트로 변환
                                if not isinstance(original_batch, list):
                                    original_batch = [original_batch]
                                if not isinstance(preprocessed_batch, list):
                                    preprocessed_batch = [preprocessed_batch]
                            # 2-요소 튜플인 경우 (original_frame, processed_frame)
                            elif len(batch_data) == 2:
                                original_batch, preprocessed_batch = batch_data
                                # 단일 프레임인 경우 리스트로 변환
                                if not isinstance(original_batch, list):
                                    original_batch = [original_batch]
                                if not isinstance(preprocessed_batch, list):
                                    preprocessed_batch = [preprocessed_batch]
                            else:
                                logger.warning(f"모델 {model_idx+1}의 입력 데이터 형식이 예상과 다릅니다: {len(batch_data)}개 요소")
                                continue
                        else:
                            logger.warning(f"모델 {model_idx+1}의 입력 데이터 형식이 예상과 다릅니다: {type(batch_data)}")
                            continue
                    else:
                        preprocessed_batch = batch_data
                        # 단일 프레임인 경우 리스트로 변환
                        if not isinstance(preprocessed_batch, list):
                            preprocessed_batch = [preprocessed_batch]
                    
                    # 바인딩 리스트 생성
                    bindings_list = []
                    for frame in preprocessed_batch:
                        try:
                            # numpy 배열로 변환 (이미 numpy 배열이면 그대로 유지)
                            if not isinstance(frame, np.ndarray):
                                frame = np.array(frame)
                                
                            bindings = self._create_bindings(configured_infer_model)
                            bindings.input().set_buffer(frame)
                            bindings_list.append(bindings)
                        except Exception as e:
                            logger.error(f"모델 {model_idx+1} 바인딩 생성 중 오류 발생: {e}")
                            import traceback
                            logger.error(f"상세 오류: {traceback.format_exc()}")
                    
                    # 결과를 저장할 출력 큐 지정
                    output_queue = model_output_queues[model_idx]
                    
                    # 모델별 콜백 함수 생성
                    def model_callback(completion_info, bindings_list, model_original_batch):
                        if completion_info.exception:
                            logger.error(f'모델 {model_idx+1} 추론 오류: {completion_info.exception}')
                        else:
                            for i, bindings in enumerate(bindings_list):
                                # 결과 처리
                                if len(bindings._output_names) == 1:
                                    result = bindings.output().get_buffer()
                                else:
                                    result = {
                                        name: np.expand_dims(
                                            bindings.output(name).get_buffer(), axis=0
                                        )
                                        for name in bindings._output_names
                                    }
                                # 결과 전송
                                output_queue.put((model_original_batch[i], result))
                                logger.debug(f"모델 {model_idx+1} 결과 큐에 추가됨")
                    
                    # 비동기 추론 실행
                    configured_infer_model.wait_for_async_ready(timeout_ms=10000)
                    job = configured_infer_model.run_async(
                        bindings_list, 
                        partial(
                            model_callback,
                            bindings_list=bindings_list,
                            model_original_batch=original_batch if self.send_original_frame else preprocessed_batch
                        )
                    )
                    logger.debug(f"모델 {model_idx+1} 비동기 추론 작업 시작됨")
                    
                except queue.Empty:
                    # 타임아웃은 정상적인 상황이므로 계속 진행
                    pass
                except Exception as e:
                    logger.error(f"모델 {model_idx+1} 처리 중 오류 발생: {e}")
                    import traceback
                    logger.error(f"상세 오류: {traceback.format_exc()}")
                
                # CPU 부하 감소를 위한 짧은 휴식
                time.sleep(0.001)
            
            # 마지막 작업이 완료될 때까지 대기
            if 'job' in locals():
                job.wait(10000)
                
        logger.info(f"HailoAsyncInference multiple_run 종료")


def load_images_opencv(images_path: str) -> List[np.ndarray]:
    """
    Load images from the specified path.

    Args:
        images_path (str): Path to the input image or directory of images.

    Returns:
        List[np.ndarray]: List of images as NumPy arrays.
    """
    import cv2
    path = Path(images_path)
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
        return [cv2.imread(str(path))]
    elif path.is_dir():
        return [
            cv2.imread(str(img)) for img in path.glob("*")
            if img.suffix.lower() in IMAGE_EXTENSIONS
        ]
    return []

def load_input_images(images_path: str):
    """
    Load images from the specified path.

    Args:
        images_path (str): Path to the input image or directory of images.

    Returns:
        List[Image.Image]: List of PIL.Image.Image objects.
    """
    from PIL import Image
    path = Path(images_path)
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
        return [Image.open(path)]
    elif path.is_dir():
        return [
            Image.open(img) for img in path.glob("*") 
            if img.suffix.lower() in IMAGE_EXTENSIONS
        ]
    return []

def validate_images(images: List[np.ndarray], batch_size: int) -> None:
    """
    Validate that images exist and are properly divisible by the batch size.

    Args:
        images (List[np.ndarray]): List of images.
        batch_size (int): Number of images per batch.

    Raises:
        ValueError: If images list is empty or not divisible by batch size.
    """
    if not images:
        raise ValueError(
            'No valid images found in the specified path.'
        )
    
    if len(images) % batch_size != 0:
        raise ValueError(
            'The number of input images should be divisible by the batch size '
            'without any remainder.'
        )


def divide_list_to_batches(
    images_list: List[np.ndarray], batch_size: int
) -> Generator[List[np.ndarray], None, None]:
    """
    Divide the list of images into batches.

    Args:
        images_list (List[np.ndarray]): List of images.
        batch_size (int): Number of images in each batch.

    Returns:
        Generator[List[np.ndarray], None, None]: Generator yielding batches 
                                                  of images.
    """
    for i in range(0, len(images_list), batch_size):
        yield images_list[i: i + batch_size]
