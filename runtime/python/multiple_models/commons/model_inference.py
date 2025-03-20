#!/usr/bin/env python3

import os
import sys
import cv2
import time
import queue
import threading
import numpy as np
from pathlib import Path
from loguru import logger
from typing import List, Tuple, Dict, Any, Optional, NamedTuple, Union
from multiprocessing import Queue, Event, Value
from abc import ABC, abstractmethod
from enum import Enum

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from commons.hailo_inference import HailoAsyncInference
from commons.model_types import ModelType, ModelOutputData, ModelInputData

class ModelInference(ABC):
    """모델 추론을 위한 추상 기본 클래스"""
    
    def __init__(
        self, 
        net_path: str, 
        labels_path: str, 
        input_queue=None, 
        output_queue=None, 
        stop_event=None, 
        output_path=None,
        batch_size=1,
        fps_shared=None,
        device_id=0
    ):
        self.net_path = net_path
        self.labels_path = labels_path
        self.input_queue = input_queue or Queue(maxsize=20)
        self.output_queue = output_queue or Queue(maxsize=20)
        self.stop_event = stop_event or Event()
        self.output_path = output_path
        self.batch_size = batch_size
        self.model_name = self.__class__.__name__
        self.frame_count = 0  # 처리한 프레임 수 추적
        self._preprocessed_frame = None  # 전처리된 프레임을 저장할 변수 추가
        self.device_id = device_id  # 장치 ID 추가
        
        # 통계 수집용 변수
        self.total_inference_time = 0
        self.total_processing_time = 0
        self.stats_lock = threading.Lock()  # 통계 업데이트를 위한 락
        
        # 성능 측정용 공유 변수
        self.fps_shared = fps_shared if fps_shared is not None else Value('d', 0.0)
        
        logger.info(f"{self.model_name} 초기화 완료: 모델={net_path}, 입력 큐={self.input_queue._maxsize}, 출력 큐={self.output_queue._maxsize}")
    
    def _get_model_type(self):
        """모델 유형 반환"""
        if 'pose' in self.__class__.__name__.lower():
            return ModelType.POSE_ESTIMATION
        elif 'object' in self.__class__.__name__.lower() or 'detection' in self.__class__.__name__.lower():
            return ModelType.OBJECT_DETECTION
        else:
            return ModelType.UNKNOWN
    
    @abstractmethod
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """프레임 전처리"""
        pass
    
    @abstractmethod
    def postprocess(self, frame: np.ndarray, inference_result: any) -> np.ndarray:
        """추론 결과 후처리 및 시각화"""
        pass
    
    def sequential_inference(self, input_data):
        """
        단일 입력 데이터에 대해 순차적으로 추론을 실행하는 메서드
        이 메서드는 외부 메인 루프에서 호출되어 단일 입력에 대한 추론을 수행합니다.
        """
        try:
            # 입력 데이터 처리
            frame_id = None
            frame = None
            processed_frame = None
            
            # 입력 데이터 형식 처리
            if isinstance(input_data, dict):
                # 딕셔너리 형식 처리
                if 'frame_id' in input_data:
                    frame_id = input_data['frame_id']
                if 'original_frame' in input_data:
                    frame = input_data['original_frame']
                if 'processed_frame' in input_data:
                    processed_frame = input_data['processed_frame']
                
                # 모델 데이터가 있는 경우
                if 'model_data' in input_data and isinstance(input_data['model_data'], dict):
                    if self.model_name in input_data['model_data']:
                        model_data = input_data['model_data'][self.model_name]
                        if 'processed_frame' in model_data:
                            processed_frame = model_data['processed_frame']
            
            elif hasattr(input_data, 'get_frame_id') and callable(input_data.get_frame_id):
                # ModelInputData 객체 처리
                frame_id = input_data.get_frame_id()
                frame = input_data.get_original_frame()
                processed_frame = input_data.get_processed_frame()
            
            elif hasattr(input_data, 'frame_id') and hasattr(input_data, 'original_frame'):
                # 직접 속성 접근
                frame_id = input_data.frame_id
                frame = input_data.original_frame
                if hasattr(input_data, 'processed_frame'):
                    processed_frame = input_data.processed_frame
            
            else:
                logger.error(f"{self.model_name} 지원되지 않는 입력 형식: {type(input_data)}")
                return None
            
            if frame is None:
                logger.error(f"{self.model_name} 프레임이 없습니다")
                return None
            
            logger.debug(f"{self.model_name} 프레임 ID {frame_id} 처리 시작")
            
            # 전처리 수행 
            preprocess_start = time.time()
            if processed_frame is not None:
                # 이미 전처리된 프레임 사용
                preprocessed = processed_frame
                logger.debug(f"{self.model_name} 프레임 ID {frame_id} 전처리 생략 (미리 전처리된 프레임 사용)")
            else:
                # 전처리된 프레임이 없으면 원본 프레임 전처리
                if frame is not None:
                    logger.debug(f"프레임 {frame_id}의 processed_frame이 None입니다. 원본 프레임에서 전처리 수행")
                    process_start = time.time()
                    processed_frame = self.preprocess(frame)
                    preprocess_time = time.time() - process_start
                    logger.debug(f"프레임 {frame_id} 전처리 완료 (소요 시간: {preprocess_time*1000:.2f}ms)")
                else:
                    logger.warning(f"프레임 {frame_id}의 processed_frame과 original_frame이 모두 None입니다. 건너뜁니다.")
                    return None
            
            # 추론 실행
            inference_start = time.time()
            inference_result = self.inference(processed_frame)
            inference_time = time.time() - inference_start
            
            # 후처리 수행
            postprocess_start = time.time()
            result = self.postprocess(frame, inference_result)
            postprocess_time = time.time() - postprocess_start
            
            # 성능 측정
            total_time = preprocess_time + inference_time + postprocess_time
            
            # 처리 통계 업데이트
            self.frame_count += 1
            
            # 성능 정보 로깅
            logger.debug(f"{self.model_name} 프레임 ID {frame_id} 처리 완료:")
            logger.debug(f"  - 전처리: {preprocess_time*1000:.2f}ms")
            logger.debug(f"  - 추론: {inference_time*1000:.2f}ms")
            logger.debug(f"  - 후처리: {postprocess_time*1000:.2f}ms")
            logger.debug(f"  - 총 시간: {total_time*1000:.2f}ms")
            
            # 결과 및 메타데이터 반환
            output_data = {
                'frame_id': frame_id,
                'model_name': self.model_name,
                'model_type': self.model_type if hasattr(self, 'model_type') else None,
                'result': result,
                'original_frame': frame,
                'inference_time': inference_time,
                'total_time': total_time
            }
            
            return output_data
        
        except Exception as e:
            logger.error(f"{self.model_name} 순차적 추론 중 오류: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def process_frames(self, input_queue, output_queue, stop_event):
        """모델 전용 입력 큐에서 데이터를 가져와 처리하고 출력 큐에 결과를 넣는 메서드"""
        logger.info(f"ModelInference.process_frames 시작 (모델: {self.net_path})")
        
        # 통계 변수 초기화
        processed_frames = 0
        skipped_frames = 0
        pending_frames = {}  # 프레임 ID를 키로 하는 대기 중인 프레임 정보 저장
        start_time = time.time()
        last_log_time = start_time
        
        try:
            # 하일로 객체 초기화
            hailo_input_queue = Queue(maxsize=20)
            hailo_output_queue = Queue(maxsize=20)
            
            hailo_inference = HailoAsyncInference(
                hef_path=self.net_path,
                input_queue=hailo_input_queue,
                output_queue=hailo_output_queue,
                batch_size=self.batch_size
            )
            
            # 비동기 처리 시작
            hailo_inference.run()
            logger.info(f"하일로 추론 엔진 초기화 성공 (모델: {self.net_path})")
            
            # 입력 처리 및 결과 처리 루프
            while not stop_event.is_set():
                # 1. 입력 큐에서 새 데이터 확인
                try:
                    input_data = input_queue.get(block=False)  # 논블로킹으로 확인
                    
                    # None 체크 (종료 신호)
                    if input_data is None:
                        logger.info("입력 큐에서 종료 신호(None)를 받았습니다.")
                        break
                    
                    # 입력 데이터에서 필요한 정보 추출
                    frame_id, original_frame, processed_frame, model_type = self._extract_input_data(input_data)
                    
                    # 필수 데이터 확인
                    if frame_id is None:
                        logger.warning("입력 데이터에 프레임 ID가 없습니다.")
                        skipped_frames += 1
                        continue
                    
                    if processed_frame is None:
                        logger.warning(f"프레임 {frame_id}의 processed_frame이 None입니다. 건너뜁니다.")
                        skipped_frames += 1
                        continue
                    
                    # 하일로 입력 큐에 바로 전송
                    logger.debug(f"프레임 {frame_id} 하일로 입력 큐에 전송")
                    
                    # 메타데이터 저장 (결과와 매칭하기 위함)
                    metadata = {
                        'frame_id': frame_id,
                        'model_type': model_type,
                        'original_frame': original_frame,
                        'timestamp': time.time()
                    }
                    
                    # 대기 프레임 저장
                    frame_key = str(frame_id)
                    pending_frames[frame_key] = metadata
                    
                    # 모델 입력으로 직접 사용 (전처리 단계 제거)
                    model_input = {'input': processed_frame}
                    hailo_input_queue.put(model_input)
                    processed_frames += 1
                    
                except queue.Empty:
                    # 입력 큐가 비어있음, 오류가 아님
                    pass
                except Exception as e:
                    logger.error(f"입력 처리 중 오류: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                
                # 2. 하일로 출력 큐에서 결과 확인
                try:
                    result = hailo_output_queue.get(block=False)  # 논블로킹으로 확인
                    
                    # 가장 오래된 대기 프레임 사용 (FIFO 순서 가정)
                    if not pending_frames:
                        logger.warning("대기 중인 프레임이 없는데 결과가 반환됨")
                        continue
                    
                    frame_key = next(iter(pending_frames))
                    metadata = pending_frames.pop(frame_key)
                    
                    frame_id = metadata.get('frame_id')
                    model_type = metadata.get('model_type')
                    original_frame = metadata.get('original_frame')
                    timestamp = metadata.get('timestamp', 0)
                    
                    logger.debug(f"프레임 {frame_id} 결과 처리 시작")
                    
                    # 후처리 시작
                    postprocess_start = time.time()
                    
                    try:
                        # 후처리 수행
                        frame_shape = None
                        if hasattr(original_frame, 'shape'):
                            frame_shape = original_frame.shape
                        elif hasattr(original_frame, 'size'):
                            width, height = original_frame.size
                            frame_shape = (height, width)
                        
                        processed_result = self.postprocess(result, frame_shape)
                        
                    except Exception as e:
                        logger.error(f"후처리 중 오류: {e}")
                        processed_result = result
                    
                    # 시간 계산
                    postprocess_time = time.time() - postprocess_start
                    inference_time = time.time() - timestamp if timestamp > 0 else 0
                    total_time = inference_time + postprocess_time
                    
                    # 결과 객체 생성
                    output_data = ModelOutputData(
                        frame_id=frame_id,
                        model_type=model_type,
                        original_frame=original_frame,
                        results=processed_result
                    )
                    
                    # 모델 메타데이터 추가
                    output_data.model_id = self.__class__.__name__
                    output_data.model_name = self.model_name
                    output_data.inference_time = inference_time
                    output_data.postprocess_time = postprocess_time
                    output_data.total_time = total_time
                    
                    # 출력 큐에 결과 전송
                    try:
                        output_queue.put(output_data)
                        logger.debug(f"프레임 {frame_id} 결과 출력 큐에 전송 성공")
                    except queue.Full:
                        logger.warning(f"출력 큐 가득 참. 프레임 {frame_id} 결과 전송 실패")
                
                except queue.Empty:
                    # 출력 큐가 비어있음, 오류가 아님
                    pass
                except Exception as e:
                    logger.error(f"결과 처리 중 오류: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                
                # 3. 성능 로깅 (5초마다)
                current_time = time.time()
                if current_time - last_log_time >= 5.0:
                    elapsed = current_time - start_time
                    fps = processed_frames / elapsed if elapsed > 0 else 0
                    
                    logger.info(f"모델 성능: {fps:.2f} FPS (처리: {processed_frames}프레임, "
                              f"건너뜀: {skipped_frames}프레임, 대기 중: {len(pending_frames)}프레임)")
                    
                    # FPS 공유 변수 업데이트 (있는 경우)
                    if hasattr(self, 'fps_shared') and self.fps_shared is not None:
                        self.fps_shared.value = fps
                    
                    last_log_time = current_time
                
                # 짧은 대기 (CPU 사용률 감소)
                time.sleep(0.001)
            
            # 정상 종료 처리
            logger.info(f"입력 처리 완료. 남은 결과 {len(pending_frames)}개 처리 중...")
            
            # 남은 대기 프레임 처리를 위한 시간 확보
            timeout_end = time.time() + 3.0  # 최대 3초 대기
            while pending_frames and time.time() < timeout_end:
                try:
                    result = hailo_output_queue.get(block=False)
                    
                    if pending_frames:
                        frame_key = next(iter(pending_frames))
                        pending_frames.pop(frame_key)
                        logger.debug(f"종료 중 남은 결과 처리: {len(pending_frames)}개 남음")
                    
                except queue.Empty:
                    time.sleep(0.01)  # 짧은 대기
                    continue
            
            # 종료 신호 전송
            logger.info("출력 큐에 종료 신호(None) 전송")
            try:
                output_queue.put(None, block=False)
            except queue.Full:
                logger.warning("출력 큐가 가득 차서 종료 신호 보내기 실패")
            
        except Exception as e:
            logger.error(f"process_frames 중 예외 발생: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        finally:
            # 하일로 자원 정리
            if 'hailo_inference' in locals() and hailo_inference is not None:
                try:
                    hailo_inference.close()
                    logger.info("하일로 추론 엔진 자원 해제 완료")
                except Exception as e:
                    logger.error(f"추론 엔진 자원 해제 중 오류: {e}")
            
            logger.info(f"추론 프로세스 종료: 총 {processed_frames}프레임 처리, {skipped_frames}프레임 건너뜀")
    
    def _extract_input_data(self, input_data):
        """입력 데이터에서 필요한 정보 추출"""
        try:
            frame_id = None
            original_frame = None
            processed_frame = None
            model_type = None
            
            # ModelInputData 객체 처리
            if isinstance(input_data, ModelInputData):
                frame_id = input_data.frame_id
                original_frame = input_data.original_frame
                processed_frame = input_data.processed_frame
                model_type = input_data.model_type
                logger.debug(f"ModelInputData 객체 처리: frame_id={frame_id}")
            
            # 딕셔너리 형식 처리
            elif isinstance(input_data, dict):
                frame_id = input_data.get('frame_id')
                original_frame = input_data.get('original_frame')
                processed_frame = input_data.get('processed_frame')
                model_type = input_data.get('model_type')
            
            # get_xxx 메서드가 있는 객체 처리
            elif hasattr(input_data, 'get_frame_id') and callable(input_data.get_frame_id):
                frame_id = input_data.get_frame_id()
                original_frame = input_data.get_original_frame()
                processed_frame = input_data.get_processed_frame()
                model_type = input_data.get_model_type() if hasattr(input_data, 'get_model_type') else None
            
            # 지원되지 않는 형식
            else:
                logger.warning(f"지원되지 않는 입력 데이터 형식: {type(input_data)}")
                return None, None, None, None
            
            return frame_id, original_frame, processed_frame, model_type
        
        except Exception as e:
            logger.error(f"데이터 추출 중 오류: {e}")
            return None, None, None, None

    def inference(self, preprocessed_frame):
        """
        전처리된 프레임에 대해 실제 모델 추론을 실행하는 메서드
        HailoAsyncInference를 사용하여 추론 수행
        """
        try:
            # HailoAsyncInference 객체 초기화
            hailo_inference = HailoAsyncInference(
                hef_path=self.net_path,
                batch_size=self.batch_size
            )
            
            # 임시 큐 생성
            temp_input_queue = queue.Queue()
            temp_output_queue = queue.Queue()
            
            # 입력 큐에 전처리된 프레임 추가
            temp_input_queue.put(preprocessed_frame)
            
            # 추론 실행 (비동기)
            hailo_inference.set_input_queue(temp_input_queue)
            hailo_inference.set_output_queue(temp_output_queue)
            hailo_inference.run()
            
            # 결과 대기
            try:
                inference_result = temp_output_queue.get(timeout=5.0)
                return inference_result
            except queue.Empty:
                logger.error(f"{self.model_name} 추론 결과 대기 중 타임아웃")
                return None
            
        except Exception as e:
            logger.error(f"{self.model_name} 추론 중 오류: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None 