import os
import sys
import time
import queue
import logging
from typing import Dict, List, Tuple, Any, Optional
from multiprocessing import Process, Queue, Event
import time
import numpy as np

# 상위 디렉토리의 utils 모듈에서 원래 HailoAsyncInference 클래스 임포트
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils import HailoAsyncInference as OriginalHailoAsyncInference

# 로거 설정
from loguru import logger

class HailoAsyncInference:
    """
    HailoAsyncInference 클래스의 확장 래퍼
    
    이 클래스는 원래의 HailoAsyncInference 클래스를 래핑하여 추가 기능을 제공합니다.
    """
    
    def __init__(self, 
                 hef_path: str, 
                 input_queue: Optional[Queue] = None, 
                 output_queue: Optional[Queue] = None, 
                 batch_size: int = 1, 
                 output_type: Optional[Dict[str, str]] = None,
                 send_original_frame: bool = True):
        """
        HailoAsyncInference 초기화
        
        Args:
            hef_path: HEF 파일 경로
            input_queue: 입력 큐 (None인 경우 내부에서 생성)
            output_queue: 출력 큐 (None인 경우 내부에서 생성)
            batch_size: 배치 크기
            output_type: 출력 데이터 타입 딕셔너리
            send_original_frame: 원본 프레임을 출력에 포함할지 여부
        """
        self.hef_path = hef_path
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.batch_size = batch_size
        # output_type은 딕셔너리 형태여야 함 (bool 타입이 들어오면 문제 발생)
        self.output_type = output_type if isinstance(output_type, dict) else None
        self.send_original_frame = send_original_frame
        self.model_name = os.path.basename(hef_path)
        
        # 원본 출력 큐 저장
        self.original_output_queue = output_queue
        
        # 래핑된 출력 큐 생성 (로깅을 위해)
        if output_queue:
            self.wrap_output_queue()
        
        # 원본 HailoAsyncInference 인스턴스 생성
        logger.debug(f"OriginalHailoAsyncInference 초기화: hef_path={hef_path}, batch_size={batch_size}, send_original_frame={send_original_frame}")
        try:
            self.hailo_inference = OriginalHailoAsyncInference(
                hef_path,
                input_queue,
                self.output_queue,  # 원래 output_queue가 아닌 래핑된 output_queue 사용
                batch_size,
                None,  # input_type은 None으로 전달
                self.output_type,  # output_type 파라미터는 반드시 딕셔너리 또는 None이어야 함
                send_original_frame
            )
            logger.debug(f"HailoAsyncInference 초기화 완료: {hef_path}")
        except Exception as e:
            logger.error(f"HailoAsyncInference 초기화 중 오류 발생: {e}")
            import traceback
            logger.error(f"상세 오류: {traceback.format_exc()}")
            raise
    
    def wrap_output_queue(self):
        """
        출력 큐를 래핑하여 로깅 추가
        """
        original_put = self.output_queue.put
        
        def wrapped_put(item, *args, **kwargs):
            # 모델 이름, 출력 타입, 크기 등 정보 로깅
            if item is None:
                logger.info(f"[{self.model_name}] 종료 신호(None)를 출력 큐에 전송")
            else:
                try:
                    # 데이터 형식에 따른 로깅
                    if isinstance(item, tuple):
                        if len(item) >= 2:
                            frame_info = ""
                            if isinstance(item[0], int):
                                frame_info = f", 프레임 ID: {item[0]}"
                            elif isinstance(item[1], np.ndarray):
                                frame_info = f", 프레임 크기: {item[1].shape}"
                            
                            result_info = ""
                            if len(item) >= 3 and isinstance(item[2], dict):
                                result_keys = list(item[2].keys())
                                result_info = f", 결과 키: {result_keys}"
                            
                            logger.info(f"[{self.model_name}] 추론 결과를 출력 큐에 전송{frame_info}{result_info}")
                        else:
                            logger.info(f"[{self.model_name}] 추론 결과를 출력 큐에 전송 (튜플 형식, 길이: {len(item)})")
                    else:
                        logger.info(f"[{self.model_name}] 추론 결과를 출력 큐에 전송 (타입: {type(item)})")
                except Exception as e:
                    logger.warning(f"[{self.model_name}] 추론 결과 로깅 중 오류: {e}")
            
            # 실제 큐에 데이터 추가
            start_time = time.time()
            result = original_put(item, *args, **kwargs)
            elapsed = time.time() - start_time
            
            # 큐가 가득 차서 대기한 경우 로깅
            if elapsed > 0.1:
                logger.warning(f"[{self.model_name}] 출력 큐가 가득 차서 {elapsed:.2f}초 대기함")
            
            return result
        
        # 원래 메서드를 저장하고 래핑된 메서드로 교체
        self.output_queue.original_put = original_put
        self.output_queue.put = wrapped_put
    
    def get_input_shape(self) -> Tuple[int, int, int]:
        """
        입력 형상 반환
        
        Returns:
            (height, width, channels) 형태의 튜플
        """
        return self.hailo_inference.get_input_shape()
    
    def get_output_shapes(self) -> Dict[str, List[int]]:
        """
        출력 형상 반환
        
        Returns:
            출력 이름을 키로 하고 형상을 값으로 하는 딕셔너리
        """
        return self.hailo_inference.get_output_shapes()
    
    def run(self):
        """
        하일로 추론 엔진 실행 (큐 기반 비동기 처리)
        입력 큐에서 데이터를 가져와 추론하고 결과를 출력 큐에 전송하는 작업을 시작합니다.
        """
        logger.debug(f"[{self.model_name}] 비동기 추론 시작")
        
        try:
            # 원본 HailoAsyncInference의 run 메서드 호출
            if hasattr(self.hailo_inference, 'run'):
                self.hailo_inference.run()  # 매개변수 없이 호출
                logger.debug(f"[{self.model_name}] 비동기 처리 시작 완료")
                return True
            else:
                logger.warning(f"[{self.model_name}] 원본 HailoAsyncInference에 run 메서드가 없습니다.")
                return False
        except Exception as e:
            logger.error(f"[{self.model_name}] 비동기 추론 시작 중 오류: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def multiple_run(self, 
                    model_input_queues: List[Queue], 
                    model_output_queues: List[Queue],
                    model_stop_events: List[Event]):
        """
        여러 모델을 동시에 실행하는 메서드
        
        Args:
            model_input_queues: 모델별 입력 큐 리스트
            model_output_queues: 모델별 출력 큐 리스트
            model_stop_events: 모델별 종료 이벤트 리스트
        """
        logger.info(f"[{self.model_name}] multiple_run 시작 (총 {len(model_input_queues)} 모델)")
        
        # 원본 multiple_run 호출
        try:
            # 원래 라이브러리에 multiple_run 메서드가 있는 경우
            if hasattr(self.hailo_inference, 'multiple_run'):
                self.hailo_inference.multiple_run(
                    model_input_queues,
                    model_output_queues,
                    model_stop_events
                )
            # run을 사용하는 경우
            else:
                logger.warning(f"[{self.model_name}] 원본 HailoAsyncInference에 multiple_run 메서드가 없습니다. run 메서드 사용을 시도합니다.")
                self.hailo_inference.run()
        except Exception as e:
            logger.error(f"[{self.model_name}] multiple_run 중 오류 발생: {e}")
            import traceback
            logger.error(f"상세 오류: {traceback.format_exc()}")
        
        logger.info(f"[{self.model_name}] multiple_run 종료")
    
    def close(self):
        """
        자원 해제
        """
        logger.debug(f"[{self.model_name}] 자원 해제")
        if hasattr(self.hailo_inference, 'close'):
            self.hailo_inference.close()
        else:
            logger.warning(f"[{self.model_name}] 원본 HailoAsyncInference에 close 메서드가 없습니다.")
    
