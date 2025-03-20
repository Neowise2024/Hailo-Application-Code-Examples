#!/usr/bin/env python3

import os
import sys
import cv2
import time
import queue
import numpy as np
from loguru import logger
from pathlib import Path
from typing import List, Optional, Any
from multiprocessing import Queue, Event
from commons.model_types import ModelType, ModelInputData

class InputProcessor:
    """입력 프레임을 처리하는 클래스"""
    
    def __init__(self, input_source="0", models=None, input_queue=None, batch_size=1, stop_event=None):
        """
        입력 프로세서 초기화
        
        주의: 이 클래스에서는 __init__ 내에서 프로세스를 자동으로 시작하지 않습니다.
        process_frames 메서드는 반드시 외부에서 명시적으로 호출해야 합니다.
        """
        logger.info(f"InputProcessor.__init__ 시작: input_source={input_source}, batch_size={batch_size}")
        self.input_source = input_source
        self.models = models or []
        # 입력 큐가 제공되지 않으면 생성
        self.input_queue = input_queue or Queue(maxsize=20)
        self.batch_size = batch_size
        self.stop_event = stop_event or Event()
        self.width = 0
        self.height = 0
        self.fps = 0
        self.cap = None
        self.frame_id = 0
        self.input_source_id = 0  # 입력 소스 ID 기본값
        self.resource_friendly = False  # 리소스 절약 모드 기본값
        
        # 비디오 캡처 설정
        try:
            logger.info("InputProcessor: 비디오 캡처 설정 시작")
            self._setup_video_capture()
            logger.info(f"InputProcessor: 비디오 캡처 설정 완료: width={self.width}, height={self.height}, fps={self.fps}")
        except Exception as e:
            logger.error(f"입력 소스 설정 중 오류: {e}")
            raise
        
        logger.info(f"InputProcessor.__init__ 완료 - 입력 큐 생성 (크기: {self.input_queue._maxsize})")
    
    def _find_available_cameras(self, max_tries=10):
        """사용 가능한 카메라 인덱스 찾기"""
        available_cameras = []
        
        logger.info(f"사용 가능한 카메라 검색 중 (최대 {max_tries}개)...")
        for i in range(max_tries):
            try:
                temp_cap = cv2.VideoCapture(i)
                if temp_cap.isOpened():
                    ret, frame = temp_cap.read()
                    if ret:
                        logger.info(f"카메라 {i} 사용 가능")
                        available_cameras.append(i)
                temp_cap.release()
            except Exception:
                pass
        
        if available_cameras:
            logger.info(f"사용 가능한 카메라: {available_cameras}")
        else:
            logger.warning("사용 가능한 카메라를 찾을 수 없습니다.")
            
        return available_cameras
    
    def _setup_video_capture(self):
        """비디오 캡처 설정 - 자동 입력 소스 감지 포함"""
        
        # 1. 문자열 '0'을 정수로 변환 시도
        if isinstance(self.input_source, str) and self.input_source.isdigit():
            camera_index = int(self.input_source)
            logger.info(f"카메라 인덱스 {camera_index} 연결 시도...")
            self.cap = cv2.VideoCapture(camera_index)
            
            # 카메라 연결 실패 시 다른 카메라 시도
            if not self.cap.isOpened():
                logger.warning(f"카메라 인덱스 {camera_index} 연결 실패, 다른 카메라 검색 중...")
                available_cameras = self._find_available_cameras()
                
                # 사용 가능한 카메라가 있으면 첫 번째 카메라 사용
                if available_cameras:
                    camera_index = available_cameras[0]
                    logger.info(f"카메라 인덱스 {camera_index} 사용")
                    self.cap = cv2.VideoCapture(camera_index)
                    self.input_source = camera_index
        
        # 2. 파일 경로인 경우
        elif isinstance(self.input_source, str) and (os.path.exists(self.input_source) or self.input_source.startswith('rtsp://')):
            logger.info(f"비디오 파일 또는 스트림 열기: {self.input_source}")
            self.cap = cv2.VideoCapture(self.input_source)
        
        # 3. 정수 인덱스인 경우
        elif isinstance(self.input_source, int):
            logger.info(f"카메라 인덱스 {self.input_source} 연결 시도...")
            self.cap = cv2.VideoCapture(self.input_source)
        
        # 4. 위 방법이 모두 실패한 경우 사용 가능한 카메라 검색
        if self.cap is None or not self.cap.isOpened():
            logger.warning(f"입력 소스 {self.input_source}를 열 수 없습니다. 사용 가능한 카메라 검색 중...")
            available_cameras = self._find_available_cameras()
            
            if available_cameras:
                camera_index = available_cameras[0]
                logger.info(f"첫 번째 사용 가능한 카메라 인덱스 {camera_index} 사용")
                self.cap = cv2.VideoCapture(camera_index)
                self.input_source = camera_index
            else:
                # 기본 테스트 이미지 생성 (파란색 화면)
                logger.warning("사용 가능한 카메라 또는 비디오 파일을 찾을 수 없습니다. 테스트 이미지를 사용합니다.")
                self._use_test_image()
                return
        
        # 비디오 속성 확인
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        logger.info(f"입력 소스 설정 완료: {self.input_source}, 크기: {self.width}x{self.height}, FPS: {self.fps}")
    
    def _use_test_image(self):
        """테스트 이미지 모드 활성화"""
        self.width = 640
        self.height = 480
        self.fps = 30
        self._test_image_mode = True
        logger.info(f"테스트 이미지 모드 활성화: 크기: {self.width}x{self.height}, FPS: {self.fps}")
    
    def _get_test_frame(self):
        """테스트 프레임 생성"""
        # 파란색 배경에 날짜/시간 텍스트가 있는 이미지 생성
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame[:, :] = (255, 0, 0)  # 파란색 배경
        
        # 현재 시간 텍스트 추가
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(
            frame, timestamp, (50, self.height//2), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
        )
        
        # 정보 텍스트 추가
        cv2.putText(
            frame, "테스트 이미지 모드", (50, self.height//2 + 50), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
        )
        
        return True, frame
    
    def process_frames(self):
        """카메라/비디오에서 프레임을 읽어 각 모델별로 전처리 후 각 모델의 입력 큐에 넣는 메서드"""
        logger.info("InputProcessor.process_frames 시작")
        
        # 통계 변수 초기화
        frame_count = 0
        skipped_frames = 0
        start_time = time.time()
        last_log_time = start_time
        
        # 비디오 파일 반복 횟수 카운터 (비디오 파일 입력 시)
        video_repeat_count = 0
        
        try:
            # 모든 모델이 활성 상태이고 종료 신호가 없는 동안 반복
            while not self.stop_event.is_set():
                # 프레임 읽기 시작 시간
                frame_read_start = time.time()
                
                # 비디오 캡처에서 프레임 읽기
                ret, frame = self.cap.read()
                
                # 프레임 읽기 종료 시간
                frame_read_time = time.time() - frame_read_start
                
                # 프레임을 읽을 수 없는 경우 처리
                if not ret:
                    # 비디오 파일인 경우 다시 시작
                    if self.input_source != 0 and os.path.isfile(str(self.input_source)):
                        logger.info(f"비디오 파일 끝에 도달. {self.video_repeat_limit}회 중 {video_repeat_count+1}회 반복")
                        video_repeat_count += 1
                        
                        # 최대 반복 횟수 확인
                        if video_repeat_count >= self.video_repeat_limit:
                            logger.info(f"비디오 반복 한도({self.video_repeat_limit}회)에 도달. 종료합니다.")
                            break
                        
                        # 비디오 캡처 재설정
                        self.cap.release()
                        self.cap = cv2.VideoCapture(self.input_source)
                        continue
                    else:
                        # 카메라에서 프레임을 읽을 수 없는 경우
                        logger.error("프레임을 읽을 수 없습니다. 종료합니다.")
                        break
                
                # 프레임 카운터 증가
                frame_count += 1
                
                # 프레임 전처리 시작 시간
                process_start = time.time()
                
                # 각 모델에 대해 프레임 처리
                for model in self.models:
                    model_name = model.__class__.__name__
                    model_type = None
                    
                    # 모델 타입 확인 (필요에 따라 모델 이름에서 추정)
                    if hasattr(model, 'model_type'):
                        model_type = model.model_type
                    elif "ObjectDetection" in model_name:
                        model_type = ModelType.OBJECT_DETECTION
                    elif "PoseEstimation" in model_name:
                        model_type = ModelType.POSE_ESTIMATION
                    else:
                        logger.warning(f"모델 {model_name}의 타입을 결정할 수 없습니다. 기본값으로 처리합니다.")
                    
                    try:
                        # 모델에 맞게 프레임 전처리
                        preprocess_start = time.time()
                        
                        # 프레임 복사 및 전처리
                        processed_frame = frame.copy()  # 항상 복사본 사용
                        
                        # 모델별 전처리 함수 호출 (모델 객체에 전처리 함수가 있는 경우)
                        if hasattr(model, 'preprocess') and callable(model.preprocess):
                            processed_frame = model.preprocess(processed_frame)
                            logger.debug(f"모델 {model_name}의 preprocess 메서드 사용하여 전처리 완료")
                        
                        # 일반적인 전처리 적용 (모델 타입에 따른 기본 전처리)
                        else:
                            # 객체 탐지 모델 전처리
                            if model_type == ModelType.OBJECT_DETECTION:
                                # 1. 크기 조정
                                if hasattr(model, 'input_shape'):
                                    target_size = model.input_shape[:2]  # (height, width)
                                else:
                                    target_size = (300, 300)  # 기본 크기
                                
                                processed_frame = cv2.resize(processed_frame, (target_size[1], target_size[0]))
                                
                                # 2. 정규화
                                processed_frame = processed_frame.astype(np.float32) / 255.0
                                
                            # 포즈 추정 모델 전처리
                            elif model_type == ModelType.POSE_ESTIMATION:
                                # 1. 크기 조정
                                if hasattr(model, 'input_shape'):
                                    target_size = model.input_shape[:2]  # (height, width)
                                else:
                                    target_size = (256, 256)  # 기본 크기
                                
                                processed_frame = cv2.resize(processed_frame, (target_size[1], target_size[0]))
                                
                                # 2. 정규화
                                processed_frame = processed_frame.astype(np.float32) / 255.0
                            
                            logger.debug(f"모델 {model_name}에 기본 전처리 적용")
                        
                        preprocess_time = time.time() - preprocess_start
                        
                        # 모델 입력 데이터 생성
                        input_data = ModelInputData(
                            frame_id=frame_count,
                            original_frame=frame,
                            processed_frame=processed_frame,
                            model_type=model_type,
                            source_id=self.input_source_id
                        )
                        
                        # 입력 큐에 데이터 넣기
                        queue_put_start = time.time()
                        
                        # 모델 입력 큐가 없으면 생성
                        if not hasattr(model, 'input_queue'):
                            logger.warning(f"모델 {model_name}에 input_queue 속성이 없습니다. 큐를 생성합니다.")
                            model.input_queue = queue.Queue(maxsize=20)
                        
                        # 큐가 가득 찼는지 확인
                        if model.input_queue.full():
                            logger.warning(f"모델 {model_name}의 입력 큐가 가득 찼습니다. 프레임 {frame_count} 건너뜁니다.")
                            skipped_frames += 1
                            continue
                        
                        # 큐에 데이터 넣기
                        model.input_queue.put(input_data, block=False)
                        
                        queue_put_time = time.time() - queue_put_start
                        
                        logger.debug(f"모델 {model_name}의 입력 큐에 프레임 {frame_count} 추가 (전처리: {preprocess_time*1000:.2f}ms, 큐 삽입: {queue_put_time*1000:.2f}ms)")
                        
                    except queue.Full:
                        logger.warning(f"모델 {model_name} 입력 큐가 가득 찼습니다. 프레임 {frame_count} 건너뜁니다.")
                        skipped_frames += 1
                    except Exception as e:
                        logger.error(f"모델 {model_name} 프레임 처리 중 오류: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                        skipped_frames += 1
                
                # 전체 프레임 처리 시간
                process_time = time.time() - process_start
                
                # 성능 로깅 (5초마다)
                current_time = time.time()
                if current_time - last_log_time >= 5.0:
                    elapsed = current_time - start_time
                    fps = frame_count / elapsed if elapsed > 0 else 0
                    
                    # 각 모델 큐 상태 로깅
                    queue_status = []
                    for model in self.models:
                        if hasattr(model, 'input_queue'):
                            q_size = model.input_queue.qsize() if hasattr(model.input_queue, 'qsize') else "N/A"
                            q_max = model.input_queue._maxsize
                            queue_status.append(f"{model.__class__.__name__}: {q_size}/{q_max}")
                    
                    queue_info = ", ".join(queue_status)
                    
                    logger.info(f"입력 처리 성능: {fps:.2f} FPS (처리: {frame_count}프레임, 건너뜀: {skipped_frames}프레임)")
                    logger.info(f"입력 큐 상태: {queue_info}")
                    
                    last_log_time = current_time
                
                # 프레임 레이트 제한 (resource_friendly 모드에서만)
                if self.resource_friendly:
                    frame_processing_time = time.time() - frame_read_start
                    sleep_time = max(0, 1.0/self.target_fps - frame_processing_time)
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                
                # 멈춤 이벤트 확인 (사용자 입력 'q' 또는 외부 중단 신호)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("사용자가 'q' 키를 눌러 종료합니다.")
                    self.stop_event.set()
                    break
            
            # 프로세스 종료 처리
            logger.info(f"입력 처리 완료: 총 {frame_count}프레임 처리, {skipped_frames}프레임 건너뜀")
            
            # 모든 모델 입력 큐에 종료 신호 전송
            for model in self.models:
                if hasattr(model, 'input_queue'):
                    try:
                        logger.info(f"모델 {model.__class__.__name__}의 입력 큐에 종료 신호(None) 전송")
                        model.input_queue.put(None, block=False)
                    except queue.Full:
                        logger.warning(f"모델 {model.__class__.__name__}의 입력 큐가 가득 차서 종료 신호를 보낼 수 없습니다.")
                        # 강제로 큐에 종료 신호 넣기 시도
                        try:
                            model.input_queue.put(None, block=True, timeout=1.0)
                            logger.info(f"모델 {model.__class__.__name__}의 입력 큐에 종료 신호 강제 전송 성공")
                        except:
                            logger.error(f"모델 {model.__class__.__name__}의 입력 큐에 종료 신호 강제 전송 실패")
                else:
                    logger.warning(f"모델 {model.__class__.__name__}에 input_queue 속성이 없습니다.")
        
        except Exception as e:
            logger.error(f"프레임 처리 중 예외 발생: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        finally:
            # 비디오 캡처 자원 해제
            if self.cap is not None:
                self.cap.release()
                logger.info("비디오 캡처 자원 해제 완료")
            
            logger.info("InputProcessor.process_frames 종료")
    
    def get_input_queue(self):
        """입력 큐 반환 메서드 (외부에서 접근용)"""
        return self.input_queue 