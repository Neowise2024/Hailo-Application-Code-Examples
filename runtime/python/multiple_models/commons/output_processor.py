#!/usr/bin/env python3

import os
import sys
import cv2
import time
import queue
import numpy as np
from multiprocessing import Queue, Event
from loguru import logger
from datetime import datetime


# 상위 디렉토리 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

# 필요한 모듈 임포트
from commons.model_types import ModelType, ModelOutputData

class OutputProcessor:
    """출력 결과를 처리하는 클래스"""
    
    def __init__(self, models=None, output_path=None, stop_event=None, use_gui=True):
        self.models = models or []
        self.output_path = output_path
        self.stop_event = stop_event or Event()
        self.results_buffer = {}  # 프레임 ID별 결과 저장을 위한 버퍼
        self.processed_frames = 0
        self.use_gui = use_gui  # GUI 사용 여부 플래그 추가
        # 최대 처리 관련 설정
        self.max_frame_window = 30  # 최대 프레임 윈도우 크기
        self.max_pending_frames = 100  # 최대 대기 프레임 수
        logger.info(f"OutputProcessor 초기화: 모델 수={len(self.models)}, 출력 경로={output_path}, GUI 사용={use_gui}")
    
    def process_results(self):
        """모델 출력 큐에서 결과를 가져와 프레임 ID별로 정렬하여 화면에 표시하는 메서드"""
        logger.info("OutputProcessor.process_results 시작")
        
        # 통계 변수
        processed_frames = 0
        skipped_frames = 0
        start_time = time.time()
        last_log_time = start_time
        
        # 각 모델의 출력에 대한 트래킹
        last_frame_id = 0
        pending_results = {}  # frame_id별로 각 모델의 결과를 저장하는 딕셔너리
        model_states = {model.__class__.__name__: {"active": True, "processed": 0} for model in self.models}
        
        try:
            # 결과 처리 UI 윈도우 생성
            if self.use_gui:
                self._setup_display_window()
                logger.info("결과 표시 윈도우 생성됨")
            
            # 모든 모델이 종료될 때까지 또는 스탑 이벤트가 설정될 때까지 반복
            while any(state["active"] for state in model_states.values()) and not self.stop_event.is_set():
                # 모든 모델의 출력 큐 확인
                for model in self.models:
                    model_name = model.__class__.__name__
                    
                    # 이미 종료된 모델은 건너뜀
                    if not model_states[model_name]["active"]:
                        continue
                    
                    try:
                        # 모델 출력 큐에서 결과 가져오기 (non-blocking)
                        try:
                            output_data = model.output_queue.get(block=False)
                        except queue.Empty:
                            # 큐가 비어있으면 계속 진행
                            continue
                        
                        # 종료 신호 확인
                        if output_data is None:
                            logger.info(f"{model_name} 모델로부터 종료 신호(None) 수신")
                            model_states[model_name]["active"] = False
                            continue
                        
                        # 프레임 ID 추출 및 결과 데이터 처리
                        frame_id = None

                        # 결과 형식에 따른 처리
                        if isinstance(output_data, ModelOutputData):
                            # ModelOutputData 객체인 경우
                            frame_id = output_data.frame_id
                            logger.debug(f"{model_name} 모델의 ModelOutputData 받음: frame_id={frame_id}")
                        elif isinstance(output_data, dict) and 'frame_id' in output_data:
                            frame_id = output_data['frame_id']
                        elif hasattr(output_data, 'get_frame_id') and callable(output_data.get_frame_id):
                            frame_id = output_data.get_frame_id()
                        else:
                            logger.warning(f"{model_name} 모델의 결과에 frame_id가 없습니다: {type(output_data)}")
                            skipped_frames += 1
                            continue
                        
                        # 모델 결과 추출
                        result = None
                        model_type = None

                        if isinstance(output_data, ModelOutputData):
                            result = output_data.results
                            model_type = output_data.model_type
                            logger.debug(f"ModelOutputData에서 결과 추출: 모델 타입={model_type}, 결과 타입={type(result)}")
                        elif hasattr(output_data, 'get_results') and callable(output_data.get_results):
                            result = output_data.get_results()
                            model_type = output_data.get_model_type() if hasattr(output_data, 'get_model_type') else None
                        elif isinstance(output_data, dict):
                            result = output_data.get('results')
                            model_type = output_data.get('model_type')
                        else:
                            logger.warning(f"알 수 없는 출력 데이터 형식: {type(output_data)}")
                            skipped_frames += 1
                            continue
                        
                        # 유효한 프레임 ID 확인
                        if frame_id is None or not isinstance(frame_id, (int, float)):
                            logger.warning(f"{model_name} 모델의 결과에 유효하지 않은 frame_id: {frame_id}")
                            skipped_frames += 1
                            continue
                        
                        # 결과 검증
                        if result is None:
                            logger.warning(f"{model_name} 모델의 결과가 None입니다")
                            skipped_frames += 1
                            continue
                        
                        # 너무 오래된 프레임 ID는 버림 (지정된 윈도우 크기보다 작은 경우)
                        if frame_id < last_frame_id - self.max_frame_window:
                            logger.warning(f"{model_name} 모델의 결과가 너무 오래됨: frame_id={frame_id}, last_frame_id={last_frame_id}")
                            skipped_frames += 1
                            continue
                        
                        # 프레임 ID별 결과 저장
                        if frame_id not in pending_results:
                            pending_results[frame_id] = {}
                        
                        # 각 모델별 결과 저장
                        pending_results[frame_id][model_name] = output_data
                        
                        # 모델 상태 업데이트
                        model_states[model_name]["processed"] += 1
                        logger.debug(f"{model_name} 모델의 frame_id {frame_id} 결과 저장됨")
                        
                    except Exception as e:
                        logger.error(f"{model_name} 모델 결과 처리 중 오류: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                
                # 완료된 프레임 결과 처리 (모든 모델의 결과가 모인 프레임)
                completed_frames = []
                for frame_id, results in pending_results.items():
                    # 모든 활성 모델의 결과가 있는지 확인
                    all_results_ready = all(
                        model.__class__.__name__ in results 
                        for model in self.models 
                        if model_states[model.__class__.__name__]["active"]
                    )
                    
                    # 모든 결과가 준비되었거나 너무 오래된 프레임인 경우 처리
                    if all_results_ready or frame_id < last_frame_id - self.max_frame_window:
                        # 결과 처리 및 화면 표시
                        self._process_frame_results(frame_id, results)
                        completed_frames.append(frame_id)
                        processed_frames += 1
                        
                        # 마지막 처리된 프레임 ID 업데이트
                        if frame_id > last_frame_id:
                            last_frame_id = frame_id
                
                # 처리 완료된 프레임 제거
                for frame_id in completed_frames:
                    del pending_results[frame_id]
                
                # 대기 중인 프레임 수가 너무 많아지는 경우, 오래된 것부터 제거
                if len(pending_results) > self.max_pending_frames:
                    oldest_frames = sorted(pending_results.keys())[:len(pending_results) - self.max_pending_frames]
                    for frame_id in oldest_frames:
                        del pending_results[frame_id]
                        logger.warning(f"대기 큐가 가득 차서 frame_id {frame_id} 결과 제거")
                        skipped_frames += 1
                
                # 성능 로깅 (5초마다)
                current_time = time.time()
                if current_time - last_log_time >= 5.0:
                    elapsed = current_time - start_time
                    fps = processed_frames / elapsed if elapsed > 0 else 0
                    
                    # 각 모델별 처리 상태 로깅
                    model_status = ", ".join([
                        f"{name}: {stats['processed']}프레임 {'(활성)' if stats['active'] else '(종료)'}"
                        for name, stats in model_states.items()
                    ])
                    
                    logger.info(f"출력 처리 성능: {fps:.2f} FPS (처리: {processed_frames}프레임, 건너뜀: {skipped_frames}프레임)")
                    logger.info(f"모델 상태: {model_status}")
                    logger.info(f"대기 중인 프레임: {len(pending_results)}개")
                    
                    last_log_time = current_time
                
                # CPU 과부하 방지를 위한 짧은 대기
                time.sleep(0.001)
                
                # 키 입력 확인 (GUI 모드에서만)
                if self.use_gui and cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("사용자가 'q' 키를 눌러 종료")
                    self.stop_event.set()
                    break
            
            logger.info(f"출력 처리 완료: 총 {processed_frames}프레임 처리, {skipped_frames}프레임 건너뜀")
            
        except Exception as e:
            logger.error(f"결과 처리 중 예외 발생: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
        finally:
            # 정리 작업
            if self.use_gui:
                cv2.destroyAllWindows()
                logger.info("출력 윈도우 정리 완료")
            
            logger.info("OutputProcessor.process_results 종료")
    
    def _process_frame_results(self, frame_id, results):
        """단일 프레임에 대한 모든 모델 결과를 처리하고 화면에 표시"""
        try:
            # 원본 프레임 가져오기 (어느 모델에서든)
            original_frame = None
            for model_name, data in results.items():
                if isinstance(data, ModelOutputData) and hasattr(data, 'original_frame'):
                    original_frame = data.original_frame
                    logger.debug(f"ModelOutputData에서 원본 프레임 추출 (모델: {model_name}, 프레임 ID: {frame_id})")
                    break
                elif hasattr(data, 'get_original_frame') and callable(data.get_original_frame):
                    original_frame = data.get_original_frame()
                    if original_frame is not None:
                        break
                elif isinstance(data, dict) and 'original_frame' in data:
                    original_frame = data['original_frame']
                    if original_frame is not None:
                        break
            
            if original_frame is None:
                logger.warning(f"프레임 ID {frame_id} 결과에 원본 프레임 없음")
                return
            
            # 결과 시각화 프레임 생성 (원본 프레임 복사)
            display_frame = original_frame.copy()
            
            # 모델별 처리 시간 기록 (로깅용)
            model_times = {}
            
            # 각 모델 결과 처리 및 시각화
            for model_name, output_data in results.items():
                try:
                    # 모델 타입 및 결과 추출
                    model_type = None
                    result = None
                    inference_time = None
                    
                    if isinstance(output_data, ModelOutputData):
                        result = output_data.results
                        model_type = output_data.model_type
                        inference_time = output_data.inference_time
                        logger.debug(f"모델 {model_name} 결과 처리 (프레임 ID: {frame_id}, 타입: {model_type})")
                    elif hasattr(output_data, 'get_results') and callable(output_data.get_results):
                        result = output_data.get_results()
                        model_type = output_data.get_model_type() if hasattr(output_data, 'get_model_type') else None
                    elif isinstance(output_data, dict):
                        result = output_data.get('results')
                        model_type = output_data.get('model_type')
                    
                    if result is None:
                        logger.warning(f"모델 {model_name}의 프레임 ID {frame_id} 결과 없음")
                        continue
                    
                    # 모델 처리 시간 기록
                    if inference_time is not None:
                        model_times[model_name] = inference_time
                    
                    # 모델 타입에 따른 시각화
                    try:
                        if model_type == ModelType.OBJECT_DETECTION or "ObjectDetection" in model_name:
                            self._visualize_object_detection(display_frame, result)
                        elif model_type == ModelType.POSE_ESTIMATION or "PoseEstimation" in model_name:
                            self._visualize_pose_estimation(display_frame, result)
                        else:
                            logger.warning(f"알 수 없는 모델 타입: {model_type}, 모델명: {model_name}")
                    except Exception as viz_error:
                        logger.error(f"모델 {model_name} 결과 시각화 중 오류: {viz_error}")
                        import traceback
                        logger.error(traceback.format_exc())
                
                except Exception as model_error:
                    logger.error(f"모델 {model_name} 결과 처리 중 오류: {model_error}")
                    import traceback
                    logger.error(traceback.format_exc())
            
            # 프레임 ID 및 처리 시간 추가
            text_lines = [f"Frame ID: {frame_id}"]
            
            # 모델별 처리 시간 표시
            for model_name, time_ms in model_times.items():
                if time_ms is not None:
                    text_lines.append(f"{model_name}: {time_ms*1000:.1f}ms")
            
            # 텍스트 정보 그리기
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            font_color = (255, 255, 255)  # 흰색
            bg_color = (0, 0, 0)  # 검은색
            
            # 텍스트 그리기
            y_offset = 20
            for line in text_lines:
                # 텍스트 크기 계산
                (text_width, text_height), _ = cv2.getTextSize(line, font, font_scale, font_thickness)
                
                # 배경 그리기
                cv2.rectangle(display_frame, (10, y_offset - text_height), (10 + text_width, y_offset + 5), bg_color, -1)
                
                # 텍스트 그리기
                cv2.putText(display_frame, line, (10, y_offset), font, font_scale, font_color, font_thickness)
                
                # 다음 줄 위치 업데이트
                y_offset += text_height + 10
            
            return display_frame
        
        except Exception as e:
            logger.error(f"프레임 결과 처리 중 오류: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _visualize_object_detection(self, frame, results):
        """객체 탐지 결과 시각화"""
        try:
            # 결과 형식에 따른 처리
            if isinstance(results, dict) and 'boxes' in results:
                boxes = results['boxes']
                labels = results.get('labels', [])
                scores = results.get('scores', [])
                
                for i, box in enumerate(boxes):
                    try:
                        if len(box) != 4:
                            logger.warning(f"박스 형식이 올바르지 않음: {box}")
                            continue
                            
                        x1, y1, x2, y2 = box
                        score = scores[i] if i < len(scores) else 0
                        label = labels[i] if i < len(labels) else ''
                        
                        # 박스 그리기
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        
                        # 라벨 및 점수 표시
                        text = f"{label}: {score:.2f}" if label else f"{score:.2f}"
                        cv2.putText(frame, text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    except Exception as box_error:
                        logger.error(f"박스 {i} 시각화 중 오류: {box_error}")
                        continue
            
            elif isinstance(results, list):
                # 리스트 형식 처리 (각 항목이 탐지 결과)
                for i, detection in enumerate(results):
                    try:
                        if isinstance(detection, dict):
                            box = detection.get('box', [])
                            label = detection.get('label', '')
                            score = detection.get('score', 0)
                            
                            if box and len(box) == 4:
                                x1, y1, x2, y2 = box
                                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                                
                                text = f"{label}: {score:.2f}" if label else f"{score:.2f}"
                                cv2.putText(frame, text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    except Exception as det_error:
                        logger.error(f"객체 {i} 시각화 중 오류: {det_error}")
                        continue
            
            else:
                logger.warning(f"지원되지 않는 객체 탐지 결과 형식: {type(results)}")
        
        except Exception as e:
            logger.error(f"객체 탐지 결과 시각화 중 오류: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _visualize_pose_estimation(self, frame, results):
        """포즈 추정 결과 시각화"""
        try:
            # 결과 형식에 따른 처리
            if isinstance(results, dict) and 'keypoints' in results:
                keypoints = results['keypoints']
                
                # 각 사람의 키포인트 처리
                for person_keypoints in keypoints:
                    # 키포인트 그리기
                    try:
                        for kp in person_keypoints:
                            if len(kp) >= 3:  # x, y, confidence
                                x, y, conf = kp[:3]
                                if conf > 0.5:  # 신뢰도 기준
                                    cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
                        
                        # 키포인트 연결선 그리기 (COCO 형식 기준)
                        # 연결선 정의
                        connections = [
                            [0, 1], [1, 3], [0, 2], [2, 4],  # 얼굴, 어깨
                            [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],  # 팔
                            [5, 11], [6, 12], [11, 12],  # 몸통
                            [11, 13], [13, 15], [12, 14], [14, 16]  # 다리
                        ]
                        
                        for conn in connections:
                            if len(conn) == 2 and max(conn) < len(person_keypoints):
                                p1 = person_keypoints[conn[0]]
                                p2 = person_keypoints[conn[1]]
                                
                                if len(p1) >= 3 and len(p2) >= 3:
                                    if p1[2] > 0.5 and p2[2] > 0.5:  # 양쪽 키포인트 모두 신뢰도가 높은 경우
                                        cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 255, 255), 2)
                    except Exception as kp_error:
                        logger.error(f"키포인트 그리기 중 오류: {kp_error}")
                        continue
            
            elif isinstance(results, list):
                # 리스트 형식 처리
                for i, person in enumerate(results):
                    try:
                        if isinstance(person, dict) and 'keypoints' in person:
                            keypoints = person['keypoints']
                            
                            # 키포인트 그리기
                            for kp in keypoints:
                                if len(kp) >= 3:
                                    x, y, conf = kp[:3]
                                    if conf > 0.5:
                                        cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
                            
                            # 연결선 그리기 (위와 동일)
                            connections = [
                                [0, 1], [1, 3], [0, 2], [2, 4],
                                [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
                                [5, 11], [6, 12], [11, 12],
                                [11, 13], [13, 15], [12, 14], [14, 16]
                            ]
                            
                            for conn in connections:
                                if len(keypoints) > max(conn):
                                    p1 = keypoints[conn[0]]
                                    p2 = keypoints[conn[1]]
                                    
                                    if len(p1) >= 3 and len(p2) >= 3:
                                        if p1[2] > 0.5 and p2[2] > 0.5:
                                            cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 255, 255), 2)
                    except Exception as person_error:
                        logger.error(f"사람 {i} 포즈 시각화 중 오류: {person_error}")
                        continue
            
            else:
                logger.warning(f"지원되지 않는 포즈 추정 결과 형식: {type(results)}")
        
        except Exception as e:
            logger.error(f"포즈 추정 결과 시각화 중 오류: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _setup_display_window(self):
        """결과 표시를 위한 디스플레이 윈도우 설정"""
        try:
            # 메인 결과 윈도우 생성
            cv2.namedWindow('결과', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('결과', 800, 600)
            logger.info("디스플레이 윈도우 설정 완료")
        except Exception as e:
            logger.error(f"디스플레이 윈도우 설정 중 오류: {e}")
            self.use_gui = False  # GUI 사용 불가 설정
            logger.warning("GUI를 사용할 수 없어 콘솔 모드로 전환합니다")

    def _process_outputs(self):
        """각 모델의 출력 큐에서 데이터를 가져와 처리"""
        logger.info("OutputProcessor._process_outputs 시작")
        
        # 통계 변수
        processed_frames = 0
        skipped_frames = 0
        
        # 프레임 ID로 인덱싱된 보류 중인 결과 딕셔너리
        # frame_id -> {model_name -> output_data}
        pending_results = {}
        
        # 모델 처리 상태 추적
        model_states = {model_name: {"processed": 0, "skipped": 0} for model_name in self.models.keys()}
        
        # 마지막으로 처리된 프레임 ID 추적
        last_frame_id = 0
        
        # 종료 이벤트가 설정될 때까지 반복
        while not self.stop_event.is_set():
            try:
                # 모든 모델에 대해 큐 확인
                for model_name, model_queue in self.input_queues.items():
                    try:
                        # 큐에서 데이터 가져오기 (블로킹 없음)
                        output_data = model_queue.get(block=False)
                        
                        # None 확인 (종료 신호)
                        if output_data is None:
                            logger.info(f"{model_name} 모델의 큐에서 종료 신호(None)를 받았습니다.")
                            continue
                        
                        # 프레임 ID 추출
                        frame_id = None
                        
                        if isinstance(output_data, ModelOutputData):
                            frame_id = output_data.frame_id
                            logger.debug(f"{model_name} 모델에서 ModelOutputData 받음: frame_id={frame_id}, model_id={output_data.model_id}")
                        elif isinstance(output_data, dict) and 'frame_id' in output_data:
                            frame_id = output_data['frame_id']
                        elif hasattr(output_data, 'get_frame_id') and callable(output_data.get_frame_id):
                            frame_id = output_data.get_frame_id()
                        else:
                            logger.warning(f"{model_name} 모델의 출력 데이터에 frame_id가 없습니다. 건너뜁니다.")
                            skipped_frames += 1
                            continue
                        
                        # 결과 추출
                        result = None
                        
                        if isinstance(output_data, ModelOutputData):
                            result = output_data.results
                        elif hasattr(output_data, 'get_results') and callable(output_data.get_results):
                            result = output_data.get_results()
                        elif isinstance(output_data, dict) and 'results' in output_data:
                            result = output_data['results']
                        else:
                            logger.warning(f"{model_name} 모델의 출력 데이터에 결과가 없습니다.")
                        
                        # 유효한 프레임 ID 확인
                        if frame_id is None or not isinstance(frame_id, (int, float)):
                            logger.warning(f"{model_name} 모델의 결과에 유효하지 않은 frame_id: {frame_id}")
                            skipped_frames += 1
                            continue
                        
                        # 결과 검증
                        if result is None:
                            logger.warning(f"{model_name} 모델의 결과가 None입니다")
                            skipped_frames += 1
                            continue
                        
                        # 너무 오래된 프레임 ID는 버림 (지정된 윈도우 크기보다 작은 경우)
                        if frame_id < last_frame_id - self.max_frame_window:
                            logger.warning(f"{model_name} 모델의 결과가 너무 오래됨: frame_id={frame_id}, last_frame_id={last_frame_id}")
                            skipped_frames += 1
                            continue
                        
                        # 프레임 ID별 결과 저장
                        if frame_id not in pending_results:
                            pending_results[frame_id] = {}
                        
                        # 각 모델별 결과 저장
                        pending_results[frame_id][model_name] = output_data
                        
                        # 모델 상태 업데이트
                        model_states[model_name]["processed"] += 1
                        logger.debug(f"{model_name} 모델의 frame_id {frame_id} 결과 저장됨")
                        
                    except queue.Empty:
                        # 큐가 비어있는 경우 (정상 상황)
                        pass
                    except Exception as e:
                        logger.error(f"{model_name} 모델의 출력 처리 중 오류: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                
                # 완료된 프레임 처리 (모든 모델의 결과가 있는 프레임)
                ready_frames = []
                
                # 완료된 프레임 찾기
                for frame_id, model_results in pending_results.items():
                    # 모든 모델의 결과가 있거나, 타임아웃(너무 오래 대기)된 경우 처리
                    all_models_ready = True if len(self.models) <= 1 else len(model_results) == len(self.models)
                    timeout_expired = frame_id < max(pending_results.keys()) - self.max_frame_delay if pending_results else False
                    
                    if all_models_ready or timeout_expired:
                        ready_frames.append(frame_id)
                        
                        # 타임아웃 경고
                        if not all_models_ready and timeout_expired:
                            missing_models = set(self.models.keys()) - set(model_results.keys())
                            logger.warning(f"프레임 ID {frame_id}의 처리 타임아웃: 누락된 모델: {missing_models}")
                
                # 정렬된 순서로 프레임 처리
                ready_frames.sort()
                
                for frame_id in ready_frames:
                    try:
                        # 비디오 녹화용 누락 프레임 처리
                        if self.output_path and self.last_processed_frame is not None:
                            # 프레임 간 간격이 너무 큰 경우 중간 프레임 채우기
                            if self.last_processed_frame_id is not None:
                                gap = frame_id - self.last_processed_frame_id
                                if gap > 1 and gap <= self.max_frame_gap:
                                    logger.debug(f"프레임 간 간격({gap}) 발견: {self.last_processed_frame_id} -> {frame_id}")
                                    for i in range(1, gap):
                                        fill_frame_id = self.last_processed_frame_id + i
                                        # 마지막 프레임으로 간격 채우기
                                        if self.video_writer is not None:
                                            self.video_writer.write(self.last_processed_frame)
                                            logger.debug(f"간격 채우기: 프레임 ID {fill_frame_id}를 마지막 프레임으로 채움")
                        
                        # 해당 프레임의 모든 모델 결과 처리
                        result_frame = self._process_frame_results(frame_id, pending_results[frame_id])
                        
                        # 결과 프레임이 있는 경우 처리
                        if result_frame is not None:
                            # 화면에 표시
                            if cv2.getWindowProperty('Output', cv2.WND_PROP_VISIBLE) >= 0:
                                cv2.imshow('Output', result_frame)
                                key = cv2.waitKey(1) & 0xFF
                                if key == 27:  # ESC 키
                                    logger.info("ESC 키 입력으로 처리 중단")
                                    self.stop_event.set()
                                    break
                            
                            # 비디오 저장
                            if self.video_writer is not None:
                                self.video_writer.write(result_frame)
                                
                            # 마지막 처리된 프레임 업데이트
                            self.last_processed_frame = result_frame
                            self.last_processed_frame_id = frame_id
                        
                        # 처리된 프레임 카운트 증가
                        processed_frames += 1
                        
                        # 마지막 처리된 프레임 ID 업데이트
                        last_frame_id = max(last_frame_id, frame_id)
                    
                    except Exception as e:
                        logger.error(f"프레임 ID {frame_id} 처리 중 오류: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                    
                    # 처리 완료된 프레임은 pending_results에서 제거
                    del pending_results[frame_id]
                
                # 주기적인 성능 로깅
                current_time = time.time()
                if current_time - self.last_log_time >= 5.0:
                    elapsed = current_time - self.start_time
                    fps = processed_frames / elapsed if elapsed > 0 else 0
                    pending_count = len(pending_results)
                    
                    # 모델별 처리 상태
                    model_stats = []
                    for model_name, stats in model_states.items():
                        model_stats.append(f"{model_name}({stats['processed']})")
                    
                    logger.info(f"처리 성능: {fps:.2f} FPS (처리: {processed_frames}프레임, 건너뜀: {skipped_frames}프레임, 대기: {pending_count}프레임)")
                    logger.info(f"모델별 처리: {', '.join(model_stats)}")
                    
                    # 대기 중인 프레임 ID 로깅
                    if pending_count > 0:
                        pending_ids = sorted(list(pending_results.keys()))
                        logger.debug(f"대기 중인 프레임 ID: {pending_ids[:10]}{'...' if len(pending_ids) > 10 else ''}")
                    
                    self.last_log_time = current_time
                
                # 잠시 대기 (CPU 부하 감소)
                time.sleep(0.001)
            
            except Exception as e:
                logger.error(f"출력 처리 루프에서 예외 발생: {e}")
                import traceback
                logger.error(traceback.format_exc()) 