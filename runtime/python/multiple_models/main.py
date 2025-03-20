#!/usr/bin/env python3

import os
import sys
import argparse
import queue
from pathlib import Path
from loguru import logger
import multiprocessing as mp
from multiprocessing import Queue, Event, Value, Process
import cv2
import time
from typing import List, Tuple, Optional, Dict, Any, NamedTuple
import numpy as np
from datetime import datetime
from enum import Enum
import signal

# 현재 디렉토리와 부모 디렉토리 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))

# 공통 모듈 임포트
# from commons.utils import setup_logging
from commons.model_types import ModelType, ModelInputData, ModelOutputData

# 커스텀 모듈 임포트 (multiple_models/commons 폴더의 모듈)
from commons.input_processor import InputProcessor
from commons.model_inference import ModelInference
from commons.pose_estimation import PoseEstimation
from commons.object_detection import ObjectDetection
from commons.output_processor import OutputProcessor

def setup_logger():
    """로그 설정 함수"""
    # 로그 디렉토리 생성
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # 현재 시간으로 로그 파일명 생성
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'inference_{current_time}.log'
    
    # 기존 로거 제거
    logger.remove()
    
    # 콘솔 출력 설정
    logger.add(sys.stderr, level="DEBUG")
    
    # 파일 출력 설정
    logger.add(
        str(log_file),
        rotation="500 MB",
        retention="10 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    )
    
    logger.info(f"로그 파일 생성: {log_file}")
    return log_file

def process_input_frames(input_processor: InputProcessor):
    """입력 프레임을 처리하는 별도의 프로세스 함수"""
    try:
        logger.info("입력 프레임 처리 프로세스 시작")
        
        # 오류 방지를 위한 추가 검사
        if input_processor is None:
            logger.error("입력 프로세서가 None입니다.")
            return
        
        if input_processor.stop_event is None:
            logger.error("stop_event가 초기화되지 않았습니다.")
            input_processor.stop_event = Event()  # 초기화되지 않은 경우 생성
            
        if input_processor.cap is None and not hasattr(input_processor, '_test_image_mode'):
            logger.error("카메라/비디오가 초기화되지 않았거나 열 수 없습니다.")
            return
            
        input_processor.process_frames()
    except Exception as e:
        import traceback
        logger.error(f"입력 프레임 처리 중 오류 발생: {str(e)}")
        logger.error(f"상세 오류: {traceback.format_exc()}")
    finally:
        logger.info("입력 프레임 처리 프로세스 종료")

def run_model_process_frames(model: ModelInference):
    """모델의 process_frames 메서드를 실행하는 함수"""
    try:
        logger.info(f"{model.__class__.__name__} 프로세스 시작")
        model.process_frames()
        logger.info(f"{model.__class__.__name__} 프로세스 정상 종료")
    except Exception as e:
        import traceback
        logger.error(f"{model.__class__.__name__} 프로세스 실행 중 오류 발생: {e}")
        logger.error(f"상세 오류: {traceback.format_exc()}")

def process_models(
    models: List[ModelInference],
    input_queue: Queue,
    output_queue: Queue,
    stop_event: Event,
    fps_shared: Value
):
    """모델들을 병렬로 처리하는 프로세스"""
    logger.info("모델 처리 프로세스 시작")
    logger.info(f"처리할 모델 수: {len(models)}")
    
    # 각 모델의 처리 프로세스 시작
    model_processes = []
    for model in models:
        process = mp.Process(
            target=run_model_process_frames,
            args=(model,)
        )
        process.start()
        model_processes.append(process)
        logger.info(f"{model.__class__.__name__} 처리 프로세스 시작됨 (PID: {process.pid})")
    
    frame_count = 0
    start_time = time.time()
    last_fps_log_time = time.time()
    
    while not stop_event.is_set():
        try:
            # 원본 프레임 받기
            frame = input_queue.get(timeout=60.0)
            if frame is None:
                logger.info("종료 신호 수신")
                break

            # 각 모델에 프레임 전달
            model_outputs = []
            for model in models:
                try:
                    # 모델의 입력 큐에 프레임 전달
                    model.input_queue.put(frame, timeout=5.0)
                    logger.debug(f"{model.__class__.__name__}에 프레임 전달 완료")
                except Exception as e:
                    logger.error(f"{model.__class__.__name__}에 프레임 전달 실패: {e}")
            
            # 결과 수집
            results = []
            for i, model in enumerate(models):
                try:
                    logger.debug(f"모델 {i+1} ({model.__class__.__name__}) 결과 대기 중...")
                    get_start = time.time()
                    result = model.output_queue.get(timeout=1.0)
                    get_time = time.time() - get_start
                    
                    if result is not None:
                        # 결과 형식 처리
                        if isinstance(result, tuple):
                            # 모델 식별자가 포함된 경우 (model_type, original_frame, inference_result)
                            if len(result) == 3:
                                model_type, original_frame, inference_result = result
                                logger.debug(f"모델 {i+1} ({model.__class__.__name__}) 결과에 모델 식별자 {model_type} 포함됨")
                                # 후처리 수행
                                processed_result = model.postprocess(original_frame, inference_result)
                            # 모델 식별자가 없는 경우 (original_frame, inference_result)
                            elif len(result) == 2:
                                original_frame, inference_result = result
                                # 후처리 수행
                                processed_result = model.postprocess(original_frame, inference_result)
                            else:
                                logger.warning(f"모델 {i+1} ({model.__class__.__name__}) 결과 형식이 예상과 다릅니다: {len(result)}개 요소")
                                continue
                        else:
                            # 단일 결과값인 경우 (이전 버전 호환성)
                            processed_result = result
                            
                        results.append(processed_result)
                        logger.debug(f"모델 {i+1} ({model.__class__.__name__}) 결과 수신 완료 (소요 시간: {get_time*1000:.2f}ms)")
                        
                        # 결과 데이터 검증
                        if isinstance(processed_result, np.ndarray):
                            logger.debug(f"모델 {i+1} 결과: 크기={processed_result.shape}, 타입={processed_result.dtype}, 값 범위=[{np.min(processed_result)}, {np.max(processed_result)}]")
                        else:
                            logger.debug(f"모델 {i+1} 결과: 타입={type(processed_result)}")
                except queue.Empty:
                    logger.debug(f"모델 {i+1} ({model.__class__.__name__}) 결과 타임아웃 (1초)")
                    continue
                except Exception as e:
                    logger.error(f"모델 {i+1} ({model.__class__.__name__}) 결과 처리 중 오류: {e}")
            
            # 결과 전송
            output_queue.put(results)
            
            # FPS 계산
            frame_count += 1
            current_time = time.time()
            if current_time - last_fps_log_time >= 5.0:
                fps = frame_count / (current_time - start_time)
                with fps_shared.get_lock():
                    fps_shared.value = fps
                logger.info(f"FPS: {fps:.2f}")
                last_fps_log_time = current_time
                
        except queue.Empty:
            logger.debug("입력 큐 타임아웃")
        except Exception as e:
            logger.error(f"모델 처리 중 오류: {e}")
    
    # 종료 처리
    logger.info("모델 처리 프로세스 종료 중...")
    
    # 모델 프로세스 종료 대기
    for i, process in enumerate(model_processes):
        process.join(timeout=5)  # 5초 타임아웃
        if process.is_alive():
            logger.warning(f"모델 프로세스 {i+1}이 응답하지 않아 강제 종료합니다")
            process.terminate()
        logger.debug(f"모델 프로세스 {i+1} 종료됨")
    
    logger.info("모든 모델 처리 프로세스 종료됨")

def handle_interrupt(sig, frame, stop_event, processes=None):
    """인터럽트 시그널(Ctrl+C, SIGTERM 등) 처리 함수"""
    logger.info(f"인터럽트 시그널 ({sig}) 받음, 정상 종료를 시작합니다.")
    
    # 종료 이벤트 설정
    stop_event.set()
    
    # 프로세스 목록이 제공된 경우 모든 프로세스에 종료 신호 전송
    if processes:
        logger.info(f"실행 중인 {len(processes)} 프로세스에 종료 신호 전송 중...")
        for p in processes:
            if p.is_alive():
                try:
                    logger.debug(f"프로세스 '{p.name}' (PID: {p.pid})에 종료 신호 전송")
                    p.terminate()
                except Exception as e:
                    logger.error(f"프로세스 '{p.name}' 종료 신호 전송 중 오류: {e}")
    
    logger.info("종료 신호 전송 완료, 정상 종료 대기 중...")
    
    # 메인 스레드가 안전하게 종료 처리할 수 있도록 약간의 시간 제공
    time.sleep(0.5)

def model_inference_process(model, input_queue, output_queue, stop_event):
    """개별 모델의 추론 프로세스 - 각 모델은 자체 입력/출력 큐를 가집니다"""
    logger.info(f"{model.__class__.__name__} 모델 추론 프로세스 시작 (PID: {os.getpid()})")
    
    try:
        # 모델에 큐 설정
        model.input_queue = input_queue
        model.output_queue = output_queue
        
        # 모델 추론 프로세스 실행
        model.process_frames(input_queue, output_queue, stop_event)
        
        logger.info(f"{model.__class__.__name__} 모델 추론 프로세스 정상 종료")
    except Exception as e:
        import traceback
        logger.error(f"{model.__class__.__name__} 모델 추론 중 오류 발생: {e}")
        logger.error(f"상세 오류: {traceback.format_exc()}")
    finally:
        # 종료 신호 전송
        try:
            output_queue.put(None, block=False)
        except:
            pass
        logger.info(f"{model.__class__.__name__} 모델 추론 프로세스 종료")

def main(args):
    """메인 함수"""
    # 로거 설정
    log_file = setup_logger()
    logger.info("=" * 80)
    logger.info("다중 모델 추론 애플리케이션 시작")
    logger.info(f"Python 버전: {sys.version}")
    logger.info(f"OpenCV 버전: {cv2.__version__}")
    logger.info(f"NumPy 버전: {np.__version__}")
    logger.info(f"로그 파일: {log_file}")
    logger.info("-" * 80)
    
    # 입력 인자 확인
    logger.info(f"입력 소스: {args.input_source}")
    logger.info(f"모델 경로: {args.models}")
    logger.info(f"배치 크기: {args.batch_size}")
    logger.info(f"출력 경로: {args.output_path}")
    logger.info(f"장치 ID: {args.device_id}")
    logger.info("-" * 80)
    
    # 종료 이벤트 생성
    stop_event = Event()
    
    # 프로세스 목록 초기화
    processes = []

    # SIGINT(Ctrl+C), SIGTERM 등에 대한 핸들러 등록은 프로세스 생성 후 아래에서 수행

    # 최종 출력 큐 생성
    final_output_queue = Queue(maxsize=100)
    
    # FPS 측정용 공유 변수
    fps_shared = Value('d', 0.0)
    
    # 실행 경로 기준으로 모델 경로 보정
    models_path = []
    for model_path in args.models:
        if not os.path.isabs(model_path):
            model_path = os.path.join(current_dir, model_path)
        models_path.append(model_path)
    
    # 모델 객체 생성
    models = []
    for model_path in models_path:
        model_name = os.path.basename(model_path)
        try:
            logger.info(f"모델 로딩 중: {model_name}")
            
            # 모델 유형에 따라 적절한 클래스 선택
            if "pose" in model_name.lower():
                model = PoseEstimation(
                    net_path=model_path, 
                    labels_path="resources/coco.txt",
                    input_queue=Queue(maxsize=20),
                    output_queue=Queue(maxsize=20),
                    stop_event=stop_event,
                    batch_size=args.batch_size
                )
                logger.info(f"포즈 추정 모델 생성됨: {model_name}")
            elif "object" in model_name.lower() or "detection" in model_name.lower():
                model = ObjectDetection(
                    net_path=model_path, 
                    labels_path="resources/coco.txt",
                    input_queue=Queue(maxsize=20),
                    output_queue=Queue(maxsize=20),
                    stop_event=stop_event,
                    batch_size=args.batch_size
                )
                logger.info(f"객체 인식 모델 생성됨: {model_name}")
            else:
                logger.warning(f"알 수 없는 모델 유형, 지원되지 않습니다: {model_name}")
                continue
            
            models.append(model)
            logger.info(f"{model.__class__.__name__} 모델 생성 및 큐 설정 완료")
        except Exception as e:
            logger.error(f"모델 '{model_name}' 초기화 중 오류 발생: {e}")
            import traceback
            logger.error(f"상세 오류: {traceback.format_exc()}")
    
    if not models:
        logger.error("유효한 모델이 없습니다. 종료합니다.")
        return
    
    # 입력 처리기 생성
    try:
        input_processor = InputProcessor(
            input_source=args.input_source,
            models=models,
            batch_size=args.batch_size,
            stop_event=stop_event
        )
        logger.info("입력 처리기 생성 완료")
    except Exception as e:
        logger.error(f"입력 처리기 초기화 중 오류 발생: {e}")
        import traceback
        logger.error(f"상세 오류: {traceback.format_exc()}")
        return
    
    # 출력 처리기 생성
    try:
        output_processor = OutputProcessor(
            models=models,
            output_path=args.output_path,
            stop_event=stop_event
        )
        logger.info("출력 처리기 생성 완료")
    except Exception as e:
        logger.error(f"출력 처리기 초기화 중 오류 발생: {e}")
        import traceback
        logger.error(f"상세 오류: {traceback.format_exc()}")
        return
    
    # 프로세스 시작
    processes = []
    
    # 입력 프로세스 시작
    try:
        input_process = Process(
            target=input_processor.process_frames,
            name="InputProcess"
        )
        input_process.start()
        processes.append(input_process)
        logger.info(f"입력 프로세스 시작됨 (PID: {input_process.pid})")
    except Exception as e:
        logger.error(f"입력 프로세스 시작 중 오류: {e}")
        stop_event.set()
        return
    
    # 각 모델별 추론 프로세스 시작
    for i, model in enumerate(models):
        try:
            model_process = Process(
                target=model_inference_process,
                args=(model, model.input_queue, model.output_queue, stop_event),
                name=f"ModelProcess_{model.__class__.__name__}"
            )
            model_process.start()
            processes.append(model_process)
            logger.info(f"모델 추론 프로세스 {i+1} 시작됨 (PID: {model_process.pid}, 모델: {model.__class__.__name__})")
        except Exception as e:
            logger.error(f"모델 추론 프로세스 {i+1} 시작 중 오류: {e}")
            import traceback
            logger.error(f"상세 오류: {traceback.format_exc()}")
    
    # 출력 프로세스 시작
    try:
        output_process = Process(
            target=output_processor.process_results,
            name="OutputProcess"
        )
        output_process.start()
        processes.append(output_process)
        logger.info(f"출력 프로세스 시작됨 (PID: {output_process.pid})")
    except Exception as e:
        logger.error(f"출력 프로세스 시작 중 오류: {e}")
        import traceback
        logger.error(f"상세 오류: {traceback.format_exc()}")
    
    # 모든 프로세스가 시작된 후 시그널 핸들러 등록 (프로세스 목록 전달)
    signal.signal(signal.SIGINT, lambda sig, frame: handle_interrupt(sig, frame, stop_event, processes))
    signal.signal(signal.SIGTERM, lambda sig, frame: handle_interrupt(sig, frame, stop_event, processes))
    logger.info("시그널 핸들러 등록 완료 (Ctrl+C 및 SIGTERM 처리)")
    
    # 메인 루프 - 프로세스 모니터링
    try:
        logger.info("모든 프로세스 시작 완료, 모니터링 중...")
        
        while not stop_event.is_set() and any(p.is_alive() for p in processes):
            time.sleep(0.1)
            
            # 모든 프로세스가 종료된 경우
            if not any(p.is_alive() for p in processes):
                logger.info("모든 프로세스가 종료되었습니다.")
                break
            
            # 비정상 종료된 프로세스 확인
            for p in processes:
                if not p.is_alive() and p.exitcode != 0:
                    logger.warning(f"프로세스 '{p.name}' (PID: {p.pid})가 비정상 종료되었습니다 (exitcode: {p.exitcode})")
            
            # 5초마다 상태 로깅
            current_time = time.time()
            if hasattr(main, 'last_status_time') and current_time - main.last_status_time >= 5.0:
                # 실행 중인 프로세스 확인
                alive_processes = [p.name for p in processes if p.is_alive()]
                logger.info(f"실행 중인 프로세스: {len(alive_processes)}/{len(processes)} - {alive_processes}")
                
                # FPS 로깅
                logger.info(f"현재 처리 속도: {fps_shared.value:.2f} FPS")
                
                main.last_status_time = current_time
            elif not hasattr(main, 'last_status_time'):
                main.last_status_time = current_time
                
    except KeyboardInterrupt:
        logger.info("Ctrl+C 감지, 정상 종료를 시작합니다.")
        stop_event.set()
    except Exception as e:
        logger.error(f"메인 루프 실행 중 오류 발생: {e}")
        import traceback
        logger.error(f"상세 오류: {traceback.format_exc()}")
        stop_event.set()
    
    # 종료 처리
    logger.info("프로세스 종료 중...")
    
    # 종료 대기 시작 시간
    termination_start = time.time()
    termination_timeout = 5.0  # 종료 대기 시간(초)
    
    # 각 프로세스 종료 대기
    while time.time() - termination_start < termination_timeout and any(p.is_alive() for p in processes):
        # 남은 프로세스 수 확인
        alive_count = sum(1 for p in processes if p.is_alive())
        if alive_count > 0:
            logger.info(f"아직 {alive_count}개 프로세스가 실행 중입니다. 대기 중...")
            time.sleep(1.0)
    
    # 시간 초과 후에도 종료되지 않은 프로세스 강제 종료
    for p in processes:
        if p.is_alive():
            logger.warning(f"프로세스 '{p.name}' (PID: {p.pid})가 응답하지 않아 강제 종료합니다.")
            try:
                # 먼저 terminate 시도
                p.terminate()
                p.join(timeout=1.0)
                
                # 여전히 살아있으면 강력한 종료 시도 (Python 3.7 이상)
                if p.is_alive() and hasattr(p, 'kill'):
                    logger.warning(f"프로세스 '{p.name}' (PID: {p.pid})를 kill()로 종료 시도.")
                    p.kill()
                    p.join(timeout=1.0)
                
                # 그래도 종료되지 않으면 OS 레벨에서 처리
                if p.is_alive():
                    logger.warning(f"프로세스 '{p.name}' (PID: {p.pid})를 OS kill 명령으로 종료 시도.")
                    import subprocess
                    try:
                        # Linux/macOS에서 SIGKILL로 강제 종료
                        subprocess.run(['kill', '-9', str(p.pid)], check=True)
                        logger.info(f"프로세스 '{p.name}' (PID: {p.pid}) OS kill 명령 실행됨.")
                        # 종료 확인을 위해 약간 대기
                        for _ in range(5):  # 최대 0.5초 대기
                            time.sleep(0.1)
                            if not p.is_alive():
                                logger.info(f"프로세스 '{p.name}' (PID: {p.pid}) 성공적으로 종료됨.")
                                break
                    except Exception as e:
                        logger.error(f"OS kill 명령 실행 중 오류: {e}")
                
                # 마지막으로 한번 더 확인
                if p.is_alive():
                    logger.error(f"프로세스 '{p.name}' (PID: {p.pid})를 모든 방법으로 종료 시도했으나 실패!")
                    # 수동 종료 명령어 안내
                    logger.error("프로세스를 수동으로 종료하려면 다음 명령을 실행하세요:")
                    logger.error(f"kill -9 {p.pid}")
            except Exception as e:
                logger.error(f"프로세스 '{p.name}' (PID: {p.pid}) 종료 중 예외 발생: {e}")
                import traceback
                logger.error(traceback.format_exc())

    # 모든 프로세스가 정상 종료되었는지 최종 확인
    alive_processes = [p for p in processes if p.is_alive()]
    if alive_processes:
        logger.warning(f"다음 {len(alive_processes)}개 프로세스가 여전히 실행 중입니다:")
        for p in alive_processes:
            logger.warning(f"  - {p.name} (PID: {p.pid})")
        
        # 사용자에게 수동 종료 방법 안내
        logger.warning("터미널에서 다음 명령을 실행하여 남은 프로세스를 강제 종료하세요:")
        for p in alive_processes:
            logger.warning(f"  kill -9 {p.pid}")
    else:
        logger.info("모든 프로세스가 성공적으로 종료되었습니다.")

    # 종료 요약 정보
    logger.info("=" * 80)
    logger.info("애플리케이션 종료 요약:")
    logger.info(f"- 총 프로세스 수: {len(processes)}")
    logger.info(f"- 정상 종료된 프로세스 수: {len(processes) - len(alive_processes)}")
    logger.info(f"- 비정상 종료된 프로세스 수: {len(alive_processes)}")
    logger.info("=" * 80)
    logger.info("애플리케이션 종료")
    logger.info("=" * 80)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='다중 모델 추론 애플리케이션')
    parser.add_argument('--models', nargs='+', required=False, 
                       default=["resources/yolov8s_pose_h8l.hef", "resources/yolov11s_object_h8l.hef"],
                       help='모델 파일 경로 목록 (.hef 파일, 기본값: resources/yolov8s_pose_h8l.hef, resources/yolov11s_object_h8l.hef)')
    parser.add_argument('--input_source', type=str, default="0",
                       help='입력 소스 (카메라 인덱스 또는 비디오 파일 경로, 기본값: 0)')
    parser.add_argument('--output_path', type=str, default="output",
                       help='출력 저장 경로 (기본값: output)')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='배치 크기 (기본값: 1)')
    parser.add_argument('--device_id', type=int, default=0,
                       help='Hailo 장치 ID (기본값: 0)')
    args = parser.parse_args()
    
    # 출력 폴더 생성
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        logger.info(f"출력 폴더 생성: {args.output_path}")
    
    main(args)
