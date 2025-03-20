#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path
import numpy as np
from loguru import logger
import multiprocessing
from multiprocessing import Process, Queue, Event, Value
import cv2
import time  # 시간 측정을 위해 추가
from hailo_platform import HEF
from commons.object_detection_utils import ObjectDetectionUtils

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import HailoAsyncInference, load_images_opencv, validate_images, divide_list_to_batches

# 전역 FPS 변수
fps_counter = 0
fps_time = time.time()
fps_value = 0.0

def parse_args() -> argparse.Namespace:
    """
    Initialize argument parser for the script.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Object Detection Example")
    parser.add_argument(
        "-n", "--net", 
        help="Path for the network in HEF format.",
        default="resources/yolov11s_h8l.hef"
    )
    parser.add_argument(
        "-i", "--input", 
        default="camera",
        help="Path to the input - either an image, a folder of images, or 'camera' for webcam input."
    )
    parser.add_argument(
        "--video-device",
        help="Specific video device path (e.g. /dev/video0, /dev/video1)",
        type=str
    )
    parser.add_argument(
        "-b", "--batch_size", 
        default=1,
        type=int,
        required=False,
        help="Number of images in one batch"
    )
    parser.add_argument(
        "-l", "--labels", 
        default="resources/coco.txt",
        help="Path to a text file containing labels."
    )
    parser.add_argument(
        "-s", "--save_stream_output",
        action="store_true",
        help="Save the output of the inference from a stream."
    )
    parser.add_argument(
        "-p", "--num_processes",
        default=2,
        type=int,
        help="Number of processes to use for preprocessing and postprocessing."
    )

    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.net):
        raise FileNotFoundError(f"Network file not found: {args.net}")
    if not os.path.exists(args.labels):
        raise FileNotFoundError(f"Labels file not found: {args.labels}")
    if args.input != "camera" and not os.path.exists(args.input):
        raise FileNotFoundError(f"Input path not found: {args.input}")

    return args

def create_output_directory() -> Path:
    """
    Create the output directory if it does not exist.

    Returns:
        Path: Path object for the output directory.
    """
    output_path = Path('output_images')
    output_path.mkdir(exist_ok=True)
    return output_path

def preprocess_input(
    images: list,
    cap_path: str,
    batch_size: int,
    input_queue: Queue,
    width: int,
    height: int,
    labels_path: str,
    stop_event: Event,
    timing_queue: Queue = None
) -> None:
    """
    Preprocess and enqueue images into the input queue as they are ready.
    """
    # 멀티프로세스에서 새 로거 설정
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add("preprocess_debug.log", rotation="500 MB", level="DEBUG")
    
    # ObjectDetectionUtils 인스턴스 생성
    utils = ObjectDetectionUtils(labels_path)
    
    logger.info("전처리 프로세스 시작")
    frame_count = 0
    total_preprocess_time = 0
    cap = None
    
    # 비디오 또는 카메라 초기화
    if cap_path is not None:
        if cap_path == "0":
            # 시스템 기본 카메라 (여러 장치 시도)
            for cam_idx in range(5):
                try:
                    cap = cv2.VideoCapture(cam_idx)
                    if cap.isOpened():
                        logger.info(f"카메라 {cam_idx} 열기 성공")
                        break
                    cap.release()
                except Exception as e:
                    logger.error(f"카메라 {cam_idx} 열기 실패: {e}")
            else:
                # 장치 파일로 직접 시도
                for video_device in ['/dev/video0', '/dev/video1', '/dev/video2', '/dev/video19', '/dev/video20']:
                    try:
                        cap = cv2.VideoCapture(video_device)
                        if cap.isOpened():
                            logger.info(f"비디오 장치 {video_device} 열기 성공")
                            break
                        cap.release()
                    except Exception as e:
                        logger.error(f"비디오 장치 {video_device} 열기 실패: {e}")
                else:
                    logger.error("어떤 카메라도 열 수 없습니다.")
                    input_queue.put(None)
                    return
        elif cap_path.startswith('/dev/'):
            # 직접 디바이스 경로 지정
            try:
                cap = cv2.VideoCapture(cap_path)
                if not cap.isOpened():
                    logger.error(f"지정된 비디오 장치 {cap_path}를 열 수 없습니다.")
                    input_queue.put(None)
                    return
                logger.info(f"비디오 장치 {cap_path} 열기 성공")
            except Exception as e:
                logger.error(f"비디오 장치 {cap_path} 열기 실패: {e}")
                input_queue.put(None)
                return
        else:
            # 숫자 인덱스나 파일 경로
            try:
                cap = cv2.VideoCapture(cap_path)
                if not cap.isOpened():
                    logger.error(f"카메라/비디오를 열 수 없습니다: {cap_path}")
                    input_queue.put(None)
                    return
            except Exception as e:
                logger.error(f"카메라/비디오 열기 실패: {e}")
                input_queue.put(None)
                return
        
        logger.info(f"카메라/비디오 입력 시작: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
        
        # 프레임 버퍼링을 최소화하기 위한 설정
        if cap is not None:
            # 프레임 버퍼링 최소화 (지연 시간 감소)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # 프레임 처리 루프
        while not stop_event.is_set():
            # 시간 측정 시작
            preprocess_start = time.time()
            
            ret, frame = cap.read()
            if not ret:
                logger.warning("프레임 읽기 실패")
                break

            frame_count += 1
            
            try:
                # 불필요한 BGR-RGB 변환 제거 (전처리 함수에서 처리)
                processed_frame = utils.preprocess(frame, width, height)
                
                # 처리 시간 계산
                preprocess_time = time.time() - preprocess_start
                total_preprocess_time += preprocess_time
                
                # 처리 시간 기록 (로깅 빈도 줄임)
                if frame_count % 10 == 0 and timing_queue is not None:
                    timing_queue.put(('preprocess', frame_count, preprocess_time))
                
                # 원본 프레임과 전처리된 프레임을 함께 전달
                input_queue.put(([frame], [processed_frame]))
            
            except Exception as e:
                logger.error(f"프레임 {frame_count} 전처리 오류: {e}")
                continue
    else:
        logger.info(f"이미지 입력 시작: {len(images)}개 이미지")
        for batch_idx, batch in enumerate(divide_list_to_batches(images, batch_size)):
            if stop_event.is_set():
                break
            
            # 시간 측정 시작
            preprocess_start = time.time()
                
            logger.debug(f"배치 {batch_idx + 1} 처리 중")
            original_batch = []
            processed_batch = []
            for img_idx, image in enumerate(batch):
                # BGR to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # 전처리
                processed_image = utils.preprocess(image_rgb, width, height)
                
                original_batch.append(image)
                processed_batch.append(processed_image)
                logger.debug(f"이미지 {img_idx + 1} 전처리 완료")
            
            # 처리 시간 계산
            preprocess_time = time.time() - preprocess_start
            total_preprocess_time += preprocess_time
            frame_count += len(batch)
            
            # 처리 시간 기록
            if timing_queue is not None:
                timing_queue.put(('preprocess_batch', batch_idx + 1, preprocess_time))
            
            logger.info(f"배치 {batch_idx + 1} 전처리 시간: {preprocess_time:.4f}초 (이미지 {len(batch)}개)")
            
            input_queue.put((original_batch, processed_batch))
            logger.debug(f"배치 {batch_idx + 1} 큐에 추가됨")

    # 정리
    if cap is not None:
        cap.release()
    
    avg_time = total_preprocess_time / max(1, frame_count)
    logger.info(f"전처리 프로세스 종료 (총 {frame_count} 프레임 처리)")
    logger.info(f"평균 전처리 시간: {avg_time:.4f}초/프레임, 초당 처리 프레임: {1/avg_time:.2f}")
    input_queue.put(None)

def postprocess_output(
    output_queue: Queue,
    display_queue: Queue,
    output_path: str,
    width: int,
    height: int,
    labels_path: str,
    save_stream_output: bool,
    stop_event: Event,
    fps_shared: Value,
    timing_queue: Queue = None
) -> None:
    """
    Process and visualize the output results.
    """
    # 멀티프로세스에서 새 로거 설정
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add("postprocess_debug.log", rotation="500 MB", level="DEBUG")
    
    # ObjectDetectionUtils 인스턴스 생성
    utils = ObjectDetectionUtils(labels_path)
    
    logger.info("후처리 프로세스 시작")
    image_id = 0
    output_path = Path(output_path)
    total_postprocess_time = 0
    
    # FPS 계산을 위한 변수들
    fps_start_time = time.time()
    fps_frame_count = 0
    
    while not stop_event.is_set():
        try:
            result = output_queue.get(timeout=1.0)
            if result is None:
                logger.info("종료 신호 수신")
                break
            
            # 실제 후처리 시간 측정 시작
            postprocess_start = time.time()

            # 프레임과 감지 결과
            original_frames, infer_results = result
            
            try:
                # 단일 이미지인 경우 처리
                if isinstance(original_frames, list) and len(original_frames) > 0:
                    original_frame = original_frames[0]
                else:
                    original_frame = original_frames
                
                # 이전 hailort 버전 호환성 처리
                if isinstance(infer_results, list):
                    if len(infer_results) == 1:
                        infer_results = infer_results[0]
                
                # 감지 결과 추출 - object_detection.py 방식으로 처리
                try:
                    detections = utils.extract_detections(infer_results)
                    
                    # FPS 계산
                    fps_frame_count += 1
                    if fps_frame_count >= 10:  # 10프레임마다 FPS 업데이트
                        fps_end_time = time.time()
                        with fps_shared.get_lock():
                            fps_shared.value = fps_frame_count / (fps_end_time - fps_start_time)
                        fps_start_time = fps_end_time
                        fps_frame_count = 0
                    
                    # 이미지에 감지 결과 그리기
                    frame_with_detections = utils.draw_detections(detections, original_frame)
                    
                    # FPS 정보 추가
                    with fps_shared.get_lock():
                        current_fps = fps_shared.value
                    cv2.putText(frame_with_detections, f"FPS: {current_fps:.1f}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # 처리 시간 계산
                    postprocess_time = time.time() - postprocess_start
                    total_postprocess_time += postprocess_time
                    
                    # 처리 시간 기록 (로깅 빈도 줄임)
                    if image_id % 10 == 0 and timing_queue is not None:
                        timing_queue.put(('postprocess', image_id + 1, postprocess_time))
                    
                    # 결과 전송
                    display_queue.put((frame_with_detections, save_stream_output))
                except Exception as e:
                    logger.error(f"감지 결과 처리 실패: {e}")
                    # 오류 발생 시 원본 프레임 전달
                    display_queue.put((original_frame, save_stream_output))
            
            except Exception as e:
                logger.error(f"프레임 {image_id + 1} 처리 중 오류 발생: {e}")
                continue

            image_id += 1

        except Exception as e:
            logger.error(f"후처리 중 오류: {e}")
            if stop_event.is_set():
                break

    avg_time = total_postprocess_time / max(1, image_id)
    logger.info(f"후처리 프로세스 종료 (총 {image_id} 프레임 처리)")
    logger.info(f"평균 후처리 시간: {avg_time:.4f}초/프레임, 초당 처리 프레임: {1/avg_time:.2f}")
    display_queue.put((None, None))

def display_frames(
    display_queue: Queue,
    output_path: str,
    cap_info: dict,
    stop_event: Event,
    timing_queue: Queue = None
) -> None:
    """
    Display frames in a separate process.
    """
    # 멀티프로세스에서 새 로거 설정
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add("display_debug.log", rotation="500 MB", level="DEBUG")
    
    logger.info("디스플레이 프로세스 시작")
    output_path = Path(output_path)
    out = None
    frame_count = 0
    
    # 비디오 캡처 정보가 있는 경우 (카메라/비디오 모드)
    is_video_mode = cap_info.get('is_video', False)
    
    if is_video_mode:
        # 디스플레이 창 설정 (fullscreen 대신 normal 사용하여 워크스테이션 성능 향상)
        cv2.namedWindow("Object Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Object Detection", 1280, 720)
        
        # 첫 번째 프레임 대기
        try:
            logger.info("첫 번째 프레임 대기 중...")
            frame, save_stream_output = display_queue.get(timeout=5.0)
            if frame is not None:
                # VideoWriter 초기화 (저장 필요한 경우)
                if save_stream_output:
                    frame_width = cap_info.get('width', 1920)
                    frame_height = cap_info.get('height', 1080)
                    fps = cap_info.get('fps', 20.0)
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    out = cv2.VideoWriter(str(output_path / 'detection_output.avi'), fourcc, fps, (frame_width, frame_height))
                    logger.info(f"비디오 저장 설정 완료: {frame_width}x{frame_height} @ {fps}fps")
                
                # 첫 번째 프레임 표시
                cv2.imshow("Object Detection", frame)
                if save_stream_output and out is not None:
                    out.write(frame)
                frame_count += 1
                logger.info("첫 번째 프레임 표시 완료")
        except Exception as e:
            logger.error(f"첫 번째 프레임을 받지 못했습니다: {e}")
            stop_event.set()
            return

    # 메인 디스플레이 루프
    total_display_time = 0
    last_fps_print_time = time.time()
    
    while not stop_event.is_set():
        try:
            # 시간 측정 시작
            display_start = time.time()
            
            frame, save_stream_output = display_queue.get(timeout=1.0)
            if frame is None:
                logger.info("종료 신호 수신")
                break

            frame_count += 1
            
            if is_video_mode:
                # 화면에 표시
                cv2.imshow("Object Detection", frame)
                if save_stream_output and out is not None:
                    out.write(frame)
                
                # 처리 시간 계산
                display_time = time.time() - display_start
                total_display_time += display_time
                
                # 처리 시간 기록 (로깅 빈도 줄임)
                if frame_count % 30 == 0 and timing_queue is not None:
                    timing_queue.put(('display', frame_count, display_time))
                    
                    # 30프레임마다 FPS 로그 출력
                    current_time = time.time()
                    if current_time - last_fps_print_time >= 5.0:  # 5초마다 FPS 정보 출력
                        fps = 30 / (current_time - last_fps_print_time)
                        logger.info(f"현재 FPS: {fps:.2f}")
                        last_fps_print_time = current_time
                
                # 1ms 대기 (CPU 사용량 감소)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("사용자 종료 요청")
                    stop_event.set()
                    break
            else:
                # 이미지 모드인 경우 파일로 저장
                output_file = output_path / f"detection_{frame_count}.jpg"
                cv2.imwrite(str(output_file), frame)
                logger.info(f"이미지 저장 완료: {output_file}")

        except Exception as e:
            logger.error(f"디스플레이 오류: {e}")
            if stop_event.is_set():
                break

    # 리소스 정리
    if is_video_mode:
        if out is not None:
            out.release()
            logger.info("비디오 저장 종료")
        cv2.destroyAllWindows()
        logger.info("디스플레이 리소스 해제")

    if frame_count > 0:
        avg_time = total_display_time / frame_count
        logger.info(f"평균 디스플레이 시간: {avg_time:.4f}초/프레임, 초당 처리 프레임: {1/avg_time:.2f}")
    
    logger.info(f"디스플레이 프로세스 종료 (총 {frame_count} 프레임 처리)")

def process_timing_info(timing_queue: Queue, stop_event: Event) -> None:
    """
    수집된 타이밍 정보를 처리하고 통계를 계산합니다.
    """
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add("timing_stats.log", rotation="500 MB", level="INFO")
    
    logger.info("타이밍 정보 처리 프로세스 시작")
    
    timing_data = {
        'preprocess': [],
        'preprocess_batch': [],
        'queue_wait': [],
        'postprocess': [],
        'display': []
    }
    
    while not stop_event.is_set():
        try:
            data = timing_queue.get(timeout=1.0)
            if data is None:
                break
                
            stage, frame_or_batch, time_taken = data
            timing_data[stage].append(time_taken)
            
            # 주기적으로 통계 계산 및 출력
            if len(timing_data['postprocess']) > 0 and len(timing_data['postprocess']) % 10 == 0:
                calculate_stats(timing_data)
                
        except Exception as e:
            if stop_event.is_set():
                break
                
    # 최종 통계 계산 및 출력
    calculate_stats(timing_data)
    logger.info("타이밍 정보 처리 프로세스 종료")

def calculate_stats(timing_data: dict) -> None:
    """
    타이밍 데이터에서 통계를 계산합니다.
    """
    for stage, times in timing_data.items():
        if len(times) > 0:
            avg_time = sum(times) / len(times)
            max_time = max(times)
            min_time = min(times)
            
            logger.info(f"{stage} 통계:")
            logger.info(f"  - 샘플 수: {len(times)}")
            logger.info(f"  - 평균 시간: {avg_time:.4f}초")
            logger.info(f"  - 최대 시간: {max_time:.4f}초")
            logger.info(f"  - 최소 시간: {min_time:.4f}초")
            logger.info(f"  - 초당 처리 가능 프레임: {1/avg_time:.2f}")

def infer(
    images: list,
    cap_path: str,
    cap_info: dict,
    net_path: str,
    batch_size: int,
    output_path: str,
    labels_path: str,
    save_stream_output: bool,
    num_processes: int
) -> None:
    """
    Run inference with HailoAsyncInference using multiprocessing.
    """
    # 멀티프로세싱 큐 및 이벤트 초기화
    input_queue = Queue(maxsize=5)  # 작은 큐 사이즈로 메모리 사용량 최적화
    output_queue = Queue(maxsize=5)
    display_queue = Queue(maxsize=5)
    timing_queue = Queue()
    stop_event = Event()
    
    # FPS 공유 변수
    fps_shared = Value('d', 0.0)
    
    # 전체 시간 측정 시작
    total_start_time = time.time()

    # HailoAsyncInference 초기화 (메인 프로세스에서 실행)
    try:
        logger.info("Hailo 추론 객체 초기화 중...")
        # send_original_frame=True로 설정하여 원본 프레임 전달
        hailo_inference = HailoAsyncInference(
            net_path, input_queue, output_queue, batch_size, send_original_frame=True
        )
        height, width, _ = hailo_inference.get_input_shape()
        logger.info(f"모델 입력 크기: {width}x{height}")
    except Exception as e:
        logger.error(f"Hailo 추론 객체 초기화 오류: {e}")
        stop_event.set()
        return

    # 프로세스 생성
    preprocess_process = Process(
        target=preprocess_input,
        name="preprocess_process",
        args=(images, cap_path, batch_size, input_queue, width, height, labels_path, stop_event, timing_queue)
    )
    
    postprocess_process = Process(
        target=postprocess_output,
        name="postprocess_process",
        args=(output_queue, display_queue, output_path, width, height, labels_path, save_stream_output, stop_event, fps_shared, timing_queue)
    )
    
    display_process = Process(
        target=display_frames,
        name="display_process",
        args=(display_queue, output_path, cap_info, stop_event, timing_queue)
    )
    
    timing_process = Process(
        target=process_timing_info,
        name="timing_process",
        args=(timing_queue, stop_event)
    )

    # 프로세스 우선순위 설정
    preprocess_process.daemon = True  # 데몬 프로세스로 설정하여 메인 프로세스 종료 시 자동 종료
    postprocess_process.daemon = True
    display_process.daemon = True
    timing_process.daemon = True

    # 프로세스 시작
    preprocess_process.start()
    postprocess_process.start()
    display_process.start()
    timing_process.start()

    try:
        # 메인 프로세스에서 추론 실행
        logger.info("메인 프로세스에서 추론 시작")
        
        # 추론 시간 측정
        inference_start = time.time()
        hailo_inference.run()
        inference_time = time.time() - inference_start
        
        logger.info(f"추론 완료. 총 추론 시간: {inference_time:.4f}초")
        
        # 프로세스가 끝날 때까지 대기
        preprocess_process.join()
        output_queue.put(None)  # 후처리 프로세스에 종료 신호 전송
        postprocess_process.join()
        display_process.join()
        
        # 타이밍 프로세스에 종료 신호 전송 및 대기
        timing_queue.put(None)
        timing_process.join()
        
        # 전체 시간 계산
        total_time = time.time() - total_start_time
        logger.info(f'추론이 성공적으로 완료되었습니다! 총 실행 시간: {total_time:.4f}초')
        logger.info(f'결과가 {output_path}에 저장되었습니다.')
      
    except Exception as e:
        logger.error(f"추론 오류: {e}")
        stop_event.set()  # 모든 프로세스에 종료 신호 보내기
        
        # 강제 종료 전 프로세스 종료 대기
        preprocess_process.join(timeout=3)
        postprocess_process.join(timeout=3)
        display_process.join(timeout=3)
        timing_process.join(timeout=3)
        
        # 종료되지 않은 프로세스 강제 종료
        if preprocess_process.is_alive():
            preprocess_process.terminate()
        if postprocess_process.is_alive():
            postprocess_process.terminate()
        if display_process.is_alive():
            display_process.terminate()
        if timing_process.is_alive():
            timing_process.terminate()
            
        sys.exit(1)

def object_detection_process() -> None:
    # 멀티프로세싱 시작 방식 설정 (Windows 호환성 위해)
    multiprocessing.set_start_method('spawn', force=True)
    
    # 로깅 설정
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add("debug.log", rotation="500 MB", level="DEBUG")
    logger.add("error.log", rotation="500 MB", level="ERROR")
    
    args = parse_args()
    logger.info(f"프로그램 시작: 입력={args.input}, 모델={args.net}, 배치 크기={args.batch_size}, 프로세스 수={args.num_processes}")
    
    # 카메라 또는 이미지 입력 처리
    cap_path = None
    cap_info = {'is_video': False}
    images = []
    
    if args.input == "camera":
        logger.info("카메라 입력 모드")
        
        # 명시적 비디오 장치가 지정된 경우
        if args.video_device:
            cap_path = args.video_device
            logger.info(f"지정된 비디오 장치 사용: {cap_path}")
        else:
            cap_path = "0"  # 카메라 인덱스
            
        cap_info = {
            'is_video': True,
            'width': 1920,
            'height': 1080,
            'fps': 30.0
        }
        # 카메라 테스트 (실제 열 수 있는지 확인)
        for cam_idx in range(5):  # 여러 카메라 인덱스 시도
            logger.info(f"카메라 인덱스 {cam_idx} 시도 중...")
            cap = cv2.VideoCapture(cam_idx)
            if cap.isOpened():
                cap_path = str(cam_idx)
                cap_info['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                cap_info['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap_info['fps'] = cap.get(cv2.CAP_PROP_FPS) or 30.0
                cap.release()
                logger.info(f"카메라 {cam_idx} 연결 성공!")
                break
            cap.release()
        else:
            # 모든 카메라 인덱스를 시도해도 실패한 경우
            logger.error("어떤 카메라도 열 수 없습니다. 샘플 이미지를 사용합니다.")
            # 기본 샘플 이미지 폴더로 전환
            sample_dir = os.path.join(os.path.dirname(__file__), "resources")
            if os.path.exists(sample_dir):
                image_files = [os.path.join(sample_dir, f) for f in os.listdir(sample_dir) 
                               if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if image_files:
                    args.input = image_files[0]
                    logger.info(f"샘플 이미지를 사용합니다: {args.input}")
                else:
                    logger.error("샘플 이미지를 찾을 수 없습니다. 프로그램을 종료합니다.")
                    return
            else:
                logger.error("샘플 이미지 폴더를 찾을 수 없습니다. 프로그램을 종료합니다.")
                return
    elif args.input.lower().endswith(('.mp4', '.avi', '.mov')):
        logger.info(f"비디오 파일 입력 모드: {args.input}")
        cap_path = args.input
        # 비디오 정보 확인
        cap = cv2.VideoCapture(args.input)
        if not cap.isOpened():
            logger.error(f"비디오 파일을 열 수 없습니다: {args.input}")
            return
        cap_info = {
            'is_video': True,
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': cap.get(cv2.CAP_PROP_FPS) or 25.0
        }
        cap.release()
        logger.info(f"비디오 파일 정보 확인 완료: {cap_info['width']}x{cap_info['height']} @ {cap_info['fps']}fps")
    else:
        logger.info(f"이미지 입력 모드: {args.input}")
        images = load_images_opencv(args.input)
        try:
            validate_images(images, args.batch_size)
            logger.info(f"이미지 로드 완료: {len(images)}개")
        except ValueError as e:
            logger.error(f"이미지 검증 실패: {e}")
            return

    output_path = create_output_directory()

    try:
        infer(
            images, cap_path, cap_info, args.net, int(args.batch_size),
            str(output_path), args.labels, args.save_stream_output, args.num_processes
        )
    except Exception as e:
        logger.error(f"추론 중 오류 발생: {e}")
        raise

if __name__ == "__main__":
    object_detection_process() 