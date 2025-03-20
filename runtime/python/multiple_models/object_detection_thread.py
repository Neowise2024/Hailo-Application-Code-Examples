#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path
import numpy as np
from loguru import logger
import queue
import threading
import cv2
from hailo_platform import HEF
from commons.object_detection_utils import ObjectDetectionUtils

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import HailoAsyncInference, load_images_opencv, validate_images, divide_list_to_batches

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
    cap: cv2.VideoCapture,
    batch_size: int,
    input_queue: queue.Queue,
    width: int,
    height: int,
    utils: ObjectDetectionUtils,
    stop_event: threading.Event
) -> None:
    """
    Preprocess and enqueue images into the input queue as they are ready.

    Args:
        images (list): List of images.
        cap (cv2.VideoCapture): Video capture object for camera input.
        batch_size (int): Number of images in one batch.
        input_queue (queue.Queue): Queue for input images.
        width (int): Model input width.
        height (int): Model input height.
        utils (ObjectDetectionUtils): Utility class for object detection.
        stop_event (threading.Event): Event to signal thread to stop.
    """
    logger.info("전처리 스레드 시작")
    frame_count = 0
    
    if cap is not None:
        logger.info(f"카메라/비디오 입력 시작: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                logger.warning("프레임 읽기 실패")
                break

            frame_count += 1
            logger.debug(f"프레임 {frame_count} 처리 중")

            # BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 전처리
            processed_frame = utils.preprocess(frame_rgb, width, height)
            
            # 원본 프레임과 전처리된 프레임을 함께 전달
            input_queue.put(([frame], [processed_frame]))
            logger.debug(f"프레임 {frame_count} 큐에 추가됨")
    else:
        logger.info(f"이미지 입력 시작: {len(images)}개 이미지")
        for batch_idx, batch in enumerate(divide_list_to_batches(images, batch_size)):
            if stop_event.is_set():
                break
                
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
            
            input_queue.put((original_batch, processed_batch))
            logger.debug(f"배치 {batch_idx + 1} 큐에 추가됨")

    logger.info(f"전처리 스레드 종료 (총 {frame_count} 프레임 처리)")
    input_queue.put(None)

def postprocess_output(
    output_queue: queue.Queue,
    display_queue: queue.Queue,
    output_path: Path,
    width: int,
    height: int,
    utils: ObjectDetectionUtils,
    save_stream_output: bool,
    stop_event: threading.Event
) -> None:
    """
    Process and visualize the output results.

    Args:
        output_queue (queue.Queue): Queue for output results.
        display_queue (queue.Queue): Queue for frames to display.
        output_path (Path): Path to save the output images.
        width (int): Image width.
        height (int): Image height.
        utils (ObjectDetectionUtils): Utility class for object detection.
        save_stream_output (bool): Whether to save stream output.
        stop_event (threading.Event): Event to signal thread to stop.
    """
    logger.info("후처리 스레드 시작")
    image_id = 0
    out = None

    while not stop_event.is_set():
        try:
            result = output_queue.get(timeout=1.0)
            if result is None:
                logger.info("종료 신호 수신")
                break

            frame, detections = result
            logger.debug(f"프레임 {image_id + 1} 후처리 시작")
            
            try:
                # detections를 딕셔너리 형태로 변환
                processed_detections = utils.postprocess(detections, threshold=0.5)
                logger.debug(f"감지된 객체 수: {processed_detections['num_detections']}")
                
                if processed_detections['num_detections'] > 0:  # 감지된 객체가 있는 경우에만 처리
                    frame_with_detections = utils.visualize(frame, processed_detections)
                    display_queue.put((frame_with_detections, save_stream_output))
                    logger.debug(f"프레임 {image_id + 1} 시각화 완료")
                else:
                    # 감지된 객체가 없는 경우 원본 프레임 표시
                    display_queue.put((frame, save_stream_output))
                    logger.debug(f"프레임 {image_id + 1} 객체 미감지")

            except Exception as e:
                logger.error(f"프레임 {image_id + 1} 처리 중 오류 발생: {e}")
                continue

            image_id += 1

        except queue.Empty:
            continue

    logger.info(f"후처리 스레드 종료 (총 {image_id} 프레임 처리)")

def display_frames(
    display_queue: queue.Queue,
    output_path: Path,
    cap: cv2.VideoCapture,
    stop_event: threading.Event
) -> None:
    """
    Display frames in the main thread.

    Args:
        display_queue (queue.Queue): Queue for frames to display.
        output_path (Path): Path to save the output images.
        cap (cv2.VideoCapture): Video capture object.
        stop_event (threading.Event): Event to signal thread to stop.
    """
    logger.info("디스플레이 스레드 시작")
    out = None
    frame_count = 0
    
    if cap is not None:
        cv2.namedWindow("Object Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Object Detection", 1280, 720)
        
        # 첫 번째 프레임 대기
        try:
            logger.info("첫 번째 프레임 대기 중...")
            frame, save_stream_output = display_queue.get(timeout=5.0)
            if frame is not None:
                # VideoWriter 초기화
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps == 0:
                    fps = 20.0
                out = cv2.VideoWriter(str(output_path / 'detection_output.avi'), fourcc, fps, (frame_width, frame_height))
                logger.info(f"비디오 저장 설정 완료: {frame_width}x{frame_height} @ {fps}fps")
                
                # 첫 번째 프레임 표시
                cv2.imshow("Object Detection", frame)
                if save_stream_output and out is not None:
                    out.write(frame)
                frame_count += 1
                logger.info("첫 번째 프레임 표시 완료")
        except queue.Empty:
            logger.error("첫 번째 프레임을 받지 못했습니다.")
            stop_event.set()
            return

    while not stop_event.is_set():
        try:
            frame, save_stream_output = display_queue.get(timeout=1.0)
            if frame is None:
                logger.info("종료 신호 수신")
                break

            frame_count += 1
            cv2.imshow("Object Detection", frame)
            if save_stream_output and out is not None:
                out.write(frame)
            logger.debug(f"프레임 {frame_count} 표시 완료")

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("사용자 종료 요청")
                stop_event.set()
                break

        except queue.Empty:
            continue

    if cap is not None:
        if out is not None:
            out.release()
            logger.info("비디오 저장 종료")
        cap.release()
        cv2.destroyAllWindows()
        logger.info("카메라/비디오 리소스 해제")

    logger.info(f"디스플레이 스레드 종료 (총 {frame_count} 프레임 표시)")

def infer(
    images: list,
    cap: cv2.VideoCapture,
    net_path: str,
    batch_size: int,
    output_path: Path,
    utils: ObjectDetectionUtils,
    save_stream_output: bool
) -> None:
    """
    Run inference with HailoAsyncInference, handle processes, and ensure proper cleanup.

    Args:
        images (list): List of images to process.
        cap (cv2.VideoCapture): Video capture object for camera input.
        net_path (str): Path to the HEF model file.
        batch_size (int): Number of images per batch.
        output_path (Path): Path to save the output images.
        utils (ObjectDetectionUtils): Utility class for object detection.
        save_stream_output (bool): Whether to save stream output.
    """
    input_queue = queue.Queue()
    output_queue = queue.Queue()
    display_queue = queue.Queue()
    stop_event = threading.Event()

    hailo_inference = HailoAsyncInference(
        net_path, input_queue, output_queue, batch_size, send_original_frame=True
    )
    height, width, _ = hailo_inference.get_input_shape()

    preprocess_thread = threading.Thread(
        target=preprocess_input,
        name="image_enqueuer",
        args=(images, cap, batch_size, input_queue, width, height, utils, stop_event)
    )
    postprocess_thread = threading.Thread(
        target=postprocess_output,
        name="image_processor",
        args=(
            output_queue, display_queue, output_path, width, height, utils,
            save_stream_output, stop_event
        )
    )

    preprocess_thread.start()
    postprocess_thread.start()

    try:
        # 추론 시작
        hailo_inference.run()
        
        # 메인 스레드에서 프레임 표시
        display_frames(display_queue, output_path, cap, stop_event)
        
        # 스레드 정리
        preprocess_thread.join()
        output_queue.put(None)
        display_queue.put((None, None))
        postprocess_thread.join()
     
        logger.info(f'Inference was successful! Results have been saved in {output_path}')
      
    except Exception as e:
        logger.error(f"Inference error: {e}")
        stop_event.set()
        preprocess_thread.join()
        postprocess_thread.join()
        os._exit(1)

def main() -> None:
    # 로깅 설정
    logger.add("debug.log", rotation="500 MB", level="DEBUG")
    logger.add("error.log", rotation="500 MB", level="ERROR")
    
    args = parse_args()
    logger.info(f"프로그램 시작: 입력={args.input}, 모델={args.net}, 배치 크기={args.batch_size}")
    
    # 카메라 또는 이미지 입력 처리
    cap = None
    images = []
    if args.input == "camera":
        logger.info("카메라 입력 모드")
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        if not cap.isOpened():
            logger.error("카메라를 열 수 없습니다")
            return
        logger.info("카메라 초기화 완료")
    elif args.input.lower().endswith(('.mp4', '.avi', '.mov')):
        logger.info(f"비디오 파일 입력 모드: {args.input}")
        cap = cv2.VideoCapture(args.input)
        if not cap.isOpened():
            logger.error(f"비디오 파일을 열 수 없습니다: {args.input}")
            return
        logger.info(f"비디오 파일 초기화 완료: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)} @ {cap.get(cv2.CAP_PROP_FPS)}fps")
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
    utils = ObjectDetectionUtils(args.labels)
    logger.info("유틸리티 초기화 완료")

    try:
        infer(
            images, cap, args.net, int(args.batch_size),
            output_path, utils, args.save_stream_output
        )
    except Exception as e:
        logger.error(f"추론 중 오류 발생: {e}")
        raise

if __name__ == "__main__":
    main() 