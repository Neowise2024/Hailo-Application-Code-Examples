#!/usr/bin/env python3

import os
import sys
import argparse
from pathlib import Path
from loguru import logger
from PIL import Image
from typing import List
from hailo_platform import HEF
from commons.pose_estimation_utils import (output_data_type2dict, PoseEstPostProcessing)
import cv2
import time
import numpy as np
import multiprocessing as mp
from multiprocessing import Queue, Event, Value
from queue import Empty
import subprocess

# Add the parent directory to the system path to access utils module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import HailoAsyncInference, load_input_images, validate_images, divide_list_to_batches


def parse_args() -> argparse.Namespace:
    """
    Initialize argument parser for the script.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Running a Hailo inference with actual images using Hailo API and OpenCV"
    )
    parser.add_argument(
        "-n", "--net",
        help="Path for the network in HEF format.",
        default="resources/yolov8s_pose_h8l.hef"
    )
    parser.add_argument(
        "-i", "--input",
        default="zidane.jpg",
        help="Path to the input - either an image, video file, folder of images, or 'camera' for webcam input."
    )
    parser.add_argument(
        "-b", "--batch_size",
        default=1,
        type=int,
        required=False,
        help="Number of images in one batch. Defaults to 1"
    )
    parser.add_argument(
        "-cn", "--class_num",
        help="The number of classes the model is trained on. Defaults to 1",
        default=1
    )

    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.net):
        raise FileNotFoundError(f"Network file not found: {args.net}")
    
    # 입력이 'camera'가 아니고 파일/디렉토리가 존재하지 않는 경우에만 에러 발생
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


def preprocess_from_cap(
    cap: cv2.VideoCapture,
    input_queue: Queue,
    width: int,
    height: int,
    post_processing: PoseEstPostProcessing,
    stop_event: Event,
    fps_shared: Value
) -> None:
    """
    카메라/비디오 프레임을 전처리하고 큐에 넣습니다.
    """
    try:
        frame_count = 0
        start_time = time.time()
        fps_update_interval = 30
        frames_to_skip = 5  # 5프레임마다 1프레임만 처리
        queue_full_count = 0
        
        # 카메라 버퍼 최소화
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % frames_to_skip != 0:
                continue

            # FPS 계산 및 출력
            if frame_count % fps_update_interval == 0:
                end_time = time.time()
                elapsed_time = end_time - start_time
                fps = frame_count / elapsed_time
                with fps_shared.get_lock():
                    fps_shared.value = fps
                logger.info(f"처리 속도: {fps:.2f} FPS (총 {frame_count}프레임, {elapsed_time:.2f}초)")

            # 프레임 리사이즈
            frame = cv2.resize(frame, (640, 640))
            
            # OpenCV BGR을 RGB로 변환
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # PIL Image로 변환
            pil_image = Image.fromarray(frame_rgb)
            # 전처리 수행
            processed_frame = post_processing.preprocess(pil_image, width, height)
            
            # 큐 상태 확인 및 처리
            try:
                input_queue.put((frame, processed_frame), timeout=0.1)
                queue_full_count = 0  # 성공적으로 넣으면 카운터 리셋
            except Exception as e:
                queue_full_count += 1
                if queue_full_count >= 3:  # 연속 3번 실패하면 잠시 대기
                    logger.warning("입력 큐가 가득 찼습니다. 잠시 대기합니다...")
                    time.sleep(0.05)  # 대기 시간 감소
                    queue_full_count = 0
                continue

    except Exception as e:
        logger.error(f"프레임 전처리 중 오류 발생: {str(e)}")
    finally:
        cap.release()
        input_queue.put(None)  # 종료 신호 전송


def postprocess_output(
    output_queue: Queue,
    output_path: Path,
    width: int,
    height: int,
    class_num: int,
    post_processing: PoseEstPostProcessing,
    stop_event: Event,
    fps_shared: Value
) -> None:
    """
    Process and visualize the output results.

    Args:
        output_queue (Queue): Queue for output results.
        output_path (Path): Path to save the output images.
        width (int): Image width.
        height (int): Image height.
        class_num (int): Number of classes.
        post_processing (PoseEstPostProcessing): Post-processing configuration.
        stop_event (Event): 종료 이벤트
        fps_shared (Value): FPS 공유 변수
    """
    image_id = 0
    cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
    
    # 성능 측정을 위한 변수들
    start_time = time.time()
    frame_count = 0
    fps_update_interval = 30
    
    while not stop_event.is_set():
        try:
            result = output_queue.get(timeout=1.0)  # 1초 타임아웃
            if result is None:
                break

            frame_count += 1
            process_start = time.time()

            processed_image, raw_detections = result
            
            # 포즈 검출 결과 추출
            detections = post_processing.extract_detections(raw_detections, height, width, class_num)
            if detections is None:
                logger.error("검출 결과가 없습니다.")
                continue
            
            # 이미지 변환 및 시각화
            image_array = np.array(processed_image)
            output_image = post_processing.draw_detections(image_array, {
                'bboxes': detections['bboxes'][0],
                'keypoints': detections['keypoints'][0],
                'joint_scores': detections['joint_scores'][0],
                'scores': detections['scores'][0]
            })
            
            # 프레임 처리 시간 측정
            process_time = time.time() - process_start
            logger.debug(f"후처리 시간: {process_time*1000:.2f}ms")
            
            # FPS 계산 및 출력
            if frame_count % fps_update_interval == 0:
                current_time = time.time()
                elapsed_time = current_time - start_time
                fps = frame_count / elapsed_time
                with fps_shared.get_lock():
                    fps_shared.value = fps
                logger.info(f"출력 속도: {fps:.2f} FPS (총 {frame_count}프레임, {elapsed_time:.2f}초)")
            
            # FPS 표시
            with fps_shared.get_lock():
                current_fps = fps_shared.value
            cv2.putText(output_image, f"FPS: {current_fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 결과 표시
            cv2.imshow("Output", output_image)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("사용자가 프로그램을 종료했습니다.")
                stop_event.set()
                break
                    
            image_id += 1
            
        except Empty:
            continue
        except Exception as e:
            logger.error(f"후처리 중 오류 발생: {str(e)}")
            continue
    
    cv2.destroyAllWindows()


def check_hailo_device():
    """
    Hailo 하드웨어 장치의 존재 여부를 확인합니다.
    """
    try:
        result = subprocess.run(['hailortcli', 'fw-control', 'identify'], 
                              capture_output=True, text=True)
        if result.returncode == 0 and "Board Name: Hailo-8" in result.stdout:
            logger.info("Hailo 장치가 정상적으로 감지되었습니다.")
            return True
        else:
            logger.error("Hailo 장치를 찾을 수 없습니다.")
            return False
    except Exception as e:
        logger.error(f"Hailo 장치 확인 중 오류 발생: {e}")
        return False


def infer(
    input_source: str,
    net_path: str,
    batch_size: int,
    class_num: int,
    output_path: Path,
    data_type_dict: dict,
    post_processing: PoseEstPostProcessing
) -> None:
    """
    Run inference with HailoAsyncInference using multiprocessing.
    """
    # Hailo 하드웨어 확인
    if not check_hailo_device():
        logger.error("Hailo 장치를 찾을 수 없습니다. 프로그램을 종료합니다.")
        return

    # 큐 크기 증가
    input_queue = Queue(maxsize=10)  # 입력 큐 크기 증가
    output_queue = Queue(maxsize=10)  # 출력 큐 크기 증가
    stop_event = Event()
    fps_shared = Value('d', 0.0)  # FPS 공유 변수
    cap = None
    hailo_inference = None

    try:
        if input_source == "camera":
            # 여러 카메라 장치 시도
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
                    raise RuntimeError("어떤 카메라도 열 수 없습니다.")

        # 버퍼 최소화
        if cap is not None:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Hailo 추론 객체 초기화 전에 큐 상태 확인
        logger.info("Hailo 추론 객체 초기화 중...")
        try:
            hailo_inference = HailoAsyncInference(
                net_path, input_queue, output_queue, batch_size,
                output_type=data_type_dict
            )
            height, width, _ = hailo_inference.get_input_shape()
            logger.info(f"모델 입력 크기: {width}x{height}")
        except Exception as e:
            logger.error(f"Hailo 추론 객체 초기화 실패: {e}")
            raise

        # 프로세스 생성
        preprocess_process = mp.Process(
            target=preprocess_from_cap,
            args=(cap, input_queue, width, height, post_processing, stop_event, fps_shared)
        )
        postprocess_process = mp.Process(
            target=postprocess_output,
            args=(output_queue, output_path, width, height, class_num, post_processing, stop_event, fps_shared)
        )

        # 프로세스 시작
        preprocess_process.start()
        postprocess_process.start()

        # 추론 실행
        logger.info("추론 시작...")
        hailo_inference.run()

        # 프로세스 종료 대기
        preprocess_process.join()
        postprocess_process.join()

        logger.info('추론이 성공적으로 완료되었습니다!')

    except Exception as e:
        logger.error(f"추론 중 오류 발생: {str(e)}")
        stop_event.set()
        if cap is not None:
            cap.release()
    finally:
        try:
            if hailo_inference is not None:
                # Hailo 리소스 정리
                hailo_inference = None
        except Exception as e:
            logger.error(f"Hailo 리소스 정리 중 오류 발생: {e}")
        
        try:
            cv2.destroyAllWindows()
        except:
            pass


def pose_estimation_process() -> None:
    """
    Main function to run the script.
    """
    # Hailo 하드웨어 확인
    if not check_hailo_device():
        logger.error("Hailo 장치를 찾을 수 없습니다. 프로그램을 종료합니다.")
        return

    args = parse_args()
    output_path = create_output_directory()
    output_type_dict = output_data_type2dict(HEF(args.net), 'FLOAT32')

    post_processing = PoseEstPostProcessing(
        max_detections=300,
        score_threshold=0.001,
        nms_iou_thresh=0.7,
        regression_length=15,
        strides=[8, 16, 32]
    )

    try:
        infer(
            args.input, args.net, int(args.batch_size), int(args.class_num),
            output_path, output_type_dict, post_processing
        )
    except Exception as e:
        logger.error(f"프로그램 실행 중 오류 발생: {e}")

if __name__ == "__main__":
    pose_estimation_process()
# End-of-file (EOF)
