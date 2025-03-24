#!/usr/bin/env python3

import argparse
import os
import sys
import cv2
from pathlib import Path
from loguru import logger
import numpy as np
from typing import List
from multiprocessing import Process, Queue, Manager
from commons.object_detection import ObjectDetection
from commons.pose_estimation import PoseEstimation
from commons.model_config import ModelConfig
import time
from PIL import Image
def parse_args() -> argparse.Namespace:
    """
    Initialize argument parser for the script.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Detection Example")
    parser.add_argument(
        "-n", "--net", 
        help="Path for the network in HEF format.",
        default="yolov7.hef"
    )
    parser.add_argument(
        "-i", "--input", 
        default="zidane.jpg",
        help="Path to the input - either an image or a folder of images."
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
        default="coco.txt",
        help="Path to a text file containing labels. If no labels file is provided, coco2017 will be used."
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

    return args



def display(display_queues: List[Queue]):
    
    logger.info("Starting multiple models display")
    
    # 원본 이미지를 그대로 사용하고 필요시 윈도우 크기만 조정합니다
    cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Output", cv2.WINDOW_NORMAL, cv2.WINDOW_NORMAL)
    
    # 초기 캔버스 생성 (실제 크기는 첫 프레임 수신 후 조정됨)
    windows = None
    image_id = 0
    
    # 화면에 맞게 윈도우 크기 조정을 위한 설정
    screen_width = 1280  # 일반적인 모니터 가로 크기
    screen_height = 720  # 일반적인 모니터 세로 크기
    
    # 디스플레이 크기 설정
    display_width = 1280
    display_height = 320
    
    # 출력 디렉토리 생성
    output_dir = "output_images/multiple_models"
    os.makedirs(output_dir, exist_ok=True)
    
    while True:        
        try:
            # 타임아웃 처리
            pose_estimation_frame = display_queues[0].get(timeout=10)
            object_detection_frame = display_queues[1].get(timeout=10)
            
            # 프레임 크기 확인 및 로깅
            pe_height, pe_width = pose_estimation_frame.shape[:2]
            od_height, od_width = object_detection_frame.shape[:2]
            logger.info(f"Pose Estimation 프레임 크기: {pe_width}x{pe_height}")
            logger.info(f"Object Detection 프레임 크기: {od_width}x{od_height}")
            
            # 첫 프레임일 경우 캔버스 크기 초기화
            if windows is None:
                # 두 프레임을 나란히 배치할 캔버스 생성 (고정 크기 사용)
                combined_width = display_width  # 1280
                combined_height = display_height  # 320
                windows = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
                
                # 화면에 맞게 윈도우 크기 조정
                cv2.resizeWindow("Output", combined_width, combined_height)
                
                logger.info(f"캔버스 크기 설정: {combined_width}x{combined_height}")
            
            # 프레임 리사이징 (NumPy 배열로 직접 처리)
            object_detection_resized = cv2.resize(object_detection_frame, (640, 320))
            pose_estimation_resized = cv2.resize(pose_estimation_frame, (640, 320))
            
            # 리사이징된 프레임을 캔버스에 배치
            windows[0:320, 0:640] = pose_estimation_resized    # 왼쪽 - 포즈 추정
            windows[0:320, 640:1280] = object_detection_resized  # 오른쪽 - 객체 감지
            
            # 출력 저장 및 표시
            cv2.imwrite(f"{output_dir}/output_{image_id:04d}.jpg", windows)
            cv2.imshow("Output", windows)
            
            logger.info(f"프레임 {image_id} 표시 완료")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            image_id += 1
        
        except Exception as e:
            logger.error(f"디스플레이 오류: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # 짧은 대기 후 계속 시도
            time.sleep(0.1)

def frame_queue_process(temp_camera_queue: Queue, cap: cv2.VideoCapture):
    logger.info("Starting frame queue size: {}".format(temp_camera_queue.qsize()))  
    try:
        while True:
            ret, frame = cap.read()
            logger.info("Frame queue process ret: {}".format(ret))
            if not ret:
                logger.info("Frame queue process end")
                break
            temp_camera_queue.put((ret, frame.copy()))
            logger.info(f"Frame sent to temp camera queue {temp_camera_queue.qsize()}")
    except Exception as e:
        logger.error(f"Error in frame queue: {e}")
        
def frame_queues_process(temp_camera_queues: List[Queue], cap: cv2.VideoCapture):
    logger.info("Starting frame queue size: {}".format(temp_camera_queues.count))  
    try:
        while True:
            ret, frame = cap.read()
            logger.info("Frame queue process ret: {}".format(ret))
            if not ret:
                logger.info("Frame queue process end")
                break
            
            for i in range(len(temp_camera_queues)):
                temp_camera_queues[i].put((ret, frame.copy()))
                logger.info(f"{i} Model Frame sent to temp camera queue {temp_camera_queues[i].qsize()}")
    except Exception as e:
        logger.error(f"Error in frame queue: {e}")

def main() -> None:
    """
    Main function to run the script.
    """
    # Parse command line arguments
    # 우선 파라미터 처리는 하지 않고 하드 코딩을 하도록 하자. 
    # args = parse_args()
   
    
    manager = Manager()
        
    
    cap = None
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    camera_queues = []
    camera_queues.append(manager.Queue(maxsize=2))
    camera_queues.append( manager.Queue(maxsize=2))    
    camera_processes = []
    
    
        
    process_result = Process(
        target=frame_queues_process,
        args=(
            camera_queues, 
            cap
        )
    )
    
    process_result.start()   
    logger.info("Starting frame queue process start")
    
    
    # 임시 모델 설정 
    object_detection_model_config = ModelConfig(
        model_name="Object Detection",
        model_path="hefs/yolov11s_object_h8l.hef",
        labels_path="hefs/coco.txt",
        batch_size=1
    )

    # 임시 모델 설정 
    pose_estimation_model_config = ModelConfig(
        model_name="Pose Estimation",
        model_path="hefs/yolov8s_pose_h8l.hef",
        batch_size=1
    )
    
    model_display_queues = []
    model_display_queues.append(manager.Queue(maxsize=2))
    model_display_queues.append(manager.Queue(maxsize=2))    
    
  
    logger.info("Starting model output display")
    # 모든 디스플레이 큐를 하나의 프로세스로 처리
    display_process = Process(
        target=display,
        args=(model_display_queues,)  # 전체 리스트를 전달
    )
    display_process.start()
    
    # 임시로 주석 처리 해뒀어 왜냐하면 object detection 모델이 정상적으로 처리 되는지 확인해야해
    pose_estimation = PoseEstimation(
        pose_estimation_model_config,
        camera_queue=camera_queues[0]
    )
    
    pose_estimation_process = Process(
        target=pose_estimation.infer,
        args=(            
            model_display_queues[0], 
        )
    )
    pose_estimation_process.start()
    
    # Start the inference
    object_detection = ObjectDetection(
        object_detection_model_config,
        camera_queue=camera_queues[1]
    )
    logger.info("Starting object detection inference")
    # 여기서 100% 멈출 꺼야 왜냐하면 여기가 동기로 처리되거든 그래서 이걸 비동기로 바꿔야 함. 
    object_detection_process = Process(
        target=object_detection.infer,
        args=(model_display_queues[1],)
    )
    object_detection_process.start()
    
    # object_detection.infer(model_display_queues[0])
    
    display_process.join()
    
    # object_detection_process.join()
    
    for camera_process in camera_processes:
        camera_process.join()
if __name__ == "__main__":
    main()
