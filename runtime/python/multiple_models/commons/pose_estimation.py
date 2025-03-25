#!/usr/bin/env python3

import os
import sys
import argparse
import cv2
import threading
import multiprocessing as mp
from multiprocessing import Process
import queue
from pathlib import Path
from loguru import logger
from PIL import Image
from typing import List
from commons.model_config import ModelConfig
from hailo_platform import HEF
from commons.pose_estimation_utils import (output_data_type2dict,
                                   check_process_errors, PoseEstPostProcessing)
import time
import numpy as np

# Add the parent directory to the system path to access utils module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import HailoAsyncInference

class PoseEstimation:
    def __init__(
        self, 
        model_config: ModelConfig,
        camera_queue: mp.Queue
    ):
        self.model_config = model_config
        self.output_type_dict = output_data_type2dict(
            HEF(self.model_config.getModelPath()),
            'FLOAT32'
        )
        self.post_processing = PoseEstPostProcessing(
            max_detections=300,
            score_threshold=0.001,
            nms_iou_thresh=0.7,
            regression_length=15,
            strides=[8, 16, 32]
        )
        self.camera_queue = camera_queue
        self.class_num = 1
                
        logger.info("Pose Estimation initialized")
        
    def create_output_directory(self) -> Path:
        """
        Create the output directory if it does not exist.

        Returns:
            Path: Path object for the output directory.
        """
        output_path = Path('output_images')
        output_path.mkdir(exist_ok=True)
        return output_path
        
    def preprocess(
        self,
        input_queue: queue.Queue,
        width: int,
        height: int,        
    ) -> None:
        """
        Preprocess and enqueue images into the input queue as they are ready.

        Args:
            images (list[Image.Image]): list of PIL.Image.Image objects.
            batch_size (int): Number of images in one batch.
            input_queue (mp.Queue): Queue for input images.
            width (int): Model input width.
            height (int): Model input height.
        """
        
        try:            
            frames = []
            processed_images = []
            
            logger.info("Pose Estimation preprocess")
            while True:
                logger.info("Pose Estimation preprocess while loop")
                ret, frame = self.camera_queue.get(timeout=10)
                logger.info("Pose Estimation preprocess while loop get")
                if not ret:
                    break
                
                frames.append(frame)
                # processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # processed_image = Image.fromarray(frame)
                processed_image = self.post_processing.preprocess_objstyle(frame, width, height)
                processed_image = Image.fromarray(processed_image)
                # logger.info(f"Pose Estimation preprocess width: {processed_image.width}, height: {processed_image.height}")
                
                processed_images.append(processed_image)
                
                if len(frames) == self.model_config.getBatchSize():
                    input_queue.put(processed_images)                                        
                    logger.info(f"pose estimation Sent camera frame to input queue {input_queue.qsize()}")                    
                    frames, processed_images = [], []
                    
            # input_queue.put(None)
            # logger.info("Pose Estimation preprocess while loop put None")
        except Exception as e:
            logger.error(f"Preprocess 오류: {e}")

    def postprocess(
        self,
        output_queue: queue.Queue,        
        display_queue: mp.Queue,
        width: int,
        height: int,
        class_num: int,
        orig_width: int = 1280,
        orig_height: int = 720
    ) -> None:
        """
        Process and visualize the output results.

        Args:
            output_queue (mp.Queue): Queue for output results.
            display_queue (mp.Queue): Queue for displaying results.
            width (int): Model input width.
            height (int): Model input height.
            class_num (int): Number of classes.
            orig_width (int, optional): Original frame width. Defaults to 1280.
            orig_height (int, optional): Original frame height. Defaults to 720.
        """
        image_id = 0
        while True:
            result = output_queue.get()
            if result is None:
                break  # Exit the loop if sentinel value is received

            processed_image, raw_detections = result
            # 포즈 추정 처리 및 시각화
            display_image = self.post_processing.postprocess_and_visualize(
                processed_image,
                raw_detections,
                'output_images/pose_estimation',
                image_id,
                height,
                width,
                class_num
            )
            
            # 원본 크기로 복원 (패딩 제거)
            display_image = self.remove_padding_and_resize(display_image, width, height, orig_width, orig_height)
            
            # 처리된 이미지를 디스플레이 큐에 추가
            display_queue.put(display_image)                
            image_id += 1
    
    def remove_padding_and_resize(self, image, model_width, model_height, orig_width, orig_height):
        """
        패딩을 제거하고 원본 프레임 크기로 복원하는 함수

        Args:
            image (numpy.ndarray): 처리된 이미지
            model_width (int): 모델 입력 너비
            model_height (int): 모델 입력 높이
            orig_width (int): 원본 프레임 너비
            orig_height (int): 원본 프레임 높이

        Returns:
            numpy.ndarray: 패딩이 제거되고 원본 크기로 복원된 이미지
        """
        try:
            # 원본 비율 계산
            orig_aspect = orig_width / orig_height
            
            # 모델 입력에 맞춰 스케일링된 크기 계산
            scale = min(model_width / orig_width, model_height / orig_height)
            scaled_width = int(orig_width * scale)
            scaled_height = int(orig_height * scale)
            
            # 패딩 오프셋 계산
            pad_x = (model_width - scaled_width) // 2
            pad_y = (model_height - scaled_height) // 2
            
            # 패딩을 제외한 실제 이미지 영역 추출
            if pad_x >= 0 and pad_y >= 0 and pad_x + scaled_width <= model_width and pad_y + scaled_height <= model_height:
                # 이미지 높이, 너비 확인
                h, w = image.shape[:2]
                
                # 모델 출력 크기가 모델 입력 크기와 다를 수 있으므로 스케일 조정
                output_scale_x = w / model_width
                output_scale_y = h / model_height
                
                # 패딩 오프셋 조정
                adjusted_pad_x = int(pad_x * output_scale_x)
                adjusted_pad_y = int(pad_y * output_scale_y)
                adjusted_scaled_width = int(scaled_width * output_scale_x)
                adjusted_scaled_height = int(scaled_height * output_scale_y)
                
                # 패딩 제거된 영역 추출
                if (adjusted_pad_y + adjusted_scaled_height <= h and 
                    adjusted_pad_x + adjusted_scaled_width <= w):
                    unpadded = image[adjusted_pad_y:adjusted_pad_y+adjusted_scaled_height, 
                                     adjusted_pad_x:adjusted_pad_x+adjusted_scaled_width]
                    
                    # 원본 크기로 리사이징
                    resized = cv2.resize(unpadded, (orig_width, orig_height))
                    logger.info(f"패딩 제거 및 리사이징 완료: {unpadded.shape} -> {resized.shape}")
                    return resized
            
            # 패딩 제거 실패 시 컬러 기반 패딩 제거 시도
            return self.remove_padding_by_color(image, orig_width, orig_height)
        
        except Exception as e:
            logger.error(f"패딩 제거 중 오류 발생: {e}")
            # 실패 시 단순 리사이징으로 복원
            return cv2.resize(image, (orig_width, orig_height))
    
    def remove_padding_by_color(self, image, orig_width, orig_height):
        """
        패딩 컬러를 기반으로 패딩을 제거하는 함수

        Args:
            image (numpy.ndarray): 처리된 이미지
            orig_width (int): 원본 프레임 너비
            orig_height (int): 원본 프레임 높이

        Returns:
            numpy.ndarray: 패딩이 제거되고 원본 크기로 복원된 이미지
        """
        try:
            h, w = image.shape[:2]
            padding_color = np.array([114, 114, 114])  # 패딩 컬러
            padding_threshold = 5  # 컬러 차이 허용 범위
            
            # 패딩 영역 감지
            y_start, y_end = 0, h-1
            x_start, x_end = 0, w-1
            
            # 상단 패딩 감지
            for y in range(h//4):  # 전체 높이의 1/4까지만 검사
                row = image[y]
                if not np.all(np.any(np.abs(row - padding_color) < padding_threshold, axis=1)):
                    y_start = y
                    break
            
            # 하단 패딩 감지
            for y in range(h-1, 3*h//4, -1):  # 전체 높이의 3/4부터 검사
                row = image[y]
                if not np.all(np.any(np.abs(row - padding_color) < padding_threshold, axis=1)):
                    y_end = y
                    break
            
            # 좌측 패딩 감지
            for x in range(w//4):  # 전체 너비의 1/4까지만 검사
                col = image[:, x]
                if not np.all(np.any(np.abs(col - padding_color) < padding_threshold, axis=1)):
                    x_start = x
                    break
            
            # 우측 패딩 감지
            for x in range(w-1, 3*w//4, -1):  # 전체 너비의 3/4부터 검사
                col = image[:, x]
                if not np.all(np.any(np.abs(col - padding_color) < padding_threshold, axis=1)):
                    x_end = x
                    break
            
            # 패딩 제거
            if y_start < y_end and x_start < x_end:
                unpadded_image = image[y_start:y_end+1, x_start:x_end+1]
                
                # 원본 크기로 리사이징
                resized = cv2.resize(unpadded_image, (orig_width, orig_height))
                logger.info(f"컬러 기반 패딩 제거 및 리사이징 완료: {unpadded_image.shape} -> {resized.shape}")
                return resized
        
        except Exception as e:
            logger.error(f"컬러 기반 패딩 제거 중 오류: {e}")
        
        # 모든 방법이 실패한 경우 단순 리사이징
        return cv2.resize(image, (orig_width, orig_height))

    def infer(
        self,
        display_queue: mp.Queue,
        orig_width: int = 1280,
        orig_height: int = 720
    ) -> None:
        """
        Run inference with HailoAsyncInference, handle processes, and ensure proper cleanup.

        Args:
            display_queue (mp.Queue): Queue for displaying results.
            orig_width (int, optional): Original camera frame width. Defaults to 1280.
            orig_height (int, optional): Original camera frame height. Defaults to 720.
        """
        input_queue = queue.Queue(maxsize=2)
        output_queue = queue.Queue(maxsize=2)
        
        hailo_inference = HailoAsyncInference(
            self.model_config.getModelPath(),
            input_queue,
            output_queue,
            self.model_config.getBatchSize(),
            # send_original_frame=True,
            output_type=self.output_type_dict
        )
        
        height, width, _ = hailo_inference.get_input_shape()
        logger.info(f"Pose Estimation infer height: {height}, width: {width}")

        preprocess = threading.Thread(
            target=self.preprocess,
            name="image_enqueuer",
            args=(
                input_queue, 
                width, 
                height, 
            )      
        )
        postprocess = threading.Thread(
            target=self.postprocess,
            name="image_processor",
            args=(
                output_queue, 
                display_queue,
                width,
                height, 
                self.class_num,
                orig_width,
                orig_height
            )
        )

        logger.info("Pose Estimation preprocess start")
        preprocess.start()
        logger.info("Pose Estimation postprocess start")
        postprocess.start()

        try:
            logger.info("Pose Estimation hailo_inference.run")
            hailo_inference.run(model_name="pose_estimation")
            logger.info("Pose Estimation preprocess.join")
            preprocess.join()
            # To signal processing process to exit
            output_queue.put(None)
            logger.info("Pose Estimation postprocess.join")
            postprocess.join()
            logger.info("Pose Estimation check_process_errors")
            check_process_errors(preprocess, postprocess)
        
            logger.info("Inference was successful!")
        
        except Exception as e:
            logger.error(f"Inference error: {e}")
            # queue.Queue 객체에는 close 메소드가 없으므로 제거
            # input_queue와 output_queue는 자동으로 가비지 컬렉션됨
            try:               
                # 최종 상태 로깅
                logger.info(f"전처리 스레드 상태: {'실행 중' if preprocess.is_alive() else '종료됨'}")
                logger.info(f"후처리 스레드 상태: {'실행 중' if postprocess.is_alive() else '종료됨'}")
                
            except Exception as thread_ex:
                logger.error(f"Thread clean-up error: {thread_ex}")

            os._exit(1)  # Force exit on error
    

# End-of-file (EOF)