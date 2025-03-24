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
                processed_image = Image.fromarray(frame)
                processed_image = self.post_processing.preprocess(processed_image, width, height)
                               
                logger.info(f"Pose Estimation preprocess width: {processed_image.width}, height: {processed_image.height}")
                
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
    ) -> None:
        """
        Process and visualize the output results.

        Args:
            output_queue (mp.Queue): Queue for output results.
            output_path (Path): Path to save the output images.
            width (int): Image width.
            height (int): Image height.
            class_num (int): Number of classes.
            post_processing (PoseEstPostProcessing): Post-processing configuration.
        """
        image_id = 0
        while True:
            result = output_queue.get()
            if result is None:
                break  # Exit the loop if sentinel value is received

            processed_image, raw_detections = result
            display_image = self.post_processing.postprocess_and_visualize(
                processed_image,
                raw_detections,
                'output_images/pose_estimation',
                image_id,
                height,
                width,
                class_num
            )
            
            display_queue.put(display_image)                

            image_id += 1

    def infer(
        self,
        display_queue: mp.Queue,
    ) -> None:
        """
        Run inference with HailoAsyncInference, handle processes, and ensure proper cleanup.

        Args:
            images (list[Image.Image]): List of images to process.
            net_path (str): Path to the HEF model file.
            batch_size (int): Number of images per batch.
            class_num (int): Number of classes.
            output_path (Path): Path to save the output images.
            data_type_dict (dict): Dictionary of layer names and data types.
            post_processing (PoseEstPostProcessing): Post-processing configuration.
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
                width,  # 원본 크기가 아닌 모델 입력 크기 사용
                height, 
                self.class_num, 
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