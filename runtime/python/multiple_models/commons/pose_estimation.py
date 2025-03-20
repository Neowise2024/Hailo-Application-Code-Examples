#!/usr/bin/env python3

import os
import sys
import cv2
import numpy as np
from PIL import Image
from loguru import logger

# 현재 디렉토리 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_inference import ModelInference

class PoseEstimation(ModelInference):
    """포즈 추정 모델 클래스"""
    
    def __init__(self, net_path: str, labels_path: str = "resources/coco.txt", input_queue=None, output_queue=None, stop_event=None, output_path=None, batch_size=1):
        super().__init__(net_path, labels_path, input_queue, output_queue, stop_event, output_path)
        from pose_estimation_utils import PoseEstPostProcessing
        self.post_processing = PoseEstPostProcessing(
            max_detections=300,
            score_threshold=0.001,
            nms_iou_thresh=0.7,
            regression_length=15,
            strides=[8, 16, 32]
        )
        self.batch_size = batch_size
        logger.info(f"포즈 추정 모델 초기화 완료: batch_size={self.batch_size}")
    
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        logger.debug(f"포즈 추정 전처리 시작: 입력 프레임 크기={frame.shape}")
        
        try:
            # 입력 크기를 640x640으로 리사이즈
            frame = cv2.resize(frame, (640, 640))
            logger.debug(f"리사이즈 완료: {frame.shape}")
            
            # OpenCV BGR을 RGB로 변환
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            logger.debug(f"RGB 변환 완료: {frame_rgb.shape}")
            
            # PIL Image로 변환
            pil_image = Image.fromarray(frame_rgb)
            logger.debug(f"PIL Image 변환 완료: {pil_image.size}, {pil_image.mode}")
            
            # 전처리 수행
            processed = self.post_processing.preprocess(pil_image, 640, 640)
            logger.debug(f"전처리 완료: {type(processed)}")
            
            return processed
            
        except Exception as e:
            logger.error(f"포즈 추정 전처리 중 오류 발생: {str(e)}")
            raise
    
    def postprocess(self, frame: np.ndarray, inference_result: any) -> np.ndarray:
        logger.debug(f"포즈 추정 후처리 시작")
        
        detections = self.post_processing.extract_detections(inference_result, frame.shape[0], frame.shape[1], 1)
        if detections is None:
            logger.warning("검출 결과가 없습니다")
            return frame
        
        output = self.post_processing.draw_detections(frame, {
            'bboxes': detections['bboxes'][0],
            'keypoints': detections['keypoints'][0],
            'joint_scores': detections['joint_scores'][0],
            'scores': detections['scores'][0]
        })
        
        return output 