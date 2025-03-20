#!/usr/bin/env python3

import os
import sys
import cv2
import numpy as np
from loguru import logger

# 현재 디렉토리 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_inference import ModelInference

class ObjectDetection(ModelInference):
    """객체 감지 모델 클래스"""
    
    def __init__(self, net_path: str, labels_path: str = "resources/coco.txt", input_queue=None, output_queue=None, stop_event=None, output_path=None, batch_size=1):
        super().__init__(net_path, labels_path, input_queue, output_queue, stop_event, output_path)
        from object_detection_utils import ObjectDetectionUtils
        self.utils = ObjectDetectionUtils(labels_path)
        self.batch_size = batch_size
        logger.info(f"객체 감지 모델 초기화 완료: batch_size={self.batch_size}")
    
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        logger.debug(f"객체 감지 전처리 시작: 입력 프레임 크기={frame.shape}")
        
        try:
            # 전처리 수행 (ObjectDetectionUtils의 preprocess 메서드 사용)
            processed = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed = self.utils.preprocess(processed, 640, 640)
            logger.debug(f"전처리 완료: {type(processed)}")
            
            return processed
                
        except Exception as e:
            logger.error(f"객체 감지 전처리 중 오류 발생: {str(e)}")
            raise
    
    def postprocess(self, frame: np.ndarray, inference_result: any) -> np.ndarray:
        logger.debug(f"객체 감지 후처리 시작")
        
        detections = self.utils.extract_detections(inference_result)
        return self.utils.draw_detections(detections, frame) 