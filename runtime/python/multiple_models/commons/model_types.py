from enum import Enum, auto
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import cv2

class ModelType(Enum):
    """모델 타입 열거형"""
    OBJECT_DETECTION = auto()
    POSE_ESTIMATION = auto()
    UNKNOWN = auto()


class ModelInputData:
    """모델 입력 데이터 클래스 (NamedTuple에서 일반 클래스로 변경)"""
    def __init__(self, frame_id: int, model_type: ModelType, original_frame: np.ndarray, processed_frame: np.ndarray = None, source_id: int = 0):
        self.frame_id = frame_id
        self.model_type = model_type
        self.original_frame = original_frame
        self.processed_frame = processed_frame
        self.source_id = source_id
    
    # 필드 접근을 위한 getter 메서드
    def get_frame_id(self) -> int:
        return self.frame_id
    
    def get_model_type(self) -> ModelType:
        return self.model_type
    
    def get_original_frame(self) -> np.ndarray:
        return self.original_frame
    
    def get_processed_frame(self) -> Optional[np.ndarray]:
        return self.processed_frame
    
    def get_source_id(self) -> int:
        return self.source_id


class ModelOutputData:
    """모델 출력 데이터 클래스 (NamedTuple에서 일반 클래스로 변경)"""
    def __init__(self, frame_id: int, model_type: ModelType, original_frame: np.ndarray, results: Dict[str, Any]):
        self.frame_id = frame_id
        self.model_type = model_type
        self.original_frame = original_frame
        self.results = results
        
        # 모델별 메타데이터 추가
        self.model_id = None  # 모델 식별자
        self.model_name = None  # 모델 이름
        self.inference_time = 0.0  # 추론 시간 (초)
        self.postprocess_time = 0.0  # 후처리 시간 (초)
        self.total_time = 0.0  # 총 처리 시간 (초)
    
    # 필드 접근을 위한 getter 메서드
    def get_frame_id(self) -> int:
        return self.frame_id
    
    def get_model_type(self) -> ModelType:
        return self.model_type
    
    def get_original_frame(self) -> np.ndarray:
        return self.original_frame
    
    def get_results(self) -> Dict[str, Any]:
        return self.results
    
    def get_model_id(self) -> str:
        return self.model_id
    
    def get_model_name(self) -> str:
        return self.model_name
    
    def get_inference_time(self) -> float:
        return self.inference_time
    
    def get_total_time(self) -> float:
        return self.total_time 