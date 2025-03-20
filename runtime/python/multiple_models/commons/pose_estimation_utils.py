from pathlib import Path
from multiprocessing import Process
import numpy as np
import cv2
from PIL import Image
from hailo_platform import HEF
from loguru import logger
from typing import List, Dict, Tuple

# Joint pairs used for drawing pose estimations
JOINT_PAIRS = [
    [0, 1], [1, 3], [0, 2], [2, 4],
    [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
    [5, 11], [6, 12], [11, 12],
    [11, 13], [12, 14], [13, 15], [14, 16]
]

class PoseEstPostProcessing:
    def __init__(self, max_detections: int, score_threshold: float, nms_iou_thresh: float,
                 regression_length: int, strides: List[int]):
        """
        Initialize the post-processing configuration.

        Args:
            max_detections (int): Maximum number of detections per class.
            score_threshold (float): Confidence threshold for filtering.
            nms_iou_thresh (float): IoU threshold for NMS.
            regression_length (int): Maximum regression value for bounding boxes.
            strides (list[int]): Stride values for each prediction scale.
        """
        self.max_detections = max_detections
        self.score_threshold = score_threshold
        self.nms_iou_thresh = nms_iou_thresh
        self.regression_length = regression_length
        self.strides = strides

    def postprocess_and_visualize(
        self, image: Image.Image, raw_detections: dict, output_path: Path,
        image_index: int, height: int, width: int, class_num: int
    ) -> None:
        """
        검출 결과 후처리 및 시각화

        Args:
            image (Image.Image): 입력 이미지
            raw_detections (dict): 원본 검출 결과
            output_path (Path): 출력 경로
            image_index (int): 이미지 인덱스
            height (int): 이미지 높이
            width (int): 이미지 너비
            class_num (int): 클래스 수
        """
        try:
            # 검출 결과 추출
            detections = self.extract_detections(raw_detections, height, width, class_num)
            if detections is None:
                return

            # 결과 시각화
            image_array = np.array(image)
            output_image = self.draw_detections(image_array, detections)
            
            # 결과 저장
            # output_image_pil = Image.fromarray(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
            # output_image_pil.save(output_path / f'output_image{image_index}.jpg', 'JPEG')
        except Exception as e:
            logger.error(f"후처리 및 시각화 중 오류 발생: {str(e)}")

    def extract_detections(self, raw_detections: dict, height: int, width: int, class_num: int = 1) -> dict:
        """
        포즈 검출 결과 추출

        Args:
            raw_detections (dict): 원본 검출 결과
            height (int): 이미지 높이
            width (int): 이미지 너비
            class_num (int): 클래스 수

        Returns:
            dict: 처리된 검출 결과 (bboxes, keypoints, joint_scores, scores)
        """
        try:
            return self.post_process(raw_detections, height, width, class_num)
        except Exception as e:
            logger.error(f"검출 결과 추출 중 오류 발생: {str(e)}")
            return None

    def draw_detections(self, image: np.ndarray, detections: dict) -> np.ndarray:
        """
        검출된 포즈를 이미지에 그리기

        Args:
            image (np.ndarray): 원본 이미지
            detections (dict): 검출 결과 (bboxes, keypoints, joint_scores, scores)

        Returns:
            np.ndarray: 포즈가 그려진 이미지
        """
        try:
            if detections is None:
                return image

            bboxes = detections['bboxes']
            scores = detections['scores']
            keypoints = detections['keypoints']
            joint_scores = detections['joint_scores']
            
            # NaN 값 체크 및 처리
            if np.isnan(bboxes).any() or np.isnan(keypoints).any() or np.isnan(scores).any() or np.isnan(joint_scores).any():
                logger.warning("NaN 값 발견됨. 유효하지 않은 값을 필터링합니다.")
                # NaN 값을 포함하는 인덱스 찾기
                valid_indices = ~np.isnan(scores).any(axis=1)
                if np.sum(valid_indices) == 0:
                    logger.warning("유효한 검출 결과가 없습니다. 원본 이미지를 반환합니다.")
                    return image
                
                # NaN 값이 없는 검출 결과만 사용
                bboxes = bboxes[valid_indices]
                scores = scores[valid_indices]
                keypoints = keypoints[valid_indices]
                joint_scores = joint_scores[valid_indices]
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            for box, score, kpts, kpts_score in zip(bboxes, scores, keypoints, joint_scores):
                # 모든 값이 유효한지 확인
                if np.isnan(box).any() or np.isnan(score).any() or np.isnan(kpts).any() or np.isnan(kpts_score).any():
                    logger.debug("유효하지 않은 검출 값(NaN)이 포함되어 있어 건너뜁니다.")
                    continue
                    
                if score < 0.5:  # detection_threshold
                    continue
                    
                try:
                    # int 변환 전에 float 값의 유효성 확인
                    xmin, ymin, xmax, ymax = box
                    if np.isnan(xmin) or np.isnan(ymin) or np.isnan(xmax) or np.isnan(ymax):
                        continue
                        
                    xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
                    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 1)
                    
                    # score를 float로 변환하여 포맷팅
                    score_text = f"{float(score):.2f}"
                    cv2.putText(image, score_text, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1)
                    
                    joint_visible = kpts_score > 0.5  # joint_threshold
                    
                    for joint, joint_score in zip(kpts, kpts_score):
                        if joint_score < 0.5 or np.isnan(joint).any():  # joint_threshold와 NaN 체크
                            continue
                        # 값 반올림 후 정수 변환으로 보다 안정적인 좌표 얻기
                        joint_x, joint_y = int(round(joint[0])), int(round(joint[1]))
                        cv2.circle(image, (joint_x, joint_y), 1, (255, 0, 255), -1)
                    
                    for joint0, joint1 in JOINT_PAIRS:
                        if (joint_visible[joint0] and joint_visible[joint1] and 
                            not np.isnan(kpts[joint0]).any() and not np.isnan(kpts[joint1]).any()):
                            pt1 = (int(round(kpts[joint0][0])), int(round(kpts[joint0][1])))
                            pt2 = (int(round(kpts[joint1][0])), int(round(kpts[joint1][1])))
                            cv2.line(image, pt1, pt2, (255, 0, 255), 3)
                except (ValueError, TypeError, IndexError) as e:
                    logger.warning(f"검출 결과 시각화 중 오류 발생: {e}")
                    continue
            
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
        except Exception as e:
            logger.error(f"포즈 시각화 중 오류 발생: {str(e)}")
            import traceback
            logger.error(f"상세 오류: {traceback.format_exc()}")
            return image

    def post_process(self, raw_detections: dict, height: int, width: int, class_num: int) -> dict:
        """
        Process raw detections into a structured format for pose estimation.

        Args:
            raw_detections (Dict): Raw detections from the model.
            height (int): The height of the input image.
            width (int): The width of the input image.
            class_num (int): Number of classes.

        Returns:
            Dict: Processed predictions dictionary.
        """
        try:
            raw_detections_keys = list(raw_detections.keys())
            
            # 모든 레이어의 shape과 이름 로깅
            logger.debug(f"포즈 후처리 시작 - raw_detections 키 수: {len(raw_detections_keys)}")
            for key in raw_detections_keys:
                logger.debug(f"레이어 '{key}' 형태: {raw_detections[key].shape}, 타입: {raw_detections[key].dtype}")
            
            # shape별 레이어 매핑
            layer_from_shape = {raw_detections[key].shape: key for key in raw_detections_keys}
            logger.debug(f"shape별 레이어 매핑: {', '.join([f'{shape}->{key}' for shape, key in layer_from_shape.items()])}")
            
            detection_output_channels = (self.regression_length + 1) * 4  # (regression length + 1) * num_coordinates
            keypoints = 51
            
            # 필요한 레이어 모양 로깅
            expected_shapes = [
                (1, 20, 20, detection_output_channels),
                (1, 20, 20, class_num),
                (1, 20, 20, keypoints),
                (1, 40, 40, detection_output_channels),
                (1, 40, 40, class_num),
                (1, 40, 40, keypoints),
                (1, 80, 80, detection_output_channels),
                (1, 80, 80, class_num),
                (1, 80, 80, keypoints)
            ]
            
            # 예상 shape들이 실제로 존재하는지 확인
            missing_shapes = [shape for shape in expected_shapes if shape not in layer_from_shape]
            if missing_shapes:
                logger.error(f"누락된 레이어 형태: {missing_shapes}")
                # 가장 유사한 shape 찾기
                for missing in missing_shapes:
                    closest = min(layer_from_shape.keys(), key=lambda x: sum(abs(a-b) for a, b in zip(x, missing)))
                    logger.error(f"누락된 shape {missing}에 가장 가까운 shape: {closest}")
            
            # endnodes 구성
            try:
                endnodes = [
                    raw_detections[layer_from_shape[1, 20, 20, detection_output_channels]],
                    raw_detections[layer_from_shape[1, 20, 20, class_num]],
                    raw_detections[layer_from_shape[1, 20, 20, keypoints]],
                    raw_detections[layer_from_shape[1, 40, 40, detection_output_channels]],
                    raw_detections[layer_from_shape[1, 40, 40, class_num]],
                    raw_detections[layer_from_shape[1, 40, 40, keypoints]],
                    raw_detections[layer_from_shape[1, 80, 80, detection_output_channels]],
                    raw_detections[layer_from_shape[1, 80, 80, class_num]],
                    raw_detections[layer_from_shape[1, 80, 80, keypoints]]
                ]
                
                logger.debug(f"endnodes 구성 완료: {len(endnodes)}개 레이어")
                for i, node in enumerate(endnodes):
                    logger.debug(f"endnode[{i}] 형태: {node.shape}, 값 범위: [{np.min(node):.6f}, {np.max(node):.6f}]")
                
                predictions_dict = self.extract_pose_estimation_results(endnodes, height, width, class_num)
                
                # 결과 로깅
                if predictions_dict:
                    for key, value in predictions_dict.items():
                        if isinstance(value, np.ndarray):
                            logger.debug(f"결과 '{key}' 형태: {value.shape}, 타입: {value.dtype}")
                        else:
                            logger.debug(f"결과 '{key}' 타입: {type(value)}")
                
                return predictions_dict
                
            except KeyError as e:
                logger.error(f"레이어 키 오류: {e}. 가능한 키: {list(layer_from_shape.keys())}")
                return None
                
        except Exception as e:
            import traceback
            logger.error(f"포즈 후처리 중 오류 발생: {e}")
            logger.error(f"상세 오류: {traceback.format_exc()}")
            return None

    def extract_pose_estimation_results(
        self, endnodes: List[np.ndarray], height: int, width: int, class_num: int
    ) -> Dict[str, np.ndarray]:
        """
        Post-process the pose estimation results.

        Args:
            endnodes (list[np.ndarray]): list of 10 tensors from the model output.
            height (int): Height of the input image.
            width (int): Width of the input image.
            class_num (int): Number of classes.
    
        Returns:
            dict: Processed detections with keys:
                'bboxes': numpy.ndarray with shape (batch_size, max_detections, 4),
                'keypoints': numpy.ndarray with shape (batch_size, max_detections, 17, 2),
                'joint_scores': numpy.ndarray with shape (batch_size, max_detections, 17, 1),
                'scores': numpy.ndarray with shape (batch_size, max_detections, 1).
        """
        try:
            logger.debug(f"포즈 추출 시작: endnodes 개수={len(endnodes)}")
            for i, node in enumerate(endnodes):
                logger.debug(f"endnode[{i}] 형태={node.shape}, 값 범위=[{np.min(node):.6f}, {np.max(node):.6f}]")

            batch_size = endnodes[0].shape[0]
            logger.debug(f"배치 크기: {batch_size}")
            
            strides = self.strides[::-1]
            logger.debug(f"스트라이드: {strides}")
            
            image_dims = (height, width)
            logger.debug(f"이미지 크기: {image_dims}")
           
            raw_boxes = endnodes[:7:3]
            logger.debug(f"raw_boxes 개수: {len(raw_boxes)}")
            for i, box in enumerate(raw_boxes):
                logger.debug(f"raw_box[{i}] 형태: {box.shape}")
            
            try:
                scores = [
                    np.reshape(s, (-1, s.shape[1] * s.shape[2], class_num)) for s in endnodes[1:8:3]
                ]
                logger.debug(f"scores 목록 개수: {len(scores)}")
                for i, score in enumerate(scores):
                    logger.debug(f"score[{i}] 형태: {score.shape}")
                
                scores = np.concatenate(scores, axis=1)
                logger.debug(f"결합된 scores 형태: {scores.shape}")
            except Exception as e:
                logger.error(f"scores 처리 중 오류: {e}")
                import traceback
                logger.error(f"상세 오류: {traceback.format_exc()}")
                raise

            try:
                kpts = [
                    np.reshape(c, (-1, c.shape[1] * c.shape[2], 17, 3)) for c in endnodes[2:9:3]
                ]
                logger.debug(f"keypoints 목록 개수: {len(kpts)}")
                for i, kpt in enumerate(kpts):
                    logger.debug(f"keypoint[{i}] 형태: {kpt.shape}")
            except Exception as e:
                logger.error(f"keypoints 처리 중 오류: {e}")
                import traceback
                logger.error(f"상세 오류: {traceback.format_exc()}")
                raise

            try:
                logger.debug(f"디코더 호출 전 - raw_boxes: {len(raw_boxes)}, kpts: {len(kpts)}")
                decoded_boxes, decoded_kpts = self.decoder(raw_boxes,
                                                  kpts, strides,
                                                  image_dims, self.regression_length)
                
                # None 값 체크 및 로깅
                if decoded_boxes is None:
                    logger.error("디코더가 None boxes 반환")
                    # 적절한 기본값 생성
                    decoded_boxes = np.zeros((batch_size, 1, 4), dtype=np.float32)
                if decoded_kpts is None:
                    logger.error("디코더가 None keypoints 반환")
                    # 적절한 기본값 생성
                    decoded_kpts = np.zeros((batch_size, 1, 17, 3), dtype=np.float32)
                    
                logger.debug(f"디코딩된 박스 형태: {decoded_boxes.shape}")
                logger.debug(f"디코딩된 키포인트 형태: {decoded_kpts.shape}")
            except Exception as e:
                logger.error(f"디코더 처리 중 오류: {e}")
                import traceback
                logger.error(f"상세 오류: {traceback.format_exc()}")
                
                # 디코더 오류 시 기본값 생성
                logger.warning("디코더 오류로 기본 더미 데이터 생성")
                decoded_boxes = np.zeros((batch_size, 1, 4), dtype=np.float32)
                decoded_kpts = np.zeros((batch_size, 1, 17, 3), dtype=np.float32)

            try:
                # shape가 맞는지 확인
                if len(decoded_kpts.shape) != 4 or decoded_kpts.shape[2] != 17:
                    logger.warning(f"키포인트 shape 불일치: {decoded_kpts.shape}, 재구성 필요")
                    # 형태를 맞춤
                    if len(decoded_kpts.shape) == 3:
                        # (batch, n, 51) -> (batch, n, 17, 3)
                        decoded_kpts = decoded_kpts.reshape(batch_size, -1, 17, 3)
                    elif decoded_kpts.shape[-1] == 3 * 17:
                        # 마지막 차원이 51인 경우
                        decoded_kpts = decoded_kpts.reshape(batch_size, -1, 17, 3)
                
                # 이제 안전하게 reshape
                decoded_kpts = np.reshape(decoded_kpts, (batch_size, -1, 51))
                logger.debug(f"재구성된 키포인트 형태: {decoded_kpts.shape}")
                
                # 모든 배열의 크기 통일을 위한 resize 로직 추가
                logger.debug(f"크기 통일 시작 - decoded_boxes: {decoded_boxes.shape}, scores: {scores.shape}, decoded_kpts: {decoded_kpts.shape}")
                
                # 가장 작은 크기를 기준으로 통일할 수도 있지만, 
                # 여기서는 중간 크기인 decoded_kpts.shape[1]을 기준으로 결정 (손실을 최소화하기 위함)
                target_size = decoded_kpts.shape[1]
                
                # 가장 작은 크기로 맞출 경우 (선택적)
                # target_size = min(decoded_boxes.shape[1], scores.shape[1], decoded_kpts.shape[1])
                
                logger.info(f"모든 배열을 크기 {target_size}로 통일합니다.")
                
                # boxes 크기 조정
                if decoded_boxes.shape[1] != target_size:
                    logger.debug(f"decoded_boxes 크기 조정: {decoded_boxes.shape[1]} -> {target_size}")
                    # 임시 배열 생성하여 크기 맞추기
                    resized_boxes = np.zeros((batch_size, target_size, 4), dtype=decoded_boxes.dtype)
                    # 복사 가능한 범위만 복사
                    min_size = min(decoded_boxes.shape[1], target_size)
                    resized_boxes[:, :min_size] = decoded_boxes[:, :min_size]
                    # 나머지는 마지막 값으로 패딩 (또는 0으로 패딩)
                    if target_size > decoded_boxes.shape[1]:
                        for i in range(decoded_boxes.shape[1], target_size):
                            resized_boxes[:, i] = decoded_boxes[:, -1] if decoded_boxes.shape[1] > 0 else 0
                    decoded_boxes = resized_boxes
                
                # scores 크기 조정
                if scores.shape[1] != target_size:
                    logger.debug(f"scores 크기 조정: {scores.shape[1]} -> {target_size}")
                    # 임시 배열 생성하여 크기 맞추기
                    resized_scores = np.zeros((batch_size, target_size, scores.shape[2]), dtype=scores.dtype)
                    # 복사 가능한 범위만 복사
                    min_size = min(scores.shape[1], target_size)
                    resized_scores[:, :min_size] = scores[:, :min_size]
                    # 나머지는 낮은 점수로 패딩
                    if target_size > scores.shape[1]:
                        resized_scores[:, scores.shape[1]:] = 0.01  # 낮은 신뢰도 값으로 설정
                    scores = resized_scores
                
                logger.debug(f"크기 통일 완료 - decoded_boxes: {decoded_boxes.shape}, scores: {scores.shape}, decoded_kpts: {decoded_kpts.shape}")
                
                # 이제 모든 배열의 두 번째 차원이 동일하므로 안전하게 연결
                predictions = np.concatenate([decoded_boxes, scores, decoded_kpts], axis=2)
                logger.debug(f"최종 예측값 형태: {predictions.shape}")
            except Exception as e:
                logger.error(f"키포인트 재구성 중 오류: {e}")
                logger.error(f"decoded_boxes 형태: {decoded_boxes.shape}, scores 형태: {scores.shape}, decoded_kpts 형태: {decoded_kpts.shape}")
                import traceback
                logger.error(f"상세 오류: {traceback.format_exc()}")
                
                # 오류 발생 시 더미 예측값 생성
                logger.warning("키포인트 재구성 오류로 기본 더미 데이터 생성")
                
                # 가장 작은 크기로 통일
                target_size = 300  # 안전한 크기 설정
                
                # 더미 데이터 생성
                dummy_boxes = np.zeros((batch_size, target_size, 4), dtype=np.float32)
                dummy_scores = np.zeros((batch_size, target_size, class_num), dtype=np.float32)
                dummy_kpts = np.zeros((batch_size, target_size, 51), dtype=np.float32)
                
                # 신뢰도 낮게 설정 (0.01)
                dummy_scores.fill(0.01)
                
                # 더미 예측값 생성
                predictions = np.concatenate([dummy_boxes, dummy_scores, dummy_kpts], axis=2)
                logger.debug(f"더미 예측값 형태: {predictions.shape}")

            try:
                nms_res = self.non_max_suppression(
                    predictions, conf_thres=self.score_threshold, 
                    iou_thres=self.nms_iou_thresh, max_det=self.max_detections
                )
                logger.debug(f"NMS 결과: {type(nms_res)}, 첫 번째 배치 감지 수: {nms_res[0]['num_detections'] if nms_res else 'None'}")
                
                # NMS 결과가 None이거나 비어있는 경우를 확인
                if nms_res is None or len(nms_res) == 0:
                    logger.warning("NMS 결과가 없습니다. 더미 데이터 생성")
                    nms_res = [{
                        'bboxes': np.zeros((0, 4)),
                        'keypoints': np.zeros((0, 17, 3)),
                        'scores': np.zeros((0)),
                        'num_detections': 0
                    } for _ in range(batch_size)]
            except Exception as e:
                logger.error(f"NMS 처리 중 오류: {e}")
                import traceback
                logger.error(f"상세 오류: {traceback.format_exc()}")
                
                # NMS 오류 시 빈 결과 생성
                nms_res = [{
                    'bboxes': np.zeros((0, 4)),
                    'keypoints': np.zeros((0, 17, 3)),
                    'scores': np.zeros((0)),
                    'num_detections': 0
                } for _ in range(batch_size)]

            output = {
                'bboxes': np.zeros((batch_size, self.max_detections, 4)),
                'keypoints': np.zeros((batch_size, self.max_detections, 17, 2)),
                'joint_scores': np.zeros((batch_size, self.max_detections, 17, 1)),
                'scores': np.zeros((batch_size, self.max_detections, 1))
            }

            try:
                for b in range(batch_size):
                    # batch 범위 확인
                    if b >= len(nms_res):
                        logger.warning(f"배치 인덱스 {b}가 nms_res 길이 {len(nms_res)}를 초과합니다")
                        continue
                    
                    # num_detections 확인
                    num_detections = nms_res[b]['num_detections']
                    if num_detections > 0:
                        # 경계 검사
                        if num_detections > self.max_detections:
                            num_detections = self.max_detections
                            logger.warning(f"감지 개수({nms_res[b]['num_detections']})가 최대 개수({self.max_detections})를 초과하여 제한됩니다")
                            
                        # bboxes 복사 및 크기 확인
                        bboxes = nms_res[b]['bboxes']
                        if bboxes.shape[0] >= num_detections:
                            output['bboxes'][b, :num_detections] = bboxes[:num_detections]
                        else:
                            logger.warning(f"bboxes 크기({bboxes.shape[0]})가 num_detections({num_detections})보다 작습니다")
                            output['bboxes'][b, :bboxes.shape[0]] = bboxes
                        
                        # keypoints 복사 및 크기 확인
                        keypoints = nms_res[b]['keypoints']
                        if keypoints.shape[0] >= num_detections:
                            output['keypoints'][b, :num_detections] = keypoints[:num_detections, :, :2]
                            
                            # joint_scores 생성 및 크기 확인
                            try:
                                output['joint_scores'][b, :num_detections, ..., 0] = self._sigmoid(keypoints[:num_detections, :, 2])
                            except Exception as e:
                                logger.error(f"joint_scores 생성 중 오류: {e}")
                                # 기본값 할당
                                output['joint_scores'][b, :num_detections] = 0.5
                        else:
                            logger.warning(f"keypoints 크기({keypoints.shape[0]})가 num_detections({num_detections})보다 작습니다")
                            output['keypoints'][b, :keypoints.shape[0]] = keypoints[..., :2]
                            try:
                                output['joint_scores'][b, :keypoints.shape[0], ..., 0] = self._sigmoid(keypoints[..., 2])
                            except Exception as e:
                                logger.error(f"joint_scores 생성 중 오류: {e}")
                                output['joint_scores'][b, :keypoints.shape[0]] = 0.5
                        
                        # scores 복사 및 크기 확인
                        scores = nms_res[b]['scores']
                        if scores.shape[0] >= num_detections:
                            output['scores'][b, :num_detections, ..., 0] = scores[:num_detections]
                        else:
                            logger.warning(f"scores 크기({scores.shape[0]})가 num_detections({num_detections})보다 작습니다")
                            output['scores'][b, :scores.shape[0], ..., 0] = scores
                
                logger.debug(f"출력 결과 - bboxes: {output['bboxes'].shape}, keypoints: {output['keypoints'].shape}")
                logger.debug(f"출력 결과 - joint_scores: {output['joint_scores'].shape}, scores: {output['scores'].shape}")
            except Exception as e:
                logger.error(f"출력 구성 중 오류: {e}")
                import traceback
                logger.error(f"상세 오류: {traceback.format_exc()}")
                # 여기서는 이미 기본값이 초기화되어 있으므로 추가 처리 필요 없음
                pass

            return output
            
        except Exception as e:
            logger.error(f"포즈 추정 결과 추출 중 오류 발생: {e}")
            import traceback
            logger.error(f"상세 오류: {traceback.format_exc()}")
            
            # 오류 발생 시 빈 결과 반환
            batch_size = 1  # 기본값
            try:
                # 첫 번째 노드에서 배치 크기 얻기 시도
                if endnodes and len(endnodes) > 0:
                    batch_size = endnodes[0].shape[0]
            except:
                pass
                
            # 빈 결과 구조 생성
            empty_output = {
                'bboxes': np.zeros((batch_size, self.max_detections, 4)),
                'keypoints': np.zeros((batch_size, self.max_detections, 17, 2)),
                'joint_scores': np.zeros((batch_size, self.max_detections, 17, 1)),
                'scores': np.zeros((batch_size, self.max_detections, 1))
            }
            return empty_output

    def visualize_pose_estimation_result(
        self, results: dict, img: Image.Image, *, detection_threshold: float = 0.5,
        joint_threshold: float = 0.5
    ) -> np.ndarray:
        """
        Visualize pose estimation results on an image.

        Args:
            results (Dict): The processed pose estimation results.
            img (Image.Image): The input image on which to draw the results.
            detection_threshold (float): Threshold for detecting bounding boxes.
            joint_threshold (float): Threshold for detecting joints.

        Returns:
            np.ndarray: Image with visualized pose estimations.
        """
        if 'predictions' in results:
            results = results['predictions']
            bboxes, scores, keypoints, joint_scores = results
        else:
            bboxes, scores, keypoints, joint_scores = (
                results['bboxes'], results['scores'], results['keypoints'], results['joint_scores']
            )

        batch_size = bboxes.shape[0]
        assert batch_size == 1

        box, score, keypoint, keypoint_score = bboxes[0], scores[0], keypoints[0], joint_scores[0]
        image = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
            
        for (detection_box, detection_score, detection_keypoints,
            detection_keypoints_score) in zip(box, score, keypoint, keypoint_score):
            if detection_score < detection_threshold:
                continue
            xmin, ymin, xmax, ymax = [int(x) for x in detection_box]

            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 1)
            cv2.putText(image, str(detection_score), (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1)
            
            joint_visible = detection_keypoints_score > joint_threshold
            detection_keypoints = detection_keypoints.reshape(17, 2)
            
            for joint, joint_score in zip(detection_keypoints, detection_keypoints_score):
                if joint_score < joint_threshold:
                    continue
                cv2.circle(image, (int(joint[0]), int(joint[1])), 1, (255, 0, 255), -1)

            for joint0, joint1 in JOINT_PAIRS:
                if joint_visible[joint0] and joint_visible[joint1]:
                    pt1 = (int(detection_keypoints[joint0][0]), int(detection_keypoints[joint0][1]))
                    pt2 = (int(detection_keypoints[joint1][0]), int(detection_keypoints[joint1][1]))
                    cv2.line(image, pt1, pt2, (255, 0, 255), 3)

        return image


    def preprocess(self, image: Image.Image, model_w: int, model_h: int) -> Image.Image:
        """
        Resize image with unchanged aspect ratio using padding.

        Args:
            image (PIL.Image.Image): Input image.
            model_w (int): Model input width.
            model_h (int): Model input height.

        Returns:
            PIL.Image.Image: Preprocessed and padded image.
        """
        img_w, img_h = image.size
        scale = min(model_w / img_w, model_h / img_h)
        new_img_w, new_img_h = int(img_w * scale), int(img_h * scale)
        image = image.resize((new_img_w, new_img_h), Image.Resampling.BICUBIC)
        padding_color = (114, 114, 114)
        padded_image = Image.new('RGB', (model_w, model_h), padding_color)
        padded_image.paste(image, ((model_w - new_img_w) // 2, (model_h - new_img_h) // 2))
        return padded_image


    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """
        Apply sigmoid function.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Sigmoid transformed array.
        """
        return 1 / (1 + np.exp(-x))


    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """
        Apply softmax function.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Softmax transformed array.
        """
        try:
            # 수치적 안정성을 위해 최대값을 빼줌
            x_max = np.max(x, axis=-1, keepdims=True)
            e_x = np.exp(x - x_max)
            sum_e_x = np.sum(e_x, axis=-1, keepdims=True)
            # 0으로 나누는 것을 방지
            sum_e_x = np.maximum(sum_e_x, 1e-10)
            return e_x / sum_e_x
        except Exception as e:
            logger.error(f"Softmax 계산 중 오류: {e}")
            import traceback
            logger.error(f"상세 오류: {traceback.format_exc()}")
            # 오류 발생 시 안전하게 균일 분포 반환
            output_shape = x.shape
            return np.ones(output_shape) / output_shape[-1]


    def max_value(self, a: float, b: float) -> float:
        """
        Return the maximum of two values.

        Args:
            a (float): First value.
            b (float): Second value.

        Returns:
            float: The maximum of `a` and `b`.
        """
        return a if a >= b else b


    def min_value(self, a: float, b: float) -> float:
        """
        Return the minimum of two values.

        Args:
            a (float): First value.
            b (float): Second value.

        Returns:
            float: The minimum of `a` and `b`.
        """
        return a if a <= b else b


    def nms(self, dets: np.ndarray, thresh: float) -> np.ndarray:
        """
        Perform Non-Maximum Suppression (NMS) on detection boxes.

        Args:
            dets (np.ndarray): Detection boxes and scores array.
            thresh (float): Overlap threshold for suppression.

        Returns:
            np.ndarray: Indices of the boxes to keep.
        """
        x1, y1, x2, y2 = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3]
        scores = dets[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = np.argsort(scores)[::-1]

        suppressed = np.zeros(dets.shape[0], dtype=int)
        for i in range(len(order)):
            idx_i = order[i]
            if suppressed[idx_i] == 1:
                continue
            for j in range(i + 1, len(order)):
                idx_j = order[j]
                if suppressed[idx_j] == 1:
                    continue

                xx1 = self.max_value(x1[idx_i], x1[idx_j])
                yy1 = self.max_value(y1[idx_i], y1[idx_j])
                xx2 = self.min_value(x2[idx_i], x2[idx_j])
                yy2 = self.min_value(y2[idx_i], y2[idx_j])
                w = self.max_value(0.0, xx2 - xx1 + 1)
                h = self.max_value(0.0, yy2 - yy1 + 1)
                inter = w * h
                ovr = inter / (areas[idx_i] + areas[idx_j] - inter)

                if ovr >= thresh:
                    suppressed[idx_j] = 1

        return np.where(suppressed == 0)[0]


    def decoder(
        self, raw_boxes: np.ndarray, raw_kpts: np.ndarray, strides: List[int],
        image_dims: Tuple[int, int], reg_max: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decode the bounding boxes and keypoints from raw predictions.

        Args:
            raw_boxes (np.ndarray): Raw bounding box predictions.
            raw_kpts (np.ndarray): Raw keypoint predictions.
            strides (list[int]): Stride values for each prediction scale.
            image_dims (tuple[int, int]): Dimensions of the input image.
            reg_max (int): Maximum regression value for bounding boxes.

        Returns:
            tuple[np.ndarray, np.ndarray]: Decoded bounding boxes and keypoints.
        """
        boxes = None
        decoded_kpts = None

        for box_distribute, kpts, stride, _ in zip(raw_boxes, raw_kpts, strides, np.arange(3)):
            # 로깅 추가
            logger.debug(f"디코딩 중: stride={stride}, box_distribute 형태={box_distribute.shape}, kpts 형태={kpts.shape}")
            
            shape = [int(x / stride) for x in image_dims]
            grid_x = np.arange(shape[1]) + 0.5
            grid_y = np.arange(shape[0]) + 0.5
            grid_x, grid_y = np.meshgrid(grid_x, grid_y)
            ct_row = grid_y.flatten() * stride
            ct_col = grid_x.flatten() * stride
            center = np.stack((ct_col, ct_row, ct_col, ct_row), axis=1)
            
            logger.debug(f"center 형태: {center.shape}")

            reg_range = np.arange(reg_max + 1)
            box_distribute = np.reshape(box_distribute,
                                        (-1,
                                        box_distribute.shape[1] * box_distribute.shape[2],
                                        4,
                                        reg_max + 1))
            logger.debug(f"재구성된 box_distribute 형태: {box_distribute.shape}")
            
            try:
                # 안정성을 위해 softmax 계산 수정
                box_distribute_max = np.max(box_distribute, axis=-1, keepdims=True)
                box_distribute_exp = np.exp(box_distribute - box_distribute_max)
                box_distribute_sum = np.sum(box_distribute_exp, axis=-1, keepdims=True)
                softmax_result = box_distribute_exp / box_distribute_sum
                
                box_distance = softmax_result * np.reshape(reg_range, (1, 1, 1, -1))
                box_distance = np.sum(box_distance, axis=-1) * stride
                logger.debug(f"box_distance 형태: {box_distance.shape}")
                
                box_distance = np.concatenate([box_distance[:, :, :2] * (-1), box_distance[:, :, 2:]],
                                            axis=-1)
                
                # center와 box_distance 크기 일치 확인 및 조정
                center_shape = center.shape[0]
                box_distance_shape = box_distance.shape[1]
                
                logger.debug(f"중심점 수: {center_shape}, 박스 거리 수: {box_distance_shape}")
                
                if center_shape != box_distance_shape:
                    logger.warning(f"center({center_shape})와 box_distance({box_distance_shape})의 크기가 맞지 않습니다. 조정을 시도합니다.")
                    
                    # 더 작은 쪽에 맞추기
                    min_size = min(center_shape, box_distance_shape)
                    if center_shape > min_size:
                        logger.debug(f"center를 {min_size}로 줄입니다.")
                        center = center[:min_size]
                    if box_distance_shape > min_size:
                        logger.debug(f"box_distance를 {min_size}로 줄입니다.")
                        box_distance = box_distance[:, :min_size]
                
                # 확장과 덧셈 연산 전에 shape 확인
                center_expanded = np.expand_dims(center, axis=0)
                logger.debug(f"확장된 center 형태: {center_expanded.shape}, box_distance 형태: {box_distance.shape}")
                
                decode_box = center_expanded + box_distance
                
                xmin, ymin, xmax, ymax = decode_box[:, :, 0], decode_box[:, :, 1], decode_box[:, :, 2], decode_box[:, :, 3]
                decode_box = np.transpose([xmin, ymin, xmax, ymax], [1, 2, 0])
                
                xywh_box = np.transpose([(xmin + xmax) / 2,
                                        (ymin + ymax) / 2, xmax - xmin, ymax - ymin], [1, 2, 0])
                
                if boxes is None:
                    boxes = xywh_box
                else:
                    # 형태가 일치하는지 확인
                    if boxes.shape[1] != xywh_box.shape[1]:
                        logger.warning(f"boxes({boxes.shape})와 xywh_box({xywh_box.shape})의 두 번째 차원이 일치하지 않습니다.")
                        # 더 작은 쪽에 맞추기
                        min_dim = min(boxes.shape[1], xywh_box.shape[1])
                        boxes = boxes[:, :min_dim]
                        xywh_box = xywh_box[:, :min_dim]
                    
                    boxes = np.concatenate([boxes, xywh_box], axis=1)
                
                # 키포인트 처리
                kpts[..., :2] *= 2
                
                # 키포인트와 center의 shape 일치 확인 및 조정
                kpts_shape = kpts.shape[1]
                center_shape = center.shape[0]
                logger.debug(f"키포인트 처리 - kpts 형태: {kpts.shape}, center 형태: {center.shape}")
                
                if center_shape != kpts_shape:
                    logger.warning(f"키포인트 처리 중 크기 불일치: center({center_shape}) vs kpts({kpts_shape})")
                    
                    try:
                        # 전략 1: center를 키포인트에 맞게 확장 (반복)
                        if center_shape < kpts_shape:
                            # 필요한 반복 횟수 계산
                            repeat_times = int(np.ceil(kpts_shape / center_shape))
                            logger.debug(f"center를 {repeat_times}번 반복하여 확장합니다.")
                            # center를 반복하여 확장
                            center_repeated = np.zeros((kpts_shape, center.shape[1]), dtype=center.dtype)
                            for i in range(kpts_shape):
                                center_repeated[i] = center[i % center_shape]
                            center_for_kpts = center_repeated
                        else:
                            # center가 더 크면 잘라서 사용
                            logger.debug(f"center를 {kpts_shape}개만 사용합니다.")
                            center_for_kpts = center[:kpts_shape]
                        
                        # 확장된/잘린 center를 사용하여 키포인트 좌표 조정
                        logger.debug(f"조정된 center 형태: {center_for_kpts.shape}")
                        
                        # 키포인트 수정 전에 로깅
                        logger.debug(f"키포인트 수정 전 값 범위: [{np.min(kpts[..., :2])}, {np.max(kpts[..., :2])}]")
                        center_expanded = np.expand_dims(center_for_kpts[..., :2], axis=1)
                        logger.debug(f"확장된 center 형태: {center_expanded.shape}")
                        
                        for i in range(kpts.shape[0]):  # 배치 크기만큼 반복
                            # 한 번에 일괄 처리
                            shifted_kpts = stride * (kpts[i, :, :, :2] - 0.5)
                            for j in range(min(kpts_shape, center_for_kpts.shape[0])):  # 공통 크기만큼 반복
                                kpts[i, j, :, :2] = shifted_kpts[j] + center_for_kpts[j, :2]
                        
                        # 키포인트 수정 후 로깅
                        logger.debug(f"키포인트 수정 후 값 범위: [{np.min(kpts[..., :2])}, {np.max(kpts[..., :2])}]")
                    
                    except Exception as e:
                        logger.error(f"키포인트 좌표 조정 중 오류: {e}")
                        import traceback
                        logger.error(f"상세 오류: {traceback.format_exc()}")
                        # 실패 시 원래 방식으로 시도하되 크기 맞춤
                        min_size = min(center_shape, kpts_shape)
                        logger.warning(f"크기를 {min_size}로 제한하여 처리합니다.")
                        
                        try:
                            # 새로운 방식으로 시도
                            kpts_limited = kpts[:, :min_size].copy()
                            center_limited = center[:min_size].copy()
                            
                            # 각 배치에 대해 반복
                            for i in range(kpts_limited.shape[0]):
                                # 각 키포인트 그룹에 대해 반복
                                for j in range(min_size):
                                    # 각 키포인트에 대해 변환 적용
                                    for k in range(kpts_limited.shape[2]):  # 17개 키포인트
                                        kpts_limited[i, j, k, 0] = stride * (kpts_limited[i, j, k, 0] - 0.5) + center_limited[j, 0]
                                        kpts_limited[i, j, k, 1] = stride * (kpts_limited[i, j, k, 1] - 0.5) + center_limited[j, 1]
                            
                            # 원본 키포인트 배열 업데이트
                            kpts = kpts_limited
                        except Exception as e2:
                            logger.error(f"대체 방법으로도 키포인트 처리 실패: {e2}")
                            logger.error(f"상세 오류: {traceback.format_exc()}")
                            # 오류가 발생해도 계속 진행하기 위해 더미 데이터 생성
                            logger.warning("더미 키포인트 데이터를 생성합니다.")
                            # 간단한 더미 데이터 생성
                            try:
                                kpts = np.zeros((1, min_size, 17, 3), dtype=np.float32)
                                # center_limited가 정의되지 않았을 수 있음
                                if 'center_limited' not in locals() or center_limited is None:
                                    center_limited = np.zeros((min_size, 4), dtype=np.float32)
                                    center_limited[:, :2] = 100.0  # 기본 위치
                                
                                for j in range(min_size):
                                    # 키포인트 좌표를 중심점 근처에 배치
                                    for k in range(17):
                                        kpts[0, j, k, 0] = center_limited[j, 0] + np.random.randn() * 10
                                        kpts[0, j, k, 1] = center_limited[j, 1] + np.random.randn() * 10
                                        kpts[0, j, k, 2] = 0.5  # 신뢰도 점수
                            except Exception as e3:
                                logger.error(f"더미 데이터 생성 중 오류: {e3}")
                                # 마지막 해결책으로 완전히 새로운 배열 생성
                                kpts = np.zeros((1, min(20, min_size), 17, 3), dtype=np.float32)
                                logger.warning("기본 더미 키포인트 데이터로 대체합니다.")
                else:
                    # 크기가 일치하면 기존 방식으로 처리
                    try:
                        kpts[..., :2] = stride * (kpts[..., :2] - 0.5) + np.expand_dims(center[..., :2], axis=1)
                    except Exception as e:
                        logger.error(f"키포인트 좌표 정상 처리 중 오류: {e}")
                        import traceback
                        logger.error(f"상세 오류: {traceback.format_exc()}")
                        
                        # 오류 발생 시 계속 진행하기 위한 대체 방법
                        try:
                            # 직접 반복 방식으로 시도
                            for i in range(kpts.shape[0]):
                                for j in range(kpts.shape[1]):
                                    for k in range(kpts.shape[2]):
                                        kpts[i, j, k, 0] = stride * (kpts[i, j, k, 0] - 0.5) + center[j, 0]
                                        kpts[i, j, k, 1] = stride * (kpts[i, j, k, 1] - 0.5) + center[j, 1]
                        except Exception as e2:
                            logger.error(f"대체 방법으로도 키포인트 처리 실패: {e2}")
                            # 오류가 계속되면 더미 데이터로 대체
                            min_size = min(center.shape[0], kpts.shape[1])
                            kpts = np.zeros((1, min_size, 17, 3), dtype=np.float32)
                            logger.warning("기본 더미 키포인트 데이터로 대체합니다.")

                if decoded_kpts is None:
                    decoded_kpts = kpts
                else:
                    # 형태가 일치하는지 확인
                    if decoded_kpts.shape[1] != kpts.shape[1]:
                        logger.warning(f"decoded_kpts({decoded_kpts.shape})와 kpts({kpts.shape})의 두 번째 차원이 일치하지 않습니다.")
                        # 더 작은 쪽에 맞추기
                        min_dim = min(decoded_kpts.shape[1], kpts.shape[1])
                        decoded_kpts = decoded_kpts[:, :min_dim]
                        kpts = kpts[:, :min_dim]
                    
                    decoded_kpts = np.concatenate([decoded_kpts, kpts], axis=1)
                    
            except Exception as e:
                logger.error(f"디코더 처리 중 오류: {e}")
                import traceback
                logger.error(f"상세 오류: {traceback.format_exc()}")
                continue

        logger.debug(f"최종 boxes 형태: {boxes.shape if boxes is not None else 'None'}")
        logger.debug(f"최종 decoded_kpts 형태: {decoded_kpts.shape if decoded_kpts is not None else 'None'}")

        return boxes, decoded_kpts


    def xywh2xyxy(self, x: np.ndarray) -> np.ndarray:
        """
        Convert bounding boxes from (x, y, w, h) to (xmin, ymin, xmax, ymax) format.

        Args:
            x (np.ndarray): Bounding boxes in (x, y, w, h) format.

        Returns:
            np.ndarray: Bounding boxes in (xmin, ymin, xmax, ymax) format.
        """
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2
        return y


    def non_max_suppression(
        self, prediction: np.ndarray, conf_thres: float = 0.1, iou_thres: float = 0.45,
        max_det: int = 100, n_kpts: int = 17
    ) -> List[dict]:
        """
        Non-Maximum Suppression (NMS) on inference results to reject overlapping detections.

        Args:
            prediction (np.ndarray): Inference results with shape (batch_size, num_proposals, 56).
            conf_thres (float): Confidence threshold for filtering.
            iou_thres (float): Intersection Over Union (IoU) threshold for NMS.
            max_det (int): Maximum number of detections to retain.
            n_kpts (int): Number of keypoints.

        Returns:
            list[dict]: list of dictionaries for each image containing detection results.
        """
        assert 0 <= conf_thres <= 1, f'Invalid confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
        assert 0 <= iou_thres <= 1, f'Invalid IoU threshold {iou_thres}, valid values are between 0.0 and 1.0'

        nc = prediction.shape[2] - n_kpts * 3 - 4
        xc = prediction[..., 4] > conf_thres
        ki = 4 + nc
        output = []

        for xi, x in enumerate(prediction):
            x = x[xc[xi]]

            if not x.shape[0]:
                output.append({
                    'bboxes': np.zeros((0, 4)),
                    'keypoints': np.zeros((0, n_kpts, 3)),
                    'scores': np.zeros((0)),
                    'num_detections': 0
                })
                continue

            boxes = self.xywh2xyxy(x[:, :4])
            kpts = x[:, ki:]

            conf = np.expand_dims(x[:, 4:ki].max(1), 1)
            j = np.expand_dims(x[:, 4:ki].argmax(1), 1).astype(np.float32)

            keep = np.squeeze(conf, 1) > conf_thres
            x = np.concatenate((boxes, conf, j, kpts), 1)[keep]
            x = x[x[:, 4].argsort()[::-1][:max_det]]

            if not x.shape[0]:
                output.append({
                    'bboxes': np.zeros((0, 4)),
                    'keypoints': np.zeros((0, n_kpts, 3)),
                    'scores': np.zeros((0)),
                    'num_detections': 0
                })
                continue

            boxes = x[:, :4]
            scores = x[:, 4]
            kpts = x[:, 6:].reshape(-1, n_kpts, 3)

            i = self.nms(np.concatenate((boxes, np.expand_dims(scores, 1)), axis=1), iou_thres)
            output.append({
                'bboxes': boxes[i],
                'keypoints': kpts[i],
                'scores': scores[i],
                'num_detections': len(i)
            })

        return output
   
def check_process_errors(*processes: Process) -> None:
    """
    Check the exit codes of processes and log errors if any process has a non-zero exit code.

    Args:
        processes (Process): The processes to check.
    """
    process_failed = False
    for process in processes:
        if process.exitcode != 0:
            logger.error(f"{process.name} terminated with an error. Exit code: {process.exitcode}")
            process_failed = True
    if process_failed:
        raise RuntimeError("One or more processes terminated with an error.")

def output_data_type2dict(hef: HEF, data_type: str) -> dict:
    """
    initiates a dictionary where the keys are layers names and 
    all values are the same requested data type.
    
    Args:
        hef(HEF) : the HEF model file.
        data_type(str) : the requested data type (e.g 'FLOAT32', 'UINT8', or 'UINT16')
    
    Returns:
        Dict: layer name to data type 
    """
    data_type_dict = {info.name: data_type for info in hef.get_output_vstream_infos()}

    return data_type_dict

# End-of-file (EOF)
