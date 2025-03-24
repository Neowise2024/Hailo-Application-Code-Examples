import cv2
import numpy as np
import queue
import os
import sys
import threading
import multiprocessing
from loguru import logger
from pathlib import Path
from typing import List
from commons.model_config import ModelConfig
from commons.object_detection_utils import ObjectDetectionUtils


# Add the parent directory to the system path to access utils module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils import HailoAsyncInference, divide_list_to_batches

class ObjectDetection:
    def __init__(
        self, 
        model_config: ModelConfig,
        camera_queue: multiprocessing.Queue 
    ):
        self.model_config = model_config
        self.camera_queue = camera_queue
        self.CAMERA_CAP_WIDTH = 1920
        self.CAMERA_CAP_HEIGHT = 1080 
        self.det_utils = ObjectDetectionUtils(self.model_config.getLabelsPath())
        

    def preprocess(
        self,
        batch_size: int,
        input_queue: queue.Queue,
        width: int,
        height: int,
        utils: ObjectDetectionUtils
    ) -> None:
        """
        Preprocess and enqueue images or camera frames into the input queue as they are ready.

        Args:
            images (List[np.ndarray], optional): List of images as NumPy arrays.
            camera (bool, optional): Boolean indicating whether to use the camera stream.
            batch_size (int): Number of images per batch.
            input_queue (queue.Queue): Queue for input images.
            width (int): Model input width.
            height (int): Model input height.
            utils (ObjectDetectionUtils): Utility class for object detection preprocessing.
        """
        self.preprocess_from_cap(
            batch_size,
            input_queue,
            width,
            height,
            utils
        )

        input_queue.put(None)  # Add sentinel value to signal end of input

    def preprocess_from_cap(
        self,         
        batch_size: int, 
        input_queue: queue.Queue, 
        width: int, 
        height: int, 
        utils: ObjectDetectionUtils
    ) -> None:
        """
        Process frames from the camera stream and enqueue them.

        Args:
            batch_size (int): Number of images per batch.
            input_queue (queue.Queue): Queue for input images.
            width (int): Model input width.
            height (int): Model input height.
            utils (ObjectDetectionUtils): Utility class for object detection preprocessing.
        """
        frames = []
        processed_frames = []

        while True:            
            ret, frame = self.camera_queue.get()
            if not ret:
                break

            frames.append(frame)
            processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frame = utils.preprocess(
                processed_frame, 
                width,
                height
            )
            processed_frames.append(processed_frame)

            if len(frames) == self.model_config.getBatchSize():
                input_queue.put((frames, processed_frames))
                processed_frames, frames = [], []
                logger.info(f"Sent camera frame to input queue {input_queue.qsize()}")
            

    def postprocess(
        self,
        output_queue: queue.Queue,
        display_queue: multiprocessing.Queue
    ) -> None:
        """
        Process and visualize the output results.

        Args:
            output_queue (queue.Queue): Queue for output results.
            camera (bool): Flag indicating if the input is from a camera.
            save_stream_output (bool): Flag indicating if the camera output should be saved.
            utils (ObjectDetectionUtils): Utility class for object detection visualization.
        """
        image_id = 0
        
        
        
        while True:
            result = output_queue.get()
            logger.info(f"object detection Output queue size: {output_queue.qsize()}")
            if result is None:
                break  # Exit the loop if sentinel value is received

            original_frame, infer_results = result

            # Deals with the expanded results from hailort versions < 4.19.0
            if len(infer_results) == 1:
                infer_results = infer_results[0]

            detections = self.det_utils.extract_detections(infer_results)

            frame_with_detections = self.det_utils.draw_detections(
                detections, original_frame,
            )
            display_queue.put(frame_with_detections)
            logger.info("object detection display queue size: {}".format(display_queue.qsize()))
            
            image_id += 1

        output_queue.task_done()  # Indicate that processing is complete


    def infer(
        self,
        display_queue: multiprocessing.Queue,
    ) -> None:
        """
        Initialize queues, HailoAsyncInference instance, and run the inference.

        Args:
            images (List[Image.Image]): List of images to process.
            net_path (str): Path to the HEF model file.
            labels_path (str): Path to a text file containing labels.
            batch_size (int): Number of images per batch.
            output_path (Path): Path to save the output images.
        """
        
        images = []        


        input_queue = queue.Queue(maxsize=2)
        output_queue = queue.Queue(maxsize=2)
        
        hailo_inference = HailoAsyncInference(
            self.model_config.getModelPath(),
            input_queue, 
            output_queue,
            self.model_config.getBatchSize(), 
            send_original_frame=True
        )
        height, width, _ = hailo_inference.get_input_shape()

        logger.info("Starting object detection preprocess init")
        preprocess_thread = threading.Thread(
            target=self.preprocess,
            args=(
                self.model_config.getBatchSize(), 
                input_queue, 
                width, 
                height, 
                self.det_utils
            )
        )
        
        postprocess_thread = threading.Thread(
            target=self.postprocess,
            args=(output_queue, display_queue)
        )
        
        logger.info("Starting object detection preprocess start")
        preprocess_thread.start()
        logger.info("Starting object detection postprocess start")
        postprocess_thread.start()
        
        logger.info("Starting object detection inference")
        hailo_inference.run(model_name="object_detection")        
        logger.info("Starting object detection preprocess join")
        preprocess_thread.join()
        
        output_queue.put(None)
        logger.info("Starting object detection postprocess join")
        postprocess_thread.join()
        
        logger.info('Inference was successful!')

