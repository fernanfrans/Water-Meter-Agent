import os 
import cv2
import shutil
import numpy as np
from tensorflow.keras.models import load_model
from ultralytics import YOLO
from detect_windows_tool import detect_windows
from recognize_digit_tool import predict_digit

class WaterMeterTools:
    def __init__(self, window_model_path = "water-meter-reading.pt", digit_model_path = "MNIST_keras_CNN.h5"):
        self.window_model = YOLO(window_model_path)
        self.digit_model = load_model(digit_model_path, compile=False)

        self.temp_dir = "agent_scratchpad"
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        os.makedirs(self.temp_dir, exist_ok=True)

    def tool_detect_windows(self, image_path, min_conf=0.5):
        """
        Tool: Detect meter windows using a specific confidence threshold.
        Args:
            image_path (str): Path to the input image.
            min_conf (float): Minimum confidence threshold for detection.
        Returns: Success/Failure message and file paths.

        """
        # Clean inputs
        image_path = str(image_path).strip().strip("'").strip('"')
        try:
            min_conf = float(min_conf)
        except:
            return "OBSERVATION: Error - Confidence threshold must be a number."

        if not os.path.exists(image_path):
            return f"OBSERVATION: Error - File {image_path} does not exist."
        
        detected_windows = detect_windows(self.window_model, image_path, min_conf=min_conf)

        if len(detected_windows) != 5:
            return f"OBSERVATION: FAILURE. Found {len(detected_windows)} windows using threshold {min_conf}. Expected exactly 5."

        # save files to temp directory
        saved_file = []
        confidences = []
        base_name = os.path.basename(image_path).split('.')[0]

        for i, (crop, conf) in enumerate(detected_windows):
            file_name = f"{base_name}_t{int(min_conf*100)}_{i}.png"
            save_path = os.path.join(self.temp_dir, file_name)
            cv2.imwrite(save_path, crop)
            saved_file.append(save_path)
            confidences.append(conf)

        return {
            "message": f"OBSERVATION: SUCCESS. Found 5 windows using threshold {min_conf}.\n"
                    f"Window Confidences: {confidences}",
            "files": saved_file
        }

    def tool_digit_recognition(self, file_paths):
        """
        Tool: Recognize digits from a LIST of image paths.
        Args:
            file_paths (List[str]): A list of file paths to cropped images.
        Returns: 
            List[dict]: A list of dictionaries containing 'digit' and 'confidence'.
        """
        # 1. Safety: Convert string representation of list to actual list if needed
        if isinstance(file_paths, str):
            import ast
            try:
                file_paths = ast.literal_eval(file_paths)
            except:
                return "OBSERVATION: Error - Input must be a valid list of file paths."

        results = []
        
        # 2. Loop through the list
        for path in file_paths:
            # Clean the path string
            path = str(path).strip().strip("'").strip('"')
            
            img = cv2.imread(path)
            if img is None:
                # Don't crash the whole batch if one image fails. Just mark it as unknown.
                results.append({'digit': '?', 'confidence': 0.0, 'error': f"Could not read {path}"})
                continue

            predicted_digit, confidence = predict_digit(self.digit_model, img)
            
            # Convert numpy types to native Python types (int/float) so JSON serializes correctly
            results.append({
                'digit': int(predicted_digit), 
                'confidence': float(confidence)
            })
            
        return results