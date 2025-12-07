# Water Meter Reading System
## Version 0

This project is a computer vision pipeline designed to automate the transcription of analog water meter readings. It utilizes a two-stage deep learning approach: **YOLOv8** for detecting digit windows (regions of interest) and a **Custom CNN (trained on MNIST)** for recognizing specific digits.

## 📂 Project Structure

* **`predict.py`**: The main entry point of the application. It coordinates the loading of models, detection of digit windows, image preprocessing, and final digit prediction to output the meter reading.
* **`detecting_window.py`**: Handles object detection using YOLO. It locates the digit windows, crops them, and sorts them from left to right to ensure the correct reading order.
* **`processing_2.py`**: Contains the image processing pipeline. It transforms raw cropped images into a format compatible with the CNN (28x28 grayscale, centered, thresholded).
* **Models (Required External Files)**:
    * `water-meter-reading.pt`: The YOLO weights file for detecting meter windows.
    * `MNIST_keras_CNN.h5`: The compiled Keras model for digit classification.

## ⚙️ Installation

1.  **Clone the repository** (or ensure all script files are in the same directory).
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Place Model Files**: Ensure `water-meter-reading.pt` and `MNIST_keras_CNN.h5` are located in the project root directory.

## 🚀 Usage

1.  Open `predict.py`.
2.  Update the `input_image` variable with the path to the water meter image you wish to process:
    ```python
    input_image = r"path/to/your/image.jpg"
    ```
3.  Run the script:
    ```bash
    python predict.py
    ```
4.  The system will:
    * Display the detected windows (press any key to proceed).
    * Display the processed digit (press any key to proceed for each digit).
    * Print the final transcribed meter reading to the console.

## 🧠 System Workflow

### Stage 1: Window Detection (`detecting_window.py`)
* **Model**: YOLOv8.
* **Logic**:
    1.  Accepts a full raw image of a water meter.
    2.  Predicts bounding boxes for digit windows using a confidence threshold of 0.5.
    3.  **Crucial Step**: Sorts the detected bounding boxes based on the `x` (horizontal) coordinate. This ensures the digits are read in the correct order (e.g., 1-2-3-4), rather than random detection order.

### Stage 2: Image Preprocessing (`processing_2.py`)
To make real-world photos compatible with an MNIST-trained model, the system applies heavy preprocessing:
1.  **Grayscale & Blur**: Reduces color noise.
2.  **Adaptive Thresholding**: Converts the image to binary (black/white), handling uneven lighting conditions.
3.  **Connected Components Analysis**: Identifies the largest connected white blob (the digit) and removes smaller noise artifacts.
4.  **ROI Extraction**: Crops tightly around the digit.
5.  **Aspect Ratio Preservation**: Resizes the digit while maintaining its shape.
6.  **Padding & Resizing**: Centers the digit in a black square and resizes it to **28x28 pixels**, matching the input shape required by the CNN.

### Stage 3: Classification (`predict.py`)
* **Model**: Keras CNN (`MNIST_keras_CNN.h5`).
* **Logic**:
    1.  Takes the 28x28 processed image.
    2.  Normalizes pixel values (0-255 $\rightarrow$ 0.0-1.0).
    3.  Performs inference to predict the digit (0-9).
    4.  Concatenates predictions into a single string (e.g., "00156").

