from detecting_window import detect_windows
from ultralytics import YOLO
import cv2
from processing_2 import process_digit_image
from tensorflow.keras.models import load_model


def main():
    water_digits = []
    # sample input image
    input_image = r"C:\Users\Administrator\DATA SCIENTIST\WIZY\Water-Meter-Agent\water-images\1745560692563-WV_4d549_20250425_135812_png.rf.848f976f8fdcd07e59cc6e1bf4635296.jpg"
    # load models
    model = YOLO("water-meter-reading.pt")
    mnist_model = load_model("MNIST_keras_CNN.h5", compile=False)

    # detect windows
    cropped_windows = detect_windows(model, input_image)
    print(f"Detected {len(cropped_windows)} windows.")
    # process each detected window
    for i, crop in enumerate(cropped_windows):
        processed_image = process_digit_image(crop)
        # display processed image for verification
        cv2.imshow("Detected Windows", processed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # predict digit
        digit_input = processed_image.astype('float32') / 255.0
        digit_input = digit_input.reshape(1, 28, 28, 1)
        prediction = mnist_model.predict(digit_input)
        predicted_digit = int(prediction.argmax())  
        water_digits.append(predicted_digit)
    meter_value = "".join(str(d) for d in water_digits)
    print("Meter reading:", meter_value)

if __name__ == "__main__":
    main()
        
