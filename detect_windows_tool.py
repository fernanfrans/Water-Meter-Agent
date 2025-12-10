from ultralytics import YOLO
import cv2
import os
import time

def detect_windows(model, image_path, show_image=True, min_conf = 0.5):
    img = cv2.imread(image_path)
    # Predict windows in the image
    results = model.predict(img, conf=min_conf, verbose=False)
    detected_items = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            # Crop window
            crop = img[y1:y2, x1:x2]

            # Store as (crop, width)
            detected_items.append((crop, x1, conf))

            # Draw box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Sort by x1 ascending
    detected_items.sort(key=lambda x: x[1])

    # Return ONLY the crops
    only_crops = [(cw[0], cw[2]) for cw in detected_items]

    return only_crops

if __name__ == "__main__":
    start_time = time.time()
    model = YOLO("water-meter-reading.pt")
    test_image_path = r"C:\Users\Administrator\DATA SCIENTIST\WIZY\Water-Meter-Agent\water-images\1745490649682-WV_60edf_20250424_183049_jpg.rf.9b97e42bca32cd51260673b25240fc53.jpg"
    cropped_windows = detect_windows(model, test_image_path)
    print(len(cropped_windows))
