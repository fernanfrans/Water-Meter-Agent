from ultralytics import YOLO
import cv2
import os
import time

def detect_windows(model, image_path, show_image=True):
    img = cv2.imread(image_path)
    # Predict windows in the image
    results = model.predict(img, conf=0.5, verbose=False)
    cropped_windows = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Crop window
            crop = img[y1:y2, x1:x2]

            # Store as (crop, width)
            cropped_windows.append((crop, x1))

            # Draw box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Sort by x1 ascending
    cropped_windows.sort(key=lambda x: x[1])

    # Show detection result
    if show_image:
        cv2.imshow("Detected Windows", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Return ONLY the crops (remove the width)
    only_crops = [cw[0] for cw in cropped_windows]

    return only_crops


def save_cropped_images(cropped_windows, pic_id, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for i, crop_image in enumerate(cropped_windows):
        output_path = os.path.join(output_folder, f"{pic_id}_{i}.png")
        cv2.imwrite(output_path, crop_image)

if __name__ == "__main__":
    start_time = time.time()
    model = YOLO("water-meter-reading.pt")
    # input_folder = r"C:\Users\Administrator\DATA SCIENTIST\WIZY\Water-Meter-Agent\water-images"
    # output_folder = r"C:\Users\Administrator\DATA SCIENTIST\WIZY\Water-Meter-Agent\water-window"

    # for i, image_name in enumerate(os.listdir(input_folder)):
    #     image_path = os.path.join(input_folder, image_name)
    #     cropped_windows = detect_windows(model, image_path)
    #     save_cropped_images(cropped_windows, f"{i}_{image_name.split('.')[0]}", output_folder)
    #     print(f"Processed image {i+1}/{len(os.listdir(input_folder))}")
    # end_time = time.time()
    # print(f"Detection completed in {end_time - start_time:.2f} seconds.")
    test_image_path = r"C:\Users\Administrator\DATA SCIENTIST\WIZY\Water-Meter-Agent\water-images\1745490649682-WV_60edf_20250424_183049_jpg.rf.9b97e42bca32cd51260673b25240fc53.jpg"
    cropped_windows = detect_windows(model, test_image_path)
    print(f"Detected {len(cropped_windows)} windows.")
