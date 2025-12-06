import cv2
import numpy as np
import os

def process_digit_image(img):
    # 1. Preprocessing the image
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to grayscale
    blur_img = cv2.GaussianBlur(gray_img, (5, 5), 0) # apply Gaussian blur
    preprocessed_img = cv2.adaptiveThreshold(
        blur_img, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        21, 
        10
    )

    # 2. Keep the largest connected component (the digit)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(preprocessed_img, connectivity=8)
    if num_labels <= 1:
        return np.zeros((28, 28), dtype=np.uint8)  # Return a blank image if no components found
    largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask_digit = np.where(labels == largest_component, 255, 0).astype('uint8')
    
    # 3. Crop the digit region
    x, y, w, h = cv2.boundingRect(mask_digit)
    digit = mask_digit[y:y+h, x:x+w]

    # 4. Resize while maintaining aspect ratio
    h, w = digit.shape
    MIN_SIZE = 20
    scale = max(MIN_SIZE / w, MIN_SIZE / h, 1)
    new_w, new_h = int(w * scale), int(h * scale)
    digit = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


    # 5. Pad to make it square
    size = max(w, h)
    top = bottom = (size - h) // 2
    left = right = (size - w) // 2
    center_digit = cv2.copyMakeBorder(digit, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

    # 6. Resize to 28x28 pixels
    final_digit = cv2.resize(center_digit, (28, 28), interpolation=cv2.INTER_AREA)
    return final_digit

if __name__ == "__main__":
    test_image_path = r"C:\Users\Administrator\DATA SCIENTIST\WIZY\Water-Meter-Agent\water-window\347_1745560954798-WV_ffe69_20250425_140234_png_3.png"
    raw_folder = r"C:\Users\Administrator\DATA SCIENTIST\WIZY\Water-Meter-Agent\water-window"
    preprocessed_output = r"C:\Users\Administrator\DATA SCIENTIST\WIZY\Water-Meter-Agent\water-window_preprocessed"
    for i, image_name in enumerate(os.listdir(raw_folder)):
        image_path = os.path.join(raw_folder, image_name)
        processed_image = process_digit_image(image_path)
        os.makedirs(preprocessed_output, exist_ok=True)
        output_path = os.path.join(preprocessed_output, image_name)
        cv2.imwrite(output_path, processed_image)
        print(f"Processed image {i+1}/{len(os.listdir(raw_folder))}")
    # processed_image = process_digit_image(test_image_path)
    # cv2.imshow("Processed Digit", processed_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()







# # === 6. Normalize (0–1 float) ===
# digit = digit.astype('float32') / 255.0

# # Optional: reshape if your model expects (1, 28, 28, 1)
# digit = np.expand_dims(digit, axis=(0, -1))