from ultralytics import YOLO
import cv2
from utils import *

# Load models
lp_detector = YOLO("weights/lp_detector.pt")
kp_detector = YOLO("weights/kp.pt")

# Load image
image_path = r'D:\github\vn_lp_recognition\sample_images\1.png'
image = cv2.imread(image_path)
image = cv2.resize(image, (720, 480))  # Resize image to 640x480 for consistency

# Detect license plates
for box in lp_detector(image)[0].boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
    
    license_plate = image[y1:y2, x1:x2]
    
    # Detect keypoints
    keypoints = kp_detector(license_plate)[0].keypoints.xy[0] # keypoints = top_left, top_right, bottom_right, bottom_left
    print("Keypoints:", keypoints)
    for (x, y) in keypoints:
        cv2.circle(license_plate, (int(x), int(y)), 2, (0, 0, 255), -1)
    warped = warp_perspective(license_plate, keypoints)
    cv2.imshow("Warped License Plate", warped)
    text = get_text(warped)
    print(text)
    cv2.putText(image, text[0], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    
cv2.imshow("License Plate Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
