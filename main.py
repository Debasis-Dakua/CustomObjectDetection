
import cv2
from ultralytics import YOLO

IMAGE_PATH = 'img2.jpg'  # Replace 'path_to_image.jpg' with the path to your image file
MODEL_PATH = 'runs/detect/train7/weights/best.pt'     # Replace 'path_to_best.pt' with the path to your "best.pt" file

# Load image
image = cv2.imread(IMAGE_PATH)

# Load YOLO model with "best.pt" file
model = YOLO(MODEL_PATH)  # load a custom model

# Perform object detection
results = model(image)

# Extract detection results for the first image
detection_results = results[0]

if len(results[0].boxes) > 0:
    # Get bounding box coordinates for the first detected object
    x_min, y_min, x_max, y_max = results[0].boxes[0].xyxy[0]

    cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)

    cv2.putText(image, 'Aquaguard', (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("detected obj", image)
else:
    print("No aquaguard detected in the image.")

cv2.waitKey(0)
cv2.destroyAllWindows()
