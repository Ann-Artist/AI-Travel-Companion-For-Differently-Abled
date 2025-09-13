import cv2
import pyttsx3
import os
from ultralytics import YOLO

# Load model
model = YOLO("yolov8s.pt")

# Text-to-Speech
engine = pyttsx3.init()
engine.setProperty("rate", 150)
engine.setProperty("volume", 1.0)

# Path to coco2017 images
image_folder = "coco2017"
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

print("Processing images from coco2017 folder...")

for img_name in image_files:
    img_path = os.path.join(image_folder, img_name)
    frame = cv2.imread(img_path)
    if frame is None:
        continue

    results = model(frame, stream=True)
    detected_objects = set()

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            detected_objects.add(label)

            # Drawing code (optional, can be commented out if no GUI)
            # x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # cv2.putText(frame, label, (x1, y1 - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    if detected_objects:
        for obj in detected_objects:
            engine.say(f"{obj} ahead")
        engine.runAndWait()

    print(f"Processed: {img_name} | Detected: {', '.join(detected_objects)}")

    # Remove GUI code for headless environments
    # cv2.imshow("Blind Assistant", frame)
    # if cv2.waitKey(0) & 0xFF == ord('q'):
    #     break

# Remove cv2.destroyAllWindows() for