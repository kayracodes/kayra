import cv2
import os
import sys
from ultralytics import YOLO
from ament_index_python.packages import get_package_share_directory

# Import your YOLO model here. Adjust the import as needed.
# from ika_vision.yolo_model import YoloModel
SIGN_ID_MAP = {
    0: "1",
    1: "10",
    2: "11",
    3: "12",
    4: "2",
    5: "3",
    6: "4",
    7: "4-END",
    8: "5",
    9: "6",
    10: "7",
    11: "8",
    12: "9",
    13: "STOP",
}

# Dummy YOLO model for demonstration. Replace with your actual model.
MODEL_PATH = os.path.join(
    get_package_share_directory("ika_vision"), "ai_models", "sign_model_v4.pt"
)
CONFIDENCE_THRESHOLD = 0.55
INFERENCE_SIZE = 832
IMG_WIDTH = 320
IMG_HEIGHT = 240


def load_yolo_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found at {MODEL_PATH}. Please check the path.")
        return None
    return YOLO(MODEL_PATH)


def yolo_predict(model, image):
    img_width, img_height = image.shape[1], image.shape[0]
    img = cv2.resize(image, (INFERENCE_SIZE, INFERENCE_SIZE))
    results = model(img, imgsz=INFERENCE_SIZE)
    boxes = []
    for result in results:
        for box in result.boxes:
            if box.conf[0] > CONFIDENCE_THRESHOLD:
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)
                w, h = x2 - x1, y2 - y1
                cls_id = int(box.cls[0])
                # Scale back to original image size
                x1_scaled = int(x1 * img_width / INFERENCE_SIZE)
                y1_scaled = int(y1 * img_height / INFERENCE_SIZE)
                x2_scaled = int(x2 * img_width / INFERENCE_SIZE)
                y2_scaled = int(y2 * img_height / INFERENCE_SIZE)
                boxes.append(
                    (
                        x1_scaled,
                        y1_scaled,
                        x2_scaled,
                        y2_scaled,
                        f"{SIGN_ID_MAP.get(cls_id, 'Unknown')}",
                        float(box.conf[0]),
                    )
                )
    return boxes


def annotate_image(image, boxes):
    for x1, y1, x2, y2, label, conf in boxes:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{label}: {conf:.2f}"
        cv2.putText(
            image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
    return image


def process_video(input_path):
    base, ext = os.path.splitext(os.path.basename(input_path))
    output_path = os.path.join(os.path.dirname(input_path), f"{base}_annotated{ext}")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Failed to open video: {input_path}")
        return

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Check if video is vertical and set rotation flag
    # rotate90 = height > width
    # if rotate90:
    #     out = cv2.VideoWriter(output_path, fourcc, fps, (height, width))
    # else:
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Replace DummyYoloModel with your actual YOLO model
    model = load_yolo_model()
    if model is None:
        print("YOLO model could not be loaded.")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {frame_count} frames...")
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # if rotate90:
        #     frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        boxes = yolo_predict(model, frame)
        annotated = annotate_image(frame, boxes)
        out.write(annotated)
        frame_idx += 1
        if frame_idx % 10 == 0:
            print(f"Processed {frame_idx}/{frame_count} frames")

    cap.release()
    out.release()
    print(f"Annotated video saved to: {output_path}")


if __name__ == "__main__":
    process_video("ika_vision/video/video37.mp4")
