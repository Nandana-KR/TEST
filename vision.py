import torch
from PIL import Image

def detect_objects(image_path, conf_thresh=0.25):
    try:
        img = Image.open(image_path)
    except Exception as e:
        raise ValueError(f"Could not open image: {e}")

    model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
    results = model(img)

    items = []
    for x1, y1, x2, y2, conf, cls in results.xyxy[0].tolist():
        if conf >= conf_thresh:
            label = results.names[int(cls)]
            items.append((label, conf))
    return items
