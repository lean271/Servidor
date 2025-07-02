from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io, logging
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

# 1) Carga tu modelo .pt
MODEL_PATH = "Modelo/best (1).pt"
model = YOLO(MODEL_PATH)

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "campo 'image' ausente"}), 400

    image = Image.open(io.BytesIO(request.files["image"].read())).convert("RGB")
    results = model.predict(source=image, conf=0.25, iou=0.45, device='cpu')

    detections = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            label  = model.names[cls_id]
            detections.append((label, conf))

    if detections:
        label, conf = max(detections, key=lambda x: x[1])
        return jsonify({
            "pest": True,
            "confidence": round(conf, 4),
            "label": label
        })
    else:
        return jsonify({
            "pest": False,
            "confidence": 0.0,
            "label": "Sin detección"
        })

    return jsonify({"detections": detections})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
