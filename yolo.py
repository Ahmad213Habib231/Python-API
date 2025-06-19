import os
import json
import tempfile
import logging
import traceback
from flask import Flask, request, Response
from ultralytics import YOLO

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# مسار الموديل
MODEL_PATH = "model/best.pt"
if not os.path.exists(MODEL_PATH):
    raise Exception(f"Model not found at {MODEL_PATH}")

print("✅ YOLO model is loading...")
model = YOLO(MODEL_PATH)
print("✅ YOLO model loaded!")

@app.route('/detect', methods=['POST'])
def detect():
    print("📥 /detect route called")

    if 'image' not in request.files:
        print("❌ No image in request")
        return Response(json.dumps({"status": False, "message": "No image provided"}), status=400)

    image_file = request.files['image']
    print("📸 Image received")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            image_file.save(temp_file.name)
            print(f"💾 Saved temp image at {temp_file.name}")

            print("🔍 Starting YOLO prediction...")
            # تشغيل على CPU لأنه لا يوجد GPU متاح
            results = model.predict(source=temp_file.name, save=False, save_txt=False, verbose=False, device='cpu')
            print("✅ Prediction done")

            detections_raw_json = results[0].to_json()  # ← التعديل هنا
            detections = json.loads(detections_raw_json)
            print("📊 Parsed results")

        os.unlink(temp_file.name)
        print("🗑️ Temp file deleted")

        return Response(json.dumps({
            "status": True,
            "message": "Detection successful",
            "count": len(detections),
            "result": detections
        }), status=200, mimetype='application/json')

    except Exception as e:
        print("❌ Exception occurred:")
        traceback.print_exc()

        return Response(json.dumps({
            "status": False,
            "message": "Detection failed",
            "error": str(e)
        }), status=500, mimetype='application/json')

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5001, use_reloader=True)
