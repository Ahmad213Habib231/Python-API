import os
import json
import tempfile
import logging
import traceback
import requests
from flask import Flask, request, Response
from ultralytics import YOLO

GOOGLE_DRIVE_FILE_ID = "1toioocY11XuKUfiwl0YsR4Q4S_lHPZvB"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "best.pt")

def download_file_from_google_drive(file_id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)
    if token:
        response = session.get(URL, params={'id': file_id, 'confirm': token}, stream=True)
    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

def ensure_model_exists():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found locally at {MODEL_PATH}. Downloading...")
        os.makedirs(MODEL_DIR, exist_ok=True)
        download_file_from_google_drive(GOOGLE_DRIVE_FILE_ID, MODEL_PATH)
        print("✅ Model downloaded successfully.")
    else:
        print("✅ Model already exists.")

# تحميل النموذج مرة واحدة فقط
ensure_model_exists()
print("✅ Loading YOLO model...")
model = YOLO(MODEL_PATH)
print("✅ YOLO model loaded!")

# إنشاء التطبيق
app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

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

            print("🔍 Running YOLO prediction...")
            results = model.predict(source=temp_file.name, save=False, save_txt=False, verbose=False, device='cpu')
            detections = json.loads(results[0].to_json())

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
    port = int(os.environ.get("PORT", 5001))
    app.run(debug=False, host='0.0.0.0', port=port)
