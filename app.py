import os
import uuid
import cv2
import pywt
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
import timm

from flask import Flask, render_template, request, jsonify, url_for

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["RESULT_FOLDER"] = os.path.join("static", "results")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["RESULT_FOLDER"], exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "model.pth"
IMG_SIZE = 299
class_names = ["fake", "real"]

def extract_prnu_array_from_pil(pil_img):
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    coeffs2 = pywt.dwt2(gray, "haar")
    LL, (LH, HL, HH) = coeffs2
    prnu = LH + HL + HH

    prnu = (prnu - np.mean(prnu)) / (np.std(prnu) + 1e-8)
    prnu = np.clip(prnu, -3, 3)
    prnu = (prnu + 3) / 6.0
    prnu = cv2.resize(prnu, (IMG_SIZE, IMG_SIZE)).astype(np.float32)
    return prnu

def save_prnu_image(prnu_array):
    prnu_img = (prnu_array * 255).clip(0, 255).astype(np.uint8)
    filename = f"prnu_{uuid.uuid4().hex}.png"
    save_path = os.path.join(app.config["RESULT_FOLDER"], filename)
    Image.fromarray(prnu_img).save(save_path)
    return url_for("static", filename=f"results/{filename}")

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

class XceptionBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "xception",
            pretrained=False,
            num_classes=0,
            global_pool="avg"
        )

    def forward(self, x):
        return self.backbone(x)

class PRNUBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):
        x = self.net(x)
        return x.view(x.size(0), -1)

class DualStreamModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.rgb_branch = XceptionBranch()
        self.prnu_branch = PRNUBranch()
        self.classifier = nn.Sequential(
            nn.Linear(2048 + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, rgb, prnu):
        f_rgb = self.rgb_branch(rgb)
        f_prnu = self.prnu_branch(prnu)
        fused = torch.cat([f_rgb, f_prnu], dim=1)
        return self.classifier(fused)

model = DualStreamModel(num_classes=2).to(device)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    MODEL_READY = True
else:
    MODEL_READY = False
    print("警告：找不到 model.pth，請把訓練好的 model.pth 放到 app.py 同一層。")

def predict_image(image_path):
    if not MODEL_READY:
        raise FileNotFoundError("找不到 model.pth，請先放入訓練好的模型權重。")

    pil_img = Image.open(image_path).convert("RGB")
    rgb_tensor = transform(pil_img).unsqueeze(0).to(device)

    prnu_array = extract_prnu_array_from_pil(pil_img)
    prnu_tensor = torch.tensor(
        np.expand_dims(prnu_array, axis=0),
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(rgb_tensor, prnu_tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    fake_prob = float(probs[0])
    real_prob = float(probs[1])
    pred_idx = int(torch.argmax(probs).item())
    pred_label = class_names[pred_idx]
    confidence = max(fake_prob, real_prob) * 100
    prnu_url = save_prnu_image(prnu_array)

    return {
        "label": pred_label,
        "confidence": round(confidence, 2),
        "fake_prob": round(fake_prob * 100, 2),
        "real_prob": round(real_prob * 100, 2),
        "prnu_url": prnu_url
    }

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict-image", methods=["POST"])
def predict_image_route():
    if "image" not in request.files:
        return jsonify({"error": "沒有收到圖片檔案"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "沒有選擇檔案"}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        return jsonify({"error": "不支援的圖片格式"}), 400

    filename = f"upload_{uuid.uuid4().hex}{ext}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    try:
        result = predict_image(save_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict-video", methods=["POST"])
def predict_video_route():
    return jsonify({
        "error": "目前模型版本只支援照片偵測；影片偵測可作為未來擴充功能。"
    }), 501

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
