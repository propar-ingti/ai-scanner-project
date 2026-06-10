from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from PIL import Image
import numpy as np
import re
import io
import joblib
from scipy.sparse import hstack

import torch
import torch.nn as nn
from torchvision import transforms

app = Flask(__name__)
CORS(app)

print("Loading Local Models...")

# ---------------- TEXT MODEL ----------------

text_model = joblib.load("models/text_detector_v2.pkl")
vectorizer = joblib.load("models/tfidf_v2.pkl")

def get_numerical_features(text):
    text = str(text)

    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]

    if len(sentences) < 2:
        burstiness = 0.0
    else:
        lengths = [len(s.split()) for s in sentences]
        burstiness = np.std(lengths) / (np.mean(lengths) + 1e-6)

    words = text.split()

    avg_word_len = (
        np.mean([len(w) for w in words])
        if words else 0
    )

    sentence_count = len(sentences)

    return [burstiness, avg_word_len, sentence_count]


# ---------------- IMAGE MODEL ----------------

class DeepDetectNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),

            nn.Linear(64 * 14 * 14, 128),
            nn.ReLU(),

            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.net(x)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

image_model = DeepDetectNet().to(device)

image_model.load_state_dict(
    torch.load(
        "models/deepdetect_v1.pth",
        map_location=device
    )
)

image_model.eval()

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5]
    )
])

print("Local Models Loaded Successfully!")


# ---------------- TEXT ROUTE ----------------

@app.route('/scan', methods=['POST'])
def scan_text():

    data = request.get_json()

    text_received = data.get('text', '')[:1500]

    if len(text_received) < 50:
        return jsonify({'ai_percentage': 0})

    try:

        X_tfidf = vectorizer.transform([text_received])

        X_num = np.array(
            [get_numerical_features(text_received)],
            dtype=np.float32
        )

        X_final = hstack([X_tfidf, X_num])

        prob = text_model.predict_proba(X_final)[0][1]

        ai_percentage = prob * 100

        print(
            f"[SERVER] Text Scanned | AI Probability: {ai_percentage:.1f}%"
        )

        return jsonify(
            {'ai_percentage': round(ai_percentage, 2)}
        )

    except Exception as e:

        print(e)

        return jsonify({'ai_percentage': 0})


# ---------------- IMAGE ROUTE ----------------

@app.route('/scan-image', methods=['POST'])
def scan_image():

    data = request.get_json()

    image_url = data.get('src', '')

    try:

        headers = {
            'User-Agent': 'Mozilla/5.0'
        }

        response = requests.get(
            image_url,
            headers=headers,
            stream=True
        )

        img = Image.open(
            response.raw
        ).convert('RGB')

        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():

            output = image_model(img_tensor)

            probabilities = torch.softmax(
                output,
                dim=1
            )

            ai_probability = probabilities[0][1].item()

        ai_percentage = ai_probability * 100

        print(
            f"[SERVER] Image Scanned | AI Probability: {ai_percentage:.1f}%"
        )

        return jsonify(
            {'ai_percentage': round(ai_percentage, 2)}
        )

    except Exception as e:

        print("Image Error:", e)

        return jsonify({
            'ai_percentage': 0,
            'error': str(e)
        })


if __name__ == '__main__':
    app.run(port=5000)
