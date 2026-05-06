"""
Fake News Detection API — Flask
Deployment target: Railway
"""

import os
import re
import string
import pickle
import functools
import torch

import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ─────────────────────────────────────────────
# 1. App & API-key setup
# ─────────────────────────────────────────────
app = Flask(__name__)

API_KEY = os.environ.get("API_KEY", "")


def require_api_key(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        incoming_key = request.headers.get("x-api-key", "")
        if not API_KEY:
            return jsonify({"error": "Server misconfigured: API_KEY env var not set."}), 500
        if incoming_key != API_KEY:
            return jsonify({"error": "Unauthorized. Invalid or missing x-api-key header."}), 401
        return f(*args, **kwargs)
    return decorated


# ─────────────────────────────────────────────
# 2. Load model & preprocessor at startup
# ─────────────────────────────────────────────
ARTIFACT_DIR = os.environ.get("ARTIFACT_DIR", "model_artifacts")
MODEL_PATH   = os.path.join(ARTIFACT_DIR, "multi_task_distilbert.pt")
PREP_PATH    = os.path.join(ARTIFACT_DIR, "preprocessor.pkl")

os.makedirs(ARTIFACT_DIR, exist_ok=True)

from huggingface_hub import hf_hub_download

HF_REPO = "JanaMostafa2/Trustera_model"
HF_TOKEN = os.environ.get("HF_TOKEN", "")

if not os.path.exists(MODEL_PATH):
    print("Downloading model from Hugging Face...")
    hf_hub_download(
        repo_id=HF_REPO,
        filename="multi_task_distilbert.pt",
        local_dir=ARTIFACT_DIR,
        token=HF_TOKEN
    )
    print("Model downloaded.")

if not os.path.exists(PREP_PATH):
    print("Downloading preprocessor from Hugging Face...")
    hf_hub_download(
        repo_id=HF_REPO,
        filename="preprocessor.pkl",
        local_dir=ARTIFACT_DIR,
        token=HF_TOKEN
    )
    print("Preprocessor downloaded.")

print(f"Loading model from {MODEL_PATH} ...")
_model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
_model.eval()
print("Model loaded.")

with open(PREP_PATH, "rb") as f:
    _prep = pickle.load(f)

_tokenizer      = _prep["tokenizer"]
_le             = _prep["label_encoder"]
_MAX_LEN        = _prep["max_len"]
_SOURCE_NAMES   = _prep.get("source_names", [])


# ─────────────────────────────────────────────
# 3. Text cleaning — IDENTICAL to notebook
# ─────────────────────────────────────────────
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = re.sub(r"^[A-Z ,]+\(reuters\)\s*[-\u2013]\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[a-z]+;", " ", text)
    text = re.sub(r"//\s*<!\[CDATA\[.*?\]\]>", " ", text, flags=re.DOTALL)
    text = re.sub(r"var\s+\w+\s*=\s*", " ", text)

    for name in _SOURCE_NAMES:
        text = re.sub(rf"\b{re.escape(name)}\b", " ", text, flags=re.IGNORECASE)

    text = text.lower()
    text = re.sub(r"\[.*?\]", " ", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\w*\d\w*", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ─────────────────────────────────────────────
# 4. Inference
# ─────────────────────────────────────────────
def predict_news(news_text: str, threshold: float = 0.5) -> dict:
    cleaned = clean_text(news_text)
    seq     = _tokenizer.texts_to_sequences([cleaned])
    padded  = pad_sequences(seq, maxlen=_MAX_LEN, padding="post")

    prob_f, prob_s = _model.predict(padded, verbose=0)

    fake_prob = float(prob_f[0][0])
    label     = "True" if fake_prob > threshold else "Fake"
    category  = _le.inverse_transform([int(np.argmax(prob_s))])[0]
    category_display = category.replace("News", "").strip().title()

    return {
        "fake_or_true": label,
        "category":     category_display,
    }


# ─────────────────────────────────────────────
# 5. Routes
# ─────────────────────────────────────────────
@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status":     "ok",
        "categories": _le.classes_.tolist(),
    })


@app.route("/predict", methods=["POST"])
@require_api_key
def predict():
    body = request.get_json(silent=True)

    if not body or "text" not in body:
        return jsonify({"error": "Request body must be JSON with a 'text' field."}), 400

    news_text = body["text"]

    if not isinstance(news_text, str) or not news_text.strip():
        return jsonify({"error": "'text' must be a non-empty string."}), 400

    try:
        result = predict_news(news_text)
        return jsonify(result), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ─────────────────────────────────────────────
# 6. Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
