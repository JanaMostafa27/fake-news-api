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
import torch.nn as nn

from flask import Flask, request, jsonify
from transformers import DistilBertTokenizerFast, DistilBertModel
from huggingface_hub import hf_hub_download

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
# 2. Model Architecture — must match training exactly
# ─────────────────────────────────────────────
class MultiTaskDistilBERT(nn.Module):
    def __init__(self, model_name, num_subjects, dropout=0.3):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained(model_name)
        hidden_size = self.distilbert.config.hidden_size  # 768

        self.dropout_shared = nn.Dropout(dropout)

        self.fake_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )
        self.subject_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_subjects),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        cls_repr = outputs.last_hidden_state[:, 0, :]  # [CLS] token → (batch, 768)
        cls_repr = self.dropout_shared(cls_repr)

        fake_logits    = self.fake_head(cls_repr).squeeze(-1)   # (batch,)
        subject_logits = self.subject_head(cls_repr)            # (batch, N)

        return fake_logits, subject_logits


# ─────────────────────────────────────────────
# 3. Download & load model artifacts at startup
# ─────────────────────────────────────────────
ARTIFACT_DIR = os.environ.get("ARTIFACT_DIR", "model_artifacts")
MODEL_PATH   = os.path.join(ARTIFACT_DIR, "multi_task_distilbert.pt")
PREP_PATH    = os.path.join(ARTIFACT_DIR, "preprocessor.pkl")

os.makedirs(ARTIFACT_DIR, exist_ok=True)

HF_REPO  = "JanaMostafa2/Trustera_model"
HF_TOKEN = os.environ.get("HF_TOKEN", "")

if not os.path.exists(MODEL_PATH):
    print("Downloading model weights from Hugging Face...")
    hf_hub_download(
        repo_id=HF_REPO,
        filename="multi_task_distilbert.pt",
        local_dir=ARTIFACT_DIR,
        token=HF_TOKEN or None,
    )
    print("Model weights downloaded.")

if not os.path.exists(PREP_PATH):
    print("Downloading preprocessor from Hugging Face...")
    hf_hub_download(
        repo_id=HF_REPO,
        filename="preprocessor.pkl",
        local_dir=ARTIFACT_DIR,
        token=HF_TOKEN or None,
    )
    print("Preprocessor downloaded.")

# ── Load preprocessor ──
print(f"Loading preprocessor from {PREP_PATH} ...")
with open(PREP_PATH, "rb") as f:
    _prep = pickle.load(f)

_le           = _prep["label_encoder"]
_MAX_LEN      = _prep["max_len"]
_SOURCE_NAMES = _prep.get("source_names", [])
_cfg          = _prep["model_config"]
_MODEL_NAME   = _cfg["model_name"]       # e.g. "distilbert-base-uncased"
_NUM_SUBJECTS = _cfg["num_subjects"]
_DROPOUT      = _cfg.get("dropout", 0.3)

# ── Load HuggingFace tokenizer (by name — same as training) ──
print(f"Loading tokenizer: {_MODEL_NAME} ...")
_tokenizer = DistilBertTokenizerFast.from_pretrained(_MODEL_NAME)

# ── Rebuild model architecture, then load state dict ──
print(f"Building model architecture ({_MODEL_NAME}, {_NUM_SUBJECTS} categories) ...")
_model = MultiTaskDistilBERT(
    model_name=_MODEL_NAME,
    num_subjects=_NUM_SUBJECTS,
    dropout=_DROPOUT,
)

print(f"Loading state dict from {MODEL_PATH} ...")
state_dict = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
_model.load_state_dict(state_dict)
_model.eval()
print("Model ready.")


# ─────────────────────────────────────────────
# 4. Text cleaning — identical to notebook
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
# 5. Inference
# ─────────────────────────────────────────────
def predict_news(news_text: str, threshold: float = 0.5) -> dict:
    cleaned = clean_text(news_text)

    # Tokenize using HuggingFace DistilBERT tokenizer
    encoding = _tokenizer(
        cleaned,
        max_length=_MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids      = encoding["input_ids"]       # (1, MAX_LEN)
    attention_mask = encoding["attention_mask"]  # (1, MAX_LEN)

    with torch.no_grad():
        fake_logits, subject_logits = _model(input_ids, attention_mask)

    # Fake/True prediction
    fake_prob = torch.sigmoid(fake_logits).item()   # probability of being TRUE (label=1)
    label     = "True" if fake_prob > threshold else "Fake"

    # Category prediction
    subject_idx      = subject_logits.argmax(dim=1).item()
    category         = _le.inverse_transform([subject_idx])[0]
    category_display = category.replace("News", "").strip().title()

    return {
        "fake_or_true": label,
        "category":     category_display,
    }


# ─────────────────────────────────────────────
# 6. Routes
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
# 7. Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
