FROM python:3.11.9-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download model artifacts at BUILD time
ARG HF_TOKEN
RUN python -c "\
from huggingface_hub import hf_hub_download; \
import os; \
os.makedirs('model_artifacts', exist_ok=True); \
hf_hub_download(repo_id='JanaMostafa2/Trustera_model', filename='multi_task_distilbert.pt', local_dir='model_artifacts', token=os.environ.get('HF_TOKEN') or None); \
hf_hub_download(repo_id='JanaMostafa2/Trustera_model', filename='preprocessor.pkl', local_dir='model_artifacts', token=os.environ.get('HF_TOKEN') or None); \
print('Artifacts downloaded successfully.')"

# Copy app code
COPY . .

EXPOSE 8080

CMD gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120
