# For a Hugging Face Docker Space (CPU). Put this file at the ROOT of the Space repo as "Dockerfile".
FROM python:3.11-slim
RUN useradd -m -u 1000 user
WORKDIR /app
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt
COPY --chown=user . .
USER user
# BASE_MODEL must be the SAME model your adapter was trained on.
ENV HOME=/home/user HF_HOME=/home/user/.cache/huggingface BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct
EXPOSE 7860
CMD ["streamlit","run","app.py","--server.port=7860","--server.address=0.0.0.0","--server.headless=true","--server.enableXsrfProtection=false","--server.enableCORS=false"]
