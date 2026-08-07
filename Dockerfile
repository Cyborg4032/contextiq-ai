FROM python:3.11-slim

# ffmpeg is required by pydub (audio conversion/chunking) and yt-dlp
# (audio extraction). It was never declared anywhere in this repo —
# on Streamlit Cloud that needs a packages.txt; here we just install
# it in the image directly.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# sentence-transformers pulls in torch as a dependency. Left to its own
# devices, pip resolves a CUDA build of torch that drags in ~2-3GB of
# unused NVIDIA libraries (Render has no GPU) — install the CPU-only
# wheel explicitly first so the requirements.txt install below reuses it.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ephemeral scratch dirs the app writes to at runtime
RUN mkdir -p downloads vector_db

# Render injects $PORT; Streamlit must bind to 0.0.0.0 on that port.
EXPOSE 8501
CMD streamlit run app.py \
    --server.port=${PORT:-8501} \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false
