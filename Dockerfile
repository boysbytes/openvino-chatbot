FROM openvino/ubuntu20_runtime:2024.6.0

# Install system dependencies
USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --upgrade pip
USER $NB_UID:$NB_GID

WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir \
    chainlit==2.1.0 \
    "optimum[openvino]==1.20.0" \
    openvino==2024.6.0 \
    transformers==4.40.0 \
    intel-extension-for-transformers==1.3.0 \
    python-dotenv

# Copy application files
COPY models/DeepSeek-R1-Distill-Qwen-1.5B-openvino/1 /app/models/DeepSeek-R1-Distill-Qwen-1.5B-openvino/1
COPY app.py . 
COPY .env .

EXPOSE 3000

# Runtime configuration
CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "3000"]
