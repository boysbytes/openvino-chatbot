FROM openvino/ubuntu20_runtime:2024.6.0

# Switch to root user for package installations
USER root

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --upgrade pip

WORKDIR /app

# Install Python dependencies with updated versions
RUN pip install --no-cache-dir \
    chainlit==2.1.0 \
    "optimum[openvino]==1.20.0" \
    openvino==2024.6.0 \
    transformers==4.40.0 \
    intel-extension-for-transformers==1.3.0

# Copy application files
COPY models/DeepSeek-R1-Distill-Qwen-1.5B-openvino/1 /app/models/DeepSeek-R1-Distill-Qwen-1.5B-openvino/1
# COPY models/Qwen2.5-0.5B-Instruct-openvino-4bit/1 /app/models/Qwen2.5-0.5B-Instruct-openvino-4bit/1
COPY app.py .

EXPOSE 3000

# Runtime configuration
CMD ["sh", "-c", "sysctl -w vm.max_map_count=262144 && chainlit run app.py --host 0.0.0.0 --port 3000"]
