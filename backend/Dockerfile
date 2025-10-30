# Use Miniforge base
FROM condaforge/miniforge3

# Set work folder
WORKDIR /app

# Copy project files
COPY environment.yml .
COPY requirements.txt .
COPY api/ api/
COPY data/ data/
COPY model/ model/
COPY scripts/ scripts/

# Create Conda env from environment file
RUN mamba env create -f environment.yml && \
    mamba clean -afy

# Install CUDA 13.0 PyTorch stack and packages
RUN set -eux && \
    pip install uv && \
    mamba run -n cs5647project uv pip install --system \
    torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

RUN set -eux && \
    mamba run -n cs5647project uv pip install --system -r requirements.txt && \
    mamba clean -afy

# Pre-download HF models at build time
ARG HUGGING_FACE_TOKEN
# persist into env
ENV HUGGING_FACE_TOKEN=${HUGGING_FACE_TOKEN}

RUN set -eux && \
    mamba run -n cs5647project python /app/scripts/entry.py

# Default entry
CMD ["mamba", "run", "-n", "cs5647project", \
     "python", "-m", "uvicorn", "api.app:app", \
     "--host", "0.0.0.0", "--port", "8000"]
