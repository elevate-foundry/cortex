FROM python:3.12-slim

WORKDIR /app

# Cortex — AI-native OS inference kernel
# Install system tools used by hardware_detect
RUN apt-get update && apt-get install -y --no-install-recommends \
    procps \
    lscpu \
    pciutils \
    && rm -rf /var/lib/apt/lists/*

# Copy source
COPY src/ /app/src/

# Default: run detect
CMD ["python", "-m", "src", "detect"]
