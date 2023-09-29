# Load pytorch image
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

COPY requirements.txt .

# Install dependencies
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git
RUN python3 -m pip install -r requirements.txt
