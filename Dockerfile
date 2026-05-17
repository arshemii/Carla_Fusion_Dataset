FROM carlasim/carla:0.9.16

USER root

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=/opt/conda/envs/carla310/bin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# Basic tools + SSH
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    wget \
    git \
    nano \
    tmux \
    bzip2 \
    openssh-server \
    && mkdir -p /var/run/sshd \
    && rm -rf /var/lib/apt/lists/*

# Install Miniforge, not Anaconda Miniconda
RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm /tmp/miniforge.sh

# Python 3.10 environment
RUN conda create -y -n carla310 python=3.10 pip \
    && conda clean -afy

# Python packages
RUN /opt/conda/envs/carla310/bin/python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && /opt/conda/envs/carla310/bin/python -m pip install --no-cache-dir \
        numpy \
        opencv-python-headless \
        scipy \
        pillow \
        tqdm \
        pyyaml \
    && /opt/conda/envs/carla310/bin/python -m pip install --no-cache-dir \
        /workspace/PythonAPI/carla/dist/carla-0.9.16-cp310-cp310-manylinux_2_31_x86_64.whl

# Download CARLA dataset generator scripts
RUN mkdir -p /workspace/scripts \
    && wget -O /workspace/scripts/gen_carla.py \
        https://raw.githubusercontent.com/arshemii/Carla_Fusion_Dataset/main/gen_carla.py \
    && wget -O /workspace/scripts/sim_carla.py \
        https://raw.githubusercontent.com/arshemii/Carla_Fusion_Dataset/main/sim_carla.py \
    && chmod 644 /workspace/scripts/gen_carla.py /workspace/scripts/sim_carla.py \
    && chown -R carla:carla /workspace/scripts

# Make Python 3.10 default
RUN ln -sf /opt/conda/envs/carla310/bin/python /usr/local/bin/python \
    && ln -sf /opt/conda/envs/carla310/bin/python /usr/local/bin/python3 \
    && ln -sf /opt/conda/envs/carla310/bin/pip /usr/local/bin/pip \
    && ln -sf /opt/conda/envs/carla310/bin/pip /usr/local/bin/pip3

# SSH configuration
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config \
    && sed -i 's/#PubkeyAuthentication yes/PubkeyAuthentication yes/' /etc/ssh/sshd_config \
    && sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config

# Startup script for RunPod
RUN cat > /start.sh <<'EOF'
#!/bin/bash
set -e

mkdir -p /root/.ssh
chmod 700 /root/.ssh

if [ -n "$PUBLIC_KEY" ]; then
    echo "$PUBLIC_KEY" > /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
fi

service ssh start

echo "SSH server started."
echo "Python path:"
which python
python --version
echo "CARLA image is ready."
echo "Use /workspace for CARLA and /data_ssd for persistent storage on RunPod."

sleep infinity
EOF

RUN chmod +x /start.sh \
    && chown -R carla:carla /opt/conda

EXPOSE 22

WORKDIR /workspace

CMD ["/start.sh"]
