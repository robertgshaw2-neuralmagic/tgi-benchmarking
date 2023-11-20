# tgi-benchmarking
Benchmarking LLMs on GPUs

# install 

Assumes ubuntu 22.04

## CUDA:

Drivers / Toolkit:
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```

Activate Path:
```bash
export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}
```

## Docker:

Add Docker's official GPG key:
```bash

sudo apt-get update
sudo apt-get install ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
```

Add the repository to Apt sources:
```bash
echo \
  "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
```

Install:

```bash
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

Verify Installed:

```bash
sudo docker run hello-world
```

## Nvidia-Docker

Configure:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list \
  && \
    sudo apt-get update
```

Install:
```bash
sudo apt-get install -y nvidia-container-toolkit
```

Configure Docker:
```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify:
```bash
sudo docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```


# run TGI

Setup enviornment variables:

```bash
mkdir data
token={YOUR_HF_TOKEN}
```

Build and run:

```bash
docker build -t tgi .
docker run --gpus all --shm-size 1g --network host -v $PWD/data:/data -v $PWD/scripts:/script -e HUGGING_FACE_HUB_TOKEN=$token -it tgi
```

# run benchmark

Download the weights

```bash
text-generation-server download-weights meta-llama/Llama-2-7b-hf
```

launch jupyter server
```bash
jupyter notebook --allow-root
```

