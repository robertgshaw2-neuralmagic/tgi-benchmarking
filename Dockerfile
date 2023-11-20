FROM ghcr.io/huggingface/text-generation-inference:latest

RUN pip install jupyter

WORKDIR /

ENTRYPOINT ["/bin/bash"]
