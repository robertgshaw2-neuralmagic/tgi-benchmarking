FROM ghcr.io/huggingface/text-generation-inference:latest

RUN pip install jupyter

COPY ./benchmark.py /usr/src/benchmark.py

ENTRYPOINT ["/bin/bash"]
