FROM dealii/dealii:latest

USER root

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev python3-venv \
    cmake ninja-build && \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install pybind11

RUN mkdir /workdir
WORKDIR /workdir

