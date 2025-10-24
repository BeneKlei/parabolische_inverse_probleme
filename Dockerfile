FROM dealii/dealii:latest

USER root

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev python3-venv \
    cmake ninja-build \
    texlive-latex-base \
    texlive-latex-extra \
    texlive-fonts-recommended \
    texlive-fonts-extra \
    dvipng \
    ghostscript \
    cm-super && \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install pybind11

USER dealii  # (or whatever the default user is in the image)
