FROM public.ecr.aws/lts/ubuntu:22.04 AS base

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONFAULTHANDLER=1
ENV DEBIAN_FRONTEND=noninteractive

# Create ideas user
RUN addgroup ideas \
    && adduser --disabled-password --home /ideas --ingroup ideas ideas

WORKDIR /ideas

RUN apt update && apt upgrade -y \
    && apt install -y software-properties-common \
    && apt install -y gcc python3-dev \
    && apt install -y libgl1-mesa-glx libglib2.0-0 \
    && apt install -y git python3-pip ffmpeg

ARG PYTHON=python3
RUN ${PYTHON} -m pip install --upgrade pip

# copy code and things we need
COPY requirements.txt ./

# install dependencies
RUN ${PYTHON} -m pip install -r requirements.txt

# link python to make it available for tool runner
RUN ln -sf /usr/bin/python3 /usr/local/bin/python

USER ideas
CMD ["/bin/bash"]
