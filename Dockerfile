FROM public.ecr.aws/lts/ubuntu:20.04 AS base

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
    && apt install -y python3.9 git python3-pip ffmpeg \
    && python3.9 -m pip install --upgrade pip

# link python to the version of python BE needs
RUN ln -s /usr/bin/python3.9 /usr/bin/python

# copy code and things we need
COPY setup.py ./

# install dependencies
RUN python3.9 -m pip install -e .

USER ideas
CMD ["/bin/bash"]
