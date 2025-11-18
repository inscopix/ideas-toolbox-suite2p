# Create base image to run analysis in
# Change the base image based on your use case
FROM public.ecr.aws/lts/ubuntu:20.04 AS base

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONFAULTHANDLER=1
ENV DEBIAN_FRONTEND=noninteractive

# Arguments for python installation
ARG PYTHON=python3.9
ARG VENV=venv
ARG PYTHON_VENV=/ideas/${VENV}/bin/python

# Create ideas user
# This is no longer necessary to do, but good practice anyways
RUN addgroup ideas \
    && adduser --disabled-password --home /ideas --ingroup ideas ideas

# Create ideas home dir
WORKDIR /ideas

# Copy python project settings
COPY pyproject.toml ./

# Install apt packages here
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
        software-properties-common \
        gcc \
        python3-dev \
        libgl1-mesa-glx \
        libglib2.0-0 \
        python3.9 \
        python3.9-venv \
        python3-pip \
        git \
        ffmpeg\
    && rm -rf /var/lib/apt/lists/* \
    # Create a venv to install python dependencies
    # This can be done globally, but using venv is best practice
    && ${PYTHON} -m venv ${VENV} \
    && ${PYTHON_VENV} -m pip install --no-cache --upgrade pip \
    && ${PYTHON_VENV} -m pip install --no-cache .

# Add venv bin to path
ENV PATH="/ideas/${VENV}/bin:${PATH}"

USER ideas
CMD ["/bin/bash"]

# Create image for testing which copies tool code and test data to
# docker image in order to facilitate unit testing in an isolated environment.
# This can also be acheived with volume mounts, but that can clutter up
# your local folder with files generated during testing.
FROM base AS test

COPY --chown=ideas ./ /ideas
