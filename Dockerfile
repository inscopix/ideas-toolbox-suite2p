# Create base image to run analysis in
# Change the base image based on your use case
FROM public.ecr.aws/lts/ubuntu:22.04 AS base

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONFAULTHANDLER=1
ENV DEBIAN_FRONTEND=noninteractive

# Create ideas user
# This is no longer necessary to do, but good practice anyways
RUN addgroup ideas \
    && adduser --disabled-password --home /ideas --ingroup ideas ideas

# Create ideas home dir
WORKDIR /ideas

# ========================== Apt Dependency Installation ===========================
RUN apt-get -y update \
    && apt-get upgrade -y --no-install-recommends \
    && apt-get install -y --no-install-recommends \
        software-properties-common \
        gcc \
        libgl1-mesa-glx \
        libglib2.0-0 \
        git \
        ffmpeg \
        wget \
    && rm -rf /var/lib/apt/lists/*

# Download cellpose model pre-trained from huggingface
# This model is usually downloaded by the cellpose package for suite2p ROI detection
# Saving the model in the image prevents download issues when running the tool in a sandbox environment
ADD --chown=ideas:ideas \
    --checksum=sha256:e1440429eb384f95afe32bcba6510f90d518eaedc917ede549bed6804004abe2 \
    https://huggingface.co/mouseland/cellpose-sam/resolve/7c61431b5fbb078f3296754bd15d9f51b320f837/cpsam \
    /ideas/.cellpose/models/cpsam

# Create a venv with uv to install python dependencies
# This can be done globally, but using venv is best practice

# ========================== Python Dependency Installation with uv ===========================

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy
# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1
# Ensure installed tools can be executed out of the box
ENV UV_TOOL_BIN_DIR=/usr/local/bin
# Install venv into in the ideas user's home
ENV UV_PROJECT_ENVIRONMENT=/ideas/venv VIRTUAL_ENV=/ideas/venv
# Allow uv to provision the pinned Python version required by the lockfile.
ENV UV_PYTHON_DOWNLOADS=automatic
# Strictly use frozen dependencies, and require cryptographic verification of each package
ENV UV_FROZEN=1 UV_REQUIRE_HASHES=1 UV_VERIFY_HASHES=1
ENV UV_CACHE_DIR=/tmp/.cache/uv

RUN --mount=from=ghcr.io/astral-sh/uv:0.11.2,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/tmp/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --no-install-project --group analysis

USER ideas

# Add venv bin to path
ENV PATH="/ideas/venv/bin:${PATH}"

CMD ["/bin/bash"]

# Create image for testing which copies tool code and test data to
# docker image in order to facilitate unit testing in an isolated environment.
# This can also be acheived with volume mounts, but that can clutter up
# your local folder with files generated during testing.
FROM base AS test

USER root

RUN --mount=from=ghcr.io/astral-sh/uv:0.11.2,source=/uv,target=/bin/uv \
--mount=type=cache,target=/tmp/.cache/uv \
--mount=type=bind,source=uv.lock,target=uv.lock \
--mount=type=bind,source=pyproject.toml,target=pyproject.toml \
uv sync --no-install-project --group analysis --group test

COPY --chown=ideas ./ /ideas

USER ideas

# Vulnerability scanning stage using Trivy
# Copies the runtime filesystem to a subdirectory to avoid overwrite conflicts with the trivy base image
FROM base AS scanner

USER root

COPY --from=aquasec/trivy:0.69.3 /usr/local/bin/trivy /usr/local/bin/trivy

RUN trivy rootfs --no-progress --ignore-unfixed --skip-files /usr/local/bin/trivy --severity CRITICAL,HIGH --exit-code 1 / \
    && touch /scan-ok

# Final stage - identical to runtime but depends on successful scan
FROM base AS final

COPY --from=scanner /scan-ok /scan-ok
