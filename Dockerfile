# ========================== Base stage for all other build stages ===========================
# Defines environment variables and installs packages necessary across all build stages
FROM public.ecr.aws/lts/ubuntu:24.04 AS base

# Standard env variables for debian and python based docker image
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONFAULTHANDLER=1
ENV DEBIAN_FRONTEND=noninteractive

# Define environment variables to control uv behavior across build stages
# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy
# Disable bytecode compilation (causes too many files open OS error)
ENV UV_COMPILE_BYTECODE=0
# Ensure installed tools can be executed out of the box
ENV UV_TOOL_BIN_DIR=/usr/local/bin
# Install venv into in the ideas user's home
ENV UV_PROJECT_ENVIRONMENT=/ideas/venv VIRTUAL_ENV=/ideas/venv
# Allow uv to download required python version based on pinned .python-version
ENV UV_PYTHON_DOWNLOADS=automatic
# Ensure the lock file is up-to-date before building image
ENV UV_LOCKED=1
# Specify directory for caching uv for reuse across builds
ENV UV_CACHE_DIR=/tmp/.cache/uv
# Specify accessible directory to install python
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python

# Create ideas user for container runtime
RUN groupadd ideas \
    && useradd -g ideas --home-dir /ideas --create-home --no-log-init ideas

# Set workdir to ideas homedir
WORKDIR /ideas

# Install and update ca certs for security
RUN apt-get -y update \
    && apt-get upgrade -y --no-install-recommends \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ========================== Builder stage to install analysis dependencies ===========================
FROM base AS builder

# Create venv to install analysis dependencies
RUN --mount=from=ghcr.io/astral-sh/uv:0.11.2,source=/uv,target=/bin/uv \
    --mount=type=cache,id=uv-cache-ideas,target=/tmp/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=.python-version,target=.python-version \
    uv sync --no-install-project --group analysis \
    # fix for pynwb api which requires write permissions to site-package dir
    && /ideas/venv/bin/python -c "import pynwb" 

# ========================== Runtime stage for tool execution ===========================
FROM base AS runtime

RUN apt-get -y update \
    && apt-get upgrade -y --no-install-recommends \
    && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        libglx-mesa0 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy python and analysis dependencies from builder stage
COPY --from=builder /ideas/venv /ideas/venv
COPY --from=builder /opt/uv-python /opt/uv-python

USER ideas

# Add venv bin to path to make `python` point to venv installation
ENV PATH="/ideas/venv/bin:${PATH}"

CMD ["/bin/bash"]

# ========================== Test stage for unit test execution ===========================
FROM runtime AS test

USER root

# Install additional python dependencies for running unit tests
RUN --mount=from=ghcr.io/astral-sh/uv:0.11.2,source=/uv,target=/bin/uv \
--mount=type=cache,id=uv-cache-ideas,target=/tmp/.cache/uv \
--mount=type=bind,source=uv.lock,target=uv.lock \
--mount=type=bind,source=pyproject.toml,target=pyproject.toml \
--mount=type=bind,source=.python-version,target=.python-version \
uv sync --no-install-project --group analysis --group test

# Copy tool code and test data for the test environment
COPY --chown=ideas ./ /ideas

USER ideas

# ========================== Scanner stage for security vulnerabilities ===========================
FROM runtime AS scanner

USER root

# Vulnerability scanning stage using Trivy
RUN --mount=from=aquasec/trivy:0.69.3,source=/usr/local/bin/trivy,target=/usr/local/bin/trivy \
    trivy rootfs --no-progress --ignore-unfixed --skip-files /usr/local/bin/trivy --severity CRITICAL,HIGH --exit-code 1 / \
    && touch /scan-ok

# ========================== Final stage - identical to runtime but depends on successful scan ===========================
FROM runtime AS final

COPY --from=scanner /scan-ok /scan-ok
