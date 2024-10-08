FROM public.ecr.aws/lts/ubuntu:20.04 AS base

# Accept PACKAGE_REQS as a build argument
ARG PACKAGE_REQS

# Set it as an environment variable for use during runtime if needed
ENV PACKAGE_REQS=$PACKAGE_REQS

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONFAULTHANDLER=1
ENV DEBIAN_FRONTEND=noninteractive

# adding this to path so that we can source python correctly. 
# this is where the python version we install using apt will
# live
ENV PATH="${PATH}:/ideas/.local/bin"


# Create ideas user
RUN addgroup ideas \
    && adduser --disabled-password --home /ideas --ingroup ideas ideas

WORKDIR /ideas


RUN apt update && apt upgrade -y \
    && apt install -y software-properties-common \
    && apt install -y gcc python3-dev \
    && apt install -y libgl1-mesa-glx libglib2.0-0 \
    && apt install -y python3.9 git python3-pip ffmpeg \
    && python3.9 -m pip install --upgrade pip \
    && python3.9 -m pip install --no-cache-dir awscli boto3 click requests

# link python to the version of python BE needs
RUN ln -s /usr/bin/python3.9 /usr/bin/python

# copy code and things we need
COPY setup.py function_caller.py user_deps.txt install_imported_code.sh ./

# install dependencies
RUN python3.9 -m pip install -e .

# install user code from git repo if needed
RUN /bin/bash install_imported_code.sh

COPY --chown=ideas toolbox /ideas/toolbox

# this is after installing the code because we don't want to
# reinstall everything if we update a command
COPY --chown=ideas commands /ideas/commands

# Mark commands as executable
# the reason we always return 0 is because we want this to succeed
# even if there are no commands in /ideas/commands/
# (which can happen in initial stages of tool dev)
RUN chmod +x /ideas/commands/* ; return 0


# copy JSON files in info
# this includes the toolbox_info.json, and annotation files
# that are used to generate output manifests
COPY --chown=ideas info /ideas/info

USER ideas
CMD ["/bin/bash"]

FROM base AS jupyter
RUN python3.9 -m pip install jupyter