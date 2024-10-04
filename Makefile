# toolbox variables
REPO=inscopix
PROJECT=ideas
MODULE=toolbox
IMAGE_NAME=suite2p
VERSION=$(shell git describe --tags --always --dirty)
IMAGE_TAG=${REPO}/${PROJECT}/${MODULE}/${IMAGE_NAME}:${VERSION}
FULL_NAME=${REPO}/${PROJECT}/${MODULE}/${IMAGE_NAME}
CONTAINER_NAME=${REPO}-${PROJECT}-${MODULE}-${IMAGE_NAME}-${VERSION}
PLATFORM=linux/amd64

# jupyter-lab configurations
ifndef JUPYTERLAB_PORT
	JUPYTERLAB_PORT=8889
endif

define run_command
    bash -c 'mkdir -p "/ideas/outputs/$1" \
        && cd "/ideas/outputs/$1" \
        && cp "/ideas/inputs/$1.json" "/ideas/outputs/$1/inputs.json" \
        && "/ideas/commands/$1.sh" \
	    && rm "/ideas/outputs/$1/inputs.json"'
endef



.PHONY: help build jupyter test clean


.DEFAULT_GOAL := build

clean:
	@echo "Cleaning up"
	-docker rm $(CONTAINER_NAME)
	-docker images | grep $(FULL_NAME) | awk '{print $$1 ":" $$2}' | grep -v $(VERSION) | xargs docker rmi



build: 
	PACKAGE_REQS=$$(if [ -f ../dev_requirements.txt ]; then cat ../dev_requirements.txt | grep -v "#" | tr '\n' ' '; else echo "ideas-public-python-utils@git+https://@github.com/inscopix/ideas-public-python-utils.git@0.0.17 isx==2.0.0"; fi) && \
	DOCKER_BUILDKIT=1 docker build . -t $(IMAGE_TAG) \
		--platform ${PLATFORM} \
		--build-arg PACKAGE_REQS="$$PACKAGE_REQS" \
		--target base; \


jupyter: clean
	@echo "Launching container with Jupyter lab..."
	PACKAGE_REQS=$$(if [ -f ../dev_requirements.txt ]; then cat ../dev_requirements.txt | grep -v "#" | tr '\n' ' '; else echo "ideas-public-python-utils@git+https://@github.com/inscopix/ideas-public-python-utils.git@0.0.17  isx==2.0.0"; fi) && \
	DOCKER_BUILDKIT=1 docker build . -t $(IMAGE_TAG)-jupyter \
		--platform ${PLATFORM} \
		--target jupyter \
		--build-arg PACKAGE_REQS="$$PACKAGE_REQS" \
	docker run -ti \
			-v $(PWD)/commands:/ideas/commands \
			-v $(PWD)/data:/ideas/data \
			-v $(PWD)/inputs:/ideas/inputs \
			-v $(PWD)/notebooks:/ideas/notebooks \
			-v $(PWD)/outputs:/ideas/outputs \
			-v $(PWD)/toolbox:/ideas/toolbox \
			-p ${JUPYTERLAB_PORT}:${JUPYTERLAB_PORT} \
			-e JUPYTERLAB_PORT=$(JUPYTERLAB_PORT) \
			--name $(CONTAINER_NAME) \
	    $(IMAGE_TAG)-jupyter \
	    jupyter-lab --ip 0.0.0.0 --port $(JUPYTERLAB_PORT) --no-browser --allow-root --NotebookApp.token="" \
	&& docker rm $(CONTAINER_NAME)

test: build clean 
	@echo "Running toolbox tests..."
	-mkdir -p $(PWD)/outputs
	docker run \
		--platform ${PLATFORM} \
		-v $(PWD)/data:/ideas/data \
		-v $(PWD)/inputs:/ideas/inputs \
		-v $(PWD)/commands:/ideas/commands \
		-w /ideas \
		-e CODEBUILD_BUILD_ID=${CODEBUILD_BUILD_ID} \
		--name $(CONTAINER_NAME) \
		${IMAGE_TAG} \
		pytest $(TEST_ARGS) 
	


run: build clean
	@bash check_tool.sh $(TOOL)
	@echo "Running the $(TOOL) tool in a Docker container. Outputs will be in /outputs/$(TOOL)"
	-rm -rf $(PWD)/outputs/
	docker run \
			--platform ${PLATFORM} \
			-v $(PWD)/data:/ideas/data \
			-v $(PWD)/inputs:/ideas/inputs \
			-v $(PWD)/commands:/ideas/commands \
			-e TC_NO_RENAME=$(TC_NO_RENAME) \
			--name $(CONTAINER_NAME) \
	    $(IMAGE_TAG) \
		$(call run_command,$(TOOL)) \
	&& docker cp $(CONTAINER_NAME):/ideas/outputs $(PWD)/outputs \
	&& docker rm $(CONTAINER_NAME)
