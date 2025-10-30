.PHONY:  clean build test

IMAGE_NAME=suite2p
ifndef LABEL
	LABEL=0.0.1
endif
IMAGE_TAG=${IMAGE_NAME}:${LABEL}
FULL_NAME=${IMAGE_NAME}
PLATFORM=linux/amd64
ifndef TARGET
	TARGET=base
endif

.DEFAULT_GOAL := build

clean:
	@echo "Cleaning up"
	-docker rm $(CONTAINER_NAME)
	-docker images | grep $(FULL_NAME) | awk '{print $$1 ":" $$2}' | grep -v $(VERSION) | xargs docker rmi

build: 
	docker build . -t $(IMAGE_TAG) \
		--platform ${PLATFORM} \
		--target ${TARGET}

test: TARGET=test
test: build 
	@echo "Running tests..."
	docker run \
		--platform ${PLATFORM} \
		--rm \
		${IMAGE_TAG} \
		pytest ${TEST_ARGS}
