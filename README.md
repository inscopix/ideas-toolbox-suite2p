# ideas-toolbox-suite2p

**Table of Contents**
- [Toolbox Description](#toolbox-description)
- [Navigating the Project Repository](#navigating-the-project-repository)
- [Usage Instructions](#usage-instructions)
  - [List available commands](#list-available-commands)
  - [Build Docker image](#build-docker-image)
  - [Clean project](#clean-project)
  - [Run a specific tool](#run-a-specific-tool)
  - [Run jupyterlab inside the docker container](#run-jupyterlab-inside-the-docker-container)
  - [Run tests](#run-tests)

## Toolbox Description
A toolbox for running suite2p-based tools on the IDEAS platform.

This toolbox is designed to run as a Docker image, which can be run on the IDEAS platform. This toolbox consists of the following tools:

- Suite2p end-to-end pipeline: Run the suite2p pipeline to extract calcium traces.
- Binary Conversion: Convert raw 2p input movie(s) into a suite2p binary file.
- Registration: Run suite2p registration on a raw suite2p binary movie.
- ROI Detection: Run suite2p ROI detection on a raw suite2p binary movie.
- ROI Extraction: Extract calcium traces from suite2p binary movie(s).
- ROI Classification: Run suite2p ROI classification on extracted ROIs.
- Spike Deconvolution: Run suite2p spike deconvolution on extracted fluorescence traces.
- Output Conversion: Output suite2p results in specific formats


## How to Get Help
- [IDEAS documentation](https://inscopix.github.io/ideas-docs/tools/suite2p/suite2p_wrapper__run_suite2p_end_to_end.html) contains detailed information on how to use the toolbox within the IDEAS platform, the parameters that can be used, and the expected output.
- If you have found a bug, we reccomend searching the [issues page](https://github.com/inscopix/ideas-toolbox-suite2p/issues) to see if it has already been reported. If not, please open a new issue.
- If you have a feature request, please open a new issue with the label `enhancement`

## Executing the Toolbox

To run the toolbox, you can use the following command:

`make run TOOL=<tool_name>`

Available tools are:

- `suite2p_wrapper__run_suite2p_end_to_end`
- `suite2p_individual_steps__suite2p_binary_conversion`
- `suite2p_individual_steps__suite2p_registration`
- `suite2p_individual_steps__suite2p_roi_detection`
- `suite2p_individual_steps__suite2p_roi_extraction`
- `suite2p_individual_steps__suite2p_roi_classification`
- `suite2p_individual_steps__suite2p_spike_deconvolution`
- `suite2p_individual_steps__suite2p_output_conversion`

The command will excute the tool with inputs specified in the `inputs` folder. The output will be saved in the `outputs` folder.

## Navigating the Project Repository

```
├── commands                # Standardized scripts to execute tools on the cloud
├── data                    # Small data files used for testing
├── info                    # Files describing the toolbox & tools for the IDEAS system
├── inputs                  # Predefined test inputs for the command scripts
│── toolbox                 # Contains all code for running and testing the tools
│   ├── tools               # Contains the individual analysis tools
│   ├── utils               # General utilities used by the tools
│   ├── tests               # Unit tests for the individual tools
└── .gitignore              # Tells Git which files & folders to ignore
│── Dockerfile              # Commands to assemble the Docker image
│── Makefile                # To automate and standardize toolbox usage
│── check_tool.sh           # Checks if tool is valid before execution
│── function_caller.py      # Executes tool for IDEAS system
│── pyproject.toml          # Configuration for the python project
│── setup.py                # Specifies dependencies of the python project
│── user_deps.txt           # Specifies user dependencies of the python project
```

## Usage Instructions

### List available commands
```
make help
```

### Build Docker image
```
make build
```

### Clean project
```
make clean
```

### Run a specific tool
```
make run TOOL="tool_name"
```

### Run jupyterlab inside the docker container
```
make run-jupyter
```

and navigate to http://localhost:8888 to get a Jupyter environment inside the docker container 

### Run tests
```
make test
```

Test arguments can be specified using `TEST_ARGS` as shown below. Refer to [pytest documentation](https://docs.pytest.org/en/7.1.x/how-to/usage.html) for supported arguments.
```
make test TEST_ARGS="-k TestWorkflows"
```