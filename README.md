# `deepcelltypes` interactive demo

Cell-type prediction for multiplexed spatial proteomic data.

To run the demo, you will need:

1. Python
2. A running docker daemon
3. The prototype model and example data

## Setup

### Docker

Ensure that your docker daemon is running and doesn't require elevated
permissions:

```bash
docker run hello-world
```

### Python

Create a Python virtual environment and install the necessary dependencies:

```bash
python -m venv demo-env
source demo-env/bin/activate  # or whichever script is appropriate for your shell
pip install -r requirements.txt
```

### Model and example data

The model is not (yet) publicly available as it is currently under development.
For access to the demo data and model prototype, direct inquiries to
vanvalenlab@gmail.com.

The model and data should be unpacked in the top-level directory of this
repository:

```bash
tar -xzf demo_data_model.tar.gz
```

## Running the demo

The demo is available in the form of a markdown-based jupyter notebook.
Launch a new session with:

```bash
jupyter notebook
```

Then right-click `presentation.md` and select `Jupytext Notebook` from the 
`Open with` menu.
