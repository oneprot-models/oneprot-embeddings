# Environment



The recommended execution environment is provided as an Apptainer/Singularity container. Rebuilding the environment from scratch involves two stages. The container itself is available on [**zenodo**](https://zenodo.org/uploads/20997998). After it is executed, an environment needs to be activated inside (Step 3 of the below workflow).

### 1. Build the base container

The base container is built from the **NVIDIA PyG 28.03** image using the Apptainer definition file:

```text
environment/pyg-28-03.def
```

This provides the NVIDIA PyG software stack, including the corresponding versions of PyTorch, PyTorch Geometric, CUDA, and Python.

### 2. Build the OneProt container

The base container is then extended with the OneProt-specific software stack using the requirements file

```text
environment/requirements_pyg_28_03.txt
```

and the corresponding Apptainer definition file

```text
environment/pyg-28-03.def
```

This produces the final `oneprot_pyg-28-03-py3-amd64.sif` container used throughout this repository.

### 3. Create the project virtual environment

Launch the container:

```bash
apptainer run --nv oneprot_pyg-28-03-py3-amd64.sif bash
```

Inside the container, create and activate a Python virtual environment, then install the project-specific dependencies:

```bash
pip install -r environment/requirements_env_dig.txt
```

Finally, install **Dive into Graphs** separately:

```bash
pip install dive-into-graphs --no-dependencies
```

This additional virtual environment contains the packages required for the downstream workflows, including the MD extension and allostery analyses.
