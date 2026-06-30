# Repository scope

This repository contains code and documentation for **three related OneProt** studies:

| Study | Status | Documentation |
|---|---:|---|
| [OneProt: Towards multi-modal protein foundation models via latent space alignment of sequence, structure, binding sites and text encoders](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1013679) | PLOS Computational Biology | [Original OneProt](#original-oneprot) |
| [When Protein Dynamics Matter: Integrating Molecular Dynamics into Protein Foundation Models](https://openreview.net/forum?id=Q6DuJPwH2U) |ICLR 2026 Foundational Models for Science | [OneProt MD extension](#md-extension) |
| [Multimodal Protein Foundation Models Reveal Dataset-Driven Separability Regimes in Allosteric Site Prediction]() | In preparation | [Allostery prediction with OneProt](#allostery-prediction-with-oneprot) |

- The **original OneProt model** aligns protein sequence, structure, binding-site, and text encoders in a shared latent space.
- The **MD extension** adds time-resolved molecular-dynamics trajectories as an additional modality.
- The **Allosteric Site prediction** study uses OneProt-derived representations for downstream analyses of allosteric versus competitive/orthosteric binding sites across a number of different separability regimes

Downstream framework is common for all three papers and described in [**Downstream**](#downstream) below

# Environment



The recommended execution environment is provided as an Apptainer/Singularity container. Rebuilding the environment from scratch involves two stages. The container itself is available on [**zenodo**](https://zenodo.org/uploads/20997998). After it is executed, an environment needs to be activated inside (Step 3 of the below workflow). Typical execution scripts can be found in `train_oneptrot_ddp.sbatch`, `collect_embeddings.sbatch` and `saprot_fit.sbatch`

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




# Original OneProt

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>


</div>

## Getting started with Lightning and Hydra
- https://lightning.ai/docs/pytorch/stable/tutorials.html lightning tutorial
- https://hydra.cc/docs/tutorials/intro/ hydra tutorial

## Description

This project is dedicated to advancing the understanding and application of various modalities related to proteins, such as sequence, structure, represented as graphs and as foldseek tokens, pockets and sequence similarity tuples, based on mutational information and multiple sequence alignments (MSA). 

We are aiming to learn aligned embeddings for different protein modalities. These different modalities can later be used on retrieval, prediction and generation tasks for proteins. 

It started as a prototype model of the Bio x ML Hackathon 2023, which won the first prize and the impact prize, and the initial version of the model is [**here**](https://github.com/svm-ai/svm-hackathon)

**The weights of the model and example datasets are available on** [**huggingface**](https://huggingface.co/HelmholtzAI-FZJ/oneprot/)

## Modalities 

- Sequence
- Structure
- Text
- Pockets
- Sequence similarity

<br>

## Dataset 
We only require paired modalities dataset. 
### Dataset curation

We used [**OpenProteinSet**](https://registry.opendata.aws/openfold/), which contains structures, sequences, and MSAs for proteins from the [**PDB**](https://www.rcsb.org) and proteins from [**UniClust30**](https://uniclust.mmseqs.com) and [**UniProtKB/Swiss-Prot**](https://www.expasy.org/resources/uniprotkb-swiss-prot). We used [**MMseqs2**](https://github.com/soedinglab/MMseqs2), to cluster the sequences with a sequence identity cut-off of 50\%, such that each cluster represents a homologous cluster in the protein fold space. We aligned the training, validation, and test splits along these sequence clusters. For each cluster representative and member, using the sequence, we find the structure from the [**AlphaFold2DB**](https://alphafold.ebi.ac.uk), the MSA from the OpenProteinSet, and the binding pocket with [**P2Rank**](https://github.com/rdk/p2rank). As we could not find an MSA and binding pocket for each protein, fewer data points for these modalities are available. Sequence similarity dataset was constructed using [**ClinVar**]( https://www.clinicalgenome.org/data-sharing/clinvar/) variant summary data and MSA data. Each data-point in the sequence similarity dataset consists of three pairs of sequences corresponding to the same protein: original sequence and sequence with a benign mutation, two distinct pathogenic sequences, original sequence and a sequence sampled from the corresponding MSA. Such dataset enforces clustering of the proteins based on their biological relevance, e.g. moves pathogenic mutations away from benign ones.

| Modality 1 | Modality 2 | Dataset Size (Train/Val/Test) |
|----------|----------|----------|
| Sequence | Structure Graph | 647781 / 1000 / 1000 |
| Sequence | Structure Token | 1000000 / 1000 / 1000 |
| Sequence | Text | 540077 / 1000 / 1000 |
| Sequence | Pockets | 335086 / 1000 / 1000|
| Sequence | Sequence similarity| 1040560 / 1000 / 1000|



<br>

## Main Ideas


- [**ImageBind**](https://arxiv.org/abs/2305.05665)
- [**CLIP**](https://arxiv.org/abs/2103.00020)

DownStream Tasks:

- [**SaProt**](https://www.biorxiv.org/content/10.1101/2023.10.01.)
<br>

# OneProt MD extension

# Allostery prediction with OneProt


This directory contains the code accompanying the manuscript [**Multimodal Protein Foundation Models Reveal Dataset-Driven Separability Regimes in Allosteric Site Prediction.**]()

The study investigates how multimodal protein foundation models can be leveraged for allosteric site prediction across datasets exhibiting markedly different levels of intrinsic separability. Using frozen representations extracted from pretrained OneProt models, we systematically evaluate the contribution of sequence, structural, molecular dynamics, and functional text modalities to downstream allosteric site classification. The benchmark spans four datasets representing distinct separability regimes, allowing us to disentangle the influence of dataset composition from encoder architecture and modality selection.

The repository provides the complete workflow for reproducing the experiments, including:

- dataset preparation,
- binding-pocket construction,
- extraction of multimodal OneProt embeddings,
- downstream MLP training and evaluation,
- generation of the figures and analyses reported in the manuscript.

Detailed workflows are provided in (**`src/allostery/`**)[https://github.com/oneprot-models/oneprot-embeddings/tree/main/src/allostery].

The accompanying Zenodo archive (https://doi.org/10.5281/zenodo.20997998) contains:

- pretrained OneProt checkpoints,
- processed binding-pocket representations (`.h5` files),
- train/validation/test splits for all benchmark datasets.

These resources enable reproduction of all experiments without regenerating the underlying pocket representations or pretrained model checkpoints.

# Downstream

## Saprot MLP Downstream Evaluation Scripts

This document describes the two Saprot-related MLP evaluation scripts:

- `src/saprot_fit_mlp.py`
- `src/saprot_fit_mlp_balanced.py`

Both scripts train a PyTorch Lightning MLP head on precomputed protein embeddings and evaluate downstream prediction tasks. They use Hydra configuration from `configs/saprot_mlp.yaml`.

## Shared Purpose

These scripts are used to benchmark embeddings on downstream biological prediction tasks. They do not generate embeddings themselves. Instead, they:

1. Load precomputed train/validation/test embeddings using `load_data(cfg)`.
2. Build an MLP classifier or regressor on top of the embeddings.
3. Train the MLP with PyTorch Lightning.
4. Evaluate on validation and test splits.
5. Save metrics with `save_results_to_csv`.

## `src/saprot_fit_mlp.py`

This is the standard MLP downstream evaluation script.

It supports multiple task types, including:

- Binary classification tasks
- Multiclass classification tasks
- Multilabel prediction tasks
- Regression tasks such as thermostability
- Sequence/pocket-style allostery tasks with masked labels

The script determines the MLP input dimension from `cfg.model_type`. For example, Saprot and ESM2-style embeddings use 1280-dimensional inputs, while other model types may use different embedding sizes. For paired or concatenated tasks, the input dimension is multiplied accordingly.

The script sweeps over MLP hyperparameters from `cfg.sweep`, including:

- Learning rate
- Batch size
- Number of epochs
- Hidden dimensions
- Dropout
- Batch normalization
- Layer normalization
- Activation function
- Residual connections
- Task name
- Model type

Use this script for the normal downstream MLP evaluation workflow.

## `src/saprot_fit_mlp_balanced.py`

This script is a class-balanced variant of `saprot_fit_mlp.py`.

It follows the same training and evaluation pipeline, but adds one important preprocessing step for selected merged binary [**allostery tasks**](#allostery-prediction-with-oneprot): during training data setup, it balances positive and negative classes by subsampling the majority class.

This is useful when merged binary datasets are strongly imbalanced and the model might otherwise learn to favor the majority class.

For balanced tasks, results are saved with a `_balanced` suffix in the task name so they can be distinguished from the standard MLP runs.

## Main Difference

`src/saprot_fit_mlp.py` uses the training data as loaded.

`src/saprot_fit_mlp_balanced.py` modifies selected binary merged allostery training sets by balancing positive and negative examples before training.

Validation and test sets are not balanced manually; they are used for evaluation as loaded.

## Typical Usage

Run from the repository root:

```bash
python src/saprot_fit_mlp.py
python src/saprot_fit_mlp_balanced.py

## Task Coverage

Both scripts span general protein benchmark tasks and allostery-specific pocket/site prediction tasks. The code still uses historical `task_name` identifiers in some places; in repository-facing documentation, these can be mapped to the current dataset names:

- `PL8` corresponds to **PPI-Site**.
- `Kinase_*` tasks correspond to **KinSite**.
- `merged_pocket*` tasks correspond to **Dual-Site**, the merged PPI-Site + KinSite dataset.
- `ASD_merged_*` tasks correspond to **AlloDiverse**, where ASD data are appended to the merged site dataset.

The task families covered by `src/saprot_fit_mlp.py` include:

- General binary protein prediction:
  - `MetalIonBinding`
  - `DeepLoc2`
  - `HumanPPI`

- Protein function / ontology prediction:
  - `EC`
  - `GO-BP`
  - `GO-MF`
  - `GO-CC`

- Localization / multiclass classification:
  - `DeepLoc10`

- Regression:
  - `ThermoStability`

- Enzyme classification:
  - `TopEnzyme`

- [**Allostery**](#allostery-prediction-with-oneprot) and pocket/site prediction:
  - `PL8` / **PPI-Site**
  - `Kinase_pocket` / **KinSite**
  - `Kinase_combined` / **KinSite**
  - `merged_pocket` / **Dual-Site**
  - `merged_pocket_sequence` / **Dual-Site**
  - `merged_pocket_binary` / **Dual-Site**
  - `merged_pocket_sequence_binary` / **Dual-Site**
  - `ASD_merged_pocket_binary` / **AlloDiverse**
  - `ASD_merged_pocket_sequence_binary` / **AlloDiverse**

- Text-augmented or concatenated allostery variants:
  - `Kinase_combined_text`
  - `Kinase_pocket_text`
  - `ASD_pockets_binary_text`
  - `ASD_pockets_sequence_binary_text`
  - `merged_pocket_binary_text`
  - `merged_pocket_sequence_binary_text`
  - `ASD_merged_pocket_binary_text`
  - `ASD_merged_pocket_sequence_binary_text`

The `src/saprot_fit_mlp_balanced.py` script covers the same broad MLP evaluation workflow, but its balancing logic is specifically applied to selected merged binary allostery tasks, especially:

- `ASD_merged_pocket_binary`
- `ASD_merged_pocket_sequence_binary`
- `ASD_merged_pocket_binary_text`
- `ASD_merged_pocket_sequence_binary_text`
- Corresponding `_comp` variants

These balanced tasks are saved with a `_balanced` suffix so they can be distinguished from the standard unbalanced MLP runs.




