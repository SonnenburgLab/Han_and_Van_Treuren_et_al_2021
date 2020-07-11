# Metabolomics Paper

This repository contains all the Jupyter notebooks used to generate the paper figures.

## Installation

To install all the required dependencies, you need to have [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html) installed.

Create and activate the `metabolomics-paper` environment from the `environment.yml` file:

```sh
conda env create -f environment.yml

conda activate metabolomics-paper
```

### Usage

Start the local Jupyter notebook server by running:

```sh
jupyter notebook
```

Each directory in this repository contains its own Jupyter notebook along with the required input files.

### Testing

To execute the unit tests, run:

```sh
pytest
```