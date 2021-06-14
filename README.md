# Scripts for training and evaluating synister networks.

## Installation

1. Singularity
```console
cd singularity
make
```
2. Conda
```
conda create -n synister python=3.6 numpy scipy cython
conda activate synister
pip install -r requirements.txt
```

## Usage

### 1. Training a new network from scratch.
#### Prepare training
```console
python prepare_training.py -d <path_to_train_dir> -e <name for the experiment/run> -t <an integer id for the training run>
```

This creates a new directory at the specified path and initialises default config files for the run.

#### Run Training
```console
cd <path_to_train_dir>/<experiment_name>/02_train/setup_t<train_id>
```
Edit config files to match architecture, database and compute resources to train with.
```console
python train.py
```
