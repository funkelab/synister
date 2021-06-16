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
pip install .
```

## Usage

### 1. Training a network.
#### Prepare training
```console
python prepare_training.py -d <base_dir> -e <experiment_name> -t <train_idn>
```

This creates a new directory at the specified path and initialises default config files for the run.

#### Run Training
```console
cd <base_dir>/<experiment_name>/02_train/setup_t<train_id>
```
Edit config files to match architecture, database and compute resources to train with. 

For example configs, training a VGG on the skeleton split, inside a singularity container, on a gpu queue see:
```
example_configs/train_config.ini

[Training]
synapse_types = gaba, acetylcholine, glutamate, serotonin, octopamine, dopamine
input_shape = 16, 160, 160
fmaps = 12
batch_size = 8
db_credentials = synister_data/credentials/db_credentials.ini
db_name_data = synister_v3
split_name = skeleton
voxel_size = 40, 4, 4
raw_container = /nrs/saalfeld/FAFB00/v14_align_tps_20170818_dmg.n5
raw_dataset = volumes/raw/s0
downsample_factors = (1,2,2), (1,2,2), (1,2,2), (2,2,2)
network = VGG
fmap_inc = 2, 2, 2, 2
n_convolutions = 2, 2, 2, 2
network_appendix = None
```

```
example_configs/worker_config.ini

[Worker]
singularity_container = synister/singularity/synister.img
num_cpus = 5
num_block_workers = 1
num_cache_workers = 5
queue = gpu_any
mount_dirs = /nrs, /scratch, /groups, /misc
```

Finally, to submit the train job with the desired number of iterations run:
```console
python train.py <num_iterations>
```
We recommend training for at least 500,000 iterations for FAVB_v3 splits.

For visualizing training progress run:
```console
tensorboard --logdir <base_dir>/<experiment_name>/02_train/setup_t<train_id>/log
```

Snapshots are written to:
```console
<base_dir>/<experiment_name>/02_train/setup_t<train_id>/snapshots
```

### 2. Validating a trained network.
#### Prepare validation runs
```console
python prepare_prediction.py -d <base_dir> -e <experiment_namen> -t <train_id> -i <iter_0> <iter_1> <iter_2> ... <iter_N> -v
```

This will create N prediction directories with appropriately initialized config files, one for each given train iteration <iter_k>. The -v flag sets the split part of the chosen split type to validation, only pulling those synapses from the DB that are tagged as validation synapses.



