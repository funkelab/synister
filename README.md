# Neurotransmitter Prediction from Electron Microscopy Images in Drosophila Melanogaster

This repository accompanies the preprint [Neurotransmitter Classification from
Electron Microscopy Images at Synaptic Sites in
Drosophila.](https://www.biorxiv.org/content/10.1101/2020.06.12.148775v2)

## Access to Neurotransmitter Predictions

Predictions for the most common neurotransmitters (GABA, acetylcholine,
glutamate, serotonin, octopamine, and dopamine) are publicly available for the
following two datasets:

### FAFB (Full Adult Fly Brain)

We used our method to predict the neurotransmitter identity of all annotated
synapses in the [FAFB](https://temca2data.org/) volume. Our predictions are
available through [FlyWire](https://flywire.ai/).

### HemiBrain

Neurotransmitter predictions for all automatically located synapses in the
[HemiBrain](https://www.janelia.org/project-team/flyem/hemibrain) are available
[here](https://storage.googleapis.com/hemibrain/v1.2/hemibrain-v1.2-tbar-neurotransmitters.feather.bz2)
in Apache Feather format.

## Training and Prediction

The following shows how to recreate our results on the FAFB volume and might be
useful to train and predict in other datasets.

### Installation

1. Singularity
```console
cd singularity
make
```

2. Conda
```
conda create -n synister -c conda-forge -c funkey python numpy scipy cython pylp pytorch-gpu
conda activate synister
pip install -r requirements.txt
pip install .
```

### Usage

#### 0. Creating a MongoDB database with the provided data.

An export of the three collections constituting the synister FAFB database used
for all described experiments can be found at ```data/fafb_v3```. The three
files contain:

1. Location, id, skid, brain region and split for each synapse
   (synapses(_v3).json).
2. Skid, neurotransmitter, hemilineage id for each skeleton
   (skeletons(_v3).json).
3. Hemilineage name, hemilineage id for each hemilineage
   (hemilineages(_v3).json).

To reproduce the experiment each json file should be imported as a collection
with names "synapses", "skeletons", "hemilineages" in one mongo database (for
additional instructions on how to import json files in a mongo db click
[here](https://docs.mongodb.com/database-tools/mongoimport/)). Dictionary keys
are field names. Provided splits can be reproduced using
```synister/split.py```, which searches for the optimally balanced split in
terms of neurotransmitter distribution for any given superset, such as
hemilineage id, skeleton id or brain region.

For training on other data, recreate the database scheme shown here (required
are a "synapses" and a "skeletons" collection) and adapt config files to match
the new database name.

#### 1. Training a network.

##### Prepare training

```console
python prepare_training.py -d <base_dir> -e <experiment_name> -t <train_id>
```

This creates a new directory at the specified path and initialises default
config files for the run.

##### Run Training

```console
cd <base_dir>/<experiment_name>/02_train/setup_t<train_id>
```

Edit config files to match architecture, database and compute resources to
train with. 

For example configs, training a VGG on the skeleton split, inside a singularity
container, on a gpu queue see:

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

#### 2. Validating a trained network.

##### Prepare validation runs

```console
python prepare_prediction.py -v -d <base_dir> -e <experiment_name> -t <train_id> -i <iter_0> <iter_1> <iter_2> ... <iter_N> 
```

This will create N prediction directories with appropriately initialized config
files, one for each given train iteration <iter_k>. The -v flag sets the split
part of the chosen split type to validation, only pulling those synapses from
the DB that are tagged as validation synapses.

```
example_configs/worker_config.ini

[Predict]
train_checkpoint = <experiment_name>/02_train/setup_t<train_id>/model_checkpoint_<iter_k>
experiment = <experiment_name>
train_number = 0
predict_number = 0
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
downsample_factors = (1, 2, 2), (1, 2, 2), (1, 2, 2), (2, 2, 2)
split_part = validation
overwrite = False
network = VGG
fmap_inc = 2, 2, 2, 2
n_convolutions = 2, 2, 2, 2
network_appendix = None
```

For most use cases the automatically initialized predict config does not
require any edits. If run as is, predictions will be written into the database
under:

```
<db_name>_predictions.<split_name>_<experiment_name>_t<train_id>_p<predict_id>
```

To start the prediction, run:
```console
cd <base_dir>/<experiment_name>/03_predict/setup_t<train_id>_p<predict_id>
python predict.py
```

If the collection already exists the script will abort. A collection can be
overwritten by setting overwrite=True in the predict config. Parallel
prediction with multiple GPUs can be done by setting num_block_workers=num_gpus
in the worker_config file. Prediction speed and expected time to finish will be
shown in the console.

For submitting multiple predictions to the cluster at once run the provided
convenience script:

```console
python start_predictions -d <base_dir> -e <experiment_name> -t <train_id> -p <predict_id_0> <predict_id_1> ... <predict_id_N>
```

#### 3. Testing a trained network.

##### Prepare test runs

```console
python prepare_prediction.py -d <base_dir> -e <experiment_name> -t <train_id> -i <iter_0> <iter_1> <iter_2> ... <iter_N>
```

Similar to validation this prepares the relevant dictionaries and config files
but sets the split part to "test".

Starting the prediction follow the same pattern as before:

```console
cd <base_dir>/<experiment_name>/03_predict/setup_t<train_id>_p<predict_id>
python predict.py
```

# Related Repositories

1. [Package for on demand predictions with a production network.](https://github.com/funkelab/synistereq)
2. [Package for high performance full brain predictions.](https://github.com/funkelab/synisterbrain)
3. [Webservice.](https://github.com/nilsec/synisterest)
