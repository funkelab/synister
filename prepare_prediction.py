import os
import sys
from shutil import copyfile, rmtree
import json
import configargparse
import configparser
from os.path import expanduser
import click
from synister.read_config import read_train_config


p = configargparse.ArgParser()
p.add('-d', '--base_dir', required=True, help='base directory for storing synister experiments')
p.add('-e', required=True, help='name of the experiment, e.g. fafb, defaults to ``base``')
p.add('-t', required=True, help='train setup number to use for this prediction')
p.add('-i', required=True, help='iteration checkpoint number to use for this prediction')
p.add('-p', required=True, help='predict number/id to use for this prediction')
p.add('-v', required=False, action='store_true', help='use validation split part')
p.add('-c', required=False, action='store_true', help='clean up - remove specified predict setup')


def set_up_environment(base_dir,
                       experiment,
                       train_number,
                       iteration,
                       predict_number,
                       clean_up,
                       validation):

    predict_setup_dir = os.path.join(os.path.join(base_dir, experiment), "03_predict/setup_t{}_p{}".format(train_number, predict_number))
    train_setup_dir = os.path.join(os.path.join(base_dir, experiment), "02_train/setup_t{}".format(train_number))
    train_checkpoint = os.path.join(train_setup_dir, "model_checkpoint_{}".format(iteration))
    train_config = os.path.join(train_setup_dir, "train_config.ini")
    train_worker_config = os.path.join(train_setup_dir, "worker_config.ini")

    if clean_up:
        if __name__ == "__main__":
            if click.confirm('Are you sure you want to remove {} and all its contents?'.format(predict_setup_dir), default=False):
                rmtree(predict_setup_dir)
            else:
                print("Abort clean up")

    if not os.path.exists(train_checkpoint):
        raise ValueError("No checkpoint at {}".format(train_checkpoint))

    if not os.path.exists(predict_setup_dir):
        os.makedirs(predict_setup_dir)
    else:
        if __name__ == "__main__":
            if click.confirm('Predict setup {} exists already, overwrite?'.format(predict_setup_dir), default=False):
                rmtree(predict_setup_dir)
                os.makedirs(predict_setup_dir)
            else:
                print("Abort.")
                return
        else:
            raise ValueError("Predict setup exists already, choose different predict number or clean up.")

    copyfile("synister/predict_pipeline.py", os.path.join(predict_setup_dir, "predict_pipeline.py"))
    copyfile("synister/predict.py", os.path.join(predict_setup_dir, "predict.py"))
    copyfile(train_worker_config, os.path.join(predict_setup_dir, "worker_config.ini"))
 
    train_config_dict = read_train_config(train_config)

    predict_config = create_predict_config(base_dir,
                                           experiment,
                                           train_number,
                                           predict_number,
                                           train_checkpoint,
                                           train_config_dict,
                                           validation)

    with open(os.path.join(predict_setup_dir, "predict_config.ini"), "w+") as f:
        predict_config.write(f)

def create_predict_config(base_dir,
                          experiment,
                          train_number,
                          predict_number,
                          train_checkpoint,
                          train_config_dict,
                          validation):

    config = configparser.ConfigParser()

    config.add_section('Predict')
    config.set('Predict', 'train_checkpoint', train_checkpoint)
    config.set('Predict', 'experiment', str(experiment))
    config.set('Predict', 'train_number', str(train_number))
    config.set('Predict', 'predict_number', str(predict_number))

    synapse_types_string = ""
    for s in train_config_dict["synapse_types"]:
        synapse_types_string += s + ", "
    synapse_types_string = synapse_types_string[:-2]

    config.set('Predict', 'synapse_types', synapse_types_string)
    config.set('Predict', 'input_shape', '{}, {}, {}'.format(train_config_dict["input_shape"][0],
                                                             train_config_dict["input_shape"][1],
                                                             train_config_dict["input_shape"][2]))
    config.set('Predict', 'fmaps', str(train_config_dict["fmaps"]))
    config.set('Predict', 'batch_size', str(train_config_dict["batch_size"]))
    config.set('Predict', 'db_credentials', str(train_config_dict["db_credentials"]))
    config.set('Predict', 'db_name_data', str(train_config_dict["db_name_data"]))
    config.set('Predict', 'split_name', str(train_config_dict["split_name"]))
    config.set('Predict', 'voxel_size', "{}, {}, {}".format(train_config_dict["voxel_size"][0],
                                                            train_config_dict["voxel_size"][1],
                                                            train_config_dict["voxel_size"][2]))
    config.set('Predict', 'raw_container', str(train_config_dict["raw_container"]))
    config.set('Predict', 'raw_dataset', str(train_config_dict["raw_dataset"]))
    config.set('Predict', 'downsample_factors', str(train_config_dict["downsample_factors"])[1:-1])
    if validation:
        config.set('Predict', 'split_part', "validation")
    else:
        config.set('Predict', 'split_part', "test")
    config.set('Predict', 'overwrite', str(False))

    return config
 

if __name__ == "__main__":
    options = p.parse_args()

    base_dir = options.base_dir
    experiment = options.e
    train_number = int(options.t)
    train_iteration = int(options.i)
    predict_number = int(options.p)
    clean_up = bool(options.c)
    validation = bool(options.v)

    set_up_environment(base_dir,
                       experiment,
                       train_number,
                       train_iteration,
                       predict_number,
                       clean_up,
                       validation)
