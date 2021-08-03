import configparser
import os
import numpy as np
import json


def read_predict_config(predict_config):
    config = configparser.ConfigParser()
    config.read(predict_config)

    cfg_dict = {}
    tmp = [s for s in config.get("Predict", "synapse_types").split(", ")]
    synapse_types = []
    for s in tmp:
        if s == "None":
            s = None
        synapse_types.append(s)
    cfg_dict["synapse_types"] = synapse_types

    cfg_dict["input_shape"] = tuple([int(c) for c in config.get("Predict", "input_shape").split(", ")])
    cfg_dict["fmaps"] = config.getint("Predict", "fmaps")
    cfg_dict["batch_size"] = config.getint("Predict", "batch_size")
    cfg_dict["db_credentials"] = config.get("Predict", "db_credentials")
    cfg_dict["db_name_data"] = config.get("Predict", "db_name_data")
    cfg_dict["split_name"] = config.get("Predict", "split_name")
    cfg_dict["voxel_size"] = tuple([int(v) for v in config.get("Predict", "voxel_size").split(", ")])
    cfg_dict["raw_container"] = config.get("Predict", "raw_container")
    cfg_dict["raw_dataset"] = config.get("Predict", "raw_dataset")
    cfg_dict["train_checkpoint"] = config.get("Predict", "train_checkpoint")
    cfg_dict["experiment"] = config.get("Predict", "experiment")
    cfg_dict["train_number"] = config.getint("Predict", "train_number")
    cfg_dict["predict_number"] = config.getint("Predict", "predict_number")

    if config.get("Predict", "neither_class") == "True":
        cfg_dict["neither_class"] = True
    else:
        cfg_dict["neither_class"] = False


    try:
        cfg_dict["split_part"] = config.get("Predict", "split_part")
    except:
        cfg_dict["split_part"] = None
    downsample_factors = config.get("Predict", "downsample_factors")
    downsample_factors = [s.strip("(").strip(")").split(",") for s in downsample_factors.split("), ")]
    cfg_dict["downsample_factors"] = []
    for factor in downsample_factors:
        f = tuple([int(k) for k in factor])
        cfg_dict["downsample_factors"].append(f)
    cfg_dict["overwrite"] = config.get("Predict", "overwrite") == "True"

    try:
        cfg_dict["network"] = config.get("Predict", "network")
    except:
        pass

    try:
        cfg_dict["fmap_inc"] = tuple([int(v) for v in config.get("Predict", "fmap_inc").split(", ")])
    except:
        pass

    try:
        cfg_dict["n_convolutions"] = tuple([int(v) for v in config.get("Predict", "n_convolutions").split(", ")])
    except:
        pass

    try:
        cfg_dict["network_appendix"] = config.get("Predict", "network_appendix")
    except:
        pass

    return cfg_dict


def read_train_config(train_config):
    config = configparser.ConfigParser()
    config.read(train_config)

    cfg_dict = {}
    cfg_dict["synapse_types"] = [s for s in config.get("Training", "synapse_types").split(", ")]
    cfg_dict["input_shape"] = tuple([int(c) for c in config.get("Training", "input_shape").split(", ")])
    cfg_dict["fmaps"] = config.getint("Training", "fmaps")
    cfg_dict["batch_size"] = config.getint("Training", "batch_size")
    cfg_dict["db_credentials"] = config.get("Training", "db_credentials")
    cfg_dict["db_name_data"] = config.get("Training", "db_name_data")
    cfg_dict["split_name"] = config.get("Training", "split_name")
    cfg_dict["voxel_size"] = tuple([int(v) for v in config.get("Training", "voxel_size").split(", ")])
    cfg_dict["raw_container"] = config.get("Training", "raw_container")
    cfg_dict["raw_dataset"] = config.get("Training", "raw_dataset")

    if config.get("Training", "neither_class") == "True":
        cfg_dict["neither_class"] = True
    else:
        cfg_dict["neither_class"] = False

    downsample_factors = config.get("Training", "downsample_factors")
    downsample_factors = [s.strip("(").strip(")").split(",") for s in downsample_factors.split("), ")]
    cfg_dict["downsample_factors"] = []
    for factor in downsample_factors:
        f = tuple([int(k) for k in factor])
        cfg_dict["downsample_factors"].append(f)

    try:
        cfg_dict["network"] = config.get("Training", "network")
    except:
        pass

    try:
        cfg_dict["fmap_inc"] = tuple([int(v) for v in config.get("Training", "fmap_inc").split(", ")])
    except:
        pass

    try:
        cfg_dict["n_convolutions"] = tuple([int(v) for v in config.get("Training", "n_convolutions").split(", ")])
    except:
        pass
    try:
        cfg_dict["network_appendix"] = config.get("Training", "network_appendix")
    except:
        pass

    return cfg_dict

def read_worker_config(worker_config):
    config = configparser.ConfigParser()
    config.read(worker_config)

    cfg_dict = {}

    # Worker
    cfg_dict["singularity_container"] = config.get("Worker", "singularity_container")
    cfg_dict["num_cpus"] = int(config.getint("Worker", "num_cpus"))
    cfg_dict["num_block_workers"] = int(config.getint("Worker", "num_block_workers"))
    cfg_dict["num_cache_workers"] = int(config.getint("Worker", "num_cache_workers"))
    cfg_dict["queue"] = config.get("Worker", "queue")
    cfg_dict["mount_dirs"] = tuple([v for v in config.get("Worker", "mount_dirs").split(", ")])

    return cfg_dict
