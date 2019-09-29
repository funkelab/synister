import configparser
import os
import numpy as np
import json

def read_train_config(train_config):
    config = configparser.ConfigParser()
    config.read(train_config)

    cfg_dict = {}
    cfg_dict["synapse_types"] = [s for s in config.get("Training", "synapse_types").split(", ")]
    cfg_dict["input_shape"] = tuple([int(c) for c in config.get("Training", "input_shape").split(", ")])
    cfg_dict["num_levels"] = config.getint("Training", "num_levels")
    cfg_dict["fmaps"] = config.getint("Training", "fmaps")
    cfg_dict["batch_size"] = config.getint("Training", "batch_size")
    cfg_dict["db_credentials"] = config.get("Training", "db_credentials")
    cfg_dict["db_name_data"] = config.get("Training", "db_name_data")
    cfg_dict["split_name"] = config.get("Training", "split_name")

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
