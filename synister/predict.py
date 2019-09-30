import os
from subprocess import check_call
from funlib.run import run, run_singularity
import logging
from synister.read_config import read_worker_config
import sys

worker_config = read_worker_config("worker_config.ini")

base_cmd = "python {}".format("predict_pipeline.py")
					  
if worker_config["singularity_container"] != "None" and worker_config["queue"] == "None":
    run_singularity(base_cmd,
                    singularity_image=worker_config["singularity_container"],
                    mount_dirs=worker_config["mount_dirs"],
                    execute=True)

elif worker_config["queue"] != "None":
    run(base_cmd,
        singularity_image=worker_config["singularity_container"],
        mount_dirs=worker_config["mount_dirs"],
        queue=worker_config["queue"],
        num_cpus=worker_config["num_cpus"],
        num_gpus=worker_config["num_block_workers"],
        batch=False,
        execute=True)

else:
    assert(worker_config["singularity_container"] == "None")
    check_call(base_cmd, shell=True)
