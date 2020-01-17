import os
from subprocess import check_call
from funlib.run import run, run_singularity
import logging
from synister.read_config import read_worker_config, read_predict_config
import sys
import threading
from synister.synister_db import SynisterDb
import time

worker_config = read_worker_config("worker_config.ini")
predict_config = read_predict_config("predict_config.ini")

base_cmd = "python {}".format("predict_pipeline.py")

num_block_workers = worker_config["num_block_workers"]

db = SynisterDb(predict_config["db_credentials"], 
                predict_config["db_name_data"])

db.initialize_prediction(predict_config["split_name"],
                         predict_config["experiment"],
                         predict_config["train_number"],
                         predict_config["predict_number"],
                         overwrite=predict_config["overwrite"],
                         validation=predict_config["split_part"] == "validation")

def monitor_prediction(predict_config,
                       interval=60):

    db = SynisterDb(predict_config["db_credentials"], predict_config["db_name_data"])
    done_0, _ = db.count_predictions(predict_config["split_name"],
                                     predict_config["experiment"],
                                     predict_config["train_number"],
                                     predict_config["predict_number"])
    start = time.time()
    while True:
        done, total = db.count_predictions(predict_config["split_name"],
                                           predict_config["experiment"],
                                           predict_config["train_number"],
                                           predict_config["predict_number"])

        time_elapsed = time.time() - start
        if done - done_0 > 0:
            eta = time_elapsed/(done - done_0) * (total - done)
            sps = (done - done_0)/time_elapsed
        else:
            eta = "NA"
            sps = "NA"
        print("{} from {} predictions done".format(done, total))
        print("Time elapsed {}".format(time_elapsed))
        print("{} samples/second".format(sps))
        print("ETA: {}".format(eta))
        time.sleep(interval)

if worker_config["singularity_container"] != "None" and worker_config["queue"] == "None":
    for worker_id in range(num_block_workers):
        thread = threading.Thread(target=run_singularity,
                                  args=(base_cmd + " {} {}".format(worker_id, num_block_workers),
                                  worker_config["singularity_container"],
                                  ".",
                                  worker_config["mount_dirs"],
                                  True,
                                  True))
        thread.start()

elif worker_config["queue"] != "None":
    for worker_id in range(num_block_workers):
        thread = threading.Thread(target=run, args=(base_cmd + " {} {}".format(worker_id, num_block_workers),
                                                    worker_config["num_cpus"],
                                                    1,
                                                    25600,
                                                    ".",
                                                    worker_config["singularity_container"],
                                                    "",
                                                    worker_config["queue"],
                                                    "",
                                                    False,
                                                    worker_config["mount_dirs"],
                                                    True,
                                                    True))
        thread.start()


else:
    assert(worker_config["singularity_container"] == "None")
    for worker_id in range(num_block_workers):
        check_call(base_cmd + " {} {}".format(worker_id, num_block_workers), shell=True)

thread = threading.Thread(target=monitor_prediction, args=(predict_config, 60))
thread.daemon = True
thread.start()
