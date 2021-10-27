from funlib.run import run, run_singularity
from subprocess import check_call
import os
from synister.read_config import \
    read_worker_config, \
    read_predict_config, \
    read_train_config
from synister.synister_db import SynisterDb
import threading
import time


def monitor_prediction(
        predict_config,
        interval=60):

    db = SynisterDb(
        predict_config["db_credentials"],
        predict_config["db_name_data"])

    done_0, _ = db.count_predictions(
        predict_config["split_name"],
        predict_config["experiment"],
        predict_config["train_number"],
        predict_config["predict_number"])

    start = time.time()

    while True:

        done, total = db.count_predictions(
            predict_config["split_name"],
            predict_config["experiment"],
            predict_config["train_number"],
            predict_config["predict_number"])

        if done == total:
            print("All predictions done.")
            return

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


def predict(predict_config, worker_config):

    db = SynisterDb(
        predict_config["db_credentials"],
        predict_config["db_name_data"])

    db.initialize_prediction(
        predict_config["split_name"],
        predict_config["experiment"],
        predict_config["train_number"],
        predict_config["predict_number"],
        overwrite=predict_config["overwrite"],
        validation=predict_config["split_part"] == "validation")

    num_block_workers = worker_config["num_block_workers"]
    singularity = worker_config["singularity_container"]
    queue = worker_config["queue"]
    singularity = singularity if singularity != "None" else None
    queue = queue if queue != "None" else None

    if singularity is not None and queue is None:

        # run locally in singularity container
        for worker_id in range(num_block_workers):

            thread = threading.Thread(
                target=run_singularity,
                args=(
                    base_cmd + " {} {}".format(worker_id, num_block_workers),
                    singularity,
                    ".",
                    worker_config["mount_dirs"],
                    True,
                    True
                )
            )
            thread.start()

    elif queue is not None:

        # run on cluster, with or without singularity
        for worker_id in range(num_block_workers):
            thread = threading.Thread(
                target=run,
                args=(
                    base_cmd + " {} {}".format(worker_id, num_block_workers),
                    worker_config["num_cpus"],
                    1,
                    25600,
                    ".",
                    singularity,
                    "",
                    queue,
                    "",
                    False,
                    worker_config["mount_dirs"],
                    True,
                    True
                )
            )
            thread.start()

    else:

        # run locally without singularity
        assert(singularity is None)
        for worker_id in range(num_block_workers):
            check_call(
                base_cmd + " {} {}".format(worker_id, num_block_workers),
                shell=True)

    thread = threading.Thread(
        target=monitor_prediction,
        args=(predict_config, 60))
    thread.daemon = True
    thread.start()


if __name__ == '__main__':

    self_path = os.path.realpath(os.path.dirname(__file__))

    worker_config = read_worker_config(
        os.path.join(self_path, "worker_config.ini"))
    predict_config_template = read_predict_config(
        os.path.join(self_path, "predict_config.ini"))

    train_number = predict_config_template['train_number']
    setup_dir = os.path.join(
        predict_config_template['train_dir'],
        f'setup_t{train_number}')
    train_config = read_train_config(
        os.path.join(setup_dir, 'train_config.ini'))

    predict_config_template.update(train_config)

    for iteration in predict_config_template['iterations']:

        base_cmd = "python {} {}".format(
            os.path.join(self_path, "predict_pipeline.py"),
            iteration)

        predict_config = dict(predict_config_template)
        predict_config['train_checkpoint'] = os.path.join(
            setup_dir,
            f'model_checkpoint_{iteration}')
        predict_config['predict_number'] = iteration

        predict(predict_config, worker_config)
