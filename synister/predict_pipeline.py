import os
import json
import numpy as np

from synister.utils import init_vgg, predict, get_raw
from synister.synister_db import SynisterDb
from synister.read_config import read_predict_config, read_worker_config

import logging
import multiprocessing
import sys

logger = logging.getLogger(__name__)


def test(worker_id,
         train_checkpoint,
         db_credentials,
         db_name_data,
         split_name,
         batch_size,
         input_shape,
         fmaps,
         downsample_factors,
         voxel_size,
         synapse_types,
         raw_container,
         raw_dataset,
         experiment,
         train_number,
         predict_number,
         num_cache_workers,
         num_block_workers,
         split_part="test",
         **kwargs):

    if not split_part in ["validation", "test"]:
        raise ValueError("'split_part' must be either 'test' or 'validation'")


    model = init_vgg(train_checkpoint,
                     input_shape,
                     fmaps,
                     downsample_factors)

    model.eval()

    logger.info('Load test sample locations from db {} and split {}...'.format(db_name_data, split_name))
    db = SynisterDb(db_credentials, db_name_data)

    logger.info('Initialize prediction writers...')
    prediction_queue = multiprocessing.JoinableQueue()

    for i in range(num_cache_workers):
        worker = multiprocessing.Process(target=prediction_writer,
                                         args=(prediction_queue,
                                               db_credentials,
                                               db_name_data,
                                               split_name,
                                               experiment,
                                               train_number,
                                               predict_number))
        #worker.daemon = True
        worker.start()


    logger.info('Start prediction...')

    locations = []
    synapses = db.get_synapses(split_name=split_name)
    predict_synapses = db.get_predictions(split_name,
                                          experiment,
                                          train_number,
                                          predict_number)

    locations = [(int(synapse["z"]), 
                  int(synapse["y"]),
                  int(synapse["x"]))
                  for synapse_id, synapse in synapses.items()
                  if synapse["splits"][split_name]==split_part and
                  predict_synapses[synapse_id]["prediction"] == None]

    loc_start = int(float(worker_id)/num_block_workers * len(locations)) 
    loc_end = int(float(worker_id + 1)/num_block_workers * len(locations))
    my_locations = locations[loc_start:loc_end]
 
    for i in range(0, len(my_locations), batch_size):
        logger.info('Predict location {}/{}'.format(i, len(my_locations)))
        locs = my_locations[i:i+batch_size]
        raw, raw_normalized = get_raw(locs,
                                      input_shape,
                                      voxel_size,
                                      raw_container,
                                      raw_dataset)
        output = predict(raw_normalized, model)

        for k in range(np.shape(output)[0]):
            loc_k = locs[k]
            out_k = output[k,:]
            loc_k_list = loc_k

            data_synapse = {"prediction": out_k.tolist(),
                            "z": loc_k_list[0],
                            "y": loc_k_list[1],
                            "x": loc_k_list[2]}

            prediction_queue.put(data_synapse)

    logger.info("Wait for write...")
    prediction_queue.join()


def prediction_writer(prediction_queue,
                      db_credentials,
                      db_name_data,
                      split_name,
                      experiment,
                      train_number,
                      predict_number):


    db = SynisterDb(db_credentials, db_name_data)
    
    while True:
        data_synapse = prediction_queue.get()

        db.write_prediction(split_name,
                            data_synapse["prediction"],
                            experiment,
                            train_number,
                            predict_number,
                            data_synapse["x"],
                            data_synapse["y"],
                            data_synapse["z"])

        prediction_queue.task_done()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    worker_id = int(sys.argv[1])
    num_block_workers = int(sys.argv[2])

    predict_config = read_predict_config("./predict_config.ini")
    worker_config = read_worker_config("./worker_config.ini")
    worker_config["worker_id"] = worker_id
    worker_config["num_block_workers"] = num_block_workers
    test(**{**predict_config, **worker_config})
