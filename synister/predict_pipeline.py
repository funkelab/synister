from funlib.learn.torch.models import Vgg3D
from synister.read_config import \
    read_worker_config, \
    read_predict_config, \
    read_train_config
from synister.synister_db import SynisterDb
from synister.utils import init_vgg, predict, get_raw
import json
import logging
import multiprocessing
import numpy as np
import os
import sys
import torch


logger = logging.getLogger(__name__)
self_path = os.path.realpath(os.path.dirname(__file__))

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

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
         neither_class, 
         split_part="test",
         output_classes=None,
         network="VGG",
         fmap_inc=(2,2,2,2),
         n_convolutions=(2,2,2,2),
         network_appendix=None,
         **kwargs):

    if not split_part in ["validation", "test"]:
        raise ValueError("'split_part' must be either 'test' or 'validation'")

    print("Network: ", network)
    if network == "VGG":
        output_classes = len(synapse_types)
        if neither_class:
            output_classes +=1
        model = Vgg3D(input_size=input_shape,
                      fmaps=fmaps,
                      downsample_factors=downsample_factors,
                      fmap_inc=fmap_inc,
                      n_convolutions=n_convolutions,
                      output_classes=output_classes)
    else:
        raise NotImplementedError("Only VGG network accesible.")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    checkpoint = torch.load(train_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

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

    logger.info(f"Predicting {len(my_locations)} locations...")
    for i in range(0, len(my_locations), batch_size):
        logger.info('Predict location {}/{}'.format(i, len(my_locations)))
        locs = my_locations[i:i+batch_size]
        raw, raw_normalized = get_raw(locs,
                                      input_shape,
                                      voxel_size,
                                      raw_container,
                                      raw_dataset)
        
        shape = tuple(raw_normalized.shape)
        raw_normalized = raw_normalized.reshape([len(locs), 1, shape[1], shape[2], shape[3]]).astype(np.float32)
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

    # signal end of data
    logger.info("Signalling end of data to prediction writers")
    for _ in range(num_cache_workers):
        prediction_queue.put(None)

    logger.info("Wait for write...")
    prediction_queue.join()
    logger.info("Done.")


def prediction_writer(prediction_queue,
                      db_credentials,
                      db_name_data,
                      split_name,
                      experiment,
                      train_number,
                      predict_number):

    logger.info("Starting prediction writer thread")

    db = SynisterDb(db_credentials, db_name_data)
    
    while True:
        data_synapse = prediction_queue.get()

        if data_synapse is None:
            logger.info("No more locations to predict, stopping prediction writer")
            prediction_queue.task_done()
            break

        db.write_prediction(split_name,
                            data_synapse["prediction"],
                            experiment,
                            train_number,
                            predict_number,
                            data_synapse["x"],
                            data_synapse["y"],
                            data_synapse["z"])

        prediction_queue.task_done()

    logger.info("Prediction writer thread stopped")

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    iteration = int(sys.argv[1])
    worker_id = int(sys.argv[2])
    num_block_workers = int(sys.argv[3])

    predict_config = read_predict_config(
        os.path.join(self_path, "predict_config.ini"))
    train_number = predict_config['train_number']
    setup_dir = os.path.join(
        predict_config['train_dir'],
        f'setup_t{train_number}')
    train_config = read_train_config(
        os.path.join(setup_dir, 'train_config.ini'))
    del train_config['batch_size']  # don't overwrite prediction batch size
    predict_config.update(train_config)
    predict_config['train_checkpoint'] = os.path.join(
        setup_dir,
        f'model_checkpoint_{iteration}')
    predict_config['predict_number'] = iteration

    worker_config = read_worker_config(os.path.join(self_path, "worker_config.ini"))
    worker_config["worker_id"] = worker_id
    worker_config["num_block_workers"] = num_block_workers
    test(**{**predict_config, **worker_config})
