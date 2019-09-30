import os
import json
import numpy as np

from synister.utils import init_vgg, predict, get_raw
from synister.synister_db import SynisterDB
from synister.read_config import read_predict_config

import logging

logger = logging.getLogger(__name__)


def test(train_checkpoint,
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
         predict_number):


    model = init_vgg(train_checkpoint,
                     input_shape,
                     fmaps,
                     downsample_factors)

    model.eval()

    logger.info('Load test sample locations from db {} and split {}...'.format(db_name_data, split_name))
    db = SynisterDB(db_credentials)

    logger.info('Start prediction...')
    predictions = []
    for synapse_type in synapse_types:
        logger.info('Predict synapse type {}...'.format(synapse_type))
        locations = db.get_synapse_locations(db_name_data,
                                             split_name,
                                             "test",
                                             tuple([synapse_type]))
 

        for i in range(0, len(locations), batch_size):
            logger.info('Predict location {}/{} ({})'.format(i, len(locations), synapse_type))
            locs = locations[i:i+batch_size]
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

                predictions.append(data_synapse)


    logger.info('Write predictions to database...')
    for data_synapse in predictions:
        db.write_prediction(db_name_data,
                            split_name,
                            data_synapse["prediction"],
                            experiment,
                            train_number,
                            predict_number,
                            data_synapse["x"],
                            data_synapse["y"],
                            data_synapse["z"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    predict_config = read_predict_config("./predict_config.ini")
    test(**predict_config)
