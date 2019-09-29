import os
import json
import numpy as np

from synister.predict import init_vgg, predict
from synister.utils import get_raw

import logging

logger = logging.getLogger(__name__)


def test(checkpoint_file,
         data_dir,
         output_file,
         batch_size,
         input_shape,
         fmaps,
         voxel_size=(40,4,4),
         synapse_types=None,
         synapse_number=None):

    synapse_types_all = [
            'gaba',
            'acetylcholine',
            'glutamate',
            'serotonin',
            'octopamine',
            'dopamine'
        ]

    if synapse_types is None:
        synapse_types = synapse_types_all


    model = init_vgg(checkpoint_file,
                     input_shape,
                     fmaps)

    model.eval()

    data_container = '/nrs/saalfeld/FAFB00/v14_align_tps_20170818_dmg.n5' 
    data_set = 'volumes/raw/s0' 

    logger.info('Load test sample locations...')
    synapse_location = []
    for synapse in synapse_types:
        filename = os.path.join(data_dir, synapse + '_test.json')
        locations = np.array(json.load(open(filename, 'r')))
        synapse_location.append((synapse, locations))

    
    logger.info('Start prediction...')
    prediction = []
    for synapse, locations in synapse_location:
        # loop through all testing sample locations of this class
        n = 0
        logger.info('Predict synapse {}'.format(synapse))
        if synapse_number is not None:
            locations = [locations[synapse_number]]
        for i in range(0, len(locations), batch_size):
            logger.info('Predict location {}/{}'.format(i, len(locations)))
            locs = locations[i:i+batch_size]
            raw, raw_normalized = get_raw(locs,
                                          input_shape,
                                          voxel_size,
                                          data_container,
                                          data_set)

            output = predict(raw_normalized, model)
            for k in range(np.shape(output)[0]):
                loc_k = locs[k]
                out_k = output[k,:]

                data_synapse = {"prediction": out_k.tolist(),
                                "loc": loc_k.tolist(),
                                "type": synapse,
                                "id": n,
                                "gt_class": synapse_types_all.index(synapse)}

                prediction.append(data_synapse)
                n += 1

    with open(output_file, "w") as f:
        json.dump(prediction, f)
