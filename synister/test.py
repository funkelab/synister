from gunpowder import *
from gunpowder.ext import torch
from gunpowder.torch import *
import torch.nn.functional as F
import daisy
import torch
import Vgg3D        # import Vgg3D from a seperate file
import sys
import os
import json
import numpy as np

def predict_all(checkpoint_file,
                data_dir,
                output_file,
                synapse_type=None,
                synapse_number=None):

    synapse_types = [
        'gaba',
        'acetylcholine',
        'glutamate',
        'serotonin',
        'octopamine',
        'dopamine'
    ]

    input_shape = Coordinate((32, 128, 128))
    voxel_size = Coordinate((40, 4, 4))
    fmaps = 32

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Vgg3D.Vgg3D(input_size=input_shape, fmaps=fmaps)
    model.to(device)

    print("Load checkpoint: ", checkpoint_file)
    checkpoint = torch.load(checkpoint_file)

    model.load_state_dict(checkpoint['model_state_dict'])

    complete_brain = daisy.open_ds(
        '/nrs/saalfeld/FAFB00/v14_align_tps_20170818_dmg.n5',
        'volumes/raw/s0')

    # load all test sample locations
    synapse_location = []
    for synapse in synapse_types:
        filename = os.path.join(data_dir, synapse + '_test.json')
        locations = np.array(json.load(open(filename, 'r')))
        synapse_location.append((synapse, locations))
    # loop through all classes (gaba,...)

    prediction = []
    if synapse_type is None:
        for synapse, locations in synapse_location:
            # loop through all testing sample locations of this class
            for loc in locations:
                raw = get_raw(input_shape,
                              voxel_size,
                              synapse,
                              loc,
                              complete_brain)

                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                output = predict(raw, model)
                data_synapse = {"prediction": output[0].tolist(),
                                "loc": loc,
                                "type": synapse}

                prediction.append(data_synapse)

        
    else:
        for synapse, locations in synapse_location:
            if synapse == synapse_type:
                if synapse_number is None:
                    for loc in locations:
                        raw = get_raw(input_shape,
                                      voxel_size,
                                      synapse,
                                      loc,
                                      complete_brain)
                        output = predict(raw, model)
                        data_synapse = {"prediction": output[0].tolist(),
                                        "loc": loc.tolist(),
                                        "type": synapse}

                        prediction.append(data_synapse)
                else:
                    loc = locations[synapse_number]
                    raw = get_raw(input_shape,
                                      voxel_size,
                                      synapse,
                                      loc,
                                      complete_brain)
                    output = predict(raw, model)
                    data_synapse = {"prediction": output[0].tolist(),
                                    "loc": loc.tolist(),
                                    "type": synapse}

                    prediction.append(data_synapse)

    with open(output_file, "w") as f:
        json.dump(prediction, f)

def predict(raw,
            model):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # [0.0, 255.0]
    raw = raw.astype(np.float32)
    # [0.0, 1.0]
    raw /= 255.0
    # [-1.0, 1.0]
    raw = raw*2.0 - 1.0

    raw_batched = raw.reshape((1,) + np.shape(raw))
    raw_batched_tensor = torch.tensor(raw_batched, device=device, requires_grad=True)
    output = model(raw=raw_batched_tensor)
    output = F.softmax(output, dim=1)

    return output, raw_batched_tensor


def get_raw(input_shape,
            voxel_size,
            synapse_type,
            loc,
            complete_brain):

    size = (input_shape*voxel_size)
    offset = loc - (input_shape/2*voxel_size)

    roi = daisy.Roi(offset, size).snap_to_grid(voxel_size, mode='closest')
    # if rounding changed the size, reset it here:
    if roi.get_shape()[0] != size[0]:
        roi.set_shape(size)

    if not complete_brain.roi.contains(roi):
        print("WARNING: synapse at %s is not contained in FAFB volume" % loc)

    # [0, 255]
    raw = complete_brain[roi].to_ndarray()

    return raw


if __name__ == "__main__":
    iteration = 99000
    run_num = 3
    checkpoint_file = "/groups/funke/home/dum/Projects/synister_exeriments/fafb/02_setups/setup01/run{}/model_checkpoint_{}".format(run_num, iteration)
    data_dir = "/groups/funke/home/dum/Projects/synister_exeriments/fafb/01_data" 
    output_file = "./test_pred_all.json"

    predict_all(checkpoint_file,
                data_dir,
                output_file,
                synapse_type='gaba',
                synapse_number=0)
