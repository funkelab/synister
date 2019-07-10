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
import h5py
from integrated_gradients import get_integrated_gradients
from test import predict, get_raw

def get_attribution(checkpoint_file,
                    data_dir,
                    output_dir,
                    baseline=None,
                    integration_steps=50,
                    synapse_type=None,
                    synapse_number=None):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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


    if baseline is None:
        baseline = np.zeros(input_shape)


    prediction = []
    if synapse_type is None:
        target_id = 0
        for synapse, locations in synapse_location:
            # loop through all testing sample locations of this class
            i = 0
            for loc in locations:
                raw = get_raw(input_shape,
                              voxel_size,
                              synapse,
                              loc,
                              complete_brain)

                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                output, average_grads, integrated_grads, _ = get_integrated_gradients(model, 
                                                                                        raw, 
                                                                                        baseline, 
                                                                                        target_id, 
                                                                                        integration_steps)

                output_file = os.path.join(output_dir, "grads_{}{}_{}.h5".format(target_id, synapse, i))
                with h5py.File(output_file, "w") as f:
                    f.create_dataset("output", data=output.detach().cpu().numpy())
                    f.create_dataset("raw", data=raw)
                    f.create_dataset("average_grads", data=average_grads)
                    f.create_dataset("location", data=loc)

                i += 1
            target_id += 1

        
    else:
        target_id = 0
        for synapse, locations in synapse_location:
            if synapse == synapse_type:
                if synapse_number is None:
                    i = 0
                    for loc in locations:
                        raw = get_raw(input_shape,
                                      voxel_size,
                                      synapse,
                                      loc,
                                      complete_brain)

                        output, average_grads, integrated_grads, _ = get_integrated_gradients(model, 
                                                                                        raw, 
                                                                                        baseline, 
                                                                                        target_id, 
                                                                                        integration_steps)

                        output_file = os.path.join(output_dir, "grads_{}{}_{}.h5".format(target_id, synapse, i))
                        with h5py.File(output_file, "w") as f:
                            f.create_dataset("output", data=output.detach().cpu().numpy())
                            f.create_dataset("raw", data=raw)
                            f.create_dataset("average_grads", data=average_grads)
                            f.create_dataset("location", data=loc)

                        i += 1
                else:
                    loc = locations[synapse_number]
                    raw = get_raw(input_shape,
                                      voxel_size,
                                      synapse,
                                      loc,
                                      complete_brain)

                    output, average_grads, integrated_grads, _ = get_integrated_gradients(model, 
                                                                                        raw, 
                                                                                        baseline, 
                                                                                        target_id, 
                                                                                        integration_steps)

                    output_file = os.path.join(output_dir, "grads_{}{}_{}.h5".format(target_id, synapse, synapse_number))
                    with h5py.File(output_file, "w") as f:
                        f.create_dataset("output", data=output.detach().cpu().numpy())
                        f.create_dataset("raw", data=raw)
                        f.create_dataset("average_grads", data=average_grads)
                        f.create_dataset("location", data=loc)

            target_id += 1



if __name__ == "__main__":
    iteration = 300000
    run_num = 4
    checkpoint_file = "/groups/funke/home/dum/Projects/synister_exeriments/fafb/02_setups/setup01/run{}/model_checkpoint_{}".format(run_num, iteration)
    data_dir = "/groups/funke/home/dum/Projects/synister_exeriments/fafb/01_data" 
    output_dir = "./attributions"

    get_attribution(checkpoint_file,
                    data_dir,
                    output_dir,
                    baseline=None,
                    integration_steps=50,
                    synapse_type='gaba',
                    synapse_number=5)
