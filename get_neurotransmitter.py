import numpy as np
import json
from synister.utils import init_vgg, predict, get_raw
from synister.catmaid_interface import Catmaid
import os
import sys
import time
import pandas
import zarr

def init_model():
    # Current model trained on all annotated synapses:
    train_checkpoint = "/nrs/funke/ecksteinn/synister_experiments/fafb_v3/02_train/setup_t8/model_checkpoint_300000"
    input_shape = (16,160,160)
    fmaps = 12
    downsample_factors = [(1, 2, 2), (1, 2, 2), (1, 2, 2), (2, 2, 2)]
    neurotransmitter_list = ["gaba", "acetylcholine", "glutamate", "serotonin", "octopamine", "dopamine"]
    output_classes = len(neurotransmitter_list)

    # Initialize model
    model = init_vgg(train_checkpoint,
                     input_shape,
                     fmaps,
                     downsample_factors,
                     output_classes)

    model_config = {"train_checkpoint": train_checkpoint,
                    "input_shape": input_shape,
                    "fmaps": fmaps,
                    "downsample_factors": downsample_factors,
                    "neurotransmitter_list": neurotransmitter_list,
                    "output_classes": output_classes}

    return model, model_config

def get_neurotransmitter(positions, 
                         model, 
                         model_config, 
                         predict_id=0, 
                         save_batches=100, 
                         output_dir="."):
    """
    positions `list of array-like of ints`:
        Synaptic postions in fafb v14 [(z0,y0,x0), (z1, y1, x1), ...]

    returns:
        list of dictionaries of neurotransmitter probabilities
    """

    neurotransmitter_list = model_config["neurotransmitter_list"]
    input_shape = model_config["input_shape"]
    output_classes = model_config["output_classes"]
    batch_size = 8

    # Disable Dropout, Batch norm etc.
    model.eval()

    # Fafb v14
    raw_container = "/nrs/saalfeld/FAFB00/v14_align_tps_20170818_dmg.n5"
    raw_dataset = "volumes/raw/s0"
    voxel_size = np.array([40,4,4])

    batch_output_dir = output_dir + "/batches"
    if not os.path.exists(batch_output_dir):
        os.makedirs(batch_output_dir)

    nt_probabilities = []
    for i in range(0, len(positions), batch_size):
        batched_positions = positions[i:i+batch_size]
        raw, raw_normalized = get_raw(batched_positions,
                                      input_shape,
                                      voxel_size,
                                      raw_container,
                                      raw_dataset)

        if i % save_batches == 0:
            zarr.save(batch_output_dir + '/batch_{}_{}.zarr'.format(predict_id, i), 
                      raw_normalized)

        output = predict(raw_normalized, model)

        # Iterate over batch and grab predictions
        for k in range(np.shape(output)[0]):
            out_k = output[k,:].tolist()
            nt_probability = {neurotransmitter_list[i]: out_k[i] for i in range(output_classes)}
            nt_probabilities.append(nt_probability)

    return nt_probabilities

def catmaid_transform(positions):
    """
    Transforms a catmaid (z,y,x) position to fafb v14 space.
    """
    return [(position[0] - 40, position[1], position[2]) for position in positions]

def get_catmaid_neurotransmitters(skids, output_dir, save_batches=100, skip_existing=True):
    cm = Catmaid()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model, model_config = init_model()
    i = 0
    for skid in skids:
        if skip_existing:
            if os.path.exists(os.path.join(output_dir, "skid_{}.json".format(skid))):
                continue

        print("Predict {} from {} skids".format(i, len(skids)))
        pos, ids = cm.get_synapse_positions(skid)
        if len(pos) > 0:
            pos_transformed = catmaid_transform(pos)
            print("Predict {} positions".format(len(pos)))
            start = time.time()
            nt_probabilities  = get_neurotransmitter(pos_transformed, 
                                                     model, 
                                                     model_config,
                                                     skid,
                                                     save_batches,
                                                     output_dir)

            nt_probabilities = [[int(ids[i]), 
                                [int(k) for k in pos_transformed[i]], 
                                nt_probabilities[i]] for i in range(len(nt_probabilities))]

            print("{} seconds per synapse".format((time.time() - start)/len(pos)))
        else:
            nt_probabilities = []

        out_file = os.path.join(output_dir, 
                                "skid_{}.json".format(skid))

        with open(out_file, "w+") as f:
            json.dump(nt_probabilities, f)

        i += 1

    out_config = os.path.join(output_dir, "model_config.json")
    with open(out_config, "w+") as f:
        json.dump(model_config, f)

def read_neuron_csv(csv_path):
    """
    The csv needs to have one column called 'skid'
    """
    data = pandas.read_csv(csv_path)
    skids = data["skid"].to_list()
    skids = list(set([int(skid) for skid in skids]))
    return skids

if __name__ == "__main__":
    skid_csv_path = sys.argv[1]
    output_dir = os.path.dirname(skid_csv_path) + "/predictions"
    skids = read_neuron_csv(skid_csv_path)
    get_catmaid_neurotransmitters(skids, output_dir)
