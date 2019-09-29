from gunpowder import *
from gunpowder.ext import torch
from gunpowder.torch import *
import json
import logging
import math
import numpy as np
import os
import sys
from funlib.learn.torch.models import Vgg3D
from synister.gp import SynapseSourceMongo, SynapseTypeSource, InspectLabels
from synister.read_config import read_train_config

torch.backends.cudnn.enabled = False

def train_until(max_iteration,
                db_credentials,
                db_name_data,
                split_name,
                synapse_types,
                input_shape,
                fmaps,
                num_levels,
                batch_size):

    input_shape = Coordinate(input_shape)

    model = Vgg3D(input_size=input_shape, fmaps=fmaps)
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4)

    raw = ArrayKey('RAW')
    synapses = PointsKey('SYNAPSES')
    synapse_type = ArrayKey('SYNAPSE_TYPE')
    pred_synapse_type = ArrayKey('PRED_SYNAPSE_TYPE')

    voxel_size = Coordinate((40, 4, 4))
    input_size = input_shape*voxel_size

    request = BatchRequest()
    request.add(raw, input_size)
    request.add(synapses, input_size/8)
    request[synapse_type] = ArraySpec(nonspatial=True)
    request[pred_synapse_type] = ArraySpec(nonspatial=True)

    fafb_source = (
        ZarrSource(
            '/nrs/saalfeld/FAFB00/v14_align_tps_20170818_dmg.n5',
            datasets={raw: 'volumes/raw/s0'},
            array_specs={raw: ArraySpec(interpolatable=True)}) +
        Normalize(raw) +
        Pad(raw, None)
    )

    sample_sources = tuple(
        (
            fafb_source,
            SynapseSourceMongo(
                db_credentials,
                db_name_data,
                split_name,
                tuple([t]),
                synapses),
            SynapseTypeSource(synapse_types, t, synapse_type)
        ) +
        MergeProvider() +
        RandomLocation(ensure_nonempty=synapses)

        for t in synapse_types
    )

    pipeline = (
        sample_sources +
        RandomProvider() +
        ElasticAugment(
            control_point_spacing=[4,40,40],
            jitter_sigma=[0,2,2],
            rotation_interval=[0,math.pi/2.0],
            prob_slip=0.05,
            prob_shift=0.05,
            max_misalign=10,
            subsample=8) +
        SimpleAugment(transpose_only=[1, 2]) +
        IntensityAugment(raw, 0.9, 1.1, -0.1, 0.1, z_section_wise=True) +
        IntensityScaleShift(raw, 2,-1) +
        PreCache(
            cache_size=40,
            num_workers=10) +
        Stack(batch_size) +
        Train(
            model,
            loss=loss,
            optimizer=optimizer,
            inputs={
                'raw': raw
            },
            target=synapse_type,
            output=pred_synapse_type,
            array_specs={
                pred_synapse_type: ArraySpec(nonspatial=True)
            },
            save_every=1000,
            log_dir='log') +
        InspectLabels(
            synapse_type,
            pred_synapse_type) +
        IntensityScaleShift(raw, 0.5, 0.5) +
        Snapshot({
                raw: 'volumes/raw',
                synapse_type: 'synapse_type',
                pred_synapse_type: 'pred_synapse_type'
            },
            every=100,
            output_filename='batch_{iteration}.hdf') +
        PrintProfilingStats(every=10)
    )

    print("Starting training...")
    with build(pipeline) as p:
        while True:
            batch = p.request_batch(request)
            if batch.iteration >= max_iteration:
                break
    print("Training finished")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    iteration = int(sys.argv[1])
    train_config = read_train_config("./train_config.ini")
    train_config["max_iteration"] = iteration
    train_until(**train_config)
