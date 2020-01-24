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
from synister.skeleton_network.gp import SynapseSkeletonSourceMongo, SkeletonIdSource, InspectLabels
from synister.read_config import read_train_config
from synister.synister_db import SynisterDb

torch.backends.cudnn.enabled = False

def train_until(max_iteration,
                db_credentials,
                db_name_data,
                split_name,
                input_shape,
                fmaps,
                downsample_factors,
                batch_size,
                voxel_size,
                raw_container,
                raw_dataset,
                output_classes=None,
                **kwargs):

    if output_classes is None:
        output_classes = len(skeleton_ids)


    db = SynisterDb(db_credentials, db_name_data)
    synapses_in_split = db.get_synapses(split_name=split_name)

    synapses_per_skeleton = {}
    for synapse_id, synapse in synapses_in_split.items():
        skid = synapse["skeleton_id"]
        if synapse["splits"][split_name] == "train":
            if not skid in list(synapses_per_skeleton.keys()):
                synapses_per_skeleton[skid] = [synapse_id]
            else:
                synapses_per_skeleton[skid].append(synapse_id)

    skeleton_ids = [skid for skid, synapses in synapses_per_skeleton.items() if len(synapses) > 100]

    if output_classes is None:
        output_classes = len(skeleton_ids)

    input_shape = Coordinate(input_shape)

    model = Vgg3D(input_size=input_shape, 
                  fmaps=fmaps, 
                  downsample_factors=downsample_factors,
                  output_classes=int(output_classes))

    model.train()

    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4)

    raw = ArrayKey('RAW')
    synapses = PointsKey('SYNAPSES')
    skeleton_id = ArrayKey('SKELETON_ID')
    pred_skeleton_id = ArrayKey('PRED_SKELETON_ID')

    voxel_size = Coordinate(tuple(voxel_size))
    input_size = input_shape*voxel_size

    request = BatchRequest()
    request.add(raw, input_size)
    request.add(synapses, input_size/8)
    request[skeleton_id] = ArraySpec(nonspatial=True)
    request[pred_skeleton_id] = ArraySpec(nonspatial=True)

    fafb_source = (
        ZarrSource(
            raw_container,
            datasets={raw: raw_dataset},
            array_specs={raw: ArraySpec(interpolatable=True)}) +
        Normalize(raw) +
        Pad(raw, None)
    )

    sample_sources = tuple(
        (
            fafb_source,
            SynapseSkeletonSourceMongo(
                db_credentials,
                db_name_data,
                split_name,
                skid,
                synapses),
            SkeletonIdSource(skeleton_ids, 
                             skid, 
                             skeleton_id)
        ) +
        MergeProvider() +
        RandomLocation(ensure_nonempty=synapses)

        for skid in skeleton_ids
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
            target=skeleton_id,
            output=pred_skeleton_id,
            array_specs={
                pred_skeleton_id: ArraySpec(nonspatial=True)
            },
            save_every=10000,
            log_dir='log') +
        InspectLabels(skeleton_id,
                      pred_skeleton_id) + 
        IntensityScaleShift(raw, 0.5, 0.5) +
        Snapshot({
                raw: 'volumes/raw',
                skeleton_id: 'skeleton_id',
                pred_skeleton_id: 'pred_skeleton_id'
            },
            every=10000,
            output_filename='batch_{iteration}.hdf') +
        PrintProfilingStats(every=1000)
    )

    print("Starting training...")
    with build(pipeline) as p:
        while True:
            batch = p.request_batch(request)
            if batch.iteration >= max_iteration:
                break
    print("Training finished")

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    #logging.getLogger('gunpowder.nodes').setLevel(logging.DEBUG)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)

    fileHandler = logging.FileHandler("train.log")
    fileHandler.setLevel(logging.DEBUG)
    logging.getLogger().addHandler(fileHandler)

    #console = logging.StreamHandler()
    #console.setLevel(logging.INFO)
    #logging.getLogger('gunpowder').addHandler(console)

    iteration = int(sys.argv[1])
    train_config = read_train_config("./train_config.ini")
    train_config["max_iteration"] = iteration
    train_until(**train_config)
