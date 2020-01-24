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
from synister.hemi_lineage_network.gp import SynapseHemiLineageSourceMongo, HemiLineageIdSource, InspectLabels
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

    db = SynisterDb(db_credentials, db_name_data)
    synapses_in_split = db.get_synapses(split_name=split_name)
    skeleton_ids = list(set([synapse["skeleton_id"] for synapse_id, synapse in synapses_in_split.items() if synapse["splits"][split_name] == "train"]))

    # Remove out of bounds skeleton
    skeleton_ids.remove(2130631)
    skeletons_in_split = db.get_skeletons(skeleton_ids=skeleton_ids)
    hemi_lineage_ids = list(set([skeleton["hemi_lineage_id"] for skeleton_id, skeleton in skeletons_in_split.items()]))

    synapses_per_hemi_lineage = {}
    for hl_id in hemi_lineage_ids:
        synapses_per_hemi_lineage[hl_id] = [synapse_id for synapse_id, synapse in db.get_synapses(hemi_lineage_id=hl_id, split_name=split_name).items() if synapse["splits"][split_name] == "train"]

    hemi_lineage_ids = [hl_id for hl_id in hemi_lineage_ids if len(synapses_per_hemi_lineage[hl_id]) > 10]

    if output_classes is None:
        output_classes = len(hemi_lineage_ids)

    input_shape = Coordinate(input_shape)

    model = Vgg3D(input_size=input_shape, 
                  fmaps=fmaps, 
                  downsample_factors=downsample_factors,
                  output_classes=output_classes)

    model.train()

    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4)

    raw = ArrayKey('RAW')
    synapses = PointsKey('SYNAPSES')
    hemi_lineage_id = ArrayKey('HEMI_LINEAGE_ID')
    pred_hemi_lineage_id = ArrayKey('PRED_HEMI_LINEAGE_ID')

    voxel_size = Coordinate(tuple(voxel_size))
    input_size = input_shape*voxel_size

    request = BatchRequest()
    request.add(raw, input_size)
    request.add(synapses, input_size/8)
    request[hemi_lineage_id] = ArraySpec(nonspatial=True)
    request[pred_hemi_lineage_id] = ArraySpec(nonspatial=True)

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
            SynapseHemiLineageSourceMongo(
                db_credentials,
                db_name_data,
                split_name,
                hlid,
                synapses),
            HemiLineageIdSource(hemi_lineage_ids, hlid, hemi_lineage_id)
        ) +
        MergeProvider() +
        RandomLocation(ensure_nonempty=synapses)

        for hlid in hemi_lineage_ids
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
            target=hemi_lineage_id,
            output=pred_hemi_lineage_id,
            array_specs={
                pred_hemi_lineage_id: ArraySpec(nonspatial=True)
            },
            save_every=10000,
            log_dir='log') +
        IntensityScaleShift(raw, 0.5, 0.5) +
        Snapshot({
                raw: 'volumes/raw',
                hemi_lineage_id: 'hemi_lineage_id',
                pred_hemi_lineage_id: 'pred_hemi_lineage_id'
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
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)

    fileHandler = logging.FileHandler("train.log")
    fileHandler.setLevel(logging.DEBUG)
    logging.getLogger().addHandler(fileHandler)

    iteration = int(sys.argv[1])
    train_config = read_train_config("./train_config.ini")
    train_config["max_iteration"] = iteration
    train_until(**train_config)
