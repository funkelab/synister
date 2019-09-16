from gunpowder import *
from gunpowder.ext import torch
from gunpowder.torch import *
import json
import logging
import math
import numpy as np
import os
import sys
from funlib.learn.torch import Vgg3D

torch.backends.cudnn.enabled = False

data_dir = '../../01_data/v2'
synapse_types = [
    'gaba',
    'acetylcholine',
    'glutamate',
    'serotonin',
    'octopamine',
    'dopamine'
]

input_shape = Coordinate((32, 128, 128))
fmaps = 32
num_levels = 4
batch_size = 8



class SynapseSource(CsvPointsSource):

    def _read_points(self):

        print("Reading %s" % self.filename)
        self.data = np.array(json.load(open(self.filename, 'r')))
        self.ndims = 3
        print("data: ", self.data.shape)


class SynapseTypeSource(BatchProvider):

    def __init__(self, synapse_types, synapse_type, array):

        n = len(synapse_types)
        i = synapse_types.index(synapse_type)

        self.label = np.int64(i)
        self.array = array

    def setup(self):

        spec = ArraySpec(
            nonspatial=True,
            dtype=np.int64)
        self.provides(self.array, spec)

    def provide(self, request):

        batch = Batch()

        spec = self.spec[self.array]
        batch.arrays[self.array] = Array(
            self.label,
            spec)

        return batch


class InspectLabels(BatchFilter):

    def __init__(self, synapse_type, pred_synapse_type):
        self.synapse_type = synapse_type
        self.pred_synapse_type = pred_synapse_type

    def process(self, batch, request):
        print("label     :", batch[self.synapse_type].data)
        print("prediction:", batch[self.pred_synapse_type].data)

def train_until(max_iteration):

    model = Vgg3D.Vgg3D(input_size=input_shape, fmaps=fmaps)
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
            SynapseSource(
                os.path.join(data_dir, t + '_train.json'),
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
    train_until(iteration)
