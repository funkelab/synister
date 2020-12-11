from gunpowder import *
from synister.synister_db import SynisterDb
from pyquaternion import Quaternion
import random
import numpy as np

class SynapseSourceMongo(CsvPointsSource):
    def __init__(self, db_credentials, 
                       db_name, 
                       split_name, 
                       synapse_type, 
                       points,
                       points_spec=None, 
                       scale=None):

        self.db = SynisterDb(db_credentials, db_name)
        self.split_name = split_name
        self.db_name = db_name
        self.synapse_type = synapse_type
        super(SynapseSourceMongo, self).__init__(filename=None,
                                                 points=points,
                                                 points_spec=points_spec,
                                                 scale=scale)

    def _read_points(self):
        if self.synapse_type != "unknown":
            print("Reading split {} from db {}".format(self.split_name, self.db_name))
            synapses = self.db.get_synapses(neurotransmitters=self.synapse_type,
                                            split_name=self.split_name)

            points = np.array([
                                [
                                    int(synapse["z"]),
                                    int(synapse["y"]),
                                    int(synapse["x"])
                                    
                                ]
                            for synapse in synapses.values()
                            if synapse["splits"][self.split_name] == "train"
                            ])
        else:
            points = get_unknown_synapse_type()

        self.data = points
        self.ndims = 3

    def get_unknown_synapse_type(self):
        """
        Samples points around known synaptic locations
        and offsets by a randomly chosen offset vector on
        the sphere.
        """
        nt_types_all = ["gaba", "acetylcholine", "glutamate", 
                        "serotonin", "octopamine", "dopamine"]
        n_type = 5000

        synapse_locs = []
        for nt in nt_types_all:
            synapses_nt = self.db.get_synapses(neurotransmitters=(nt,), 
                                               split_name=self.split_name)
            points_nt = [[
                            [
                                int(synapse["z"]),
                                int(synapse["y"]),
                                int(synapse["x"])
                                
                            ]
                        for synapse in synapses.values()
                        if synapse["splits"][self.split_name] == "train"
                        ]]

            random.shuffle(points_nt)
            n = min(len(points_nt), n_type)
            synapse_locs.extend(points_nt[:n])

    random_offsets = self.get_random_offsets(len(synapse_locs))
    synapse_locs = np.array(synapse_locs)
    synapse_locs += random_offsets
    return synapse_locs

    def sample_trig(self, npoints):
        theta = 2*np.pi*np.random.rand(npoints)
        phi = np.arccos(2*np.random.rand(npoints)-1)
        x = np.cos(theta) * np.sin(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(phi)
        return np.array([z,y,x])

    def sample_radii(self, npoints, d_min=2000, d_max=4000):
        return np.random.randint(d_min, high=d_max, size=npoints)

    def get_random_offset(self, npoints):
        direction_samples = self.sample_trig(n_points)
        radii_samples = sample_radii(npoints)
        offset_samples = direction_samples * radii_samples
        return offset_samples.T


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


class AddChannelDim(BatchFilter):

    def __init__(self, array):
        self.array = array

    def process(self, batch, request):

        if self.array not in batch:
            return

        batch[self.array].data = batch[self.array].data[np.newaxis]

