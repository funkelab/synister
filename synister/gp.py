from gunpowder import *
from synister.synister_db import SynisterDb
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

        self.data = points
        self.ndims = 3

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

