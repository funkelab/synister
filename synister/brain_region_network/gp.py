from gunpowder import *
from synister.synister_db import SynisterDb
import numpy as np

class SynapseBrainRegionSourceMongo(CsvPointsSource):
    def __init__(self, db_credentials, 
                       db_name, 
                       split_name, 
                       brain_region_id, 
                       points,
                       points_spec=None, 
                       scale=None):

        self.db = SynisterDb(db_credentials, db_name)
        self.split_name = split_name
        self.db_name = db_name
        self.brain_region_id = brain_region_id
        super(SynapseBrainRegionSourceMongo, self).__init__(filename=None,
                                                            points=points,
                                                            points_spec=points_spec,
                                                            scale=scale)

    def _read_points(self):
        print("Reading split {} from db {}".format(self.split_name, self.db_name))
        synapses = self.db.get_synapses(split_name=self.split_name)

        points = np.array([
                            [
                                int(synapse["z"]),
                                int(synapse["y"]),
                                int(synapse["x"])
                                
                            ]
                        for synapse in synapses.values()
                        if (synapse["splits"][self.split_name] == "train"
                            and len(synapse["brain_region"]) == 1 
                            and synapse["brain_region"][0] == self.brain_region_id)
                        ])

        self.data = points
        self.ndims = 3

class BrainRegionIdSource(BatchProvider):
    def __init__(self, brain_region_ids, brain_region_id, array):
        n = len(brain_region_ids)
        i = brain_region_ids.index(brain_region_id)

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
    def __init__(self, brain_region_id, pred_brain_region_id):
        self.brain_region_id = brain_region_id
        self.pred_brain_region_id = pred_brain_region_id

    def process(self, batch, request):
        print("label     :", batch[self.brain_region_id].data)
        print("prediction:", batch[self.pred_brain_region_id].data)
