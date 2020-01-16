from gunpowder import *
from synister.synister_db import SynisterDb
import numpy as np

class SynapseSkeletonSourceMongo(CsvPointsSource):
    def __init__(self, db_credentials, 
                       db_name, 
                       split_name, 
                       skeleton_id, 
                       points,
                       points_spec=None, 
                       scale=None):

        self.db = SynisterDb(db_credentials, db_name)
        self.split_name = split_name
        self.db_name = db_name
        self.skeleton_id = skeleton_id
        super(SynapseSkeletonSourceMongo, self).__init__(filename=None,
                                                 points=points,
                                                 points_spec=points_spec,
                                                 scale=scale)

    def _read_points(self):
        print("Reading split {} from db {}".format(self.split_name, self.db_name))
        synapses = self.db.get_synapses(skeleton_ids=[self.skeleton_id],
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

class SkeletonIdSource(BatchProvider):
    def __init__(self, skeleton_ids, skeleton_id, array):
        n = len(skeleton_ids)
        i = skeleton_ids.index(skeleton_id)

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
    def __init__(self, skeleton_id, pred_skeleton_id):
        self.skeleton_id = skeleton_id
        self.pred_skeleton_id = pred_skeleton_id

    def process(self, batch, request):
        print("label     :", batch[self.skeleton_id].data)
        print("prediction:", batch[self.pred_skeleton_id].data)
