from gunpowder import *
from synister.synister_db import SynisterDb
import numpy as np

class SynapseHemiLineageSourceMongo(CsvPointsSource):
    def __init__(self, db_credentials, 
                       db_name, 
                       split_name, 
                       hemi_lineage_id, 
                       points,
                       points_spec=None, 
                       scale=None):

        self.db = SynisterDb(db_credentials, db_name)
        self.split_name = split_name
        self.db_name = db_name
        self.hemi_lineage_id = hemi_lineage_id
        super(SynapseHemiLineageSourceMongo, self).__init__(filename=None,
                                                 points=points,
                                                 points_spec=points_spec,
                                                 scale=scale)

    def _read_points(self):
        print("Reading split {} from db {}".format(self.split_name, self.db_name))
        synapses = self.db.get_synapses(hemi_lineage_id=self.hemi_lineage_id,
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

class HemiLineageIdSource(BatchProvider):
    def __init__(self, hemi_lineage_ids, hemi_lineage_id, array):
        n = len(hemi_lineage_ids)
        i = hemi_lineage_ids.index(hemi_lineage_id)

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
    def __init__(self, hemi_lineage_id, pred_hemi_lineage_id):
        self.hemi_lineage_id = hemi_lineage_id
        self.pred_hemi_lineage_id = pred_hemi_lineage_id

    def process(self, batch, request):
        print("label     :", batch[self.hemi_lineage_id].data)
        print("prediction:", batch[self.pred_hemi_lineage_id].data)
