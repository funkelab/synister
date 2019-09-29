from gunpowder import *
from synister.synister_db import SynisterDB
import numpy as np

class SynapseSourceCsv(CsvPointsSource):

    def _read_points(self):

        print("Reading %s" % self.filename)
        self.data = np.array(json.load(open(self.filename, 'r')))
        self.ndims = 3
        print("data: ", self.data.shape)


class SynapseSourceMongo(CsvPointsSource):
    def __init__(self, db_credentials, 
                       db_name, 
                       split_name, 
                       synapse_type, 
                       points,
                       points_spec=None, 
                       scale=None):

        self.db = SynisterDB(db_credentials)
        self.split_name = split_name
        self.db_name = db_name
        self.synapse_type = synapse_type
        super(SynapseSourceMongo, self).__init__(filename=None,
                                                 points=points,
                                                 points_spec=points_spec,
                                                 scale=scale)

    def _read_points(self):
        print("Reading split {} from db {}".format(self.split_name, self.db_name))
        points = np.array(self.db.get_synapse_locations(self.db_name,
                                                        self.split_name,
                                                        "train",
                                                        self.synapse_type), dtype=np.float32)

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

