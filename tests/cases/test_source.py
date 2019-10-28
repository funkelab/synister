import unittest
from synister.gp import SynapseSourceMongo
from synister.synister_db import SynisterDb
from gunpowder import *
import os

class SynapseSourceMongoTestCase(unittest.TestCase):
    def setUp(self):
        self.db_credentials = os.path.join(os.path.abspath(os.path.dirname(__file__)) + "/../../db_credentials.ini")
        self.db_name = "synister_v2_refactor"
        self.split_name = "neuron"
        self.points = PointsKey('SYNAPSES')
        self.db = SynisterDb(self.db_credentials, self.db_name)
        self.neurotransmitters = [
            ('gaba',),
            #('acetylcholine',), TODO: Fix position query for large number of objects. DB hangs for this.
            ('glutamate',),
            ('dopamine',),
            ('octopamine',),
            ('serotonin',),
        ]

    def runTest(self):
        for synapse_type in self.neurotransmitters:
            source = SynapseSourceMongo(self.db_credentials,
                                        self.db_name,
                                        self.split_name,
                                        synapse_type,
                                        self.points)

            
            source.setup()
            points = source.data
            
            print("query pos...")
            synapses = self.db.get_synapses(positions=points)

            print("get skeletons...")
            skeletons = self.db.get_skeletons()

            n = 1
            for synapse in synapses.values():
                self.assertTrue(synapse["splits"][self.split_name] == "train")
                nt = skeletons[synapse["skeleton_id"]]["nt_known"]
                self.assertTrue(nt == synapse_type)
                n += 1


            synapses_in_split = self.db.get_synapses(split_name=self.split_name, neurotransmitters=synapse_type)
            all_synapse_ids_in_train = [id_ for id_, s in synapses_in_split.items() if s["splits"][self.split_name] == "train"]
            synapse_ids_retrieved = [s for s in synapses]
            self.assertTrue(sorted(all_synapse_ids_in_train) == sorted(synapse_ids_retrieved))

if __name__ == "__main__":
    unittest.main()
