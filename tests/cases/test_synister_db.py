import unittest
from synister.synister_db import SynisterDb
import os
from pymongo import MongoClient

class DbSetupTestCase(unittest.TestCase):
    def setUp(self):
        self.db_credentials = os.path.join(os.path.abspath(os.path.dirname(__file__)) + "/../../db_credentials.ini")
        self.db = SynisterDb(self.db_credentials, "synister_v2_refactor")

class GetSynapsesTestCase(DbSetupTestCase):
    def runTest(self):
        test_client = MongoClient(self.db.auth_string)
        db = test_client[self.db.db_name]
        synapse_collection = db["synapses"]

        all_synapses = self.db.get_synapses()
        self.assertTrue(len(all_synapses) == synapse_collection.count_documents({}))

        on_skeleton_16 = self.db.get_synapses(skeleton_ids=[16])
        self.assertTrue(len(on_skeleton_16) == synapse_collection.count_documents({"skeleton_id": 16}))

        on_skeleton_16_or_12578 = self.db.get_synapses(skeleton_ids=[16, 12578])
        self.assertTrue(len(on_skeleton_16_or_12578) == synapse_collection.count_documents({"skeleton_id": 16}) +\
                                                        synapse_collection.count_documents({"skeleton_id": 12578}))

        ach = self.db.get_synapses(neurotransmitters=("acetylcholine",))
        
        skeleton_collection = db["skeletons"]
        ach_skeleton_ids = [sk["skeleton_id"] for sk in skeleton_collection.find({"nt_known": ["acetylcholine"]})]
        ach_synapses = synapse_collection.count_documents({"skeleton_id": {"$in": ach_skeleton_ids}})
        self.assertTrue(len(ach)==ach_synapses)

        on_skeleton_16_or_12578_and_ach = self.db.get_synapses(skeleton_ids=[16, 12578], neurotransmitters=("acetylcholine",))

        synapses = synapse_collection.count_documents({"skeleton_id": {"$in": ach_skeleton_ids}})
        self.assertTrue(len(on_skeleton_16_or_12578_and_ach) == 1704)

        on_pos = self.db.get_synapses(positions=[(217400, 164242, 438817)])
        self.assertTrue(len(on_pos) == 1 and list(on_pos.keys())[0] == 999188)

        on_skeleton_16_or_12578_and_ach_on_pos = self.db.get_synapses(skeleton_ids=[16, 12578], neurotransmitters=("acetylcholine",), positions=[(217400, 164242, 438817)])
        self.assertTrue(on_skeleton_16_or_12578_and_ach_on_pos == on_pos)

        on_id = self.db.get_synapses(synapse_ids=[999188])
        self.assertTrue(on_id == on_skeleton_16_or_12578_and_ach_on_pos == on_pos)

        on_hemi_lineage_ALAD1 = self.db.get_synapses(hemi_lineage_name="ALAD1")
        on_hemi_lineage_id_0 = self.db.get_synapses(hemi_lineage_id=0)
        on_hemi_lineage_id_0_and_name_ALAD1 = self.db.get_synapses(hemi_lineage_id=0, hemi_lineage_name="ALAD1")
        self.assertTrue(on_hemi_lineage_ALAD1 == on_hemi_lineage_id_0 == on_hemi_lineage_id_0_and_name_ALAD1)

        on_split_neuron = self.db.get_synapses(split_name="neuron")
        train = [s for s in on_split_neuron if on_split_neuron[s]["splits"]["neuron"]=="train"]
        test = [s for s in on_split_neuron if on_split_neuron[s]["splits"]["neuron"]=="test"]
        self.assertTrue(len(train) + len(test) == len(on_split_neuron))


        on_all = self.db.get_synapses(skeleton_ids=[16, 12578], neurotransmitters=("acetylcholine",), 
                                      positions=[(217400, 164242, 438817)], hemi_lineage_name="ALAD1",
                                      hemi_lineage_id=0, split_name="neuron")

        self.assertTrue(len(on_all) == 1)

        on_none = self.db.get_synapses(skeleton_ids=[16, 12578], neurotransmitters=("acetylcholine",), 
                                      positions=[(217400, 164242, 438817)], hemi_lineage_name="ALAD1",
                                      hemi_lineage_id=1, split_name="neuron")
  
        self.assertFalse(on_none)

class GetSkeletonsTestCase(DbSetupTestCase):
    def runTest(self):
        all_skeletons = self.db.get_skeletons()
        self.assertTrue(len(all_skeletons)==1928)
        
        skeleton_16 = self.db.get_skeletons(skeleton_ids=[16])
        self.assertTrue(len(skeleton_16) == 1 and list(skeleton_16.keys())[0]==16)

        skeleton_16_or_12578 = self.db.get_skeletons(skeleton_ids=[16, 12578])
        self.assertTrue(len(skeleton_16_or_12578) == 2 and sorted(list(skeleton_16_or_12578.keys())) == sorted([16, 12578]))

        gaba_skeletons = self.db.get_skeletons(neurotransmitters=("gaba",))
        self.assertTrue(len(gaba_skeletons) == 106)

        synapse_id_skeletons = self.db.get_skeletons(synapse_ids=[999188, 97954])
        self.assertTrue(len(synapse_id_skeletons)==1 and list(synapse_id_skeletons.keys())[0] == 16)

        position_skeletons = self.db.get_skeletons(positions=[(217400, 164242, 438817)])
        self.assertTrue(len(synapse_id_skeletons)==1 and list(synapse_id_skeletons.keys())[0] == 16)

        hemi_lineage_name_skeletons = self.db.get_skeletons(hemi_lineage_name="ALAD1")
        hemi_lineage_id_skeletons = self.db.get_skeletons(hemi_lineage_id=0)
        hemi_lineage_id_and_name_skeletons = self.db.get_skeletons(hemi_lineage_id=0, hemi_lineage_name="ALAD1")
        self.assertTrue(len(hemi_lineage_name_skeletons)==len(hemi_lineage_id_skeletons)==len(hemi_lineage_id_and_name_skeletons)==85)
   
class GetHemiLineagesTestCase(DbSetupTestCase):
    def runTest(self):
        hls = self.db.get_hemi_lineages()
        self.assertTrue(len(hls) == 70)

class InitializePredictionsTestCase(DbSetupTestCase):
    def runTest(self):
        """TODO"""
        self.assertTrue(False)

class WritePredictionsTestCase(DbSetupTestCase):
    def runTest(self):
        """TODO"""
        self.assertTrue(False)

class CountPredictionsTestCase(DbSetupTestCase):
    def runTest(self):
        """TODO"""
        self.assertTrue(False)

class MakeSplitTestCase(DbSetupTestCase):
    def runTest(self):
        """TODO"""
        self.assertTrue(False)

if __name__ == "__main__":
    unittest.main()
