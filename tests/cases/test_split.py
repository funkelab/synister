import unittest
from synister.synister_db import SynisterDb
from synister.split import find_optimal_split
import os

class DbSetupTestCase(unittest.TestCase):
    def setUp(self):
        self.db_credentials = os.path.join(os.path.abspath(os.path.dirname(__file__)) + "/../../db_credentials.ini")
        self.db = SynisterDb(self.db_credentials, "synister_v2_refactor")


class FindOptimalSplitTestCase(DbSetupTestCase):
    def runTest(self):
        hemi_lineages = self.db.get_hemi_lineages()

        neurotransmitters = [
            ('gaba',),
            ('acetylcholine',),
            ('glutamate',),
            ('dopamine',),
            ('octopamine',),
            ('serotonin',),
        ]

        hemi_lineages_by_synapse_id = {}
        nt_by_synapse_id = {}
        synapse_ids = []

        for nt in neurotransmitters:
            synapses = self.db.get_synapses(neurotransmitters=nt)
            for synapse_id in synapses:
                nt_by_synapse_id[synapse_id] = nt
                synapse_ids.append(synapse_id)

        for hl_id in hemi_lineages:
            synapses = self.db.get_synapses(hemi_lineage_id=hl_id)
            for synapse_id in synapses:
                hemi_lineages_by_synapse_id[synapse_id] = hl_id

        train, test = find_optimal_split(synapse_ids=synapse_ids,
                           superset_by_synapse_id=hemi_lineages_by_synapse_id,
                           nt_by_synapse_id=nt_by_synapse_id,
                           neurotransmitters=neurotransmitters,
                           supersets=hemi_lineages,
                           train_fraction=0.8)

        # Test for superset overlap:
        self.assertFalse(set(list(train.keys())) & set(list(test.keys())))

        # Test if we can manually reproduce the fractions:
        train_synapses = {}
        test_synapses = {}

        for hl_id in train:
            train_synapses = {**train_synapses, **self.db.get_synapses(hemi_lineage_id=hl_id)}

        for hl_id in test:
            test_synapses = {**test_synapses, **self.db.get_synapses(hemi_lineage_id=hl_id)}


        train_synapses_by_nt = {nt: [] for nt in neurotransmitters}
        skeletons = self.db.get_skeletons()

        for synapse_id, synapse in train_synapses.items():
            skeleton = skeletons[synapse["skeleton_id"]]
            nt = skeleton["nt_known"]

            if nt in train_synapses_by_nt:
                train_synapses_by_nt[nt].append({synapse_id: synapse})

        for nt, synapses in train_synapses_by_nt.items():
            len(train_synapses_by_nt)
        
if __name__ == "__main__":
    unittest.main()
