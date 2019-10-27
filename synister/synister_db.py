from pymongo import MongoClient, IndexModel, ASCENDING
from configparser import ConfigParser
from copy import deepcopy
import logging
import time
from itertools import permutations

logger = logging.getLogger(__name__)

unknown_hemi_lineage_names = [
    'NA',
    'NA1',
    'NONE',
    'NEW',
    'NEW2_POSTERIOR',
    'NONE'
] + ['UNKNOWN%d' % d for d in range(10)]


class SynisterDb(object):
    def __init__(self, credentials, db_name):
        with open(credentials) as fp:
            config = ConfigParser()
            config.read_file(fp)
            self.credentials = {}
            self.credentials["user"] = config.get("Credentials", "user")
            self.credentials["password"] = config.get("Credentials", "password")
            self.credentials["host"] = config.get("Credentials", "host")
            self.credentials["port"] = config.get("Credentials", "port")

        self.auth_string = 'mongodb://{}:{}@{}:{}'.format(self.credentials["user"],
                                                          self.credentials["password"],
                                                          self.credentials["host"],
                                                          self.credentials["port"])


        self.collections = ["synapses", "skeletons", "hemi_lineages"]


        self.synapse = {"x": None,
                        "y": None,
                        "z": None,
                        "synapse_id": None,
                        "skeleton_id": None,
                        "source_id": None,
                        "splits": None}

        self.skeleton = {"skeleton_id": None,
                       "hemi_lineage_id": None,
                       "nt_known": None}

        self.hemi_lineage = {"hemi_lineage_id": None,
                             "hemi_lineage_name": None,
                             "nt_guess": None}

        self.prediction = {"synapse_id": None,
                           "prediction": None}

        self.db_name = db_name


    def __get_client(self):
        client = MongoClient(self.auth_string, connect=False)
        return client

    def __get_db(self, db_name=None):
        if db_name is None:
            db_name = self.db_name
        client = self.__get_client()
        db = client[db_name]
        return db

    def __generate_synapse(self, x, y, z, synapse_id, skeleton_id, source_id):
        synapse = deepcopy(self.synapse)
        synapse["x"] = x
        synapse["y"] = y
        synapse["z"] = z
        synapse["synapse_id"] = synapse_id
        synapse["skeleton_id"] = skeleton_id
        synapse["source_id"] = str(source_id).upper()
        return synapse

    def __generate_skeleton(self, skeleton_id, hemi_lineage_id, nt_known):
        skeleton = deepcopy(self.skeleton)
        skeleton["skeleton_id"] = skeleton_id
        skeleton["hemi_lineage_id"] = hemi_lineage_id
        if isinstance(nt_known, list):
            skeleton["nt_known"] = sorted([str(nt).lower() for nt in nt_known])
        else:
            skeleton["nt_known"] = [str(nt_known).lower()]
        return skeleton

    def __generate_hemi_lineage(self, hemi_lineage_id, hemi_lineage_name, nt_guess):
        hemi_lineage = deepcopy(self.hemi_lineage)
        hemi_lineage["hemi_lineage_id"] = hemi_lineage_id
        hemi_lineage["hemi_lineage_name"] = str(hemi_lineage_name)
        if isinstance(nt_guess, list):
            hemi_lineage["nt_guess"] = sorted([str(nt).lower() for nt in nt_guess])
        else:
            hemi_lineage["nt_guess"] = [str(nt_guess).lower()]

        return hemi_lineage


    def __consolidate_unknown(self, hemi_lineage_name):
        # find unknown_ids:
        if hemi_lineage_name in unknown_hemi_lineage_names:
            return None
        return hemi_lineage_name

    def __get_skeleton_ids_query(self, skeleton_ids):
        query = {'skeleton_id': {'$in': skeleton_ids}}
        return query

    def __get_neurotransmitters_query(self, neurotransmitters):
        # Detach from db ordering
        nt_permutations = list(permutations(neurotransmitters))
        query = {"$or": 
                 [{'nt_known': nt_permutation} 
                     for nt_permutation in nt_permutations
                    ]
                 }
        return query

    def __get_positions_query(self, positions):
        query = {"$or": [{"$and": [{"z": round(z)},
                                   {"y": round(y)}, 
                                   {"x": round(x)}]} for z,y,x in positions]}
        return query

    def __get_synapse_ids_query(self, synapse_ids):
        query = {"synapse_id": {"$in": synapse_ids}}
        return query

    def __get_hemi_lineage_id_query(self, hemi_lineage_id):
        query = {"hemi_lineage_id": hemi_lineage_id}
        return query

    def __get_hemi_lineage_name_query(self, hemi_lineage_name):
        # Get hemi_lineage id:
        hemi_lineages_collection = self.__get_db()["hemi_lineages"]
        hemi_lineages = [doc for doc in hemi_lineages_collection.find({"hemi_lineage_name": hemi_lineage_name.upper()})]
        assert(len(hemi_lineages) == 1)
        hemi_lineage_id = hemi_lineages[0]["hemi_lineage_id"]
        return self.__get_hemi_lineage_id_query(hemi_lineage_id)

    def __get_split_name_query(self, split_name):
        query = {"$or": [{"splits.{}".format(split_name): "train"}, 
                         {"splits.{}".format(split_name): "test"}]}
        return query

    def create(self, overwrite=False):
        logger.info("Create new synister db {}".format(self.db_name))
        db = self.__get_db()

        if overwrite:
            for collection in self.collections:
                logger.info("Overwrite {}.{}...".format(self.db_name, collection))
                db.drop_collection(collection)

        # Synapses
        synapses = db["synapses"]
        logger.info("Generate indices...")
        synapses.create_index([("z", ASCENDING), ("y", ASCENDING), ("x", ASCENDING)],
                                name="pos",
                                sparse=True)

        synapses.create_index([("synapse_id", ASCENDING)],
                                name="synapse_id",
                                sparse=True)

        synapses.create_index([("skeleton_id", ASCENDING)],
                                name="skeleton_id",
                                sparse=True)


        # Skeletons
        skeletons = db["skeletons"]
        logger.info("Generate indices...")

        skeletons.create_index([("skeleton_id", ASCENDING)],
                                 name="skeleton_id",
                                 sparse=True)

        skeletons.create_index([("hemi_lineage_id", ASCENDING)],
                                 name="hemi_lineage_id",
                                 sparse=True)

        # Hemi lineages
        hemi_lineages = db["hemi_lineages"]
        hemi_lineages.create_index([("hemi_lineage_id", ASCENDING)],
                                     name="hemi_lineage_id",
                                     sparse=True)

    def copy(self, new_db_name):
        if new_db_name == self.db_name:
            raise ValueError("Choose new db name for copy")
        client = self.__get_client()
        client.admin.command('copydb',
                             fromdb=self.db_name,
                             todb=new_db_name)

    def rename_collection(self, old_collection, new_collection):
        db = self.__get_db()
        db[old_collection].rename(new_collection)

    def get_synapses(self, skeleton_ids=None, neurotransmitters=None, positions=None, 
                     synapse_ids=None, hemi_lineage_name=None, hemi_lineage_id=None,
                     split_name=None):
        '''Get all the synapses in the DB.

        Args:

            skeleton_ids (list of int, optional):

                Return only synapses for the given skeletons.

            neurotransmitters (tuple of string, optional):

                Return only synapses that have the given combination of
                neurotransmitters.

            synapse_ids (list of ints, optional):

                Return only synapses with the given synapse_id

            positions (list of tuples of int (``z``, ``y``, ``x``), optional):
                
                Return only synapses with the given position

            hemi_lineage_name (string, optional):

                Return only synapses of the given hemi_lineage name

            hemi_lineage_id (int, optional):
                
                Return only synapses of the given hemi_lineage_id

            split_name (string, optional):

                Return only synapses of the given split.
                 
        Returns:

            Dictionary from synapse ID to position (``x``, ``y``, ``z``),
            skeleton (``skeleton_id``), brain region (``brain_region``) 
            and splits.
        '''

        db = self.__get_db()

        # Get skeleton_ids:
        if hemi_lineage_name is not None:
            # Get skeleton IDs for hemi lineage
            skeleton_collection = db['skeletons']
            query = self.__get_hemi_lineage_name_query(hemi_lineage_name)
            result = skeleton_collection.find(query, 
                                              projection=['skeleton_id'])

            hemi_skeleton_ids = list(n["skeleton_id"] for n in result)

            if skeleton_ids is None:
                skeleton_ids = hemi_skeleton_ids
            else:
                skeleton_ids = list(set(skeleton_ids) & set(hemi_skeleton_ids))

        if hemi_lineage_name is not None:
            skeleton_collection = db['skeletons']
            query = self.__get_hemi_lineage_name_query(hemi_lineage_name)
            result = skeleton_collection.find(query, 
                                              projection=['skeleton_id'])

            hemi_skeleton_ids = list(n["skeleton_id"] for n in result)

            if skeleton_ids is None:
                skeleton_ids = hemi_skeleton_ids
            else:
                skeleton_ids = list(set(skeleton_ids) & set(hemi_skeleton_ids))

        if hemi_lineage_id is not None:
            skeleton_collection = db['skeletons']
            query = self.__get_hemi_lineage_id_query(hemi_lineage_id)
            result = skeleton_collection.find(query, 
                                              projection=['skeleton_id'])

            hemi_skeleton_ids = list(n["skeleton_id"] for n in result)

            if skeleton_ids is None:
                skeleton_ids = hemi_skeleton_ids
            else:
                skeleton_ids = list(set(skeleton_ids) & set(hemi_skeleton_ids))

        if neurotransmitters is not None:
            if not isinstance(neurotransmitters, tuple):
                raise TypeError("Neurotransmitters must be a tuple of strings")

            # get skeleton IDs for neurotransmitters
            skeleton_collection = db['skeletons']
            query = self.__get_neurotransmitters_query(neurotransmitters)
            result = skeleton_collection.find(query,
                                              projection=['skeleton_id'])
            nt_skeleton_ids = list(n['skeleton_id'] for n in result)

            # intersect with skeleton_ids
            if skeleton_ids is None:
                skeleton_ids = nt_skeleton_ids
            else:
                skeleton_ids = list(set(skeleton_ids) & set(nt_skeleton_ids))

        # Construct query
        query = [{}]

        if synapse_ids is not None:
            query += [self.__get_synapse_ids_query(synapse_ids)]

        if positions is not None:
            query += [self.__get_positions_query(positions)]

        if skeleton_ids is not None:
            query += [self.__get_skeleton_ids_query(skeleton_ids)]

        if split_name is not None:
            query += [self.__get_split_name_query(split_name)]

        synapse_collection = db['synapses']
        query = {"$and": [q for q in query]}
        result = synapse_collection.find(query)

        synapses = {
            synapse['synapse_id']: {
                k: synapse[k]
                for k in [
                    'x', 'y', 'z',
                    'skeleton_id',
                    'brain_region',
                    'splits'
                ]
            }
            for synapse in result
        }

        return synapses

    def get_skeletons(self, skeleton_ids=None, neurotransmitters=None,
                      synapse_ids=None, positions=None, hemi_lineage_name=None,
                      hemi_lineage_id=None):
        '''Get all the skeletons in the DB.

        Args:

            skeleton_ids (list of int, optional):

                Return only skeletons for the given skeleton ids.

            neurotransmitters (tuple of string, optional):

                Return only skeletons that have the given combination of
                neurotransmitters.

            synapse_ids (list of ints, optional):

                Return only skeletons that have synapses with the given synapse_ids

            positions (list of tuples of int (z, y, x)):
                
                Return only skeletons that have synapses with the given position

            hemi_lineage_name (string):

                Return only skeletons of the given hemi_lineage name

            hemi_lineage_id (int):

                Return only skeletons of the given hemi_lineage id
                 

        Returns:

            Dictionary from skeleton ID to hemi_lineage_id and nt_known.
        '''

        db = self.__get_db()

        get_synapses_kwargs = {}
        if synapse_ids is not None:
            get_synapses_kwargs = {**get_synapses_kwargs, **{"synapse_ids": synapse_ids}}

        if positions is not None:
            get_synapses_kwargs = {**get_synapses_kwargs, **{"positions": positions}}

        if get_synapses_kwargs:
            synapses = self.get_synapses(**get_synapses_kwargs)
            synapse_skeleton_ids = set([s["skeleton_id"] for s in synapses.values()])

            if skeleton_ids is None:
                skeleton_ids = list(synapse_skeleton_ids)
            else:
                skeleton_ids = list(synapse_skeleton_ids & set(skeleton_ids))

        # Construct query
        query = [{}]

        if skeleton_ids is not None:
            query += [self.__get_skeleton_ids_query(skeleton_ids)]

        if neurotransmitters is not None:
            if not isinstance(neurotransmitters, tuple):
                raise TypeError("Neurotransmitters must be a tuple of strings")

            query += [self.__get_neurotransmitters_query(neurotransmitters)]

        if hemi_lineage_name is not None:
            query += [self.__get_hemi_lineage_name_query(hemi_lineage_name)]

        if hemi_lineage_id is not None:
            query += [self.__get_hemi_lineage_id_query(hemi_lineage_id)]

        skeleton_collection = db["skeletons"]
        query = {"$and": [q for q in query]}
        result = skeleton_collection.find(query)

        skeletons = {
            skeleton['skeleton_id']: {
                'hemi_lineage_id': skeleton['hemi_lineage_id'],
                'nt_known': tuple(sorted(skeleton['nt_known']))
            }
            for skeleton in result
        }

        return skeletons

    def get_hemi_lineages(self):
        '''Returns a list of all hemi-lineages in the DB.'''

        db = self.__get_db()

        hemi_lineage_collection = db["hemi_lineages"]
        result = hemi_lineage_collection.find({})
        
        hemi_lineages = {
                hl["hemi_lineage_id"]: {
                    "nt_guess": sorted(hl["nt_guess"]), 
                    "hemi_lineage_name": self.__consolidate_unknown(hl["hemi_lineage_name"])
                }
                for hl in result
        }

        return hemi_lineages

    def initialize_prediction(self, 
                              split_name,
                              experiment,
                              train_number,
                              predict_number):

        db = self.__get_db(self.db_name + "_predictions")
        predictions = db["{}_{}_t{}_p{}".format(split_name, 
                                                experiment,
                                                train_number,
                                                predict_number)]

        # Existence check:
        if predictions.count_documents({}) > 0:
            db.drop_collection(predictions)

        synapses_in_split = self.get_synapses(split_name=split_name)

        train_synapses = []
        test_synapses = []
        for synapse_id, synapse in synapses_in_split.items():
            if synapse["splits"][split_name] == "train":
                train_synapses.append(synapse_id)
            elif synapse["splits"][split_name] == "test":
                test_synapses.append(synapse_id)
            else:
                raise ValueError("Split corrupted, abort")
        
        prediction_documents = []
        for synapse_id in test_synapses:
            prediction_document = deepcopy(self.prediction)
            prediction_document["synapse_id"] = synapse_id
            prediction_documents.append(prediction_document)

        predictions.insert_many(prediction_documents)

    def write_prediction(self, 
                         split_name,
                         prediction,
                         experiment,
                         train_number,
                         predict_number,
                         x,
                         y,
                         z):

        db = self.__get_db(self.db_name + "_predictions")
        predictions = db["{}_{}_t{}_p{}".format(split_name, 
                                                experiment,
                                                train_number,
                                                predict_number)]

        # Update prediction:
        synapse_in_db = self.get_synapses(positions=[(z,y,x)])

        assert(len(synapse_in_db) == 1)
        synapse_id = list(synapse_in_db.keys())[0]

        result = predictions.update_one({"synapse_id": synapse_id},
                                        {"$set": {"prediction": list(prediction)}})

        if not (result.matched_count == 1):
            raise ValueError("Error, none or multiple matching synapses in split {}".format(split_name))
 
    def count_predictions(self,
                          split_name,
                          experiment,
                          train_number,
                          predict_number):

        db = self.__get_db(self.db_name + "_predictions")
        predictions = db["{}_{}_t{}_p{}".format(split_name, 
                                                experiment,
                                                train_number,
                                                predict_number)]

        total = predictions.count_documents({})
        done = predictions.count_documents({"prediction": {"$ne": None}})

        return done, total

    def make_split(self,
                   split_name,
                   train_synapse_ids,
                   test_synapse_ids):
        
        db = self.__get_db()
        synapses = db["synapses"]

        synapses.update_many({"synapse_id": {"$in": train_synapse_ids}},
                             {"$set": {"splits.{}".format(split_name): "train"}})

        synapses.update_many({"synapse_id": {"$in": test_synapse_ids}},
                             {"$set": {"splits.{}".format(split_name): "test"}})
