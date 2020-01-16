from pymongo import MongoClient, IndexModel, ASCENDING
from configparser import ConfigParser
from copy import deepcopy
import logging
import time
from itertools import permutations
import os
from iteration_utilities import duplicates

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


        self.collections = ["synapses", "skeletons", "hemi_lineages", "meta"]

        self.synapse = {"x": None,
                        "y": None,
                        "z": None,
                        "synapse_id": None,
                        "skeleton_id": None,
                        "splits": None,
                        "prepost": None,
                        "meta_id": None}

        self.skeleton = {"skeleton_id": None,
                         "hemi_lineage_id": None,
                         "nt_known": None}

        self.hemi_lineage = {"hemi_lineage_id": None,
                             "hemi_lineage_name": None,
                             "nt_guess": None}

        self.prediction = {"synapse_id": None,
                           "prediction": None}

        self.meta = {"meta_id": None,
                     "group": None,
                     "tracer": None}

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

    def __generate_synapse(self, x, y, z, synapse_id, skeleton_id, prepost=None, meta_id=None):
        synapse = deepcopy(self.synapse)
        synapse["x"] = x
        synapse["y"] = y
        synapse["z"] = z
        synapse["synapse_id"] = synapse_id
        synapse["skeleton_id"] = skeleton_id
        synapse["meta_id"] = meta_id
        synapse["prepost"] = prepost
        return synapse

    def __generate_skeleton(self, skeleton_id, hemi_lineage_id, nt_known):
        skeleton = deepcopy(self.skeleton)
        skeleton["skeleton_id"] = skeleton_id
        skeleton["hemi_lineage_id"] = hemi_lineage_id
        if isinstance(nt_known, list):
            skeleton["nt_known"] = sorted([str(nt).lower() for nt in nt_known])
        else:
            if skeleton["nt_known"] is None:
                pass
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
            if hemi_lineage["nt_guess"] is None:
                pass
            else:
                hemi_lineage["nt_guess"] = [str(nt_guess).lower()]
        return hemi_lineage

    def __generate_meta(self, meta_id, group, tracer):
        meta = deepcopy(self.meta)
        meta["meta_id"] = int(meta_id)
        if not group is None:
            meta["group"] = group.lower()
        if not tracer is None:
            meta["tracer"] = tracer.lower()
        return meta

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
        query = {"$or": [{"$and": [{"z": int(round(z))},
                                   {"y": int(round(y))}, 
                                   {"x": int(round(x))}]} for z,y,x in positions]}
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
                         {"splits.{}".format(split_name): "test"},
                         {"splits.{}".format(split_name): "validation"}]}
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

    def write_synapse(self, synapse_id, skeleton_id, x, y, z, prepost=None, meta_id=None):
        db = self.__get_db()
        synapse_collection = db["synapses"]
        synapse_document = self.__generate_synapse(x, y, z, synapse_id, skeleton_id, prepost, meta_id)
        synapse_collection.insert_one(synapse_document)

    def write_skeleton(self, skeleton_id, hemi_lineage_id, nt_known):
        db = self.__get_db()
        skeleton_collection = db["skeletons"]
        skeleton_document = self.__generate_skeleton(skeleton_id, hemi_lineage_id, nt_known)
        skeleton_collection.insert_one(skeleton_document)

    def write_hemi_lineage(self, hemi_lineage_id, hemi_lineage_name, nt_guess):
        db = self.__get_db()
        hemi_lineage_collection = db["hemi_lineages"]
        hemi_lineage_document = self.__generate_hemi_lineage(hemi_lineage_id, hemi_lineage_name, nt_guess)
        hemi_lineage_collection.insert_one(hemi_lineage_document)

    def write_meta(self, meta_id, group, tracer):
        db = self.__get_db()
        meta_collection = db["meta"]
        meta_document = self.__generate_meta(meta_id, group, tracer)
        meta_collection.insert_one(meta_document)

    def write_many(self, synapses=None, skeletons=None, hemi_lineages=None, metas=None):
        db = self.__get_db()
        if synapses is not None:
            synapse_documents = [self.__generate_synapse(**synapse) for synapse in synapses]
            synapse_collection = db["synapses"]
            synapse_collection.insert_many(synapse_documents)
        if skeletons is not None:
            skeleton_documents = [self.__generate_skeleton(**skeleton) for skeleton in skeletons]
            skeleton_collection = db["skeletons"]
            skeleton_collection.insert_many(skeleton_documents)
        if hemi_lineages is not None:
            hemi_lineage_documents = [self.__generate_hemi_lineage(**hemi_lineage) for hemi_lineage in hemi_lineages]
            hemi_lineage_collection = db["hemi_lineages"]
            hemi_lineage_collection.insert_many(hemi_lineage_documents)
        if metas is not None:
            meta_documents = [self.__generate_meta(**meta) for meta in metas]
            meta_collection = db["meta"]
            meta_collection.insert_many(meta_documents)

    def validate_synapses(self):
        db = self.__get_db()
        synapse_collection = db["synapses"]
        all_synapses = [s for s in synapse_collection.find({})]

        # Check for duplicates:
        synapse_ids = [s["synapse_id"] for s in all_synapses]
        synapse_locs = [(s["x"], s["y"], s["z"]) for s in all_synapses]

        duplicate_synapse_ids = list(duplicates(synapse_ids))
        duplicate_synapse_locs = list(duplicates(synapse_locs))

        # Check that skeleton exists:
        skeleton_collection = db["skeletons"]
        unmatched_synapses = []
        for synapse in all_synapses:
            if skeleton_collection.count_documents({"skeleton_id": synapse["skeleton_id"]}) == 0:
                unmatched_synapses.append(synapse["synapse_id"])

        return {"id_duplicates": duplicate_synapse_ids,
                "loc_duplicates": duplicate_synapse_locs,
                "no_skid_match": unmatched_synapses}

    def validate_skeletons(self):
        db = self.__get_db()
        skeleton_collection = db["skeletons"]
        all_skeletons = [s for s in skeleton_collection.find({})]

        # Check for duplicates:
        skeleton_ids = [s["skeleton_id"] for s in all_skeletons]
        duplicate_skeleton_ids = list(duplicates(skeleton_ids))

        # Check that hemi_lineage exists:
        hemi_lineage_collection = db["hemi_lineages"]
        unmatched_skeletons = []
        for skeleton in all_skeletons:
            if hemi_lineage_collection.count_documents({"hemi_lineage_id": skeleton["hemi_lineage_id"]}) == 0:
                unmatched_skeletons.append(skeleton["skeleton_id"])

        return {"id_duplicates": duplicate_skeleton_ids,
                "no_hlid_match": unmatched_skeletons}

    def validate_hemi_lineages(self):
        db = self.__get_db()
        hemi_lineage_collection = db["hemi_lineages"]
        all_hemi_lineages = [h for h in hemi_lineage_collection.find({})]

        # Check for duplicates:
        hemi_lineage_ids = [h["hemi_lineage_id"] for h in all_hemi_lineages]
        hemi_lineage_names = [h["hemi_lineage_name"] for h in all_hemi_lineages]
        duplicate_hemi_lineage_ids = list(duplicates(hemi_lineage_ids))
        duplicate_hemi_lineage_names = list(duplicates(hemi_lineage_names))

        return {"id_duplicates": duplicate_hemi_lineage_ids,
                "name_duplicates": duplicate_hemi_lineage_names}

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

        try:
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
        except KeyError:
            result = synapse_collection.find(query)
            synapses = {
                synapse['synapse_id']: {
                    k: synapse[k]
                    for k in [
                        'x', 'y', 'z',
                        'skeleton_id',
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
                'nt_known': tuple(sorted(skeleton['nt_known'])) if skeleton['nt_known'] is not None else skeleton["nt_known"]
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
                    "nt_guess": tuple(sorted(hl['nt_guess'])) if hl['nt_guess'] is not None else hl["nt_guess"],
                    "hemi_lineage_name": self.__consolidate_unknown(hl["hemi_lineage_name"])
                }
                for hl in result
        }

        return hemi_lineages

    def get_predictions(self,
                        split_name,
                        experiment,
                        train_number,
                        predict_number):
        """Get all predictions in given run


            Returns:

                Dictionary of synapse_ids to predictions.

        """
        db = self.__get_db(self.db_name + "_predictions")
        prediction_collection = db["{}_{}_t{}_p{}".format(split_name, 
                                                          experiment,
                                                          train_number,
                                                          predict_number)]

        result = prediction_collection.find({})

        predictions = {p["synapse_id"]: {"prediction": p["prediction"]}
                       for p in result}

        return predictions
        
    def initialize_prediction(self, 
                              split_name,
                              experiment,
                              train_number,
                              predict_number,
                              overwrite=False,
                              validation=False):

        db = self.__get_db(self.db_name + "_predictions")
        predictions = db["{}_{}_t{}_p{}".format(split_name, 
                                                experiment,
                                                train_number,
                                                predict_number)]

        # Existence check:
        if predictions.count_documents({}) > 0:
            if overwrite:
                db.drop_collection(predictions)
            else:
                return 0 

        synapses_in_split = self.get_synapses(split_name=split_name)

        train_synapses = []
        test_synapses = []
        validation_synapses = []
        for synapse_id, synapse in synapses_in_split.items():
            if synapse["splits"][split_name] == "train":
                train_synapses.append(synapse_id)
            elif synapse["splits"][split_name] == "test":
                test_synapses.append(synapse_id)
            elif synapse["splits"][split_name] == "validation":
                validation_synapses.append(synapse_id)
            else:
                raise ValueError("Split corrupted, abort")

        if validation:
            test_synapses = validation_synapses
        
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

    def init_splits(self):
        db = self.__get_db()
        synapse_collection = db["synapses"]

        synapse_collection.update_many({}, {"$set": {"splits": {}}})

    def make_split(self,
                   split_name,
                   train_synapse_ids,
                   test_synapse_ids,
                   validation_synapse_ids=None):

        self.remove_split(split_name)
 
        db = self.__get_db()
        synapse_collection = db["synapses"]

        synapse_collection.update_many({"synapse_id": {"$in": train_synapse_ids}},
                             {"$set": {"splits.{}".format(split_name): "train"}})

        synapse_collection.update_many({"synapse_id": {"$in": test_synapse_ids}},
                             {"$set": {"splits.{}".format(split_name): "test"}})

        if validation_synapse_ids is not None:
            synapse_collection.update_many({"synapse_id": {"$in": validation_synapse_ids}},
                                           {"$set": {"splits.{}".format(split_name): "validation"}})


    def remove_split(self,
                     split_name):

        db = self.__get_db()
        synapse_collection = db["synapses"]

        synapse_collection.update_many({},
                                        {"$unset": 
                                        {"splits.{}".format(split_name): ""}
                                        }
                                       )

    def create_queryable(self, documents):
        db_name = "queryable"
        db = self.__get_db(db_name)
        collection_name = str(os.getpid())
        db.drop_collection(collection_name)
        collection = db[collection_name]
        collection.insert_many(documents)
        return collection

    def destroy_queryable(self, queryable):
        db = self.__get_db("queryable")
        db.drop_collection(queryable)

    def update_synapse(self,
                       synapse_id,
                       key,
                       value):
        db = self.__get_db()
        synapses = db["synapses"]
        synapses.update_one({"synapse_id": synapse_id},
                            {"$set": {key: value}})
