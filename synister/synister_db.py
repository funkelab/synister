from pymongo import MongoClient, IndexModel, ASCENDING
from configparser import ConfigParser
from copy import deepcopy
import logging
import time

logger = logging.getLogger(__name__)

unknown_hemi_lineage_names = [
    'NA',
    'NONE',
    'NEW'
] + ['UNKNOWN%d' % d for d in range(10)]


class SynisterDB(object):
    def __init__(self, credentials, db_name):
        with open(credentials) as fp:
            config = ConfigParser()
            config.readfp(fp)
            self.credentials = {}
            self.credentials["user"] = config.get("Credentials", "user")
            self.credentials["password"] = config.get("Credentials", "password")
            self.credentials["host"] = config.get("Credentials", "host")
            self.credentials["port"] = config.get("Credentials", "port")

        self.auth_string = 'mongodb://{}:{}@{}:{}'.format(self.credentials["user"],
                                                          self.credentials["password"],
                                                          self.credentials["host"],
                                                          self.credentials["port"])


        self.collections = ["synapses", "neurons", "supers"]


        self.synapse = {"x": None,
                        "y": None,
                        "z": None,
                        "synapse_id": None,
                        "skeleton_id": None,
                        "source_id": None}

        self.neuron = {"skeleton_id": None,
                       "super_id": None,
                       "nt_known": None}

        self.super = {"super_id": None,
                      "nt_guess": None}

        self.prediction = {"synapse_id": None,
                           "prediction": None}

        self.db_name = db_name

    def get_neurons(self):

        neurons = self.get_collection('neurons')
        neurons = {
            neuron['skeleton_id']: {
                # TODO: remove [0] when DB updated to have only one super (->
                # hemi_lineage)
                'super_id': self.__consolidate_unknown(neuron['super_id'][0]),
                'nt_known': tuple(sorted(neuron['nt_known']))
            }
            for neuron in neurons
        }

        return neurons

    def get_synapses(self, skeleton_ids=None, neurotransmitters=None):
        '''Get all the synapses in the DB.

        Args:

            skeleton_ids (list of int, optional):

                Return only synapses for the given skeletons.

            neurotransmitters (tuple of string, optional):

                Return only synapses that have the given combination of
                neurotransmitters.

        Returns:

            Dictionary from synapse ID to position (``x``, ``y``, ``z``),
            skeleton (``skeleton_id``), and brain region (``brain_region``).
        '''

        db = self.__get_db()

        if neurotransmitters is not None:

            # get skeleton IDs for neurotransmitters
            neuron_collection = db['neurons']
            result = neuron_collection.find({
                    'nt_known': tuple(sorted(neurotransmitters))
                },
                projection=['skeleton_id'])
            nt_skeleton_ids = list(n['skeleton_id'] for n in result)

            # intersect with skeleton_ids
            if skeleton_ids is None:
                skeleton_ids = nt_skeleton_ids
            else:
                skeleton_ids = list(set(skeleton_ids) & set(nt_skeleton_ids))

        synapse_collection = db['synapses']

        if skeleton_ids is None:
            result = synapse_collection.find({})
        else:
            result = synapse_collection.find({'skeleton_id': {'$in': skeleton_ids}})

        synapses = {
            synapse['synapse_id']: {
                k: synapse[k]
                for k in [
                    'x', 'y', 'z',
                    'skeleton_id',
                    'brain_region'
                ]
            }
            for synapse in result
        }

        return synapses

    def __consolidate_unknown(self, name):

        if name in unknown_hemi_lineage_names:
            return None
        return name

    def get_synapse_by_position(self, x, y, z):
        db = self.__get_db()
        synapses = db["synapses"]

        matching_synapses = synapses.find({"$and": [{"z": round(z)},{"y": round(y)}, {"x": round(x)}]})
        synapse_documents = []
        for synapse in matching_synapses:
            synapse_documents.append(synapse)
        
        if len(synapse_documents) > 1:
            raise ValueError(
                "Database compromised, two synapses with position "
                "({}, {}, {}) in {}".format(
                        x, y, z, self.db_name))

        return synapse_documents[0]


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
        if predictions.find({}).count() > 0:
            db.drop_collection(predictions)
        
        train_synapses, test_synapses = self.read_split(self.db_name,
                                                        split_name)

        prediction_documents = []
        for synapse in test_synapses:
            prediction_document = deepcopy(self.prediction)
            prediction_document["synapse_id"] = synapse["synapse_id"]
            prediction_documents.append(prediction_document)

        predictions.insert_many(prediction_documents)


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

        total = predictions.find({}).count()
        done = predictions.find({"prediction": {"$ne": None}}).count()

        return done, total


    def update_synapse(self, 
                       synapse_id,
                       key,
                       value):

        db = self.__get_db()
        synapses = db["synapses"]
        synapses.update_one({"synapse_id": synapse_id},
                      {"$set": {key: value}})


    def remove_from_split(self,
                          split_name,
                          synapse_id):

        db = self.__get_db()
        synapses = db["synapses"]

        synapses.update_one(
          {"synapse_id": synapse_id},
          {"$unset": {split_name:1}}
          )


    def get_brain_regions(self):

        synapses = self.get_collection("synapses")
        
        stats = {"size": len(synapses)}
        brain_regions = set([tuple(synapse["brain_region"]) for synapse in synapses])
        stats["distinct_brain_regions"] = len(brain_regions)
        brain_regions = {brain_region: 0 for brain_region in brain_regions}

        for synapse in synapses:
            brain_regions[tuple(synapse["brain_region"])] += 1

        stats["brain_regions"] = brain_regions

        return stats

    def get_hemi_lineages(self):
        '''Returns a list of all hemi-lineage names in the DB.'''

        neurons = self.get_neurons()
        hemi_lineages = set(n['super_id'] for n in neurons.values())
        if None in hemi_lineages:
            hemi_lineages.remove(None)
            has_unknown = True
        else:
            has_unknown = False

        hemi_lineages = sorted(list(hemi_lineages))

        if has_unknown:
            hemi_lineages.append(None)

        return hemi_lineages

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
        synapse_in_db = self.get_synapse_by_position(self.db_name,
                                                     x,
                                                     y,
                                                     z)

        result = predictions.update_one({"synapse_id": synapse_in_db["synapse_id"]},
                                        {"$set": {"prediction": list(prediction)}})

        if not (result.matched_count == 1):
            raise ValueError("Prediction failed to update, none or multiple matching synapses in split {}".format(split_name))
    

    def get_collection(self, collection_name):
        db = self.__get_db()
        collection = db[collection_name]

        collection_iterator = collection.find({})
        collection_documents = [c for c in collection_iterator]
        return collection_documents

    def get_neurotransmitters(self):
        neurons = self.get_collection("neurons")
        supers = self.get_collection("supers")

        nt_known = []
        for n in neurons:
            nts = tuple([nt for nt in n["nt_known"]])
            nt_known.append(nts)

        nt_guess = []
        for s in supers:
            nts = tuple([nt for nt in s["nt_guess"]])
            nt_guess.append(nts)

        return nt_known, nt_guess


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


        # Neurons
        neurons = db["neurons"]
        logger.info("Generate indices...")

        neurons.create_index([("skeleton_id", ASCENDING)],
                               name="skeleton_id",
                               sparse=True)

        # Supers
        supers = db["supers"]

    def add_synapse(self, x, y, z, synapse_id, skeleton_id,
                    nt_known, source_id, nt_guess=None, super_id=None):
        """
        Add a new synapse to the database. This entails generating 
        or adding a synapse, a neuron and optionally a super
        that specificies neurotransmitter guesses. The id 
        of the super should correspond to its identity. 
        Super ids are case insensitive. Existing supers 
        or neurons cannot be modified or overwritten with this 
        method, i.e. supers and neuron transmitter ids 
        have to be consistent before adding a new
        synapse.
        """

        if not isinstance(synapse_id, int):
            raise ValueError("Synapse id must be an integer type")
        if not isinstance(skeleton_id, int):
            raise ValueError("Skeleton id must be an integer type")

        if super_id is not None:
            if not isinstance(super_id, str):
                raise ValueError("Super id must be string or None type")

            super_id = super_id.upper()

        if not isinstance(source_id, str):
            raise ValueError("Source id must be a string")

        source_id = source_id.upper()

        if not isinstance(x, int):
            raise ValueError("x must be integer")
        if not isinstance(y, int):
            raise ValueError("y must be integer")
        if not isinstance(z, int):
            raise ValueError("z must be integer")
        if not isinstance(self.db_name, str):
            raise ValueError("db_name must be str")

        logger.info("Add synapse in DB {}".format(self.db_name))

        db = self.__get_db()

        synapses_to_insert = []
        neurons_to_insert = []
        supers_to_insert = []

        logger.info("Update synapses...")
        synapses = db["synapses"]
        synapse_entry = self.__generate_synapse(x, y, z, 
                                               synapse_id, 
                                               skeleton_id,
                                               source_id)

        synapse_in_db = synapses.find({"synapse_id": synapse_entry["synapse_id"]})
        assert(synapse_in_db.count()<=1)

        if synapse_in_db.count() == 1:
            for doc in synapse_in_db:
                x_known_in_db = doc["x"]
                y_known_in_db = doc["y"]
                z_known_in_db = doc["z"]
                skeleton_id_in_db = doc["skeleton_id"]
                source_id_in_db = doc["source_id"]

                x_to_add = synapse_entry["x"]
                y_to_add = synapse_entry["y"]
                z_to_add = synapse_entry["z"]
                skeleton_id_to_add = synapse_entry["skeleton_id"]
                source_id_to_add = synapse_entry["source_id"]

                if x_known_in_db != x_to_add:
                    if abs(x_known_in_db - x_to_add)<=19:
                        print("Almost Match detected, allow...")
                    else:
                        raise ValueError("Synapse {} already in db but new x position does not match (db: {}, to add: {})".format(synapse_id, x_known_in_db, x_to_add))

                if y_known_in_db != y_to_add:
                    if abs(y_known_in_db - y_to_add)<=19:
                        print("Almost Match detected, allow...")
                    else:
                        raise ValueError("Synapse {} already in db but new y position does not match (db: {}, to add: {})".format(synapse_id, y_known_in_db, y_to_add))

                if z_known_in_db != z_to_add:
                    if abs(z_known_in_db - z_to_add)<=19:
                        print("Almost Match detected, allow...")
                    else:
                        raise ValueError("Synapse {} already in db but new z position does not match (db: {}, to add: {})".format(synapse_id, z_known_in_db, z_to_add))

                if skeleton_id_in_db != skeleton_id_to_add:
                    raise ValueError("synapse {} already in db but assigned to a different skeleton".format(synapse_id))

        else: # count == 0
            synapses.insert_one(synapse_entry)

        logger.info("Update neurons...")
        neurons = db["neurons"]
        neuron_entry = self.__generate_neuron(skeleton_id, super_id, nt_known)

        neuron_in_db = neurons.find({"skeleton_id": neuron_entry["skeleton_id"]})
        assert(neuron_in_db.count()<=1)
        
        if neuron_in_db.count() == 1:
            for doc in neuron_in_db:
                nt_known_in_db = set(doc["nt_known"])
                super_id_in_db = doc["super_id"]

                nt_known_to_add = set(neuron_entry["nt_known"])
                super_id_to_add = neuron_entry["super_id"]
                
                if nt_known_in_db != nt_known_to_add:
                    if nt_known_to_add == set(['none']):
                        pass
                    elif nt_known_in_db == set(['none']):
                        print("Update nt known")
                        neurons.update_one({"skeleton_id": neuron_entry["skeleton_id"]}, {"$set": {"nt_known": neuron_entry["nt_known"]}})
                    else:
                        raise ValueError("neuron {} already in db but has different known neurotransmitters (db: {}, to add: {})".format(skeleton_id, nt_known_in_db, nt_known_to_add))

                if set(super_id_in_db) != set(super_id_to_add):
                    if not (set(super_id_to_add) == set(['NONE']) or "NA" in super_id_to_add[0]):
                        raise ValueError("neuron {} already in db but assigned to a different super {} (new), {} (in db)".format(skeleton_id,super_id_to_add, super_id_in_db))
                    else:
                        print("Super id not known in new ({}) but known in db ({}), keep known".format(super_id_to_add, super_id_in_db))

        else: # count == 0
            neurons.insert_one(neuron_entry)

        logger.info("Update supers...")
        if super_id is None:
            assert(nt_guess is None)
        else:
            supers = db["supers"]
            super_entry = self.__generate_super(super_id, nt_guess)
            
            super_in_db = supers.find({"super_id": super_id})
            assert(super_in_db.count()<=1)

            if super_in_db.count() == 1:
                for doc in super_in_db:
                    nt_guess_in_db = set(doc["nt_guess"])
                    nt_guess_to_add = set(doc["nt_guess"])

                    if nt_guess_in_db != nt_guess_to_add:
                        raise ValueError("super {} already in db but has different neurotransmitter guess".format(super_id))
 

            else: # count == 0
                supers.insert_one(super_entry)



    def get_synapse(self, 
                    synapse_id):

        db = self.__get_db()

        synapses = db["synapses"]
        matching_synapses = synapses.find({"synapse_id": synapse_id})
        synapse_documents = []
        for synapse_doc in matching_synapses:
            synapse_documents.append(synapse_doc)

        if len(synapse_documents) > 1:
            raise ValueError("Found more than one synapses with id {}, abort.".format(synapse_id))
        elif len(synapse_documents) == 0:
            raise ValueError(
                "No synapse with id {} in db {}".format(synapse_id, self.db_name))

        synapse = synapse_documents[0]

        neurons = db["neurons"]
        matching_neurons = neurons.find({"skeleton_id": synapse["skeleton_id"]})
        neuron_documents = []
        for neuron_doc in matching_neurons:
            neuron_documents.append(neuron_doc)

        if len(neuron_documents) > 1:
            raise ValueError("Found more than one neuron with skid {} for synapse {}".format(synapse["skeleton_id"], 
                                                                                             synapse["synapse_id"]))
        elif len(neuron_documents) == 0:
            raise ValueError(
                "No neuron with skid {} in db {}".format(
                    synapse["skeleton_id"], self.db_name))
        neuron = neuron_documents[0]


        if neuron["super_id"] != ["NONE"]:
            supers = db["supers"]
            matching_supers = supers.find({"super_id": neuron["super_id"]})
            super_documents = []
            for super_doc in matching_supers:
                super_documents.append(super_doc)

            if len(super_documents) > 1:
                raise ValueError("Found more than one super with super_id {} for synapse {}".format(neuron["super_id"], 
                                                                                                    synapse["synapse_id"]))
            elif len(super_documents) == 0:
                raise ValueError(
                    "No super with super_id {} in db {}".format(
                        neuron["super_id"], self.db_name))
            super_ = super_documents[0]

        else:
            super_ = {"super_id": "NONE",
                      "nt_guess": ["NONE"]}

        synapse = {**synapse, **neuron, **super_}

        return synapse

    
    
    def make_split(self,
                   split_name,
                   train_synapse_ids,
                   test_synapse_ids):

        
        db = self.__get_db()
        synapses = db["synapses"]

        synapses.update_many({"synapse_id": {"$in": train_synapse_ids}},
                             {"$set": {split_name: "train"}})

        synapses.update_many({"synapse_id": {"$in": test_synapse_ids}},
                             {"$set": {split_name: "test"}})


    def read_split(self, 
                   split_name):

        db = self.__get_db()
        synapses = self.get_collection("synapses")

        train_synapses = []
        test_synapses = []
        for synapse in synapses:
            try:
                if synapse[split_name] == "train":
                    train_synapses.append(synapse)
                elif synapse[split_name] == "test":
                    test_synapses.append(synapse)
                else:
                    raise ValueError("Split {} corrupted, abort".format(split_name))
            except KeyError:
                pass

        return train_synapses, test_synapses


    def get_synapse_locations(self, split_name, split, neurotransmitter):
        """
        neurotransmitter: tuple
        """
        assert(isinstance(neurotransmitter, tuple))

        if not split in ["train", "test"]:
            raise ValueError("Split must be either train or test")

        nt_known, nt_guess = self.get_neurotransmitters(self.db_name)
        nts = nt_known
        if not neurotransmitter in nts:
            raise ValueError("{} not in database.".format(neurotransmitter))

        synapses = self.get_synapses_by_nt(self.db_name,
                                           [neurotransmitter])
        
        locations = []
        for synapse in synapses[neurotransmitter]:
            in_split = False
            try:
                in_split = (synapse[split_name] == split)
            except:
                pass

            if in_split:
                location = [int(synapse["z"]), 
                            int(synapse["y"]), 
                            int(synapse["x"])]
                locations.append(location)
       
        return locations


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

    def __generate_neuron(self, skeleton_id, super_id, nt_known):
        neuron = deepcopy(self.neuron)
        neuron["skeleton_id"] = skeleton_id
        neuron["super_id"] = [str(super_id).upper()]
        if isinstance(nt_known, list):
            neuron["nt_known"] = [str(nt).lower() for nt in nt_known]
        else:
            neuron["nt_known"] = [str(nt_known).lower()]
        return neuron

    def __generate_super(self, super_id, nt_guess):
        """
        A super is a superset of neurons and thus
        includes, but is not limited to, hemilineages.
        """

        super_ = deepcopy(self.super)
        super_["super_id"] = str(super_id)
        if isinstance(nt_guess, list):
            super_["nt_guess"] = [str(nt).lower() for nt in nt_guess]
        else:
            super_["nt_guess"] = [str(nt_guess).lower()]

        return super_
