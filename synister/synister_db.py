from pymongo import MongoClient, IndexModel, ASCENDING
from configparser import ConfigParser
from copy import deepcopy
import logging

logger = logging.getLogger(__name__)


class SynisterDB(object):
    def __init__(self, credentials):
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


        # Extend if needed
        self.super_classes = ["NA", "HEMI"]


    def create(self, db_name, overwrite=False):
        logger.info("Create new synister db {}".format(db_name))
        db = self.__get_db(db_name)

        if overwrite:
            for collection in self.collections:
                logger.info("Overwrite {}.{}...".format(db_name, collection))
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
        neuron["super_id"] = str(super_id).upper()
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

    def add_synapse(self, db_name, x, y, z, synapse_id, skeleton_id,
                    nt_known, source_id, nt_guess=None, super_id=None, mode="unique"):
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

        modes = ["unique", "overwrite"]

        if not mode in modes:
            raise ValueError("Mode has to be one of {}".format(modes))

        """
        A duplicate is a synapse with matching synapse id.

        unique: Fail upon insert of duplicate synapse 
        that doesn't match records. If records match
        the synapse is skipped.

        overwrite: Overwrite records of duplicate
        synapses.

        Does not affect supers and neurons. All added 
        synapses must be consistent with already present 
        supers or must have a different i.e. new super/skeleton_id.
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
        if not isinstance(db_name, str):
            raise ValueError("db_name must be str")

        logger.info("Add synapse in DB {} in {} mode...".format(db_name, mode))

        db = self.__get_db(db_name)

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

                if mode == "unique": 
                    assert(x_known_in_db == x_to_add)
                    assert(y_known_in_db == y_to_add)
                    assert(z_known_in_db == z_to_add)
                    assert(skeleton_id_in_db == skeleton_id_to_add)
                    assert(source_id_in_db == source_id_to_add)

                else: # mode == overwrite
                    synapses.update_one({"synapse_id": synapse_id}, 
                                        {"$set": [{"x": x_to_add}, 
                                                  {"y": y_to_add}, 
                                                  {"z": z_to_add},
                                                  {"skeleton_id": skeleton_id_to_add},
                                                  {"source_id": source_id_to_add}
                                                 ]
                                        })
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

                assert(nt_known_in_db == nt_known_to_add)
                assert(super_id_in_db == super_id_to_add)

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

                    assert(nt_guess_in_db == nt_guess_to_add)

            else: # count == 0
                supers.insert_one(super_entry)


    def __get_client(self):
        client = MongoClient(self.auth_string, connect=False)
        return client

    def __get_db(self, db_name):
        client = self.__get_client()
        db = client[db_name]
        return db

    def __get_collection(self, db_name, collection, overwrite=False):
        logger.info("Get client...")
        client = self.get_client()

        db = self.get_db(db_name)
        collections = db.collection_names()

        if collection in collections:
            if overwrite:
                logger.info("Warning, overwrite collection {}...".format(collection))
                self.create_collection(name_db=db_name,
                                       collection=collection,
                                       overwrite=True)

                # Check that collection is empty after overwrite:
                assert(db[collection].find({}).count() == 0)

            else:
                logger.info("Collection already exists, request {}.{}...".format(db_name, collection))
        else:
            logger.info("Collection does not exist, create...")
            self.create_collection(name_db=db_name,
                                   collection=collection,
                                   overwrite=False)

        collection_handle = db[collection]

        return collection_handle
