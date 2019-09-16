from pymongo import MongoClient, IndexModel, ASCENDING
from configparser import ConfigParser
from copy import deepcopy
import logging

logger = logging.getLogger(__name__)


class SynisterDB(object):
    def __init__(self, credentials):
        with open(credentials) as fp:
            config = ConfigParser.ConfigParser()
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


        self.collections = ["synapses", "neurons", "hemilineages"]


        self.synapse = {"x": None,
                        "y": None,
                        "z": None,
                        "synapse_id": None,
                        "skeleton_id": None}

        self.neuron = {"skeleton_id": None,
                       "hemi_lineage_id": None,
                       "nt_known": None}

        self.hemilineage = {"hemi_lineage_id": None,
                            "nt_guess": None}


    def create(self, db_name, overwrite=False):
        logger.info("Create new synister db {}".format(name_db))
        client = self.get_client()
        db = self.get_db(name_db)

        if overwrite:
            for collection in self.collections:
                logger.info("Overwrite {}.{}...".format(name_db, collection))
                db.drop_collection(collection)

        # Synapses
        synapses = db["synapses"]
        logger.info("Generate indices...")
        synapses.create_index([("z", ASCENDING), ("y", ASCENDING), ("x", ASCENDING)],
                                name="pos",
                                sparse=True)

        synapses.create_index(("synapse_id", ASCENDING),
                           name="synapse_id",
                           sparse=True)

        synapses.create_index(("skeleton_id", ASCENDING),
                               name="skeleton_id",
                               sparse=True)


        # Neurons
        neurons = db["neurons"]
        logger.info("Generate indices...")

        neurons.create_index(("skeleton_id", ASCENDING),
                               name="skeleton_id",
                               sparse=True)

        neurons.create_index(("hemilineage_id", ASCENDING),
                               name="hemilineage_id",
                               sparse=True)


        # Hemilineages
        hemilineagess = db["hemilineages"]
        logger.info("Generate indices...")

        hemilineages.create_index(("hemilineage_id", ASCENDING),
                                   name="hemilineage_id",
                                   sparse=True)

    def generate_synapse(self, x, y, z, synapse_id, skeleton_id):
        synapse = deepcopy(self.synapse)
        synapse["x"] = str(x)
        synapse["y"] = str(y)
        synapse["z"] = str(z)
        synapse["synapse_id"] = str(synapse_id)
        synapse["skeleton_id"] = str(skeleton_id)
        return synapse

    def generate_neuron(self, skeleton_id, hemi_lineage_id, nt_known):
        neuron = deepcopy(self.neuron)
        neuron["skeleton_id"] = str(skeleton_id)
        neuron["hemi_lineage_id"] = str(hemi_lineage_id)
        try:
            neuron["nt_known"] = [str(nt) for nt in nt_known]
        except:
            neuron["nt_known"] = [str(nt_known)]
        return neuron

    def generate_hemilineage(self, hemi_lineage_id, nt_guess):
        hemilineage = deepcopy(self.hemilineage)
        hemilineage["hemi_lineage_id"] = hemi_lineage_id
        try:
            hemilineage["nt_guess"] = [str(nt) for nt in nt_guess]
        except:
            hemilineage["nt_guess"] = [str(nt_guess)]
        return hemilineage
    
    def add(self, synapses=[], neurons=[], hemilineages=[]):
        client = self.get_client()
        db = self.get_db(name_db)

        logger.info("Add {} synapses, {} neurons and {} hemilineages...".format(len(synapses),
                                                                                len(neurons),
                                                                                len(hemilineages)))

        if synapses:
            collection = db["synapses"]
            collection.insert_many(synapses)

        if neurons:
            collection = db["neurons"]
            collection.insert_many(neurons)

        if hemilineages:
            collection = db["hemilineages"]
            collection.insert_many(hemilineages)


    def __get_client(self):
        client = MongoClient(self.auth_string, connect=False)
        return client

    def __get_db(self, name_db):
        client = self.get_client()
        db = client[name_db]
        return db

    def __get_collection(self, name_db, collection, overwrite=False):
        logger.info("Get client...")
        client = self.get_client()

        db = self.get_db(name_db)
        collections = db.collection_names()

        if collection in collections:
            if overwrite:
                logger.info("Warning, overwrite collection {}...".format(collection))
                self.create_collection(name_db=name_db,
                                       collection=collection,
                                       overwrite=True)

                # Check that collection is empty after overwrite:
                assert(db[collection].find({}).count() == 0)

            else:
                logger.info("Collection already exists, request {}.{}...".format(name_db, collection))
        else:
            logger.info("Collection does not exist, create...")
            self.create_collection(name_db=name_db,
                                   collection=collection,
                                   overwrite=False)

        collection_handle = db[collection]

        return collection_handle
