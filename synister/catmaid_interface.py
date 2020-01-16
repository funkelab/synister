import pymaid
import configparser
import os
import numpy as np
import vispy
import plotly.io as pio
import plotly
import json
import random
from synister.synister_db import SynisterDb
pio.renderers.default = "browser"



class Catmaid(object):
    def __init__(self, credentials=os.path.join(os.path.abspath(os.path.dirname(__file__)), "../catmaid_credentials.ini")):
        self.credentials = credentials
        self.instance = self.__get_instance(self.credentials)
        self.volumes = self.__get_volumes()

    def __get_instance(self, credentials):
        with open(credentials) as fp:
            config = configparser.ConfigParser()
            config.readfp(fp)
            user = config.get("Credentials", "user")
            password = config.get("Credentials", "password")
            token = config.get("Credentials", "token")

            rm = pymaid.CatmaidInstance('https://neuropil.janelia.org/tracing/fafb/v14/',
                                        user,
                                        password,
                                        token)
        return rm


    def __get_volumes(self):
        # All volumes in instance
        volumes = pymaid.get_volume()

        # Filter out trash (user id 55 is safe)
        volumes = volumes.loc[volumes['user_id']==55]

        # Filter out v14 prefixes:
        volumes = volumes.loc[~volumes["name"].str.contains('14')]

        volumes = list(volumes["name"])

        return volumes


    def get_volume(self, positions):
        """
        2D array: [[x, y, z], (...), ...]
                                    in (Catmaid != FAFB) world
                                    coordinates.

        returns: For each position a list of associated
        brain regions.
        """

        volumes = pymaid.in_volume(x=positions,
                                   volume=self.volumes)
        return volumes


    def plot_neurons(self, skeleton_ids, volumes=[], connectors=True):
        """
        Call with python -i (interactive mode)
        """
        volumes = [pymaid.get_volume(v) for v in volumes]
        data = pymaid.plot3d(skeleton_ids + volumes, connectors=connectors, backend='plotly')
        for trace in data["data"]:
            if "showlegend" in trace:
                trace["showlegend"] = False
        data["layout"]["scene"]["camera"] = {'eye': {'x': 0, 'y': 2, 'z': 0}}
        data["layout"]["scene"]["xaxis"]["range"] = [-700000,-350000]
        data["layout"]["scene"]["xaxis"]["dtick"] = 50000
        data["layout"]["scene"]["yaxis"]["range"] = [-200000, -50000]
        data["layout"]["scene"]["yaxis"]["dtick"] = 50000
        data["layout"]["scene"]["zaxis"]["range"] = [-350000, -50000]
        data["layout"]["scene"]["zaxis"]["dtick"] = 50000
        data["layout"]["scene"]["aspectmode"] = "cube"

        plotly.offline.plot(data, filename="plot_neurons", image="svg")

    def plot_brain_regions(self, test_brain_regions, train_brain_regions, connectors=True):
        test_volumes = [pymaid.get_volume(v, color=(255,0,0,.2)) for v in test_brain_regions]           #test is red
        train_volumes = [pymaid.get_volume(v, color=(0,255,0,.2)) for v in train_brain_regions]         #train is green
        data = pymaid.plot3d(test_volumes + train_volumes + overlap_volumes, connectors=connectors, backend='plotly')
        plotly.offline.plot(data, filename="plot_brain_regions_catmaid")

    def plot_split_by_neuron(self, db_credentials, db_name, split_name, neurotransmitter, test_boolean):
        db = SynisterDb(db_credentials, db_name)
        test_skids = set()
        train_skids = set()

        synapses = db.get_synapses(neurotransmitters=(neurotransmitter,))

        for synapse in synapses.values():
            skid = synapse["skeleton_id"]
            if synapse["splits"][split_name] == "train":
                train_skids.add(skid)
            else:
                test_skids.add(skid)

        test_skids = list(test_skids)
        train_skids = list(train_skids)
        if len(test_skids) > 25:
            test_skids = random.sample(test_skids, 25)
        if len(train_skids) >25:
            train_skids = random.sample(train_skids, 25)

        if test_boolean == True:
            self.plot_neurons(test_skids)
        else:
            self.plot_neurons(train_skids)

    def write_split_to_file(self, db_credentials, db_name, split_name, neurotransmitters, path):
        db = SynisterDb(db_credentials, db_name)

        f = open(path+"{}_by_nt.txt".format(split_name), "w+")
        all_test = set()
        all_train = set()
        not_in_catmaid = set()

        for nt in neurotransmitters:
            test_skids = set()
            train_skids = set()
            overlap = set()
            test_brain_regions = set()
            train_brain_regions = set()
            test_synapses = set()
            train_synapses = set()

            synapses = db.get_synapses(neurotransmitters=(nt,))

            for synapse_id, synapse in synapses.items():
                skid = synapse["skeleton_id"]

                if skid not in all_test and skid not in all_train:                  #Checks if is in Catmaid
                    try:
                        pymaid.get_neurons([skid])

                        if synapse["splits"][split_name] == "train":
                            train_skids.add(skid)
                            all_train.add(skid)
                            train_brain_regions.update(synapse["brain_region"])
                            train_synapses.add(synapse_id)
                            if skid in test_skids:
                                overlap.add(skid)
                        else:
                            test_skids.add(skid)
                            all_test.add(skid)
                            test_brain_regions.update(synapse["brain_region"])
                            test_synapses.add(synapse_id)
                            if skid in train_skids:
                                overlap.add(skid)


                    except:
                        not_in_catmaid.add(skid)
                        print("Skid not in catmaid {}".format(skid))





            test_skids = list(test_skids)
            train_skids = list(train_skids)

            f.write("Neurotransmitter: {}\n".format(nt))
            f.write("Test skids: {}\n".format(test_skids))
            f.write("Train skids: {}\n".format(train_skids))
            f.write("Overlap: {}\n".format(overlap))
            if split_name == "brain_region":
                f.write("Test brain regions: {}\n".format(test_brain_regions))
                f.write("Train brain regions: {}\n".format(train_brain_regions))
                # f.write("Test synapses: {}\n".format(test_synapses))
                # f.write("Train synapses: {}\n".format(train_synapses))
            f.write("\n\n")

        f.write("All test: {}\n".format(list(all_test)))
        f.write("All train: {}\n".format(list(all_train)))
        print(not_in_catmaid)


    def write_synapse_locations(self, db_credentials, db_name, split_name, neurotransmitters, path):
        db = SynisterDb(db_credentials, db_name)

        data = {}

        for nt in neurotransmitters:
            data[nt] = {"test":{},"train":{}}

            synapses = db.get_synapses(neurotransmitters=(nt,))

            for synapse_id, synapse in synapses.items():
                skid = synapse["skeleton_id"]
                synapse_location = tuple((synapse["x"]/10000.0, synapse["z"]/10000.0, synapse["y"]/-10000.0))

                if skid not in data[nt][synapse["splits"][split_name]]:
                    data[nt][synapse["splits"][split_name]][skid] = []

                data[nt][synapse["splits"][split_name]][skid].append(synapse_location)

        with open(path+"{}_synapse_locations.json".format(split_name), "w") as f:
            json.dump(data, f)
