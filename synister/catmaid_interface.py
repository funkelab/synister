import pymaid
import configparser
import os
import numpy as np
import vispy
import plotly.io as pio
import plotly
import random
from synister.synister_db import SynisterDB
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


    def plot_neurons(self, skeleton_ids, volumes=[], connectors=True):
        """
        Call with python -i (interactive mode)
        """
        volumes = [pymaid.get_volume(v) for v in volumes]
        data = pymaid.plot3d(skeleton_ids + volumes, connectors=connectors, backend='plotly')
        plotly.offline.plot(data, filename="plot_neurons", image="png")

    def plot_brain_regions(self, test_brain_regions, train_brain_regions, connectors=True):
        test_volumes = [pymaid.get_volume(v, color=(255,0,0,.2)) for v in test_brain_regions]           #test is red
        train_volumes = [pymaid.get_volume(v, color=(0,255,0,.2)) for v in train_brain_regions]         #train is green
        data = pymaid.plot3d(test_volumes + train_volumes + overlap_volumes, connectors=connectors, backend='plotly')
        plotly.offline.plot(data, filename="plot_brain_regions_catmaid", image="png")

    def plot_split_by_neuron(self, db_credentials, db_name, split_name, neurotransmitter):
        db = SynisterDB(db_credentials)
        skids = set()

        synapses = db.get_synapses_by_nt(db_name, [(neurotransmitter,)])[(neurotransmitter,)]
        for synapse in synapses:
            skid = synapse["skeleton_id"]
            skids.add(skid)
        skids = list(skids)
        if len(skids) > 25:
            skids = random.sample(skids, 25)
        self.plot_neurons(skids)
