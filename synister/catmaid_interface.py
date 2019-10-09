import pymaid
import configparser
import os
import numpy as np
import vispy
import plotly.io as pio
import plotly
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
        plotly.offline.plot(data) 
