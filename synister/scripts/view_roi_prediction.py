import daisy
import neuroglancer
import numpy as np
import sys
import os
import configparser
from funlib.show.neuroglancer import add_layer, ScalePyramid
import configargparse
try:
    from synister.synister_db import SynisterDB
    branch = "master"
except ImportError:
    from synister.synister_db import SynisterDb
    branch = "refactor"
import synister.evaluate
import webbrowser

neuroglancer.set_server_bind_address('0.0.0.0')
p = configargparse.ArgParser()

p.add('-r', required=False,
      help='raw dataset path',
      default='/nrs/saalfeld/FAFB00/v14_align_tps_20170818_dmg.n5')
p.add('-p', required=False,
      help='prediction container', default="")
p.add('-d', required=False,
      help='prediction dataset', default="")

options = p.parse_args()
raw_file = options.r
prediction = None
if options.p and options.d:
    prediction_file = options.p
    prediction_dset = options.d
    prediction = daisy.open_ds(prediction_file, prediction_dset)


raw = [
    daisy.open_ds(raw_file, 'volumes/raw/s%d'%s)
    for s in range(17)
]


viewer = neuroglancer.Viewer()
with viewer.txn() as s:
    if prediction is not None:
        add_layer(s, prediction, 'prediction')
    add_layer(s, raw, 'raw')

print(viewer)
