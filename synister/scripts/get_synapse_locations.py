from synister.catmaid_interface import Catmaid
import numpy as np
from get_neurotransmitter import catmaid_transform
import configargparse
import csv

p = configargparse.ArgParser()

p.add("-s", required=True,
      help="FAFB skid")
p.add("-o", required=True,
      help="out file")
p.add("-n", action='store_true',
      required=False,
      help="neuroglancer format")

options = p.parse_args()
skid = int(options.s)
out_file = options.o
ng_format = bool(options.n)

print(skid, out_file, ng_format)

cm = Catmaid()
positions, ids = cm.get_synapse_positions(skid)
positions = catmaid_transform(positions)
voxel_size = np.array([40,4,4])
header = ["z", "y", "x"]

if ng_format:
    positions = [p[::-1]/voxel_size[::-1] for p in positions]
    header = ["x", "y", "z"]

with open(out_file, 'w') as f:
    file_writer = csv.writer(f)
    file_writer.writerow(header)
    for k in range(len(positions)):
        file_writer.writerow([positions[k][i] for i in range(3)])
