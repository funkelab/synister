from synister import SynisterDb
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument(
    "--database",
    type=str,
    help="MongoDB database name (will be overwritten)",
    default="synister_fafb_v3",
)
parser.add_argument("--credentials", "-c", type=str, help="MongoDB credential file")

if __name__ == "__main__":

    args = parser.parse_args()
    db = SynisterDb(args.credentials, args.database)
    db.create(overwrite=True)

    collections = {
        "hemi_lineages": {
            "source": "hemi_lineages_v3.json",
            "record_template": "hemi_lineage",
            "id_field": "hemi_lineage_id",
        },
        "skeletons": {
            "source": "skeletons_v3.json",
            "record_template": "skeleton",
            "id_field": "skeleton_id",
        },
        "synapses": {
            "source": "synapses_v3.json",
            "record_template": "synapse",
            "id_field": "synapse_id",
        },
    }

    records = {}
    for coll, props in collections.items():
        with open(props["source"], "r") as f:
            documents = json.load(f)
        records[coll] = [
            {**getattr(db, props["record_template"]), props["id_field"]: int(k), **v}
            for k, v in documents.items()
        ]

    db.write(**records)
