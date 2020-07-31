from funlib.run import run, run_singularity
from subprocess import check_call
import os
import sys


if __name__ == "__main__":
    skid_csv_path = sys.argv[1]
    queue = sys.argv[2]
    self_path = os.path.realpath(os.path.dirname(__file__))
    base_cmd = "python {} {}".format(os.path.join(self_path, "get_neurotransmitter.py"), skid_csv_path)

    if queue != "None":
        run(base_cmd,
            5,
            1,
            25600,
            ".",
            None,
            "",
            queue,
            "",
            False,
            ["/nrs, /scratch, /groups, /misc"],
            True,
            True)
    else:
        check_call(base_cmd, shell=True)
