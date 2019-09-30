import os
import sys
from shutil import copyfile, rmtree
import configargparse
import configparser
import click

p = configargparse.ArgParser()
p.add('-d', '--base_dir', required=True, 
      help='base directory for storing synister experiments')
p.add('-e', required=True, help='name of the experiment, e.g. fafb')
p.add('-t', required=True, help='train number/id for this particular run')
p.add('-c', required=False, action='store_true', help='clean up - remove specified train setup')

def set_up_environment(base_dir,
                       experiment,
                       train_number,
                       clean_up=False):
    ''' Sets up the directory structure and config file for 
        training a network for microtubule prediction.

    Args:

        base_dir (``string``):

            The base directory for storing all micron related experiments and data.

        experiment (``string``):

            The name of the experiment this training run belongs to.

        train_number (``int``):

            The number/id of the training run.

        clean_up (``bool``):

            If true removes the specified train directory
    '''


    base_dir = os.path.expanduser(base_dir)
    setup_dir = os.path.join(base_dir, experiment, "02_train/setup_t{}".format(train_number))

    if clean_up:
        if __name__ == "__main__":
            if click.confirm('Are you sure you want to remove {} and all its contents?'.format(setup_dir), default=False):
                rmtree(setup_dir)
            else:
                print("Abort clean up.")
                return
        else:
            rmtree(setup_dir)
    else:
        if not (os.path.exists(setup_dir)):
            try:
                os.makedirs(setup_dir)
            except:
                raise ValueError("Cannot create setup {}, path invalid".format(setup_dir))
        else:
            raise ValueError("Cannot create setup {}, setup exists already.".format(setup_dir))

        this_dir = os.path.dirname(__file__)
        copyfile(os.path.join(this_dir, "synister/train.py"), os.path.join(setup_dir, "train.py"))
        copyfile(os.path.join(this_dir, "synister/train_pipeline.py"), os.path.join(setup_dir, "train_pipeline.py"))
        
        train_config = create_train_config()

        worker_config = create_worker_config(mount_dirs="/nrs, /scratch, /groups, /misc",
                                             singularity=os.path.abspath("singularity/synister.img"),
                                             queue=None)

        with open(os.path.join(setup_dir, "train_config.ini"), "w+") as f:
            train_config.write(f)

        with open(os.path.join(setup_dir, "worker_config.ini"), "w+") as f:
            worker_config.write(f)



def create_train_config():

    default_synapse_types = [
        'gaba',
        'acetylcholine',
        'glutamate',
        'serotonin',
        'octopamine',
        'dopamine']

    config = configparser.ConfigParser()

    config.add_section('Training')
    synapse_types_string = ""
    for s in default_synapse_types:
        synapse_types_string += s + ", "
    synapse_types_string = synapse_types_string[:-2]
    config.set('Training', 'synapse_types', synapse_types_string)
    config.set('Training', 'input_shape', '32, 128, 128')
    config.set('Training', 'fmaps', '32')
    config.set('Training', 'batch_size', '8')
    config.set('Training', 'db_credentials', str(None))
    config.set('Training', 'db_name_data', str(None))
    config.set('Training', 'split_name', str(None))
    config.set('Training', 'voxel_size', "40, 4, 4")
    config.set('Training', 'raw_container', "/nrs/saalfeld/FAFB00/v14_align_tps_20170818_dmg.n5")
    config.set('Training', 'raw_dataset', "volumes/raw/s0")
    config.set('Training', 'downsample_factors', "(2,2,2), (2,2,2), (2,2,2), (2,2,2)")
 
    return config


def create_worker_config(mount_dirs,
                         singularity,
                         queue):

    config = configparser.ConfigParser()
    config.add_section('Worker')
    if singularity == None or singularity == "None" or not singularity:
        config.set('Worker', 'singularity_container', str(None))
    else:
        config.set('Worker', 'singularity_container', str(singularity))
    config.set('Worker', 'num_cpus', str(5))
    config.set('Worker', 'num_block_workers', str(1))
    config.set('Worker', 'num_cache_workers', str(5))
    if queue == None or queue == "None" or not queue:
        config.set('Worker', 'queue', str(None))
    else:
        config.set('Worker', 'queue', str(queue))
    if mount_dirs == None or mount_dirs == "None" or not mount_dirs:
        config.set('Worker', 'mount_dirs', "None")
    else:
        config.set('Worker', 'mount_dirs', mount_dirs)
    return config



if __name__ == "__main__":
    options = p.parse_args()

    base_dir = options.base_dir
    experiment = options.e
    train_number = int(options.t)
    clean_up = bool(options.c)
    set_up_environment(base_dir,
                       experiment,
                       train_number,
                       clean_up)
