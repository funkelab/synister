from setuptools import setup

setup(
    name='synister',
    version='0.1',
    description='Neurotransmitter classification',
    url='https://github.com/funkelab/synister',
    author='Funkelab',
    packages=[
        'synister',
        'synister.skeleton_network',
        'synister.hemi_lineage_network',
        'synister.brain_region_network'
        ])
