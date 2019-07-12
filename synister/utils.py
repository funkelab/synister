import daisy
import numpy as np
import logging

logger = logging.getLogger(__name__)

def get_raw(locs,
            size,
            voxel_size,
            data_container,
            data_set):
    """
    Get raw crops from the specified
    dataset.

    locs(``list of tuple of ints``):

        list of centers of location of interest

    size(``tuple of ints``):
        
        size of cropout in voxel

    voxel_size(``tuple of ints``):

        size of a voxel

    data_container(``string``):

        path to data container (e.g. zarr file)

    data_set(``string``):

        corresponding data_set name, (e.g. raw)

    """

    raw = []
    size = daisy.Coordinate(size)
    voxel_size = daisy.Coordinate(voxel_size)
    size_nm = (size*voxel_size)
    dataset = daisy.open_ds(data_container,
                            data_set)

    for loc in locs:
        offset_nm = loc - (size/2*voxel_size)
        roi = daisy.Roi(offset_nm, size_nm).snap_to_grid(voxel_size, mode='closest')

        if roi.get_shape()[0] != size[0]:
            roi.set_shape(size_nm)

        if not dataset.roi.contains(roi):
            logger.WARNING("Location %s is not fully contained in dataset" % loc)
            return

        raw.append(dataset[roi].to_ndarray())

    raw = np.stack(raw)
    raw = raw.astype(np.float32)
    raw_normalized = raw/255.0
    raw_normalized = raw_normalized*2.0 - 1.0
    return raw, raw_normalized
