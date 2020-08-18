import daisy
import numpy as np
import torch.nn.functional as F
import torch
from funlib.learn.torch.models import Vgg3D
import logging

logger = logging.getLogger(__name__)

def predict(raw_batched,
            model):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    raw_batched_tensor = torch.tensor(raw_batched, device=device)
    output = model(raw=raw_batched_tensor)
    output = F.softmax(output, dim=1)
    return output


def init_vgg(checkpoint_file,
             input_shape,
             fmaps,
             downsample_factors=[(2,2,2), (2,2,2), (2,2,2), (2,2,2)],
             output_classes=None):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if output_classes is None:
        output_classes = 6

    model = Vgg3D(input_size=input_shape, fmaps=fmaps,
                  downsample_factors=downsample_factors,
                  output_classes=output_classes)
    model.to(device)
    logger.info("Init vgg with checkpoint {}".format(checkpoint_file))
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


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
        loc = daisy.Coordinate(tuple(loc))
        offset_nm = loc - (size/2*voxel_size)
        roi = daisy.Roi(offset_nm, size_nm).snap_to_grid(voxel_size, mode='closest')

        if roi.get_shape()[0] != size[0]:
            roi.set_shape(size_nm)

        if not dataset.roi.contains(roi):
            logger.warning("Location %s is not fully contained in dataset" % loc)
            return

        raw.append(dataset[roi].to_ndarray())

    raw = np.stack(raw)
    raw = raw.astype(np.float32)
    raw_normalized = raw/255.0
    raw_normalized = raw_normalized*2.0 - 1.0
    return raw, raw_normalized
