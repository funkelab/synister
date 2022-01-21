import daisy
import numpy as np
import torch.nn.functional as F
import torch
from funlib.learn.torch.models import Vgg3D
import logging
from multiprocessing import Pool, TimeoutError

logger = logging.getLogger(__name__)

def predict(raw_batched,
            model):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    raw_batched_tensor = torch.tensor(raw_batched, device=device)
    output = model(raw_batched_tensor)
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


def get_raw_parallel(locs,
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

    pool = Pool(processes=len(locs))
    raw = []
    size = daisy.Coordinate(size)
    voxel_size = daisy.Coordinate(voxel_size)
    size_nm = (size*voxel_size)
    dataset = daisy.open_ds(data_container,
                            data_set)

    raw_workers = [pool.apply_async(fetch_from_ds, 
                            (dataset, loc, voxel_size, size, size_nm))
                            for loc in locs]
    raw = [w.get(timeout=60) for w in raw_workers]
    pool.close()
    pool.join()

    raw = np.stack(raw)
    raw = raw.astype(np.float32)
    raw_normalized = raw/255.0
    raw_normalized = raw_normalized*2.0 - 1.0
    return raw, raw_normalized

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

    if tuple(dataset.voxel_size) != tuple(voxel_size):
        dataset.voxel_size = voxel_size
        roi_shape = dataset.roi.get_shape()
        roi_offset = dataset.roi.get_offset()
        roi_shape_phys = roi_shape * voxel_size[::-1]
        roi_offset_phys = roi_offset * voxel_size[::-1]
        dataset.roi = daisy.Roi(roi_offset_phys, roi_shape_phys)

    for loc in locs:
        loc = daisy.Coordinate(tuple(loc))
        offset_nm = loc - (size/2*voxel_size)
        roi = daisy.Roi(offset_nm, size_nm).snap_to_grid(voxel_size, mode='closest')

        if roi.get_shape()[0] != size[0]:
            roi.set_shape(size_nm)

        if not dataset.roi.contains(roi):
            logger.warning(f"Location {loc} is not fully contained in dataset")
            return None, None

        raw.append(dataset[roi].to_ndarray())

    raw = np.stack(raw)
    raw = raw.astype(np.float32)
    raw_normalized = raw/255.0
    raw_normalized = raw_normalized*2.0 - 1.0
    return raw, raw_normalized

def get_array(data_container,
              data_set,
              begin,
              end,
              context=(0,0,0)):

    context = np.array(context)
    roi = daisy.Roi(begin - context/2, 
                    end - begin + context)
    dataset = daisy.open_ds(data_container,
                            data_set)
    data_array = dataset[roi].to_ndarray()
    return data_array

def get_raw_dense(locs,
                  size,
                  data_array,
                  data_array_offset,
                  voxel_size):
    """
    Get raw crops from the specified
    data array.
    locs(``list of tuple of ints``):
        list of centers of location of interest
    """

    locs = [(l-data_array_offset)/voxel_size + np.array(size)/2 for l in locs]

    raw = []
    for loc in locs:
        loc = np.array(loc)
        raw.append(data_array[int(loc[0] - size[0]/2):int(loc[0] + size[0]/2),
                              int(loc[1] - size[1]/2):int(loc[1] + size[1]/2),
                              int(loc[2] - size[2]/2):int(loc[2] + size[2]/2)])

    raw = np.stack(raw)
    raw = raw.astype(np.float32)
    raw_normalized = raw/255.0
    raw_normalized = raw_normalized*2.0 - 1.0
    return raw, raw_normalized


def fetch_from_ds(dataset, loc, voxel_size, size, size_nm):
    loc = daisy.Coordinate(tuple(loc))
    offset_nm = loc - (size/2*voxel_size)
    roi = daisy.Roi(offset_nm, 
                    size_nm).snap_to_grid(voxel_size, mode='closest')

    if roi.get_shape()[0] != size[0]:
        roi.set_shape(size_nm)

    if not dataset.roi.contains(roi):
        logger.warning(f"Location {loc} is not fully contained in dataset")
        return None

    return dataset[roi].to_ndarray()

    
