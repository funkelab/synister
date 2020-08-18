from get_neurotransmitter import get_neurotransmitter, init_model
import argparse
import numpy as np
import zarr

parser = argparse.ArgumentParser()
parser.add_argument(
    '--location', '-l',
    type=int,
    nargs='+',
    help="Center location in nm")
parser.add_argument(
    '--step', '-s',
    type=int,
    nargs='+',
    help="Step-size of predictions in nm")
parser.add_argument(
    '--context', '-c',
    type=int,
    nargs='+',
    help="Number of steps to predict in each direction")
parser.add_argument(
    '--out', '-o',
    type=str,
    help="Name of output zarr container.")

def predict_in_roi(begin, end, step):

    model, model_config = init_model()

    z, y, x = np.meshgrid(
        range(begin[0], end[0], step[0]),
        range(begin[1], end[1], step[1]),
        range(begin[2], end[2], step[2]),
        indexing='ij')
    depth, height, width = z.shape

    positions = np.stack([z.flatten(), y.flatten(), x.flatten()], axis=1)
    print(f"predicting {len(positions)} locations...")

    predictions = get_neurotransmitter(
        positions,
        model,
        model_config)

    predictions = np.array([
        np.array([p[nt] for p in predictions]).reshape(depth, height, width)
        for nt in model_config["neurotransmitter_list"]
    ])

    return predictions

if __name__ == '__main__':

    args = parser.parse_args()

    center = np.array(args.location)

    step = np.array(args.step)
    center = center//step * step
    context = np.array(args.context) * step

    f = zarr.open(args.out)

    predictions = predict_in_roi(center - context, center + context, step)

    f['prediction'] = predictions
    f['prediction'].attrs['offset'] = list(int(x) for x in center - context)
    f['prediction'].attrs['resolution'] = list(int(x) for x in step)
