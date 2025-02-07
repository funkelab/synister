import logging
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
import zarr
import hydra
import hydra.utils as hu
from omegaconf import DictConfig
from torch import nn


class ZarrDataset(torch.utils.data.Dataset):
    def __init__(self, zarr_array, locations, ids, shape=(64, 64, 64), transform=None, voxel_size=None):
        """
        Torch dataset getting locations from a zarr array.

        Parameters
        ----------
        zarr_array : zarr.core.Array
            Zarr array containing the data.
        locations : np.ndarray
            Array of locations to extract from the zarr array, assumed to be in the same order as the zarr array.
            So, if the zarr array is organized as (z, y, x), the locations should be in the same order.
        ids : np.ndarray
            Array of identifiers for the locations.
        shape : tuple
            Shape of the output array. The dataset will return a crop, centered at the location, of this shape.
        transform : callable
            Transform to apply to the data.
        voxel_size : tuple
            Voxel size of the data. If None, we assume isotropic data, and locations in voxel coordinates.
            If not None, we assume locations in world coordinates - the locations will be rounded to the nearest voxel.
        """
        self.zarr_array = zarr_array
        self.locations = locations
        self.voxel_size = voxel_size
        self.ids = ids
        assert len(locations) == len(ids)
        self.shape = shape
        self.transform = transform

    def __len__(self):
        return len(self.locations)
    
    def _voxel_location(self, location):
        """
        Convert a location in world coordinates to voxel coordinates.
        """
        if self.voxel_size is None:
            return location
        return np.round(location / self.voxel_size).astype(int)

    def __getitem__(self, idx):
        center = self._voxel_location(self.locations[idx])
        corner = center - (np.array(self.shape) // 2)
        array = np.array(
            self.zarr_array[
                corner[0] : corner[0] + self.shape[0],
                corner[1] : corner[1] + self.shape[1],
                corner[2] : corner[2] + self.shape[2],
            ]
        )[np.newaxis]
        this_id = int(self.ids[idx])
        if self.transform:
            return self.transform(array), this_id
        return array, this_id


class Transform:
    def __init__(self, mean, std, max_value=255):
        self.mean = mean
        self.std = std
        self.max_value = max_value

    def __call__(self, sample):
        return torch.from_numpy(
            ((sample / self.max_value) - self.mean) / self.std
        ).float()


def save_output(results, identifiers, class_names_ordered, output_file):
    to_out = pd.DataFrame(np.vstack(results), columns=class_names_ordered)
    to_out["id"] = np.hstack(identifiers)
    to_out.to_feather(output_file)


@torch.no_grad()
def validate(
    model,
    val_gt_location,
    container,
    dataset,
    num_transmitters: int = 6,
    experiment_dir: str = None,
    checkpoint_number=100000,
    nt_name="nt_name",
    point_id="point_id",
    num_workers=12,
    input_shape=(80, 80, 80),
    batch_size=32,
    num_output_splits=10,
    num_partitions: int = 1,
    partition_id: int = 1,
    output_dir: str = None,
    class_names_ordered=None,
    voxel_size=None,
):
    """
    Run prediction on validation data.

    Parameters
    ----------
    model : torch.nn.Module
        Model to validate.
    val_gt_location : str
        Location of the validation ground truth. Expected to be a feather file.
    container : str
        Zarr container location.
    dataset : str
        Name of the dataset in the zarr container.
    num_transmitters : int
        Number of transmitters.
    experiment_dir : str
        Directory containing the experiment.
        Checkpoints are expected to be there, under "checkpoints".
        Evaluation results will be saved here, under "eval", unless output_dir is specified.
    nt_name : str
        Name of the column containing the transmitter names.
    point_id : str
        Name of the column containing the point ids.
    num_workers : int
        Number of workers to use for the dataloader.
    input_shape : tuple
        Shape of the input to the model.
    batch_size : int
        Batch size for validation.
    num_output_splits : int
        Number of splits to save the output in.
        This ensures that the output does not take up too much memory.
    num_partitions : int
        Number of partitions to split the validation data into for distributed inference.
    partition_id : int
        ID of the partition to validate (1-indexed).
    output_dir : str
        Directory to store the evaluation results in. 
        If None, the results are stored in the experiment directory.
    class_names_ordered : list
        List of class names in the order they are output by the model.
        If None, we expect the data to have ground truth transmitter names.
        In that case, we will order them alphabetically.
    """
    # Metadata
    experiment_dir = Path(experiment_dir)
    checkpoint_dir = experiment_dir / "checkpoints"
    checkpoint = f"{checkpoint_dir}/model_checkpoint_{checkpoint_number}"
    assert Path(checkpoint).exists(), f"Checkpoint {checkpoint} does not exist"
    # location to store output
    if output_dir is None:
        eval_dir = experiment_dir / "eval"
    else:
        eval_dir = Path(output_dir)
    eval_dir.mkdir(exist_ok=True, parents=True)

    # Load the validation ground truth
    logging.info("Reading data...")
    df = pd.read_feather(val_gt_location)
    # Get coordinate locations
    locations = df[["z", "y", "x"]].values
    point_ids = df[point_id].values
    # Get the class names ordered
    if nt_name is not None:
        class_names_ordered = sorted(df[nt_name].unique())
        assert len(class_names_ordered) == num_transmitters
    elif class_names_ordered is None:
        raise ValueError("class_names_ordered must be provided if nt_name is None")

    # Split the validation data into partitions
    partition_suffix = ''
    if num_partitions > 1:
        assert partition_id <= num_partitions
        num_samples = len(locations)
        samples_per_partition = np.ceil(
            num_samples / num_partitions).astype(int)
        start_idx = (partition_id - 1) * samples_per_partition
        end_idx = start_idx + samples_per_partition
        end_idx = min(end_idx, num_samples)
        locations = locations[start_idx:end_idx, :]
        point_ids = point_ids[start_idx:end_idx]

        partition_padding = len(str(num_partitions))
        partition_suffix = f'_{partition_id:0{partition_padding}d}'

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load checkpoint
    model.load_state_dict(torch.load(checkpoint)["model_state_dict"])
    # Move to device, and use DataParallel if available
    num_gpus = torch.cuda.device_count()
    logging.info(f"There are {num_gpus} GPUs available")
    model.to(device)
    model = nn.DataParallel(model, device_ids=list(range(num_gpus)))
    # Set to eval mode
    model.eval()

    z = zarr.open(container, mode="r")[dataset]
    assert isinstance(z, zarr.core.Array)
    # Get datatype
    datatype = z.dtype
    max_value = np.iinfo(datatype).max
    transform = Transform(mean=0.5, std=0.5, max_value=max_value)

    zarr_dataset = ZarrDataset(
        z, locations, point_ids, shape=input_shape, transform=transform, voxel_size=voxel_size
    )
    dataloader = torch.utils.data.DataLoader(
        zarr_dataset,
        batch_size=batch_size,
        drop_last=False,
        shuffle=False,
        num_workers=num_workers * num_gpus,
    )

    samples_per_output = np.ceil(len(dataloader) / num_output_splits).astype(int)

    results = []
    identifiers = []
    i = 0
    with torch.inference_mode():
        for inputs, ids in tqdm(dataloader, total=len(dataloader)):
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs = torch.nn.functional.softmax(outputs, dim=1)
            ids = ids.cpu().numpy().astype(np.int64)  # Ensure full precision for IDs
            identifiers.append(ids)
            results.append(outputs.cpu().numpy())
            if len(results) >= samples_per_output:
                # Save the results
                output_suffix = f"_{i:0{len(str(num_output_splits))-1}d}"
                fn = f"predictions_{checkpoint_number}{partition_suffix}{output_suffix}.feather"
                output_file = eval_dir / fn
                save_output(results, identifiers, class_names_ordered, output_file)
                results = []
                identifiers = []
                i += 1
                logging.info(f"Intermediate results saved to {output_file}")

    # Save the last results
    if len(results) > 0:
        output_suffix = f"_{i:0{len(str(num_output_splits))-1}d}"
        fn = f"predictions_{checkpoint_number}{partition_suffix}{output_suffix}.feather"
        output_file = eval_dir / fn
        save_output(results, identifiers, class_names_ordered, output_file)
        logging.info(f"Intermediate results saved to {output_file}")

@hydra.main(config_path="../config", config_name="validate")
def main(cfg: DictConfig):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    # Remove logging from gunpowder
    logging.getLogger("gunpowder").setLevel(logging.WARNING)

    # Load the model
    model = hu.instantiate(cfg.model, _convert_="partial")
    # training data
    validation_data = cfg.gt.val

    num_partitions = 1
    partition_id = 1
    if "num_partitions" in cfg:
        num_partitions = cfg.num_partitions
    if "partition_id" in cfg:
        partition_id = cfg.partition_id
    output_dir = None
    if "output_dir" in cfg:
        output_dir = cfg.output_dir

    # run validation
    validate(
        model=model,
        # Get the data location
        val_gt_location=validation_data,
        **cfg.validate,
        num_partitions=num_partitions,
        partition_id=partition_id,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
