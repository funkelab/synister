import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pathlib import Path
import logging
from omegaconf import DictConfig
import hydra


def split_data(
    base: str = "/groups/flyem/data/scratchspace/flyemflows/cns-full/neurotransmitters/malecns-tbars-neurotransmitter-groundtruth-2024-12-17.feather",
    val_size: float = 0.2,
    random_state: int = 42,
    train: str = None,
    val: str = None,
    body_id: str = "body",
    nt_name: str = "nt_name",
    point_id: str = "point_id",
    num_neurotransmitters: int = 6,
):
    """
    Split the ground truth data into training and validation sets, stratified by neurotransmitter type.

    Parameters
    ----------
    base : str
        Location of the ground truth data. Expected to be a feather file.
    val_size : float
        Proportion of the data to include in the validation set.
    random_state : int
        Random seed for reproducibility.
    train: str
        Location to save the training set. Will be saved as a feather file.
    val: str
        Location to save the validation set. Will be saved as a feather file.
    body_id: str
        Name of the column containing the body IDs.
    nt_name: str
        Name of the column containing the neurotransmitter names.
    point_id: str
        Name of the column containing the point IDs, the "unique" identifier for each location.
    num_neurotransmitters: int
        Number of neurotransmitters in the data. Used to check if the data is correct.
    """
    gt = pd.read_feather(base)
    # Print some basic information
    logging.info(f"Total number of synapses: {len(gt)}")
    logging.info(gt.nt_name.value_counts())
    # Get the body IDs
    body_ids = gt[[body_id, nt_name]].sort_values(body_id).drop_duplicates()
    # Check if any body id has more than one neurotransmitter
    assert body_ids.body.value_counts().max() == 1

    # Splitting the body IDs into training and validation sets, stratified by neurotransmitter type
    train_body, val_body = train_test_split(
        body_ids,
        test_size=val_size,
        stratify=body_ids[nt_name],
        random_state=random_state,
    )
    # Ensure that there is no overlap between the training and validation sets
    assert len(set(train_body[body_id]) & set(val_body[body_id])) == 0
    # Split the ground truth data into training and validation sets
    train_gt = gt[gt[body_id].isin(train_body[body_id])]
    val_gt = gt[gt[body_id].isin(val_body[body_id])]
    # Ensure that there is no overlap between the training and validation sets
    assert len(set(train_gt[point_id]) & set(val_gt[point_id])) == 0
    # Ensure that the number of neurotransmitters is correct
    assert len(train_gt.nt_name.unique()) == num_neurotransmitters
    assert len(val_gt.nt_name.unique()) == num_neurotransmitters
    for df in [train_gt, val_gt]:
        # Convert the neurotransmitter names to integers
        df["neurotransmitter"] = df["nt_name"].astype("category").cat.codes
    # TODO ensure that z,y,x columns are called "z", "y", "x"
    # Save
    if train is not None:
        logging.info(f"Saving the training set to {train}")
        train_gt.to_feather(train)
    if val is not None:
        logging.info(f"Saving the validation set to {val}")
        val_gt.to_feather(val)
    return train_gt, val_gt


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):
    # The information for this script is in cfg.gt
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    # Split the data
    train_gt, val_gt = split_data(**cfg.gt)
    # Some basic information as output
    # Get counts for training set
    print("Training set:", len(train_gt))
    print(train_gt.nt_name.value_counts())
    print("Training set, normalized")
    print(train_gt.nt_name.value_counts() / len(train_gt))
    # Get counts for validation set
    print("Validation set:", len(val_gt))
    print(val_gt.nt_name.value_counts())
    print("Validation set, normalized")
    print(val_gt.nt_name.value_counts() / len(val_gt))


if __name__ == "__main__":
    main()
