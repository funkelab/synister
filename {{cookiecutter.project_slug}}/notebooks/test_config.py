"""
Get hyperparameters from configuration file with hydra.
"""

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from pprint import pprint


@hydra.main(version_base=None, config_path="../config", config_name="config")
def test_config(cfg: DictConfig) -> None:
    print(f"Output directory: {HydraConfig.get().runtime.output_dir}")


if __name__ == "__main__":
    test_config()
