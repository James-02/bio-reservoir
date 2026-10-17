"""The data path reproduces the v0.1 data: the preprocessed datasets and the v0.1
cross-validation pools (seed 6337, 4,015 rows) match the checksums of the v0.1 inputs.
The references were taken from a fresh Kaggle download (dataset version 1) on 17 Oct 2026,
whose CSV files are byte-identical to the copies used for v0.1."""

import hashlib

import numpy as np
import pytest

from utils import config
from utils.preprocessing import dataset_sha256, load_ecg_data

V0_1 = {
    "five-class": {
        "cache": "ecg_data.npz",
        "dataset": "cd3ece0610daa08a9b5a9424496428104c98c423eb444e0b1474019ce12fe098",
        "pool": "6e63ecebba75c280277fdb0a2fd0e466b89f46f41e8282ab3269599d77776356",
    },
    "binary": {
        "cache": "binary_ecg_data.npz",
        "dataset": "815a424445a6ade744db391157b7866c35cf1a162983cd595e8acfd8dd583746",
        "pool": "5bdbb22dd8cd94d08b773f2bcbe5d9f241d7465ed53825fa3fd12a241238ab0e",
    },
}

ECG_DIR = config.data_dir() / "ecg"
CAN_LOAD = (ECG_DIR / "ecg_data.npz").is_file() or bool(
    config.get("KAGGLE_KEY") or config.get("KAGGLE_HEARTBEAT_DIR"))

pytestmark = pytest.mark.skipif(not CAN_LOAD, reason="needs data/ecg or Kaggle credentials")


def _arrays_sha256(*arrays) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(np.asarray(array))
        digest.update(str((array.dtype.str, array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


@pytest.mark.parametrize("task", sorted(V0_1))
def test_data_matches_v0_1(task):
    pool = load_ecg_data(rows=4015, binary=task == "binary", balance_classes=True,
                         max_per_class=None, noise_rate=0.0, noise_ratio=0.0,
                         scaler_type="sequence_zscore", seed=config.SEED)
    assert dataset_sha256(ECG_DIR / V0_1[task]["cache"]) == V0_1[task]["dataset"]
    assert _arrays_sha256(*pool) == V0_1[task]["pool"]
