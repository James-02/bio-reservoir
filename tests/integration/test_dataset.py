"""The data path reproduces the v0.1 data: the datasets the v0.1 runs read and the v0.1
cross-validation pools (seed 6337, 4,015 rows) drawn from them match the checksums of the
v0.1 inputs. The preparation command rebuilds the v0.1 five-class dataset from a fresh
Kaggle download (dataset version 1, 17 Oct 2026; its CSV files are byte-identical to the
copies used for v0.1).

The v0.1 binary runs read another binary file than the one the preparation command writes:
a pre-balanced pool of 37,714 beats written by an earlier loader, kept as
data/ecg_archive/binary_ecg_data.npz in the v0.1 workspace. The prepared binary dataset
(all 109,446 beats) is checked against its own reference."""

import hashlib

import numpy as np
import pytest

from utils import config
from utils.preprocessing import dataset_sha256, load_ecg_data

V0_1 = {
    "five-class": {
        "file": "ecg/ecg_data.npz",
        "dataset": "cd3ece0610daa08a9b5a9424496428104c98c423eb444e0b1474019ce12fe098",
        "pool": "6e63ecebba75c280277fdb0a2fd0e466b89f46f41e8282ab3269599d77776356",
    },
    "binary": {
        "file": "ecg_archive/binary_ecg_data.npz",
        "dataset": "522e747d5730d0ef6bcff2035c23f5ca8ca22faf1a4a1a366059f3931bd7dae1",
        "pool": "67a48d5a6e67806c21422bd62695d5cc12401f18d883b4f56bc2b1d826a601f4",
    },
}
PREPARED_BINARY = {
    "file": "ecg/binary_ecg_data.npz",
    "dataset": "815a424445a6ade744db391157b7866c35cf1a162983cd595e8acfd8dd583746",
    "pool": "5bdbb22dd8cd94d08b773f2bcbe5d9f241d7465ed53825fa3fd12a241238ab0e",
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


def _check(reference, binary):
    path = config.data_dir() / reference["file"]
    if path.parent != ECG_DIR and not path.is_file():
        pytest.skip(f"{reference['file']} is not in the data folder")
    pool = load_ecg_data(rows=4015, binary=binary, balance_classes=True, max_per_class=None,
                         noise_rate=0.0, noise_ratio=0.0, scaler_type="sequence_zscore",
                         seed=config.SEED, data_dir=str(path.parent),
                         **{"binary_save_file" if binary else "save_file": path.name})
    assert dataset_sha256(path) == reference["dataset"]
    assert _arrays_sha256(*pool) == reference["pool"]


@pytest.mark.parametrize("task", sorted(V0_1))
def test_data_matches_v0_1(task):
    _check(V0_1[task], binary=task == "binary")


def test_prepared_binary_data_matches_its_reference():
    _check(PREPARED_BINARY, binary=True)
