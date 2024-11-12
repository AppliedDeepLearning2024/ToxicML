import sys
sys.path.append('.')

from ToxicMl.Dl.GraphDataset import ClintoxDataset, DatasetType
from pathlib import Path

clintox_base_path = Path.cwd() / "data" / "clintox" / "downsampled"
def test_clintoc_dataset():
    dataset = ClintoxDataset(clintox_base_path, DatasetType.TRAIN)
    assert len(dataset) == 117

    dataset = ClintoxDataset(clintox_base_path, DatasetType.TEST)
    assert len(dataset) == 45

    dataset = ClintoxDataset(clintox_base_path, DatasetType.EVAL)
    assert len(dataset) == 29