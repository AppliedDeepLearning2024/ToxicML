from torch.utils.data.dataset import Dataset
from pathlib import Path
from enum import Enum

class DatasetType(Enum):
    TRAIN = "train"
    TEST = "test"
    EVAL = "eval"



class ClintoxDataset(Dataset):
    def __init__(self, data_dir: Path,  datasetType: DatasetType, transform=None,  target_transform=None):
        self.img_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        self.datasetType = datasetType
        self.filepath = (data_dir / datasetType.value).with_suffix(".csv")
        if not self.filepath.is_file():
            raise ValueError(f"{self.filepath} is not a valid file")
        self.num_rows = self._get_file_size(self.filepath)
    
    def _get_file_size(self, filepath: Path):
        with open(filepath) as f:
            return sum(1 for line in f)

    def __len__(self):
        return self.num_rows
    
    def __getitem__(self, idx):
        return 0