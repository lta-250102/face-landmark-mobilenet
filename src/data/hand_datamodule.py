from torch.utils.data import DataLoader, Dataset, random_split
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image
from typing import Any, Tuple
from lightning import LightningDataModule
import torch
import albumentations as A
import numpy as np
import pickle


class HandDataset(Dataset):
    def __init__(self, path, data_dir: str) -> None:
        super().__init__()
        self.config = pickle.load(open(path, 'rb'))
        # self.images = []
        self.labels = []
        self.data_dir = data_dir

        for idx, data in self.config.items():
            # self.images.append(Image.open(data_dir + '/color/%.5d.png' % idx))
            self.labels.append(data['uv_vis'][:, :2])
                
    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Any:
        transforms = A.Compose([
            A.Resize(224, 224),
            A.Normalize(),
            ToTensorV2(),
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
        
        image = Image.open(self.data_dir + 'color/%.5d.png' % index)
        landmarks = self.labels[index]
        transformed = transforms(
            image=np.array(image.convert('RGB')), 
            keypoints=landmarks)
        image = transformed['image']
        landmarks = torch.tensor(transformed['keypoints'], dtype=torch.float32).flatten() / 224
        return image, landmarks
    
class HandDataModule(LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 32, num_workers: int = 4, split: Tuple[int, int] = (0.8, 0.2)) -> None:
        super().__init__()
        self.data_dir = data_dir + 'RHD_published_v2/'
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split = split

    def setup(self, stage: str = None) -> None:
        train_dataset = HandDataset(self.data_dir + 'training/anno_training.pickle', self.data_dir + 'training/')
        self.test_set = HandDataset(self.data_dir + 'evaluation/anno_evaluation.pickle', self.data_dir + 'evaluation/')
        self.train_set, self.val_set = random_split(train_dataset, self.split, generator=torch.Generator().manual_seed(42))
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
    