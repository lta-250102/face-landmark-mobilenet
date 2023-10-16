from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import transforms
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image
from xml.etree import ElementTree as ET
from typing import Any, Tuple
from lightning import LightningDataModule
import torch
import albumentations as A
import numpy as np


class FaceDataset(Dataset):
    class RawRecord:
        image: Image
        width: int
        height: int
        box_top: int
        box_left: int
        box_width: int
        box_height: int
        landmarks: list[int]

    def __init__(self, path, data_dir: str) -> None:
        super().__init__()
        self.images = ET.parse(path).getroot()[2]
        self.data_dir = data_dir

        self.transforms = A.Compose([
            A.Resize(244, 244),
            A.Normalize(),
            ToTensorV2(),
        ], keypoint_params=A.KeypointParams(format='xy'))
        
    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Any:
        data = self.images[index]
        landmarks = []
        for landmark in data[0]:
            landmarks.append((float(landmark.attrib['x']), float(landmark.attrib['y'])))
        transformed = self.transforms(
            image=np.array(Image.open(self.data_dir + data.attrib['file']).convert('RGB')), 
            keypoints=landmarks)
        image = transformed['image']
        landmarks = torch.tensor(transformed['keypoints']).flatten() / 244
        return image, landmarks
    
class FaceDataModule(LightningDataModule):
    def __init__(
        self,
        train_val_split: Tuple[int, int] = (0.8, 0.2),
        data_dir: str = "./data/",
        batch_size: int = 64,
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        self.num_workers = num_workers
        self.train_path = data_dir + 'ibug_300W_large_face_landmark_dataset/labels_ibug_300W_train.xml'
        self.test_path = data_dir + 'ibug_300W_large_face_landmark_dataset/labels_ibug_300W_test.xml'
        train_dataset = FaceDataset(self.train_path, data_dir=data_dir + 'ibug_300W_large_face_landmark_dataset/')
        (self.data_train, self.data_val) = random_split(
            dataset=train_dataset,
            lengths=train_val_split,
            generator=torch.Generator().manual_seed(42),
        )
        self.data_test: Dataset = FaceDataset(self.test_path, data_dir=data_dir + 'ibug_300W_large_face_landmark_dataset/')

        self.batch_size_per_device = batch_size

    @property
    def num_classes(self) -> int:
        return 136

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.num_workers,
            shuffle=False,
        )


if __name__ == "__main__":
    _ = FaceDataModule()
