from typing import Any, Dict, Optional, Tuple

import torch
from PIL import Image
from lightning import LightningDataModule
from xml.etree import ElementTree as ET
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import transforms


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
        images = ET.parse(path).getroot()[2]
        self.data_dir = data_dir

        # data transformations
        self.transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.Grayscale(num_output_channels=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.records = []
        for image in images:
            record = self.RawRecord()
            record.image = Image.open(self.data_dir + image.attrib['file'])
            record.width = int(image.attrib['width'])
            record.height = int(image.attrib['height'])
            record.box_top = int(image[0].attrib['top'])
            record.box_left = int(image[0].attrib['left'])
            record.box_width = int(image[0].attrib['width'])
            record.box_height = int(image[0].attrib['height'])
            record.landmarks = []
            for landmark in image[0]:
                record.landmarks.append(int(landmark.attrib['x']))
                record.landmarks.append(int(landmark.attrib['y']))
            self.records.append(record)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Any:
        record: FaceDataset.RawRecord = self.records[index]
        # image = record.image.crop((record.box_left, record.box_top, record.box_left + record.box_width, record.box_top + record.box_height))
        image = self.transforms(record.image)
        landmarks = torch.tensor(record.landmarks) / 255.0
        return image, landmarks

class FaceDataModule(LightningDataModule):
    def __init__(
        self,
        train_val_split: Tuple[int, int] = (0.8, 0.2),
        data_dir: str = "./data/",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.train_path = data_dir + 'ibug_300W_large_face_landmark_dataset/labels_ibug_300W_train.xml'
        self.test_path = data_dir + 'ibug_300W_large_face_landmark_dataset/labels_ibug_300W_test.xml'
        train_dataset = FaceDataset(self.train_path, data_dir=data_dir + 'ibug_300W_large_face_landmark_dataset/')
        (self.data_train, self.data_val) = random_split(
            dataset=train_dataset,
            lengths=train_val_split,
            generator=torch.Generator().manual_seed(42),
        )
        # self.data_train: Dataset = FaceDataset(self.train_path)
        # self.data_val: Dataset = FaceDataset(self.test_path)
        self.data_test: Dataset = FaceDataset(self.test_path, data_dir=data_dir + 'ibug_300W_large_face_landmark_dataset/')

        self.batch_size_per_device = batch_size

    @property
    def num_classes(self) -> int:
        """Get the number of classes.
        """
        return 136

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        # load and split datasets only if not loaded already
        pass

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    _ = FaceDataModule()
