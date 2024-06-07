'''
Abstraction for LongCIU data.
This doesn't allow that much customization, its just to reproduce the simple UNet baseline training.

Customize as you want!
'''
import os
import json
import torch
import random
import numpy as np
from typing import Optional, Dict, Any, Tuple, Callable, List, Union
from sanitize_filename import sanitize
from torch.utils.data import Dataset
import SimpleITK as sitk
import pytorch_lightning as pl
from torch.utils.data import DataLoader


class TrainTransform():
    '''
    Just simple random patching
    '''
    def __init__(self, 
                 patch_size: int = 256):
        self.patch_size = patch_size

    def __call__(self, 
                 x: Union[np.ndarray, torch.Tensor], 
                 y: Union[np.ndarray, torch.Tensor]) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
        C, Y, X = x.shape
        pos_y, pos_x = random.randint(0, Y - self.patch_size), random.randint(0, X - self.patch_size)
        return x[:, pos_y:pos_y + self.patch_size, pos_x:pos_x + self.patch_size], y[:, pos_y:pos_y + self.patch_size, pos_x:pos_x + self.patch_size]



class DataMaskSlicer(Dataset):
    '''
    Abstracts reading from an in-memory data and mask array,
    following idxs for indexing the correct slices and applying transform on the slices.
    '''    
    def __init__(self, name: str, data: np.ndarray, mask: np.ndarray, idxs: List[int], transform: Optional[Callable]):
        '''
        data, mask and idxs and transform should be defined externally
        '''
        self.name = name
        self.data = data
        self.mask = mask
        self.idxs = idxs
        self.transform = transform

        print(f"Initialized {name} with {len(self.idxs)} slices and transform: {self.transform}")

    def __len__(self) -> int:
        return len(self.idxs)
    
    def __getitem__(self, i: int) -> Tuple[np.ndarray, np.ndarray]:
        idx = self.idxs[i]
        data, mask = np.expand_dims(self.data[idx], 0), np.expand_dims(self.mask[idx], 0)

        if self.transform is not None:
            data, mask = self.transform(data, mask)

        metadata = {"ID": f"{self.name}_{idx}",
                    "idx": idx}

        return data, mask, metadata


class LongCIUDataModule(pl.LightningDataModule):
    '''
    Attempt to abstract all data loading, removing redundant steps
    '''
    def __init__(self, 
                 data_dir: str, 
                 num_workers: int,
                 train_batch_size: int,
                 eval_batch_size: int,
                 train_transform: Optional[object] = None, 
                 eval_transform: Optional[object] = None):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.data_dir = data_dir
        self.train_transform = train_transform
        self.eval_transform = eval_transform
    
    def prepare_data(self):
        '''
        Since dataset is small, we already load it into RAM here.
        '''
        print(f"LongCIUDataModule prepare_data called")
        with open(os.path.join(self.data_dir, "longciu_splits.json"), 'r') as splits_file:
            self.splits = json.load(splits_file)

        self.data_path = os.path.join(self.data_dir, "longciu_img.nii.gz")
        self.mask_path = os.path.join(self.data_dir, "longciu_STAPLE_tgt.nii.gz")
        self.data, self.mask = sitk.GetArrayFromImage(sitk.ReadImage(self.data_path)), sitk.GetArrayFromImage(sitk.ReadImage(self.mask_path))
        
        # You can customize this preprocessing if you want!
        MIN, MAX = -1024, 600
        self.data = np.clip(self.data, MIN, MAX)
        self.data = (self.data - MIN)/(MAX - MIN)
        self.data, self.mask = self.data.astype(np.float32), self.mask.astype(np.float32)
        
    def setup(self, stage):
        '''
        Since data is very small, everything is in memory at all times, accesible in self.data and self.mask.
        "stage" uses the indexes in longciu_splits.json to index the correct slices for each split.

        stage: one of train, val, test or all. All will return all slices.
        transform: object that takes as input both image and target and returns a tuple of transformed image and target
        '''
        print(f"WARNING: PL template stage {stage} argument ignored. Initializing all three splits.")
        self.train_dataset = DataMaskSlicer("train_longciu", self.data, self.mask, self.splits["train"], self.train_transform)
        self.val_dataset = DataMaskSlicer("val_longciu", self.data, self.mask, self.splits["val"], self.eval_transform)
        self.test_dataset = DataMaskSlicer("test_longciu", self.data, self.mask, self.splits["test"], self.eval_transform)
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.eval_batch_size, num_workers=self.num_workers, shuffle=False)
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.eval_batch_size, num_workers=self.num_workers, shuffle=False)


if __name__ == "__main__":
    # Test routine
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid

    longciu_datamodule = LongCIUDataModule(data_dir="../data", train_batch_size=4, eval_batch_size=4, num_workers=1, 
                                           train_transform=TrainTransform(), eval_transform=None)
    longciu_datamodule.prepare_data()
    longciu_datamodule.setup()
    dataloaders = {stage: getattr(longciu_datamodule, f"{stage}_dataloader")() for stage in ["train", "val", "test"]}

    for name, dataloader in dataloaders.items():
        for x, y, m in dataloader:
            plt.figure()
            plt.subplot(1, 3, 1)
            plt.imshow(make_grid(x, nrow=2).permute(1, 2, 0))
            plt.subplot(1, 3, 2)
            plt.imshow(make_grid(y/2, nrow=2).permute(1, 2, 0))
            plt.subplot(1, 3, 3)
            plt.imshow(make_grid(torch.where(y > 0, y/2, x), nrow=2).permute(1, 2, 0))
            plt.suptitle(str(m))
            plt.show()
            