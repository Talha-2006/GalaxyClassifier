import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class GalaxyDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Build the full image path
        galaxy_id = self.dataframe.loc[idx, 'GalaxyID']
        img_path = os.path.join(self.img_dir, f"{galaxy_id}.jpg")

        # Load image and convert to RGB
        image = Image.open(img_path).convert("RGB")

        # Apply transforms if any
        if self.transform:
            image = self.transform(image)

        # Get the label (0, 1, or 2)
        label = self.dataframe.loc[idx, 'label']
        label = torch.tensor(int(label), dtype=torch.long)

        return image, label