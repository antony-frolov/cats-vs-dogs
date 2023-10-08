import zipfile
from pathlib import Path

import requests
import torch
import torchvision
from torchvision import transforms
from tqdm.auto import tqdm

DATA_DIR = Path("data")
NUM_WORKERS = 4
SIZE_H = SIZE_W = 96
BATCH_SIZE = 256
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


transformer = transforms.Compose(
    [
        transforms.Resize((SIZE_H, SIZE_W)),  # scaling images to fixed size
        transforms.ToTensor(),  # converting to tensors
        transforms.Normalize(IMAGE_MEAN, IMAGE_STD),  # normalize image data per-channel
    ]
)


def download_data(url: str, fname: str, chunk_size=1024):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=f"Downloading data: {fname}",
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)


def get_dataset(folder):
    return torchvision.datasets.ImageFolder(DATA_DIR.joinpath(folder), transform=transformer)


def get_dataloader(dataset, shuffle=False):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
    )


def get_train_val_test_dataloaders():
    if not DATA_DIR.exists():
        download_data("https://www.dropbox.com/s/gqdo90vhli893e0/data.zip?dl=1", "data.zip")
        zipfile.ZipFile("data.zip").extractall(DATA_DIR)
    return (
        get_dataloader(get_dataset("train_11k"), shuffle=True),
        get_dataloader(get_dataset("val")),
        get_dataloader(get_dataset("test_labeled")),
    )
