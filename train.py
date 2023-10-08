import torch
import torch.nn as nn
import torchvision
from nvitop import select_devices
from torchinfo import summary

from cats_vs_dogs.runner import Runner
from cats_vs_dogs.utils import get_train_val_test_dataloaders

SIZE_H = SIZE_W = 96
NUM_CLASSES = 2
EPOCH_NUM = 2
LEARNING_RATE = 1e-3
EMBEDDING_SIZE = 128


def get_dense_model():
    model = nn.Sequential()

    model.add_module("flatten", nn.Flatten(start_dim=1))

    model.add_module("1-dense", nn.Linear(3 * SIZE_H * SIZE_W, 256))
    model.add_module("1-relu", nn.ReLU())
    model.add_module("1-dropout", nn.Dropout(0.1))

    model.add_module("2-dense", nn.Linear(256, EMBEDDING_SIZE))
    model.add_module("2-relu", nn.ReLU())
    model.add_module("2-dropout", nn.Dropout(0.1))

    model.add_module("3-dense-logits", nn.Linear(EMBEDDING_SIZE, NUM_CLASSES))

    return model


def get_resnet_model():
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


if __name__ == "__main__":
    device = torch.device(select_devices(sort=True)[0] if torch.cuda.is_available() else "cpu")

    train_dataloader, val_dataloader, test_dataloader = get_train_val_test_dataloaders()

    model = get_resnet_model()
    print(summary(model, (1, 3, SIZE_H, SIZE_W)))

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    ckpt_path = "checkpoints/model.ckpt"

    runner = Runner(
        model=model,
        optimizer=optimizer,
        device=device,
        ckpt_path=ckpt_path,
    )

    runner.train(train_dataloader, val_dataloader, n_epochs=EPOCH_NUM)
