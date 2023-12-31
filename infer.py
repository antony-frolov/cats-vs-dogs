from pathlib import Path

import torch
from nvitop import select_devices

from cats_vs_dogs.runner import Runner
from cats_vs_dogs.utils import get_train_val_test_dataloaders

PREDICTIONS_DIR = Path("predictions")

if __name__ == "__main__":
    device = torch.device(select_devices(sort=True)[0] if torch.cuda.is_available() else "cpu")

    _, val_dataloader, test_dataloader = get_train_val_test_dataloaders()

    ckpt_path = "checkpoints/model.ckpt"

    model = torch.load(ckpt_path)

    runner = Runner(
        model=model,
        device=device,
        ckpt_path=ckpt_path,
    )

    val_stats = runner.validate(val_dataloader, phase_name="val")
    test_predictions = runner.predict(test_dataloader, phase_name="test")
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    test_predictions.to_csv(PREDICTIONS_DIR.joinpath("test.csv"))
