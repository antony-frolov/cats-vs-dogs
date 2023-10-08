import timeit
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm


class Runner:
    def __init__(self, model, device, optimizer=None, ckpt_path=None):
        self.model = model.to(device)
        self.optimizer = optimizer
        if optimizer is not None:
            self.optimizer.zero_grad()
        self.device = device
        self.ckpt_path = Path(ckpt_path)
        self.ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        self.epoch = 0
        self.output = None
        self.metrics = None
        self._global_step = 0
        self._set_events()
        self._top_val_accuracy = -1
        self.log_dict = {
            "train": [],
            "val": [],
            "test": [],
        }

    def _set_events(self):
        self._phase_name = ""
        self.events = {
            "train": defaultdict(list),
            "val": defaultdict(list),
            "test": defaultdict(list),
        }

    def _reset_events(self, event_name):
        self.events[event_name] = defaultdict(list)

    def forward(self, inputs):
        return {"logits": self.model(inputs)}

    def run_criterion(self, labels):
        logits = self.output["logits"]

        loss = torch.nn.functional.cross_entropy(logits, labels)

        scores = F.softmax(logits, 1).detach().cpu().numpy()[:, 1].tolist()
        labels = labels.detach().cpu().numpy().ravel().tolist()

        self.events[self._phase_name]["loss"].append(loss.detach().cpu().numpy())
        self.events[self._phase_name]["scores"].extend(scores)
        self.events[self._phase_name]["labels"].extend(labels)

        return loss

    def save_checkpoint(self):
        val_accuracy = self.metrics["accuracy"]
        if val_accuracy > self._top_val_accuracy and self.ckpt_path is not None:
            self._top_val_accuracy = val_accuracy
            torch.save(self.model, open(self.ckpt_path, "wb"))

    def output_log(self):
        scores = np.array(self.events[self._phase_name]["scores"])
        labels = np.array(self.events[self._phase_name]["labels"])

        assert len(labels) > 0, print("Label list is empty")
        assert len(scores) > 0, print("Score list is empty")
        assert len(labels) == len(scores), print("Label and score lists are of different size")

        self.metrics = {
            "loss": np.mean(self.events[self._phase_name]["loss"]),
            "accuracy": accuracy_score(labels, np.int32(scores > 0.5)),
            "f1": f1_score(labels, np.int32(scores > 0.5)),
        }
        print(f"{self._phase_name}: ", end="")
        print(" | ".join([f"{k}: {v:.4f}" for k, v in self.metrics.items()]))

        self.save_checkpoint()

    def _run_batch(self, inputs):
        self._global_step += len(inputs)
        self.output = self.forward(inputs)

    def _run_epoch(self, dataloader, train_phase=True, output_log=False):
        self.model.train(train_phase)

        for batch in tqdm(
            dataloader,
            desc="Training" if train_phase else "Evaluation",
            leave=False,
        ):
            inputs, labels = batch

            self._run_batch(inputs.to(self.device))

            with torch.set_grad_enabled(train_phase):
                loss = self.run_criterion(labels.to(self.device))

            if train_phase:
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

        self.log_dict[self._phase_name].append(np.mean(self.events[self._phase_name]["loss"]))

        if output_log:
            self.output_log()

    def train(self, train_dataloader, val_dataloader, n_epochs):
        assert self.optimizer is not None
        for _epoch in range(n_epochs):
            start_time = timeit.default_timer()
            self.epoch += 1
            print(f"epoch {self.epoch:3d}/{n_epochs:3d} started")

            self._set_events()
            self._phase_name = "train"
            self._run_epoch(train_dataloader, train_phase=True)

            print(f"epoch {self.epoch:3d}/{n_epochs:3d} took {timeit.default_timer() - start_time:.2f}s")

            self._phase_name = "val"
            self.validate(val_dataloader)
            self.save_checkpoint()

    @torch.no_grad()  # we do not need to save gradients during validation
    def validate(self, dataloader, phase_name="val"):
        self._phase_name = phase_name
        self._reset_events(phase_name)
        self._run_epoch(dataloader, train_phase=False, output_log=True)
        return self.metrics
