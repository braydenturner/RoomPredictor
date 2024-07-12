import lightning as L
from torchmetrics.functional import accuracy
from typing import Optional
import tensorboard
from lightning.pytorch.loggers import TensorBoardLogger

import pandas as pd
from pandas import DataFrame
import numpy as np

import torch
from torch import nn
from torch.utils.data import random_split, DataLoader, Dataset


class Classifier(L.LightningModule):

    def __init__(self, num_hidden, num_outputs, lr: float, dropout=0.2):
        super().__init__()
        self.save_hyperparameters()
        self.model = nn.Sequential(
            nn.LazyLinear(num_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.LazyLinear(num_outputs)
        )
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, X):
        return self.model(X)

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X)
        loss: torch.Tensor = self.loss(y_hat, y)
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        self.log("val_loss", loss)
        self.log("val_accuracy", acc)

    def test_step(self, batch, batch_idx):
        _, acc = self._shared_eval_step(batch, batch_idx)
        self.log("test_accuracy", acc)

    def _shared_eval_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X)
        loss: torch.Tensor = self.loss(y_hat, y)
        acc = accuracy(preds=y_hat, target=y, task="multiclass", num_classes=self.hparams.num_outputs)

        return loss, acc

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)
        return optimizer


class Data(Dataset):

    def __init__(self, df: DataFrame, target_column: str):
        self.X = torch.from_numpy(df.loc[:, df.columns != target_column].to_numpy().astype(np.float32))
        self.y = torch.from_numpy(df.loc[:, target_column].to_numpy().astype(np.int8))

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.y)


class RoomPredictorDataModule(L.LightningDataModule):

    def __init__(self, file: str, ratios: list = [.8, .1, .1], batch_size: int = 8):
        super().__init__()
        self.directory = "../data"
        self.file = file
        self.target_column = "room"
        self.batch_size = batch_size

        self.df: Optional[DataFrame] = None

        self.ratios = ratios
        self.train: Optional[Dataset] = None
        self.val: Optional[Dataset] = None
        self.test: Optional[Dataset] = None

    def prepare_data(self) -> None:
        def custom_round(x, base=1):
            return base * round(float(x) / base)

        data = pd.read_csv(f"{self.directory}/{self.file}")
        data["time_rounded"] = data["time"].apply(lambda x: custom_round(x, base=.5))
        deduped_data = data.groupby(["time_rounded", "id", "room"], as_index=False)["rssi"].mean()
        columns = np.append(["room"], deduped_data["id"].unique())
        new_data = pd.DataFrame(columns=columns)

        rooms = deduped_data["room"].unique()
        room_labels = dict()
        label_rooms = dict()

        for i, room in enumerate(rooms):
            room_labels[room] = i
            label_rooms[i] = room

        for name, group in deduped_data.groupby("time_rounded"):
            if group["room"].nunique() > 1:
                print(f"Skipping {name}")

            row = dict()
            row["room"] = room_labels[group["room"].unique()[0]]  # Convert room to int for training

            for i, r in group.iterrows():
                row[r["id"]] = (r["rssi"] + 100) / 100

            for c in columns:
                # Set id's not found in this time as lowest value
                if c not in row:
                    row[c] = 0

            new_data = new_data._append(
                row,
                ignore_index=True
            )

        self.df = new_data

    def setup(self, stage: str):
        train, val, test = self.split(self.ratios)

        self.train = Data(train.dataset, target_column=self.target_column)
        self.val = Data(val.dataset, target_column=self.target_column)
        self.test = Data(test.dataset, target_column=self.target_column)

    def split(self, ratios: list):
        train_perc, val_perc, test_perc = ratios
        amount = len(self.df)

        test_amount = (int(amount * test_perc) if test_perc is not None else 0)
        val_amount = (int(amount * val_perc) if val_perc is not None else 0)
        train_amount = amount - test_amount - val_amount

        train, val, test = random_split(
            self.df,
            (train_amount, val_amount, test_amount),
            # If we have a seed, pass that in to a torch generator
            generator=(
                torch.Generator().manual_seed(self.seed)
                if self.seed
                else None))

        return train, val, test

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False)
