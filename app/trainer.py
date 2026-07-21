from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from torch_geometric.loader import NeighborLoader
from torch_geometric.data import HeteroData

from torch.utils.tensorboard import SummaryWriter

import mlflow

from loguru import logger


# -------------------------
# CONFIG
# -------------------------

@dataclass
class TrainConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    device: str
    mixed_precision: bool
    patience: int


# -------------------------
# TRAINER
# -------------------------

class GNNTrainer:
    """
    Production-ready trainer for heterogeneous GNN models.
    """

    def __init__(
        self,
        model,
        data: HeteroData,
        config: TrainConfig,
        output_dir: str = "outputs",
    ):
        self.model = model
        self.data = data
        self.config = config
        self.output_dir = output_dir

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu"
        )

        self.model = self.model.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
        )

        self.scaler = torch.cuda.amp.GradScaler(enabled=config.mixed_precision)

        self.writer = SummaryWriter(log_dir=os.path.join(output_dir, "tensorboard"))

        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Training device: {self.device}")

    # -------------------------
    # CHECKPOINT
    # -------------------------

    def save_checkpoint(self, epoch: int):
        path = os.path.join(self.output_dir, f"model_epoch_{epoch}.pt")

        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "epoch": epoch,
            },
            path,
        )

        logger.info(f"Checkpoint saved: {path}")

    # -------------------------
    # LOSS FUNCTION (SELF-SUPERVISED NODE EMBEDDING)
    # -------------------------

    def compute_loss(self, out_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Contrastive-style loss approximation:
        aligns node embeddings across types.
        """

        losses = []

        keys = list(out_dict.keys())

        if len(keys) < 2:
            return torch.tensor(0.0, device=self.device)

        base = out_dict[keys[0]]

        for k in keys[1:]:
            target = out_dict[k]

            min_size = min(base.size(0), target.size(0))

            loss = F.mse_loss(
                base[:min_size],
                target[:min_size],
            )

            losses.append(loss)

        return torch.stack(losses).mean()

    # -------------------------
    # TRAIN LOOP
    # -------------------------

    def train(self):
        logger.info("Starting training...")

        mlflow.set_experiment("BankGraphAI_GNN")

        best_loss = float("inf")
        patience_counter = 0

        with mlflow.start_run():

            mlflow.log_params({
                "epochs": self.config.epochs,
                "lr": self.config.learning_rate,
                "batch_size": self.config.batch_size,
            })

            for epoch in range(self.config.epochs):

                self.model.train()

                loader = NeighborLoader(
                    self.data,
                    num_neighbors=[15, 10],
                    batch_size=self.config.batch_size,
                    shuffle=True,
                )

                epoch_loss = 0.0
                steps = 0

                for batch in loader:

                    batch = batch.to(self.device)

                    self.optimizer.zero_grad()

                    with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):

                        out = self.model(batch)
                        loss = self.compute_loss(out)

                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    epoch_loss += loss.item()
                    steps += 1

                avg_loss = epoch_loss / max(steps, 1)

                logger.info(f"Epoch {epoch} | Loss: {avg_loss:.4f}")

                mlflow.log_metric("loss", avg_loss, step=epoch)
                self.writer.add_scalar("loss/train", avg_loss, epoch)

                # -------------------------
                # EARLY STOPPING
                # -------------------------

                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                    self.save_checkpoint(epoch)
                else:
                    patience_counter += 1

                if patience_counter >= self.config.patience:
                    logger.info("Early stopping triggered.")
                    break

        logger.info("Training completed.")