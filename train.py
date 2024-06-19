import os
from typing import final
from pyannote.audio import Model
from argparse import ArgumentParser, Namespace
from pyannote.database import get_protocol, FileFinder, registry
from pyannote.audio.tasks import (
    PixIT,
    VoiceActivityDetection,
)
import pytorch_lightning as pl
from pyannote.audio.models.separation import ToTaToNet
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
import torch
from types import MethodType
import argparse

pl.seed_everything(42)

def main(args):
    preprocessors = {"audio": FileFinder()}
    registry.load_database(
        args.database_dir
    )
    dataset = registry.get_protocol(
        "AMI-SDM.SpeakerDiarization.only_words", {"audio": FileFinder()}
    )
    segmentation_task = PixIT(
        dataset,
        duration=5.0,
        batch_size=16,
        max_speakers_per_chunk=3,
        separation_loss_weight=0.5,
        num_workers=4,
    )
    segmentation_model = ToTaToNet(
        task=segmentation_task,
        linear={"hidden_size": 64, "num_layers": 2},
    )

    def configure_optimizers(self):
        optimizer_wavlm = torch.optim.Adam(self.wavlm.parameters(), lr=1e-5)
        other_params = list(
            filter(lambda kv: "wavlm" not in kv[0], self.named_parameters())
        )
        optimizer_rest = torch.optim.Adam(dict(other_params).values(), lr=3 * 1e-4)
        return [
            optimizer_wavlm,
            optimizer_rest,
        ]

    segmentation_model.configure_optimizers = MethodType(
        configure_optimizers, segmentation_model
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    checkpoint = ModelCheckpoint(
        dirpath=None,
        monitor="loss/val",
        mode="min",
        save_top_k=1,
        every_n_epochs=1,
        save_last=True,
        save_weights_only=False,
        verbose=True,
    )
    callbacks = [checkpoint, lr_monitor]
    callbacks.append(
        EarlyStopping(monitor="loss/val", mode="min", patience=10, verbose=True)
    )

    trainer = pl.Trainer(
        max_epochs=100,
        callbacks=callbacks,
        accelerator="gpu",
        devices=1,
        num_nodes=1,
    )
    trainer.fit(segmentation_model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Add your command line arguments here
    parser.add_argument("--database-dir", type=str, required=True)
    args = parser.parse_args()
    main(args)