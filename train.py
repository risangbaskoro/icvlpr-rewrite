import argparse
import os

import torch

from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import ICVLPDataset
from decoder import GreedyCTCDecoder
from metrics import LetterNumberRecognitionRate
from model import LPRNet, SpatialTransformerLayer, LocNet
from utils import TColor


class Trainer:
    def __init__(self):
        self.args = None
        self.logger = None

        self.device = None

        self.ds_train = None
        self.dl_train = None
        self.ds_val = None
        self.dl_val = None

        self.model = None
        self.optimizer = None
        self.loss_fn = None
        self.acc_fn = None
        self.lr_scheduler = None

        self.epoch = None

        self.avg_loss = 0.0
        self.avg_acc = 0.0
        self.val_avg_loss = 0.0
        self.val_avg_acc = 0.0

        self.parse_args()
        self.resolve_device()
        self.log_args()

        self.init_dataset()
        self.init_model()

        self.init_logger()

        self.run()

        self.cleanup()

    @staticmethod
    def log(message, level="info"):
        if level == "info":
            message = f"{TColor.OKBLUE}info{TColor.ENDC}: {message}"
        elif level == "error":
            message = f"{TColor.FAIL}error{TColor.ENDC}: {message}"
        print(f"{message}")

    def parse_args(self):
        parser = argparse.ArgumentParser(
            description="Train ICVLPR model on Indonesian Commercial Vehicle License Plate dataset"
        )

        # Training Setting
        parser.add_argument(
            "--device", type=str, default=None, help="Device to use for training"
        )
        parser.add_argument(
            "--learning-rate", type=float, default=0.001, help="Initial learning rate"
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=32,
            help="Batch size for training in each iteration",
        )
        parser.add_argument(
            "--epoch-start", type=int, default=0, help="Start epoch number"
        )
        parser.add_argument(
            "--epoch-end", type=int, default=1500, help="End epoch number"
        )
        parser.add_argument(
            "--learning-rate-scheduler-step",
            type=int,
            default=700,
            help="Learning rate scheduler step",
        )
        parser.add_argument(
            "--save-last",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Save last checkpoint of the run",
        )

        # Checkpoint
        parser.add_argument(
            "--checkpoint",
            type=str,
            help="If set, will restore the model from the checkpoint path and continue training",
        )
        parser.add_argument(
            "--checkpoint-dir",
            type=str,
            default="checkpoints",
            help="Directory to save checkpoints",
        )
        parser.add_argument(
            "--checkpoint-save-interval",
            type=int,
            default=100,
            help="Save checkpoint interval",
        )
        parser.add_argument(
            "--checkpoint-prefix",
            type=str,
            default="epoch",
            help="Checkpoint prefix when saving",
        )

        # Logging wandb
        parser.add_argument(
            "--wandb",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Enable wandb logging",
        )
        parser.add_argument("--run-id", type=str, default=None, help="Run ID for wandb")

        # Spatial Transformer Network
        parser.add_argument(
            "--stn-enable-at", type=int, default=300, help="Enable STN at epoch"
        )

        self.args = parser.parse_args()

    def resolve_device(self):
        if self.args.device is None:
            self.args.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(self.args.device)

    def log_args(self):
        print("-" * 20)
        for key, value in vars(self.args).items():
            print(f"{TColor.GREEN}{key:<35}: {TColor.BOLD}{value}{TColor.ENDC}")
        print("-" * 20)

    def init_dataset(self):
        self.log("Initializing dataset...")

        def pad_target_sequence(batch):
            """Collate function for the dataloader.

            Automatically adds padding to the target of each batch.
            """
            # Extract samples and targets from the batch
            samples, targets = zip(*batch)

            # Pad the target sequences to the same length
            padded_targets = pad_sequence(targets, batch_first=True, padding_value=0)

            # Return padded samples and targets
            return torch.stack(samples), padded_targets

        img_transforms = transforms.Compose(
            [
                transforms.RandomAffine(
                    degrees=(-5, 5),
                    translate=(0.07, 0.05),
                    scale=(0.7, 1),
                    shear=(-10, 10),
                ),
                transforms.RandomPerspective(
                    distortion_scale=0.3,
                    p=0.5,
                ),
            ]
        )

        self.ds_train = ICVLPDataset(
            "data",
            subset="train",
            transform=img_transforms,
            download=True,
        )
        self.dl_train = DataLoader(
            self.ds_train,
            batch_size=self.args.batch_size,
            shuffle=True,
            collate_fn=pad_target_sequence,
        )
        self.ds_val = ICVLPDataset(
            "data",
            subset="val",
            transform=img_transforms,
        )
        self.dl_val = DataLoader(
            self.ds_val,
            batch_size=self.args.batch_size,
            shuffle=False,
            collate_fn=pad_target_sequence,
        )
        self.log(f"Train Dataset Length: {len(self.ds_train)}")
        self.log(f"Val Dataset Length: {len(self.ds_val)}")
        self.log("Datasets initialized.")

    def init_model(self):
        self.log("Initializing model...")

        loc = LocNet()
        stn = SpatialTransformerLayer(
            localization=loc, align_corners=False
        )  # TODO: Experiment with align_corners=True
        self.model = LPRNet(stn=stn).to(self.device)

        if self.args.checkpoint:
            self.log(f"Restoring model from checkpoint: {self.args.checkpoint}")
            self.log(
                self.model.load_state_dict(
                    torch.load(self.args.checkpoint, map_location=self.device)
                )
            )

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.args.learning_rate,
        )
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.args.learning_rate_scheduler_step,
            gamma=0.1,
        )

        self.loss_fn = nn.CTCLoss(blank=0, zero_infinity=False, reduction="sum")

        self.decoder = GreedyCTCDecoder(blank=0)
        self.acc_fn = LetterNumberRecognitionRate(
            decoder=self.decoder, blank=0, reduction="sum"
        )

        self.log("Model initialized.")

    def init_logger(self):
        if not self.args.wandb:
            return

        import wandb

        config = {
            "learning-rate": self.args.learning_rate,
            "batch-size": self.args.batch_size,
            "epoch-start": self.args.epoch_start,
            "epoch-end": self.args.epoch_end,
            "epoch": abs(self.args.epoch_end - self.args.epoch_start),
            "dataset": self.ds_train.__class__.__name__,
            "loss": self.loss_fn.__class__.__name__,
            "optimizer": "Adam",
        }

        self.logger = wandb.init(
            project="icvlpr",
            config=config,
            id=self.args.run_id,
            resume="allow",
        )

    def log_epoch(self):
        if self.logger:
            self.logger.log(
                {
                    "loss/train": self.avg_loss,
                    "loss/val": self.val_avg_loss,
                    "acc/train": self.avg_acc,
                    "acc/val": self.val_avg_acc,
                },
                step=self.epoch,
            )

    def calculate_loss(self, logits, targets):
        target_lengths = targets.ne(0).sum(dim=1)
        input_lengths = torch.full(
            size=(logits.size(0),), fill_value=logits.size(1), dtype=torch.long
        )
        log_probs = logits.permute(2, 0, 1)
        log_probs = F.log_softmax(log_probs, dim=2)

        return self.loss_fn(
            log_probs=log_probs,
            targets=targets,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
        )

    def activate_stn(self):
        if not self.model.using_stn and self.args.stn_enable_at >= self.epoch:
            self.model.use_stn(True)
            self.log("Spatial Transformer Network activated.")

    def cleanup(self):
        if self.args.save_last and self.args.epoch_end == self.epoch:
            self.save_model()

        if self.logger:
            self.logger.finish()

    def run(self):
        for epoch in range(self.args.epoch_start, self.args.epoch_end):
            self.epoch = epoch + 1

            # Training
            self.activate_stn()
            self.model.train()
            self.train()

            # Validation
            self.model.eval()
            self.eval()

            # Log and save

            # Learning Rate Scheduler Step
            self.lr_scheduler.step()

    def train(self):
        running_loss = 0.0
        running_acc = 0.0

        for i, (images, targets) in enumerate(
            pbar := tqdm(
                self.dl_train,
                desc=f"Epoch {self.epoch}/{self.args.epoch_end}",
                unit="step",
                leave=False,
                position=0,
            )
        ):
            images = images.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.calculate_loss(logits, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            running_acc += self.acc_fn(logits, targets)

            pbar.set_postfix(
                loss=f"{running_loss / (i + 1):.4f}",
                acc=f"{running_acc / (i + 1):.4f}",
                val_loss=f"{self.val_avg_loss:.4f}",
                val_acc=f"{self.val_avg_acc:.4f}",
            )

        self.avg_loss = running_loss / len(self.dl_train)
        self.avg_acc = running_acc / len(self.dl_train)

    @torch.inference_mode()
    def eval(self):
        running_loss = 0.0
        running_acc = 0.0

        for i, (images, targets) in enumerate(
            pbar := tqdm(
                self.dl_val,
                desc=f"Epoch {self.epoch}/{self.args.epoch_end}",
                unit="step",
                leave=True,
                position=0,
            )
        ):
            images = images.to(self.device)
            targets = targets.to(self.device)

            logits = self.model(images)
            loss = self.calculate_loss(logits, targets)

            running_loss += loss.item()
            running_acc += self.acc_fn(logits, targets)

            pbar.set_postfix(
                loss=f"{self.avg_loss:.4f}",
                acc=f"{self.avg_acc:.4f}",
                val_loss=f"{running_loss / (i + 1):.4f}",
                val_acc=f"{running_acc / (i + 1):.4f}",
            )

        self.val_avg_loss = running_loss / len(self.dl_val)
        self.val_avg_acc = running_acc / len(self.dl_val)


if __name__ == "__main__":
    Trainer()
