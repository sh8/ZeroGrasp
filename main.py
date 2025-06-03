import os

import ocnn
import torch as th
from torch import optim
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from torchmetrics.functional.classification import (
    binary_f1_score,
    binary_accuracy,
    binary_recall,
    binary_precision,
)
from torchvision.ops import sigmoid_focal_loss

import webdataset as wds

from zerograsp.utils.config import parse_config
from zerograsp.utils.dataset import make_sample_wrapper, decode_depth
from zerograsp.nets import ZeroGrasp

th.set_float32_matmul_precision("medium")
th.backends.cuda.matmul.allow_tf32 = True

INTRINSICS_K = {
    "mirage": [
        [572.41136339, 0.0, 325.2611084],
        [0.0, 573.57043286, 242.04899588],
        [0.0, 0.0, 1.0],
    ],
    "graspnet": [
        [927.1697387695312, 0.0, 651.3150634765625],
        [0.0, 927.3668823242188, 349.621337890625],
        [0.0, 0.0, 1.0],
    ],
    "ycb_video": [
        [1066.778, 0.0, 312.9869],
        [0.0, 1067.487, 241.3109],
        [0.0, 0.0, 1.0],
    ],
    "hope": [[1390.53, 0.0, 964.957], [0.0, 1386.99, 522.586], [0.0, 0.0, 1.0]],
    "hb": [[537.4799, 0.0, 318.8965], [0.0, 536.1447, 238.3781], [0.0, 0.0, 1.0]],
    "woven_easy": [
        [610.1778394083658, 0.0, 640.0],
        [0.0, 610.1778394082355, 512.0],
        [0.0, 0.0, 1.0],
    ],
    "woven_normal": [
        [610.1778394083658, 0.0, 640.0],
        [0.0, 610.1778394082355, 512.0],
        [0.0, 0.0, 1.0],
    ],
    "woven_hard": [
        [610.1778394083658, 0.0, 640.0],
        [0.0, 610.1778394082355, 512.0],
        [0.0, 0.0, 1.0],
    ],
}


class BaseTrainer(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_name = config.model_name
        self.model = ZeroGrasp(config)

    def forward(self, batch):
        output = self.model(batch)

        occs = output["occs"]
        gt_occs = output["gt_occs"]
        loss_occ = 0
        stats_dict = {}
        for i, (occ, gt_occ) in enumerate(zip(occs, gt_occs)):
            depth = i + self.config.min_lod
            loss_occ += F.cross_entropy(occ, gt_occ)
            preds = occ.argmax(1).long()
            acc = binary_accuracy(preds, gt_occ.long())
            rec = binary_recall(preds, gt_occ.long())
            pre = binary_precision(preds, gt_occ.long())
            f1 = binary_f1_score(preds, gt_occ.long())
            stats_dict.update(
                {
                    f"acc_{depth}": acc,
                    f"rec_{depth}": rec,
                    f"pre_{depth}": pre,
                    f"f1_{depth}": f1,
                }
            )
        loss_dict = {"loss_occ": loss_occ}

        signal = output["signal"]
        gt_signal = output["gt_signal"]

        loss_nrm = th.mean(th.sum((gt_signal[:, :3] - signal[:, :3]) ** 2, dim=1))
        loss_sdf = th.mean((gt_signal[:, 3:4] - signal[:, 3:4]) ** 2)
        loss_dict["loss_nrm"] = loss_nrm
        loss_dict["loss_sdf"] = loss_sdf

        if self.config.predict_grasp:
            valid = gt_signal[:, 4:5]
            loss_validity = th.mean((valid - signal[:, 4:5]) ** 2)
            loss_dict["loss_validity"] = 5.0 * loss_validity

            valid = valid.squeeze(-1) > 0.1
            if valid.sum() > 0:
                loss_quality = th.mean(
                    (gt_signal[valid, 5:6] - signal[valid, 5:6]) ** 2
                )
                loss_tangent_1 = th.sum(
                    (gt_signal[valid, 6:9] - signal[valid, 6:9]) ** 2, dim=1
                )
                loss_tangent_1 = th.sum(
                    (gt_signal[valid, 6:9] - signal[valid, 6:9]) ** 2, dim=1
                )
                loss_tangent_2 = th.sum(
                    (-gt_signal[valid, 6:9] - signal[valid, 6:9]) ** 2, dim=1
                )
                loss_tangent = th.mean(th.minimum(loss_tangent_1, loss_tangent_2))
                loss_approach = th.mean(
                    th.sum((gt_signal[valid, 9:12] - signal[valid, 9:12]) ** 2, dim=1)
                )
                loss_standoff = th.mean(
                    (gt_signal[valid, 12:13] - signal[valid, 12:13]) ** 2
                )
                loss_width = th.mean(
                    (gt_signal[valid, 13:14] - signal[valid, 13:14]) ** 2
                )
                loss_dict["loss_quality"] = 5.0 * loss_quality
                loss_dict["loss_tangent"] = loss_tangent
                loss_dict["loss_approach"] = loss_approach
                loss_dict["loss_standoff"] = 2.5 * loss_standoff
                loss_dict["loss_width"] = 2.5 * loss_width
            else:
                loss_dict["loss_quality"] = 0.0
                loss_dict["loss_tangent"] = 0.0
                loss_dict["loss_approach"] = 0.0
                loss_dict["loss_standoff"] = 0.0
                loss_dict["loss_width"] = 0.0

        if "pm" in output:
            global_step = self.trainer.global_step
            anneal_step = global_step % self.config.kl_loss_cycle_len
            anneal_start = 0
            anneal_end = (
                self.config.kl_loss_cycle_len // 2
            )  # optimize full weight for second half of cycle
            anneal_weight = (anneal_step - anneal_start) / (anneal_end - anneal_start)
            anneal_weight = 1.0 if anneal_weight > 1.0 else anneal_weight
            pm, pv, qm, qv = output["pm"], output["pv"], output["qm"], output["qv"]
            element_wise = 0.5 * (
                th.log(pv) - th.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1
            )
            loss_kl_unweighted = element_wise.sum(-1).mean()
            # loss_dict['loss_kl'] = self.config.kl_weight * loss_kl_unweighted
            loss_dict["loss_kl"] = (
                anneal_weight * self.config.kl_weight * loss_kl_unweighted
            )
            stats_dict["kl_weight"] = anneal_weight
            stats_dict["loss_kl_unweighted"] = (
                self.config.kl_weight * loss_kl_unweighted
            )

        return loss_dict, stats_dict

    def training_step(self, batch, batch_idx):
        loss_dict, stats_dict = self(batch)
        loss = 0.0
        for name, value in loss_dict.items():
            loss += value
            self.log(
                f"train_{name}",
                value,
                on_step=True,
                prog_bar=True,
                sync_dist=True,
                rank_zero_only=True,
            )
        for name, value in stats_dict.items():
            self.log(
                f"train_{name}",
                value,
                on_step=True,
                prog_bar=True,
                sync_dist=True,
                rank_zero_only=True,
            )
        self.log(
            f"train_loss",
            loss,
            on_step=True,
            prog_bar=True,
            sync_dist=True,
            rank_zero_only=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss_dict, stats_dict = self(batch)
        loss = 0.0
        for name, value in loss_dict.items():
            loss += value
            self.log(
                f"valid_{name}",
                value,
                on_step=True,
                prog_bar=True,
                sync_dist=True,
                rank_zero_only=True,
            )
        for name, value in stats_dict.items():
            self.log(
                f"valid_{name}",
                value,
                on_step=True,
                prog_bar=True,
                sync_dist=True,
                rank_zero_only=True,
            )
        self.log(
            f"valid_loss",
            loss,
            on_step=True,
            prog_bar=True,
            sync_dist=True,
            rank_zero_only=True,
        )
        return loss

    def configure_optimizers(self):
        if self.config.optimizer == "Adam":
            optimizer = optim.Adam(self.parameters(), lr=self.config.lr)
        elif self.config.optimizer == "AdamW":
            optimizer = optim.AdamW(
                self.parameters(),
                lr=self.config.lr,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise Exception(f"{self.config.optimizer} is not supported!")

        scheduler = th.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.config.scheduler_step,
            gamma=self.config.scheduler_decay,
        )

        return [optimizer], [
            {"scheduler": scheduler, "interval": "step", "frequency": 1}
        ]

    def train_dataloader(self):
        url = self.config.train_dataset_url
        if url.startswith("s3"):
            url = f"pipe:aws s3 cp {url} -"
        batch_size = self.config.batch_size
        num_workers = self.config.num_workers
        max_epochs = self.config.max_epochs
        dataset_size = self.config.train_dataset_size
        iter_per_epoch = dataset_size // (batch_size * self.trainer.num_devices)

        dataset = (
            wds.WebDataset(
                url,
                nodesplitter=wds.split_by_node,
                handler=wds.warn_and_continue,
                shardshuffle=True,
            )
            .decode(decode_depth, "pil", handler=wds.warn_and_continue)
            .map(
                make_sample_wrapper(
                    self.config, K=INTRINSICS_K[self.config.train_dataset_name]
                ),
                handler=wds.warn_and_continue,
            )
            .batched(batch_size)
        )

        dataloader = (
            wds.WebLoader(
                dataset,
                batch_size=None,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=False,
            )
            .repeat(max_epochs)
            .with_epoch(iter_per_epoch)
            .with_length(iter_per_epoch)
        )

        return dataloader

    def val_dataloader(self):
        url = self.config.val_dataset_url
        if url.startswith("s3"):
            url = f"pipe:aws s3 cp {url} -"
        dataset_size = self.config.val_dataset_size

        dataset = (
            wds.WebDataset(
                url,
                nodesplitter=wds.split_by_node,
                handler=wds.warn_and_continue,
                shardshuffle=False,
            )
            .decode(decode_depth, "pil")
            .map(
                make_sample_wrapper(
                    self.config, K=INTRINSICS_K[self.config.val_dataset_name]
                ),
                handler=wds.warn_and_continue,
            )
            .batched(1)
        )

        dataloader = (
            wds.WebLoader(
                dataset, batch_size=None, shuffle=False, num_workers=0, pin_memory=False
            )
            .with_epoch(dataset_size // (self.trainer.num_devices))
            .with_length(dataset_size // (self.trainer.num_devices))
        )

        return dataloader

    def on_load_checkpoint(self, checkpoint):
        if self.config.fine_tuning:
            print("Reset the optimizer states")
            checkpoint["optimizer_states"] = []


def main():
    config = parse_config()

    if config.mode == "training":
        model = BaseTrainer(config)
        # Store configurations in WandB
        checkpoint_path = os.path.join(
            "checkpoints", config.project_name, config.run_name
        )
        callbacks = [
            ModelCheckpoint(
                dirpath=checkpoint_path,
                save_top_k=-1,
                save_on_train_epoch_end=True,
                every_n_train_steps=config.checkpoint_every_n_steps,
            )
        ]
        logger = WandbLogger(
            project=config.project_name,
            entity="tri",
            name=config.run_name,
            log_model=True,
        )
    else:
        config.batch_size = 1
        callbacks = None
        logger = TensorBoardLogger(save_dir=".")

    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        logger=logger,
        log_every_n_steps=config.log_every_n_steps,
        strategy="ddp_find_unused_parameters_true",
        precision=32,
        check_val_every_n_epoch=1,
        gradient_clip_val=0.5,
        callbacks=callbacks,
    )

    if trainer.global_rank == 0:
        logger.experiment.config.update(config)
    trainer.fit(model=model, ckpt_path=config.checkpoint)


if __name__ == "__main__":
    main()
