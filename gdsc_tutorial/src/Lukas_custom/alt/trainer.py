from glob import glob

import lightning as L
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, precision_recall_fscore_support 
from torch_audiomentations import ApplyImpulseResponse


class TrainModule(L.LightningModule):
    def __init__(self, model, loss_fn, optimizer_name, optimizer_hparams, cfg):
        """
        Inputs:
            model_name - Name of the model/CNN to run. Used for creating the model (see function below)
            model_hparams - Hyperparameters for the model, as dictionary.
            optimizer_name - Name of the optimizer to use. Currently supported: Adam, SGD
            optimizer_hparams - Hyperparameters for the optimizer, as dictionary. This includes learning rate, weight decay, etc.
        """
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters(ignore=["model", 'loss_fn'])
        # Create model
        self.model = model
        # Create loss module
        self.loss_fn = loss_fn
        self.impulse = ApplyImpulseResponse(p=cfg.impulse_prob, p_mode="per_example", 
                                            sample_rate=cfg.sample_rate, mode="per_example",
                                            compensate_for_propagation_delay=True, ir_paths=glob("../../data/irs/*/*/mono/*"))

    def forward(self, wave):
        # Forward function that is run when visualizing the graph
        return self.model(wave)

    def configure_optimizers(self):
        # We will support Adam or SGD as optimizers.
        if self.hparams.optimizer_name == "Adam":
            # AdamW is Adam with a correct implementation of weight decay (see here
            # for details: https://arxiv.org/pdf/1711.05101.pdf)
            optimizer = optim.AdamW(self.parameters(), **self.hparams.optimizer_hparams)
        elif self.hparams.optimizer_name == "SGD":
            optimizer = optim.SGD(self.parameters(), **self.hparams.optimizer_hparams)
        else:
            assert False, f'Unknown optimizer: "{self.hparams.optimizer_name}"'

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=1.0e-9,
        )
        return [optimizer] , [scheduler]

    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        wave, labels = batch['wave'], batch['labels']
        batch_size = wave.shape[0]
        wave = self.impulse(wave.view(batch_size, 1, -1)).view(batch_size, -1)
        preds = self.model(wave)
        loss = self.loss_fn(preds, labels)
        #loss = loss.mean(dim=1).sum()
        acc = (preds.argmax(dim=-1) == labels.argmax(dim=-1)).float().mean()
        precision, recall, f1, _ = precision_recall_fscore_support(labels.argmax(dim=-1).detach().cpu().numpy(), 
                                                                   preds.argmax(dim=-1).detach().cpu().numpy(), average="macro",
                                                                  zero_division=1)

        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log("train_acc", acc, on_step=False, on_epoch=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_precision", precision, on_step=True, on_epoch=True)
        self.log("train_recall", recall, on_step=False, on_epoch=True)
        self.log("train_f1", f1, on_step=False, on_epoch=True, prog_bar=True)

        return loss  # Return tensor to call ".backward" on

    def validation_step(self, batch, batch_idx):
        wave, labels = batch['wave'], batch['labels']
        preds = self.model(wave).argmax(dim=-1)
        acc = (labels.argmax(dim=-1) == preds).float().mean()
        precision, recall, f1, _ = precision_recall_fscore_support(labels.argmax(dim=-1).detach().cpu().numpy(), 
                                                                   preds.detach().cpu().numpy(), average="macro",
                                                                  zero_division=1)

        # By default logs it per epoch (weighted average over batches)
        self.log("val_acc", acc, on_step=False, on_epoch=True)
        self.log("val_precision", precision, on_step=False, on_epoch=True)
        self.log("val_recall", recall, on_step=False, on_epoch=True)
        self.log("val_f1", f1, on_step=False, on_epoch=True, prog_bar=True)
        
    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs).argmax(dim=-1)
        acc = (labels == preds).float().mean()
        # By default logs it per epoch (weighted average over batches), and returns it afterwards
        self.log("test_acc", acc)