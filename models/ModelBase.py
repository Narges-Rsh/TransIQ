import os
import torch
import torchmetrics
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import yaml

class ModelBase(pl.LightningModule):
    def __init__(self, classes, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        metrics = {
            'F1': torchmetrics.classification.MulticlassF1Score(num_classes=len(classes))
        }
        metrics = torchmetrics.MetricCollection(metrics)
        self.val_metrics = metrics.clone(f"val/")
        self.test_metrics = metrics.clone(f"test/")
        self.cm_metric = torchmetrics.classification.MulticlassConfusionMatrix(len(classes), normalize='true')

        self.classes = classes

    def on_train_start(self):
        if self.global_step==0: 
            init_logs = {k: 0 for k in self.val_metrics.keys()}
            init_logs.update({k: 0 for k in self.test_metrics.keys()})
            self.logger.log_hyperparams(self.hparams, init_logs)

    def on_test_start(self):
        
        test_len = len(self.trainer.datamodule.ds_test.indices)
        self.outputs_list = torch.zeros((test_len, len(self.classes)))
        self.targets_list = torch.zeros((test_len))
        self.snr_list = torch.zeros((test_len, self.trainer.datamodule.n_rx))

    def training_step(self, batch, batch_nb):
        output = self.forward(**batch)
        loss = self.loss(output, batch['y'])
        self.logger.log_metrics({'train/loss': loss, 'epoch': self.current_epoch}, self.global_step)
        self.log("train/loss", loss, prog_bar=True, logger=False)
        return loss

    def validation_step(self, batch, batch_nb):
        output = self.forward(**batch)
        self.val_metrics.update(output, batch['y'])
        self.cm_metric.update(output, batch['y'])

        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        # Confusion Matrix
        mpl.use("Agg")
        fig = plt.figure(figsize=(13, 13))
        cm = self.cm_metric.compute().cpu().numpy()
        self.cm_metric.reset()
        ax = sns.heatmap(cm, annot=True, fmt=".2f", cbar=False)
        # labels, title and ticks
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix')
        ax.xaxis.set_ticklabels(self.classes, rotation=90)
        ax.yaxis.set_ticklabels(self.classes , rotation=0)
        plt.tight_layout()
        if self.logger is not None:
            self.logger.experiment.add_figure("val/cm", fig, global_step=self.global_step)
       
        
    def test_step(self, batch, batch_nb):
        output = self.forward(**batch)
        self.test_metrics.update(output, batch['y'])
        self.cm_metric.update(output, batch['y'])
        
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True, prog_bar=True)
        
        batch_size = self.trainer.datamodule.batch_size
        batch_idx = batch_nb*batch_size
        self.outputs_list[batch_idx:batch_idx+batch_size] = output.detach().cpu().clone()
        self.targets_list[batch_idx:batch_idx+batch_size] = batch['y'].detach().cpu().clone()
        self.snr_list[batch_idx:batch_idx+batch_size] = batch['snr'].detach().cpu().clone()

    def on_test_epoch_end(self):
        # Confusion Matrix
        mpl.use("Agg")
        fig = plt.figure(figsize=(13, 13))
        cm = self.cm_metric.compute().cpu().numpy()
        self.cm_metric.reset()
        ax = sns.heatmap(cm, annot=True, fmt=".2f", cbar=False)
        # labels, title and ticks
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix')
        ax.xaxis.set_ticklabels(self.classes, rotation=90)
        ax.yaxis.set_ticklabels(self.classes , rotation=0)
        plt.tight_layout()
        if self.logger is not None:
            self.logger.experiment.add_figure("test/cm", fig, global_step=self.global_step)

        # CM -> CSV
        pd.DataFrame(cm, index=self.classes, columns=self.classes).to_csv(os.path.join(self.logger.log_dir, f"test_cm.csv"))
        
        # SNR plot
        test_snr = torch.round(self.snr_list.max(1)[0].flatten())
        SNRs, snr_counts = torch.unique(test_snr[test_snr<20], return_counts=True)
        F1s = []
        for snr in SNRs:
            ind = test_snr == snr
            F1s.append(torchmetrics.functional.classification.multiclass_f1_score(self.outputs_list[ind], self.targets_list[ind], len(self.classes)))
        F1s = torch.stack(F1s)

        self.graph_F1 = F1s
        self.graph_snr = SNRs

        self.outputs_list = self.outputs_list.zero_()

        fig = plt.figure(figsize=(8, 4))
        ax = fig.subplots()
        color = 'tab:blue'
        ax.plot(SNRs, F1s, linestyle='-', marker='o', color=color)
        ax.set_title('SNR F1')
        ax.set_xlabel('SNR')
        ax.set_ylabel('F1', color=color)
        ax.tick_params(axis='y', labelcolor=color)
        ax.set_ylim(0,1)
        ax.grid(True)

        if self.logger is not None:
            self.logger.experiment.add_figure("test/snr_f1", fig, global_step=self.global_step)

            csv_df = pd.DataFrame({"snr": SNRs, "f1": F1s, "count":  snr_counts})
            fpath = os.path.join(self.logger.log_dir, f"test_snr.csv")
            csv_df.to_csv(fpath, index=False)

        # Dump scalar metrics to YAML
        test_metrics_dict = self.test_metrics.compute()
        test_metrics_dict = {k:float(v) for k,v in test_metrics_dict.items()}
        with open(os.path.join(self.logger.log_dir, f"test_metrics.yaml"), 'w') as f:
            yaml.dump(test_metrics_dict, f)
