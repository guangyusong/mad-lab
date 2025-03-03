import torch
import typing as tp
import pytorch_lightning as pl
import torchmetrics as met
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from mad.metrics import Accuracy


class PLModelWrap(pl.LightningModule):
    """
    PyTorch Lightning model wrapper.
    
    Args:
        model (nn.Module): Model to wrap.
        mad_config (MADConfig): MAD configuration.
        metrics (list, optional): List of metrics to use.
    """

    def __init__(self, model, mad_config, metrics: list=['acc', 'ppl']):
        super().__init__()
        self.model = model
        self.mad_config = mad_config
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.mad_config.target_ignore_index)
        self.instantiate_metrics(metrics=metrics)
        self.save_hyperparameters('mad_config')

    def instantiate_metrics(self, metrics: list) -> None:
        mets = []
        for m in metrics:
            if m=='acc':
                mets.append(
                    Accuracy(
                        num_classes=self.model.vocab_size,
                        ignore_index=self.mad_config.target_ignore_index
                    )
                )
            elif m=='ppl':
                mets.append(met.text.Perplexity(ignore_index=self.mad_config.target_ignore_index))
            elif isinstance(m, met.Metric):
                mets.append(m)
            else:
                raise ValueError(f"invalid metric: {m}, must be one of 'acc', 'ppl' or a torchmetrics metric instance")

        mets = met.MetricCollection(mets)
        self.train_metrics = mets.clone(prefix='train/')
        self.test_metrics = mets.clone(prefix='test/')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def step(self,
        batch: tuple,
        batch_idx: int
    ) -> tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.loss_fn(
            outputs.view(-1, outputs.size(-1)),
            targets.view(-1)
        )
        return loss, outputs, targets
    
    def phase_step(self,
        batch: tuple,
        batch_idx: int,
        phase: str='train'
    ) -> tp.Dict[str, tp.Union[torch.Tensor, tp.Any]]:
        loss, outputs, targets = self.step(batch, batch_idx)
        self.log(f'{phase}/Loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        metrics = getattr(self, f'{phase}_metrics')(outputs, targets)
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return {'loss': loss, "outputs": outputs, "targets": targets}
    
    def training_step(self,
        batch: tuple,
        batch_idx: int
    ) -> tp.Dict[str, tp.Union[torch.Tensor, tp.Any]]:
        return self.phase_step(batch, batch_idx, phase='train')
    
    def validation_step(self,
        batch: tuple,
        batch_idx: int
    ) -> tp.Dict[str, tp.Union[torch.Tensor, tp.Any]]:
        # We currently do not use any validation data, only train/test
        return self.phase_step(batch, batch_idx, phase='test')

    def test_step(self,
        batch: tuple,
        batch_idx: int
    ) -> tp.Dict[str, tp.Union[torch.Tensor, tp.Any]]:
        return self.phase_step(batch, batch_idx, phase='test')

    def configure_optimizers(self) -> tp.Union[torch.optim.Optimizer, tp.Dict[str, tp.Any]]:
        # optimizer:
        # Get optimizer parameters from config
        optimizer_eps = getattr(self.mad_config, 'optimizer_eps', 1e-18) 
        beta1 = getattr(self.mad_config, 'beta1', 0.9)
        beta2 = getattr(self.mad_config, 'beta2', 0.999)
        use_param_grouping = getattr(self.mad_config, 'use_param_grouping', True)
        
        if self.mad_config.optimizer == 'adamw':
            if use_param_grouping:
                # RWKV-7 style parameter grouping
                # ENHANCED: More sophisticated parameter grouping like canonical RWKV-7
                lr_1x = []   # base learning rate
                lr_2x = []   # 2x learning rate
                lr_3x = []   # 3x learning rate
                lr_decay = [] # weight decay group
                
                # Get model dimension parameter to use as threshold
                model_dim = 0
                for name, param in self.named_parameters():
                    if 'embedding' in name and 'weight' in name:
                        model_dim = param.shape[1]
                        break
                
                if model_dim == 0:
                    for name, param in self.named_parameters():
                        if len(param.shape) >= 2:
                            model_dim = max(model_dim, param.shape[1])
                
                for name, param in self.named_parameters():
                    if not hasattr(self.mad_config, 'layerwise_lr') or self.mad_config.layerwise_lr <= 0:
                        # If layerwise_lr not enabled, everything goes to lr_1x except weight decay candidates
                        if len(param.shape) >= 2 and 'weight' in name and self.mad_config.weight_decay > 0:
                            lr_decay.append(name)
                        else:
                            lr_1x.append(name)
                    else:
                        if ("_w1" in name) or ("_w2" in name):
                            lr_1x.append(name)
                        elif len(param.shape) >= 2 and 'weight' in name and self.mad_config.weight_decay > 0:
                            lr_decay.append(name)
                        elif ("time_mix" in name) or ("time_maa" in name):
                            lr_1x.append(name)
                        elif ("time_decay" in name) or ("att.w0" in name):
                            lr_2x.append(name)
                        elif "time_first" in name:
                            lr_3x.append(name)
                        else:
                            lr_1x.append(name)
                
                param_dict = {n: p for n, p in self.named_parameters()}
                optim_groups = [
                    {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "lr": self.mad_config.lr}
                ]
                
                if len(lr_2x) > 0 and hasattr(self.mad_config, 'layerwise_lr') and self.mad_config.layerwise_lr > 0:
                    optim_groups.append(
                        {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "lr": 2.0 * self.mad_config.lr}
                    )
                
                if len(lr_3x) > 0 and hasattr(self.mad_config, 'layerwise_lr') and self.mad_config.layerwise_lr > 0:
                    optim_groups.append(
                        {"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0, "lr": 3.0 * self.mad_config.lr}
                    )
                
                if len(lr_decay) > 0 and self.mad_config.weight_decay > 0:
                    optim_groups.append(
                        {"params": [param_dict[n] for n in lr_decay],
                         "weight_decay": self.mad_config.weight_decay,
                         "lr": self.mad_config.lr}
                    )
                
                optimizer = torch.optim.AdamW(
                    optim_groups,
                    lr=self.mad_config.lr,  
                    betas=(beta1, beta2),  
                    eps=optimizer_eps,     
                    weight_decay=0.0,      
                    amsgrad=False
                )
            else:
                optimizer = torch.optim.AdamW(
                    self.parameters(),
                    lr=self.mad_config.lr,
                    weight_decay=self.mad_config.weight_decay,
                    betas=(beta1, beta2), 
                    eps=optimizer_eps     
                )
        elif self.mad_config.optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.mad_config.lr,
                weight_decay=self.mad_config.weight_decay
            )
        else:
            raise ValueError(f"invalid optimizer: {self.mad_config.optimizer}")
        
        # scheduler:
        if self.mad_config.scheduler == 'none':
            return optimizer
        elif self.mad_config.scheduler == 'cosine':
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.mad_config.epochs,
                eta_min=self.mad_config.min_lr,
                last_epoch=-1
            )
            return {'optimizer': optimizer, 'scheduler': scheduler}
        elif self.mad_config.scheduler == 'plateau':
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                patience=self.mad_config.plateau_patience,
                factor=self.mad_config.plateau_factor,
                min_lr=self.mad_config.min_lr,
                verbose=True
            )
            return {'optimizer': optimizer, 'scheduler': scheduler, 'monitor': "test/Loss_epoch"}
        else:
            raise ValueError(f"invalid scheduler: {self.mad_config.scheduler}")