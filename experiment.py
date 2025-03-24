import argparse
#import os
#import random
#from glob import glob
#random.seed(42)
#torch.manual_seed(42)

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger

from data import get_modelnet_module
from light_pointnet2 import PointNet2



parser = argparse.ArgumentParser(description='Run PointNet++ on ModelNet Experiments.')


parser.add_argument('--task', type=str) # "ModelNet10","ModelNet40"

parser.add_argument('--modelnet_version', choices=['ModelNet10', 'ModelNet40'], default='ModelNet10')
parser.add_argument('--sample_points', type=int, default=2048, choices=range(256,4097))
parser.add_argument('--ratio1', type=float, default=0.75)
parser.add_argument('--ratio2', type=float, default=0.33)
parser.add_argument('--radius1', type=float, default=0.48)
parser.add_argument('--radius2', type=float, default=0.24)

parser.add_argument('--epochs', type=int, default=20, choices=range(500))
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--dropout', type=float, default=0.1)

parser.add_argument('--batch_size', type=int, default=20, choices=range(1,129))
parser.add_argument('--num_workers', type=int, default=6, choices=range(1,11))

config = parser.parse_args()

"""
classes = sorted([
    x.split(os.sep)[-2]
    for x in glob(os.path.join(
        config.modelnet_dataset_alias, "raw", '*', ''
    ))
]) """

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

datamodule = get_modelnet_module(
    version=config.modelnet_version,
    sample_points=config.sample_points,
    batch_size=config.batch_size,
    num_workers=config.num_workers
    )

model = PointNet2(
    datamodule=datamodule,
    set_abstraction_ratio_1=config.ratio1,
    set_abstraction_ratio_2=config.ratio2,
    set_abstraction_radius_1=config.radius1,
    set_abstraction_radius_2=config.radius2,
    dropout=config.dropout,
    learning_rate=config.lr
    ).to(device)

wandb_logger = WandbLogger(
        project="pointnet2_modelnet",
        group=config.modelnet_version,
        name="standard",
        #log_model=True,
        config=vars(config),
        offline=True
    )

trainer = Trainer(
    #callbacks=callbacks, TODO: Implement callbacks
    logger=wandb_logger,
    accelerator='gpu',  #'auto',
    devices=[0], #'auto',
    #enable_checkpointing=True, TODO: Implement checkpointing
    max_epochs=config.epochs
    )

trainer.fit(model, train_dataloaders=datamodule.train_dataloader(), val_dataloaders=datamodule.val_dataloader())