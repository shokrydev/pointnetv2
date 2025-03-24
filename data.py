from torch_geometric.data.lightning.datamodule import LightningDataset
from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T


def get_modelnet_module(version="ModelNet10", sample_points=1024, batch_size=16, num_workers=6, augmentation=None):
    pre_transform = T.NormalizeScale()

    if augmentation is None:
        transform = T.SamplePoints(sample_points)
    #TODO implement augmentation; Do it separately to avoid augmenting the validation set

    train_dataset = ModelNet(
        root=version,
        name=version[-2:],
        train=True,
        transform=transform,
        pre_transform=pre_transform
    )

    val_dataset = ModelNet(
        root=version,
        name=version[-2:],
        train=False,
        transform=transform,
        pre_transform=pre_transform
    )
    
    modelnet_module = LightningDataset(train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)

    return modelnet_module

if __name__ == "__main__":
    m10 = get_modelnet_module()
    print(m10)