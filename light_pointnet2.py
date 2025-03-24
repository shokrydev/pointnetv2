import torch
import torch.nn.functional as F
import lightning.pytorch as L
from torch_geometric.nn import MLP, fps, global_max_pool, radius
from torch_geometric.nn import PointNetConv as PointConv

class SetAbstraction(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch

class GlobalSetAbstraction(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch

class PointNet2(L.LightningModule):
    def __init__(
        self,
        datamodule,
        set_abstraction_ratio_1 = 0.75, # [0.1-1.0]
        set_abstraction_ratio_2 = 0.33, # [0.1-1.0]
        set_abstraction_radius_1 = 0.48, # [0.1-1.0]
        set_abstraction_radius_2 = 0.24, # [0.1-1.0]
        dropout = 0.1,
        learning_rate = 0.001
    ):
        super().__init__()

        self.datamodule = datamodule

        # Input channels account for both `pos` and node features.
        self.sa1_module = SetAbstraction(
            set_abstraction_ratio_1,
            set_abstraction_radius_1,
            MLP([3, 64, 64, 128])
        )
        self.sa2_module = SetAbstraction(
            set_abstraction_ratio_2,
            set_abstraction_radius_2,
            MLP([128 + 3, 128, 128, 256])
        )
        self.sa3_module = GlobalSetAbstraction(MLP([256 + 3, 256, 512, 1024]))

        self.mlp = MLP([1024, 512, 256, 10], dropout=dropout, norm=None)

        self.learning_rate = learning_rate

    def forward(self, data):
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out

        return self.mlp(x).log_softmax(dim=-1)

    def training_step(self, batch, batch_idx):
        prediction = self(batch)
        loss = F.nll_loss(prediction, batch.y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        prediction = self(batch)
        loss = F.nll_loss(prediction, batch.y)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        prediction = self(batch)
        loss = F.nll_loss(prediction, batch.y)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate
        )
        #TODO: Add learning rate scheduler
        return optimizer
    
    def train_dataloader(self):
        return self.datamodule.train_dataloader()

    def val_dataloader(self):
        return self.datamodule.val_dataloader()

    def test_dataloader(self):
        #TODO
        pass

