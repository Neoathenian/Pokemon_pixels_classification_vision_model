import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import os

from torchmetrics import Accuracy, Precision, Recall, F1Score, Specificity, AUROC, MatthewsCorrCoef, ConfusionMatrix, AUROC, AveragePrecision

class Block(nn.Module):
    def __init__(self, in_ch, out_ch,stride_conv=1):
        super().__init__()
        #We put padding on the Conv3D so that we maintain the same depth
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3,padding=[1,1],stride=stride_conv)
        self.relu  = nn.LeakyReLU(negative_slope=0.01)
        self.bn1 =  nn.InstanceNorm2d(out_ch,eps=1.0e-05,momentum=0.1,affine=True,track_running_stats=False)
    def forward(self, x):
        return self.bn1(self.relu(self.conv1(x)))

# Define the CNN architecture
class Net(pl.LightningModule):
    name="CNN_3"
    num_classes=2
    def __init__(self):
        super(Net, self).__init__()
        self.block1=Block(3,16)
        self.block2=Block(16,32)
        self.block3=Block(32,64)
        self.block4=Block(64,128)

        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, self.num_classes)
        self.softmax=nn.Softmax(dim=1)
        self.loss=nn.CrossEntropyLoss()


        self.metrics_classification = {
            'train_acc': Accuracy(num_classes=self.num_classes, task='multiclass'),
            'train_precision': Precision(num_classes=self.num_classes, task='multiclass'),
            'train_recall': Recall(num_classes=self.num_classes, task='multiclass'),
            'train_f1': F1Score(num_classes=self.num_classes, task='multiclass'),
            'train_specificity': Specificity(num_classes=self.num_classes, task='multiclass'),
            'train_mcc': MatthewsCorrCoef(num_classes=self.num_classes, task='multiclass'),
        }

        self.metrics_probs={
            'train_auroc': AUROC(num_classes=self.num_classes, task='multiclass'),
            'train_aupr': AveragePrecision(num_classes=self.num_classes, task='multiclass'),
        }
        
    def forward(self, x):
        batch_size = x.size(0)  # Get the batch size
        x = self.pool(self.block1(x))
        x = self.pool(self.block2(x))
        x = self.pool(self.block3(x))
        x = self.pool(self.block4(x))
        x = x.view(batch_size, -1)  # Flatten the tensor without using x.view()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.softmax(x)
        return x


    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y) #F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, on_epoch=True,on_step=False,prog_bar=True,logger=True)

        # Compute and log additional metrics
        preds = y_hat.argmax(dim=1)

        for name, metric in self.metrics_classification.items():
            self.log(name, metric.to(y.device)(preds, y), on_epoch=True, on_step=False, prog_bar=True, logger=True)

        for name, metric in self.metrics_probs.items():
            self.log(name, metric.to(y.device)(y_hat, y), on_epoch=True, on_step=False, prog_bar=True, logger=True)


        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.RAdam(self.parameters(), lr = 0.001, weight_decay=0.00001)  #optim.Adam(self.parameters(), lr=0.01)
        return optimizer

    def validation_step(self, batch, batch_idx):
        #Here´s the code for the validation step (right now it´s the same as the training step)
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y) #F.cross_entropy(y_hat, y)
        self.log('val_loss', loss, on_epoch=True,on_step=False,prog_bar=True,logger=True)

        # Compute and log additional metrics
        preds = y_hat.argmax(dim=1)

        for name, metric in self.metrics_classification.items():
            self.log(name, metric.to(y.device)(preds, y), on_epoch=True, on_step=False, prog_bar=True, logger=True)

        for name, metric in self.metrics_probs.items():
            self.log(name, metric.to(y.device)(y_hat, y), on_epoch=True, on_step=False, prog_bar=True, logger=True)

        return loss
    
    def test_step(self, batch, batch_idx):
        #Here´s the code for the test step (right now it´s the same as the training step)
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y) #F.cross_entropy(y_hat, y)
        self.log('test_loss', loss, on_epoch=True,on_step=False,prog_bar=True,logger=True)

        # Compute and log additional metrics
        preds = y_hat.argmax(dim=1)

        for name, metric in self.metrics_classification.items():
            self.log(name, metric.to(y.device)(preds, y), on_epoch=True, on_step=False, prog_bar=True, logger=True)

        for name, metric in self.metrics_probs.items():
            self.log(name, metric.to(y.device)(y_hat, y), on_epoch=True, on_step=False, prog_bar=True, logger=True)

        return loss