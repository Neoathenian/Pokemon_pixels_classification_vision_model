<<<<<<< HEAD
import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics import Accuracy, Precision, Recall, F1Score, Specificity, MatthewsCorrCoef, AUROC, AveragePrecision
import pytorch_lightning as pl
from torch.optim import Adam

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Net(pl.LightningModule):
    name="ResNet_1"
    num_classes=10
    def __init__(self, block, num_blocks,):
        super(Net, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, self.num_classes)

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
        self.loss=nn.CrossEntropyLoss()


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        batch_size = x.size(0)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(batch_size, -1)
        out = self.linear(out)
        return out

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
        print("Pre")
        loss = self.loss(y_hat, y) #F.cross_entropy(y_hat, y)
        print("Pre2")
        self.log('val_loss', loss, on_epoch=True,on_step=False,prog_bar=True,logger=True)
        print("Pos")

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
def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])
=======
import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics import Accuracy, Precision, Recall, F1Score, Specificity, MatthewsCorrCoef, AUROC, AveragePrecision
import pytorch_lightning as pl
from torch.optim import Adam

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Net(pl.LightningModule):
    name="ResNet_1"
    num_classes=10
    def __init__(self, block, num_blocks,):
        super(Net, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, self.num_classes)

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
        self.loss=nn.CrossEntropyLoss()


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        batch_size = x.size(0)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(batch_size, -1)
        out = self.linear(out)
        return out

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
        print("Pre")
        loss = self.loss(y_hat, y) #F.cross_entropy(y_hat, y)
        print("Pre2")
        self.log('val_loss', loss, on_epoch=True,on_step=False,prog_bar=True,logger=True)
        print("Pos")

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
def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])
>>>>>>> 4c95344 (.)
