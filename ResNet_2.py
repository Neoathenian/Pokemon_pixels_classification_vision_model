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
    name="ResNet_1_17"
    num_classes=17
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

        #average macro es que calcula la medida para cada clase como binaria y luego hace la media
        self.metrics_classification_raw = {
            'acc': Accuracy(num_classes=self.num_classes, task='multiclass',average="none"),
            'precision': Precision(num_classes=self.num_classes, task='multiclass',average="none"),
            'recall': Recall(num_classes=self.num_classes, task='multiclass',average="none"),
            'f1': F1Score(num_classes=self.num_classes, task='multiclass',average="none"),
            'specificity': Specificity(num_classes=self.num_classes, task='multiclass',average="none"),
            'mcc': MatthewsCorrCoef(num_classes=self.num_classes, task='multiclass',average="none"),
        }       

        self.metrics_probs_raw={
            'auroc': AUROC(num_classes=self.num_classes, task='multiclass',average="none"),
            'aupr': AveragePrecision(num_classes=self.num_classes, task='multiclass',average="none"),
        }

        self.metrics_classification_mean = {
            'acc': Accuracy(num_classes=self.num_classes, task='multiclass',average="macro"),
            'precision': Precision(num_classes=self.num_classes, task='multiclass',average="macro"),
            'recall': Recall(num_classes=self.num_classes, task='multiclass',average="macro"),
            'f1': F1Score(num_classes=self.num_classes, task='multiclass',average="macro"),
            'specificity': Specificity(num_classes=self.num_classes, task='multiclass',average="macro"),
            'mcc': MatthewsCorrCoef(num_classes=self.num_classes, task='multiclass',average="macro"),
        }       

        self.metrics_probs_mean={
            'auroc': AUROC(num_classes=self.num_classes, task='multiclass',average="macro"),
            'aupr': AveragePrecision(num_classes=self.num_classes, task='multiclass',average="macro"),
        }

        #This is gonna be useful for validation
        self.probs = []
        self.targets = []


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
        self.log('train_loss', loss.float(), on_epoch=True,on_step=False,prog_bar=True,logger=True)

        self.log_stuff(y_hat, y, save_name="train_")

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.RAdam(self.parameters(), lr = 0.001, weight_decay=0.00001)  #optim.Adam(self.parameters(), lr=0.01)
        return optimizer


    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('val_loss', loss.float(), on_epoch=True,on_step=False,prog_bar=False,logger=True)

        # Store predictions and targets for all batches
        self.probs.append(y_hat)
        self.targets.append(y)
        
        return loss


    def on_validation_epoch_end(self):
        all_preds = torch.cat(self.probs, dim=0)
        all_targets = torch.cat(self.targets, dim=0)

        # Concatenate predictions and targets from all batches
        self.log_stuff(all_preds, all_targets, save_name="val_")


        self.probs = []
        self.targets = []


    def log_stuff(self, probs, targets, save_name="",prog_bar=False):
        preds = probs.argmax(dim=1)

        for name, metric in self.metrics_classification_mean.items():
            self.log(save_name+name, metric.to(targets.device)(preds, targets).float(), on_epoch=True, on_step=False, prog_bar=prog_bar, logger=True)

        for name, metric in self.metrics_classification_raw.items():
            metric_result = metric.to(targets.device)(preds, targets).float()
            if metric_result.dim() != 0:  # Check if metric_result is not a 0-d tensor
                for i, class_result in enumerate(metric_result):
                    self.log(f"{save_name}{name}_class{i}", class_result, on_epoch=True, on_step=False, prog_bar=prog_bar, logger=True)
            #else:
            #    self.log(save_name+name, metric_result, on_epoch=True, on_step=False, prog_bar=False, logger=True)

        
        for name, metric in self.metrics_probs_mean.items():
            self.log(save_name+name, metric.to(targets.device)(probs, targets).float(), on_epoch=True, on_step=False, prog_bar=prog_bar, logger=True)

        for name, metric in self.metrics_probs_raw.items():
            metric_result = metric.to(targets.device)(probs, targets).float()
            if metric_result.dim() != 0:  # Check if metric_result is not a 0-d tensor
                for i, class_result in enumerate(metric_result):
                    self.log(f"{save_name}{name}_class{i}", class_result, on_epoch=True, on_step=False, prog_bar=prog_bar, logger=True)
            #else:
            #    self.log(save_name+name, metric_result, on_epoch=True, on_step=False, prog_bar=False, logger=True)
    
    
    
def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])
