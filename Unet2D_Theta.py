import torch.nn as nn
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision
import pickle
import os
import torch.optim as optim
import json


import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import os

from torchmetrics import Accuracy, Precision, Recall, F1Score, Specificity, AUROC, MatthewsCorrCoef, ConfusionMatrix, AUROC, AveragePrecision

#In this model we add conv layers at the beginning to turn the image from 800x800->200x200

class Block(nn.Module):
    def __init__(self, in_ch, out_ch,stride_conv=1):
        super().__init__()
        #We put padding on the Conv3D so that we maintain the same depth
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3,padding=[1,1],stride=stride_conv)
        self.relu  = nn.LeakyReLU(negative_slope=0.01)
        self.bn1 =  nn.InstanceNorm2d(out_ch,eps=1.0e-05,momentum=0.1,affine=True,track_running_stats=False)
        
                
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3,padding=[1,1],stride=stride_conv)
        self.bn2 = nn.InstanceNorm2d(out_ch,eps=1.0e-05,momentum=0.1,affine=True,track_running_stats=False)
    
    def forward(self, x):
        return self.bn2(self.relu(self.conv2(self.bn1(self.relu(self.conv1(x))))))

class Encoder(nn.Module):
    def __init__(self, chs=(64,128,256,512)):
        super().__init__()
        self.Preproc=Block(3, chs[0],stride_conv=2)
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        #A lo largo de las dimensiones H,W, queremos que se divida por 2
        self.pool       = nn.MaxPool2d(3,padding=[1,1],stride=[2,2])
    
    def forward(self, x):
        ftrs = []
        x=self.Preproc(x)
        ftrs.append(x)
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs


class Decoder(nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.chs         = chs
        self.upconvs    = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], kernel_size=2,dilation=[1,1],
                                                padding=[0,0],stride=[2,2]) for i in range(len(chs)-1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)]) 
        
        self.depreproc= nn.Sequential( 
            nn.ConvTranspose2d(chs[-1], chs[-1], kernel_size=2,dilation=[1,1],padding=[0,0],stride=[2,2]),
            Block(chs[-1], chs[-1]),
            nn.ConvTranspose2d(chs[-1], chs[-1], kernel_size=2,dilation=[1,1],padding=[0,0],stride=[2,2]),
            Block(chs[-1], chs[-1])                                            
        )
        
    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            #chs=(1024, 512, 256, 128, 64)
            x        = self.upconvs[i](x)
            #We need to crop in cases where chs is very long
            x = self.crop(x,encoder_features[i].shape)
            x        = torch.cat([x, encoder_features[i]], dim=1)#This adds their channels together (e.g. 128+128=256)
            x        = self.dec_blocks[i](x)
        return self.depreproc(x)
    
    #Not using this crop function butĺl be necessary in the future
    def crop(self, x,shape):
        _,_, H_i, W_i = x.shape#Final shape sizes
        _,_, H_f, W_f = shape#Final shape sizes
        
        x=torch.narrow(x,2,(H_i-H_f)//2,(H_f+H_i)//2)
        x=torch.narrow(x,3,(W_i-W_f)//2,(W_f+W_i)//2)
        return x
    
    #Not using this crop function butĺl be necessary in the future
    def crop(self, x,shape):
        _,_, H_i, W_i = x.shape#Final shape sizes
        _,_, H_f, W_f = shape#Final shape sizes
        
        x=torch.narrow(x,2,(H_i-H_f)//2,(H_f+H_i)//2)
        x=torch.narrow(x,3,(W_i-W_f)//2,(W_f+W_i)//2)
        return x

#inputs to the net must be of the form: (N,C,D,H,W) (for a single image we take N=1)
class Net(pl.LightningModule):
    name="Unet2D_Theta"
    num_classes=2
    def __init__(self,chs=(3,64,128),frozen_encoder=False):
        super().__init__()

        self.encoder = Encoder(chs)

        if frozen_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False


        self.decoder = Decoder(chs[1:][::-1])#The decoder doesn´t take it to 1, but the step before
        self.fc1 = nn.Linear(16384, 1024)
        self.fc2 = nn.Linear(1024, 64)
        self.fc3 = nn.Linear(64, self.num_classes)
        #AQUI FALTA UN SOFTMAX
        self.softmax=nn.Softmax(dim=1)#we want the probs over channels
        self.chs=chs
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
        enc_output = self.encoder(x)
        x = self.decoder(enc_output[-1], enc_output[:-1][::-1])
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
        optimizer = torch.optim.RAdam(self.parameters(), lr = 0.00001, weight_decay=0.00001)  #optim.Adam(self.parameters(), lr=0.01)
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