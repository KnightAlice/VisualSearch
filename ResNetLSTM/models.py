import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision.models as models

class Actor(nn.Module):
    def __init__(self,grid_size):
        super().__init__()
        self.layer1=nn.Conv2d(3,64,3,3,padding=1)#self.layer1=nn.Conv2d(3,64,3,3)
        self.layer2=nn.Conv2d(64,128,3,3,padding=1)#self.layer2=nn.Conv2d(64,128,3,3)
        self.layer3=nn.Conv2d(128,256,3,3,padding=1)#self.layer3=nn.Conv2d(128,256,3,3)
        self.layer4=nn.Conv2d(256,64,2,2,padding=2)#self.layer4=nn.Conv2d(256,128,3,3)
        self.layer5=nn.Conv2d(64,1,1,1)
        self.norm=nn.BatchNorm2d(1)#128
        #self.linear=nn.Linear(512,grid_size**2) #no linear but maxpool
        self.CNNs=nn.ModuleList([self.layer1,self.layer2,self.layer3,self.layer4,self.layer5])
        self.flatten=nn.Flatten()
        self.relu=nn.ReLU(inplace=True)
        
        
    def forward(self,x):
        for layer in self.CNNs:
            x=self.relu(layer(x))
        #x=self.linear(self.flatten(self.norm(x)))
        x=self.flatten(self.norm(x))

        return x

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1=nn.Conv2d(3,64,3,3)
        self.layer2=nn.Conv2d(64,128,3,3)
        self.layer3=nn.Conv2d(128,256,3,3)
        self.layer4=nn.Conv2d(256,128,3,3)
        self.maxpool=nn.MaxPool2d(2)
        self.norm=nn.BatchNorm2d(128)
        #self.linear=nn.Linear(128,1)
        self.linear=nn.Linear(128,18) #since coco-search has 18 cates
        self.CNNs=nn.ModuleList([self.layer1,self.layer2,self.layer3,self.layer4,self.maxpool])
        self.flatten=nn.Flatten()
        self.dropout=nn.Dropout(0.3)
        #self.softmax=nn.Softmax(dim=0)
        
    def forward(self,x):
        for layer in self.CNNs:
            x=F.relu(layer(x))
        x=self.dropout(x)
        x=self.norm(x)
        x=self.flatten(x)
        x=self.linear(x)
        #x=self.softmax(x)
        
        return x

class Critic_vgg(nn.Module):
    def __init__(self,weights=models.VGG16_Weights.IMAGENET1K_V1):
        super().__init__()
        self.vgg=models.vgg16(weights=weights)
        
        self.relu=nn.ReLU(inplace=True)
        self.read_out=nn.Linear(1000,18)
        

    def forward(self,x):
        output=self.read_out(self.relu(self.vgg(x)))
        return output

class ACMerge_vgg(nn.Module):
    def __init__(self,weights=models.VGG16_Weights.IMAGENET1K_V1,grid_size=6):
        super().__init__()
        self.vgg=models.vgg16(weights=weights)
        self.actor_read_out=nn.Conv2d(3840,1,1,1)
        self.relu=nn.ReLU(inplace=True)
        self.value_read_out=nn.Linear(1000,18)
        self.flatten=nn.Flatten()
       
        self.grid_size=grid_size
        
    def forward(self,x):
        layer_collection=[]
        for layer in self.vgg.features:
            x=layer(x)
            if 'ReLU' in str(layer) and x.shape[1]>128: #filter out low level features
                layer_collection.append(x.detach()) #grad unrequired

        saliency=self.saliency_map(layer_collection) #3840 channels
        act=self.actor_read_out(saliency).squeeze().flatten(start_dim=-2,end_dim=-1) #e.g. 6x6->36 for Categorial to operate correctly, won't effect position info
        
        x=self.flatten(self.vgg.avgpool(x))
        x=self.vgg.classifier(x)
        value=self.value_read_out(self.relu(x))

        #print(act)

        return act,value

    def saliency_map(self,layer_collection):
        new_maps=[]
        for map in layer_collection:
            new_maps.append(F.upsample(map,(self.grid_size,self.grid_size)))
        rescaled_maps=torch.concatenate(new_maps,dim=1)
        #print(rescaled_maps.shape)
        return rescaled_maps

class ResNetLSTM(nn.Module):
    def __init__(self,grid_size=6,resnet_weight=models.ResNet18_Weights.IMAGENET1K_V1):
        super().__init__()
        self.resnet=models.resnet18(weights=resnet_weight)
        self.lstm_map=nn.LSTMCell(input_size=1000,hidden_size=grid_size**2)
        self.lstm_state=nn.LSTMCell(input_size=1000,hidden_size=1)
        self.relu=nn.ReLU(inplace=True)
        self.grid_size=grid_size

    def forward(self,x,ht1_0=None,ct1_0=None,ht2_0=None,ct2_0=None):
        x=self.resnet(x)
        if ht1_0==None and ct1_0==None:
            ht1_0=torch.zeros(x.shape[0],self.grid_size**2).to(x.device)
            ct1_0=torch.zeros(x.shape[0],self.grid_size**2).to(x.device)
            ht2_0=torch.zeros(x.shape[0],1).to(x.device)
            ct2_0=torch.zeros(x.shape[0],1).to(x.device)
        h1,c1=self.lstm_map(self.relu(x),(ht1_0,ct1_0))
        h2,c2=self.lstm_state(self.relu(x),(ht2_0,ct2_0))
        return h1,c1,h2,c2