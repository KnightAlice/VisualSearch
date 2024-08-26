import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision.models as models
#import components

class Actor(nn.Module):
    def __init__(self,grid_size):
        '''Fucking stride assigned'''
        raise ValueError

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
    def __init__(self,vgg_weights=models.VGG16_Weights.IMAGENET1K_V1,grid_size=6):
        super().__init__()
        self.vgg=models.vgg16(weights=vgg_weights)
        #self.actor_read_out=nn.Conv2d(3840,1,1,1)
        self.actor_read_out_fake=nn.Conv2d(3840,18,1,1)
        #nn.init.constant_(self.actor_read_out_fake.weight,1.0)
        self.relu=nn.ReLU(inplace=True)
        self.value_read_out=nn.Linear(1000,18)
        self.flatten=nn.Flatten()
       
        self.grid_size=grid_size
        
    def forward(self,x,target_id=None):
        batch_size=x.shape[0]
        layer_collection=[]
        for layer in self.vgg.features:
            x=layer(x)
            if 'ReLU' in str(layer) and x.shape[1]>128: #filter out low level features
                layer_collection.append(x.detach()) #grad unrequired

        saliency=self.saliency_map(layer_collection) #3840 channels
        #print(self.actor_read_out_fake.weight.shape)
        act=self.actor_read_out_fake(saliency.squeeze()).flatten(start_dim=-2,end_dim=-1)[torch.arange(batch_size),target_id] #e.g. 6x6->36 for Categorial to operate correctly, won't effect position info

        x=self.flatten(self.vgg.avgpool(x))
        x=self.vgg.classifier(x)
        value=self.value_read_out(self.relu(x))

        return act,value

    def saliency_map(self,layer_collection):
        new_maps=[]
        for map in layer_collection:
            new_maps.append(F.upsample(map,(self.grid_size,self.grid_size)))
        rescaled_maps=torch.concatenate(new_maps,dim=1)
        #print(rescaled_maps.shape)
        return rescaled_maps
        
class ACMerge_vgg_trainer(nn.Module):
    '''for pretrain IVSN'''
    def __init__(self,weights=models.VGG16_Weights.IMAGENET1K_V1,grid_size=6):
        super().__init__()
        self.vgg=models.vgg16(weights=weights)
        self.actor_read_out=nn.Conv2d(3840,1,1,1)
        self.actor_read_out_fake=nn.Conv2d(3840,18,1,1)
        #nn.init.constant_(self.actor_read_out_fake.weight,1.0)
        self.relu=nn.ReLU(inplace=True)
        self.value_read_out=nn.Linear(1000,18)
        self.flatten=nn.Flatten()
       
        self.grid_size=grid_size
        
    def forward(self,x,target_id=None):
        layer_collection=[]
        for layer in self.vgg.features:
            x=layer(x)
            if 'ReLU' in str(layer) and x.shape[1]>128: #filter out low level features
                layer_collection.append(x.detach()) #grad unrequired

        saliency=self.saliency_map(layer_collection) #3840 channels
        #print(self.actor_read_out_fake.weight.shape)
        act=self.actor_read_out_fake(saliency.squeeze()).flatten(start_dim=-2,end_dim=-1).mean(dim=-1) #e.g. 6x6->36 for Categorial to operate correctly, won't effect position info

        x=self.flatten(self.vgg.avgpool(x))
        x=self.vgg.classifier(x)
        value=self.value_read_out(self.relu(x))

        return act,value

    def saliency_map(self,layer_collection):
        new_maps=[]
        for map in layer_collection:
            new_maps.append(F.upsample(map,(self.grid_size,self.grid_size)))
        rescaled_maps=torch.concatenate(new_maps,dim=1)
        #print(rescaled_maps.shape)
        return rescaled_maps
        
class ACMerge_resnet(nn.Module):
    def __init__(self,weights=models.ResNet18_Weights.IMAGENET1K_V1,grid_size=7):
        super().__init__()
        self.resnet=models.resnet18(weights=weights)
        self.actor_read_out=nn.Conv2d(512,18,1,1)
        self.relu=nn.ReLU(inplace=True)
        self.value_read_out=nn.Linear(512,18)
        self.flatten=nn.Flatten()
       
        self.grid_size=grid_size
        
    def forward(self,x,target_id=None):
        batch_size=x.shape[0]
        layer_collection=[]
        for layer_name,layer in self.resnet.named_children():
          
            if 'avg' in layer_name: #filter out low level features
                break #grad unrequired
            x=layer(x)
            #if 'layer' in layer_name:
            #    layer_collection.append(x.detach())

        #saliency=self.saliency_map(layer_collection) 
        act=self.actor_read_out(x).squeeze().flatten(start_dim=-2,end_dim=-1)[torch.arange(batch_size),target_id] #e.g. 6x6->36 for Categorial to operate correctly, won't effect position info 
        x=self.flatten(self.resnet.avgpool(x))
        value=self.value_read_out(self.relu(x))

        #print(act.shape)
        #print(value.shape)

        return act,value

    def saliency_map(self,layer_collection):
        new_maps=[]
        for map in layer_collection:
            new_maps.append(F.upsample(map,(self.grid_size,self.grid_size)))
        rescaled_maps=torch.concatenate(new_maps,dim=1)
        print(rescaled_maps.shape)
        return rescaled_maps
        
class ACMerge_resnet_trainer(nn.Module):
    def __init__(self,weights=models.ResNet18_Weights.IMAGENET1K_V1,grid_size=7):
        super().__init__()
        self.resnet=models.resnet18(weights=weights)
        self.actor_read_out=nn.Conv2d(512,18,1,1)
        self.relu=nn.ReLU(inplace=True)
        self.value_read_out=nn.Linear(512,18)
        self.flatten=nn.Flatten()
       
        self.grid_size=grid_size
        
    def forward(self,x):
        batch_size=x.shape[0]
        layer_collection=[]
        for layer_name,layer in self.resnet.named_children():
          
            if 'avg' in layer_name: #filter out low level features
                break #grad unrequired
            x=layer(x)
            #if 'layer' in layer_name:
            #    layer_collection.append(x.detach())

        #saliency=self.saliency_map(layer_collection) 
        act=self.actor_read_out(x).squeeze().flatten(start_dim=-2,end_dim=-1).mean(dim=-1) #e.g. 6x6->36 for Categorial to operate correctly, won't effect position info 
        x=self.flatten(self.resnet.avgpool(x))
        value=self.value_read_out(self.relu(x))

        #print(act)

        return act,value

    def saliency_map(self,layer_collection):
        new_maps=[]
        for map in layer_collection:
            new_maps.append(F.upsample(map,(self.grid_size,self.grid_size)))
        rescaled_maps=torch.concatenate(new_maps,dim=1)
        print(rescaled_maps.shape)
        return rescaled_maps