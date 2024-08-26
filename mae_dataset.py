import os
import numpy as np
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader,Dataset
from torchvision.transforms import v2
import torch.nn.functional as F
from PIL import Image
import torch

def to_onehot(input_label,num_classes):
    memo={}
    pos=0
    return_list=[]
    for item in input_label:
        if item in memo:
            key=memo.get(item)
        else:
            key=pos
            memo.update({item:pos})
            pos=pos+1
            
        one_hot=np.zeros(num_classes)
        one_hot[key]=1
        return_list.append(one_hot)        
    return return_list,memo
    

class imagenet_dataset(Dataset):
    def __init__(self, img_size):
        train_dataset=[['image','class']]

        self.category=0
        self.img_size=img_size
        
        for item in os.listdir(r'./data/miniImageNet'):
            a=np.array(os.listdir(os.path.join('./data/miniImageNet',item)))
            b=np.array([item for x in range(len(a))])
            b=b.reshape(len(b),1)
            a=a.reshape(len(a),1)
            c=np.concatenate((a,b),axis=1)
            train_dataset=np.concatenate((train_dataset,c),axis=0)
            self.category=self.category+1

        #print(train_dataset[1:,])
        xy=train_dataset[1:,]
        #np.random.shuffle(xy)

        self.x=xy[:,0]
        self.y=xy[:,1]
        self.y_=xy[:,1]
        self.y,self.memo=to_onehot(self.y,self.category)
        self.n_samples=xy.shape[0]
        print("Data Preparation Done")
        
    def __getitem__(self,index):
        transform=transforms.v2.Compose([transforms.v2.ToImage(), 
                                         transforms.v2.ToDtype(torch.float32, scale=True),
                                         transforms.v2.Resize((self.img_size, self.img_size))])

        x_path=os.path.join('./data/miniImageNet',self.y_[index],self.x[index])
        im=Image.open(x_path)
        img=transform(im)
        if img.shape[0]==1:
            img=img.repeat(3,1,1)
        if img.shape[0]==4:
            img=img[:3]
        label=torch.from_numpy(self.y[index])

        return img,label
        
    def __len__(self):
        return self.n_samples


def get_miniImageNetDataLoader(batch_size, img_size, shuffle):
    id=imagenet_dataset(img_size)
    dataloader=DataLoader(dataset=id,batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True)
    print("Data Loaded.")
    return dataloader, id.memo
