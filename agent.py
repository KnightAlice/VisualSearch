import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from models import Actor, Critic, Critic_vgg, ACMerge_vgg, ACMerge_resnet


#Agent serve as both a policy maker and a value estimator

class Agent(nn.Module):

    def __init__(self,device,actor_path,critic_path,grid_size=6,disable_critic=False,recog_threshold=0.5):
        super().__init__()
        self.grid_size=grid_size
        self.device=device
        self.disable_critic=disable_critic
        #self.acmerge=ACMerge_vgg(grid_size=grid_size).to(device)
        self.acmerge=ACMerge_resnet(grid_size=grid_size).to(device)
        print(self.acmerge.value_read_out.load_state_dict(torch.load(critic_path)))
        self.acmerge.actor_read_out.weight=torch.load(actor_path)
        #self.actor=Actor(grid_size).to(device)
        self.recog_threshold=recog_threshold
        #self.critic=Critic().to(device)
        #if disable_critic==False:
        #    self.critic=Critic_vgg().to(device)
        #    self.critic.load_state_dict(torch.load(critic_path))
        #self.critic.require_grad=False

    def initial(self):
        """Initial recurrent state for given number of environments."""
        pass

    def forward(self,x,target_id):
        #map x,target_id to act_probs and state_values
        '''Target_values: whether the agent finds the target or relevant context, so it's not traditional Value
                          but something encourage the agent to stay to the target
        '''

        #if multi_timestep, squeeze the 0,1 dim and then reshape
        
        x=x.to(self.device)
        time_step=0
        batch_size=x.shape[0]
        if len(x.shape)==5:
            time_step=x.shape[1]
            x=torch.flatten(x,start_dim=0,end_dim=1)
            target_id=torch.flatten(target_id.repeat(time_step))

        #ac merge
        act_batch,state_values=self.acmerge(x,target_id)
        #act_batch=self.actor(x)

        if self.disable_critic==False:
            state_values=F.softmax(state_values,dim=1) #pretrained to recognize categories, adding softmax to score the output
            target_vec=F.one_hot(target_id,num_classes=state_values.shape[1]).to(self.device) #target->one hot vector
            target_values=torch.sum(state_values*target_vec,dim=1)#element_wise mul->prob
            '''recog_threshold to filter those observations less likely to be the target, and thus assign no rewards'''
            target_values[torch.where(target_values<self.recog_threshold)]=0.0

            self.last_state=torch.clone(state_values)
        else:#disabled
            target_values=torch.zeros_like(target_id).to(self.device)

        
            
        if time_step!=0:
            act_batch=act_batch.view(batch_size,time_step,self.grid_size**2)
            target_values=target_values.view(batch_size,time_step)

        act_probs=F.softmax(act_batch, dim=-1)

        return act_probs,target_values,state_values #epistemic_value
    
    def predict(self, obs, state, deterministic=False, **kwargs):
        raise NotImplementedError
        pi, _, state = self.forward(obs, state, **kwargs)
        
        if deterministic:
            return th.argmax(pi.probs), state
        else:
            return pi.sample(), state