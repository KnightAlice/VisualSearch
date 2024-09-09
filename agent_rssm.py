import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from models import Actor, Critic, Critic_vgg, ACMerge_vgg, ACMerge_resnet
from mae_components_no_cls import *


#Agent serve as both a policy maker and a value estimator

class Agent(nn.Module):

    def __init__(self, device, config, disable_critic=False, recog_threshold=0.5):
        super().__init__()
        self.grid_size=config['Environment']['grid_size'] #action grid size
        self.patch_size=config['PrefixCNN']['patch_size']
        self.device=device
        self.embed_dim=config['ViTEncoder']['embed_dim']
        self.disable_critic=disable_critic

        self.percept_grid_size=2*config['Environment']['radius']//self.patch_size #perception grid

        self.mae_encoder = MaskedViTEncoder(config, img_size= 2*config['Environment']['radius'], patch_size=self.patch_size, embed_dim=config['ViTEncoder']['embed_dim'], device=device).to(device)
        print(config['ViTEncoder']['weight_path'])
        print(self.mae_encoder.load_state_dict(torch.load(config['ViTEncoder']['weight_path']), strict=False))
        self.mae_encoder.requires_grad = False

        self.shift_encoder =  ShiftTransformer(config, img_size=2*config['Environment']['radius'], patch_size=self.patch_size, embed_dim=config['ShiftTransformer']['embed_dim']-1, device=device).to(device)
        print(config['ShiftTransformer']['weight_path'])
        print(self.shift_encoder.load_state_dict(torch.load(config['ShiftTransformer']['weight_path']), strict=False))
        self.shift_encoder.requires_grad = False

        self.mae_decoder = MaskedViTDecoder(config, img_size=2*config['Environment']['radius'], patch_size=self.patch_size, encoder_embed_dim=config['ViTEncoder']['embed_dim'], decoder_embed_dim=config['ViTDecoder']['embed_dim'], device=device, masked_decoder_loss=False, edge_emphasize_loss=False).to(device)
        print(config['ViTDecoder']['weight_path'])
        print(self.mae_decoder.load_state_dict(torch.load(config['ViTDecoder']['weight_path']), strict=False))
        self.shift_encoder.requires_grad = False
        
        self.mae_readout = Block(dim=config['ViTEncoder']['embed_dim'], num_heads=1, mlp_ratio=4, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=0, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None).to(device)
        '''
        self.gru_input_size=config['ViTEncoder']['embed_dim'] #512
        self.grucell = nn.GRUCell(input_size=self.gru_input_size, hidden_size=self.gru_input_size, bias=True, device=device, dtype=None)
        '''

        #self.actor(one step) = argmax G{pi} => -D_KL-ln(o|C)
    
    def initial(self):
        """Initial recurrent state for given number of environments."""
        pass

    def forward(self, x, target_id, prev_state=None, prev_action=None):
        #if multi_timestep, squeeze the 0,1 dim and then reshape
        x=x.to(self.device)
        time_step=0
        batch_size=x.shape[0]

        if len(x.shape)==5:
            time_step=x.shape[1]
            x=torch.flatten(x,start_dim=0,end_dim=1)
            target_id=torch.flatten(target_id.repeat(time_step))
            
        if prev_action==None:
            #center by default
            prev_action = (self.percept_grid_size//2*self.percept_grid_size+self.percept_grid_size//2) * torch.ones([batch_size, 2], device=self.device)

        if prev_state==None:
            prev_state = 10 * torch.randn([batch_size, self.percept_grid_size**2, self.embed_dim], device=self.device)
            
        assert prev_state!=None
        assert prev_action!=None #pre-define needed
        
        #map x,target_id to act_probs and state_values
        '''Target_values: whether the agent finds the target or relevant context, so it's not traditional Value
                          but something encourage the agent to stay to the target
        '''

        """Step 1
        x into encoder --> token embeddings
        """
        embedded, _ = self.mae_encoder.forward_encoder(x, mask_ratio=0.0)
        #torch.Size([B, 196, 512])
        
        """Step 2-> get priors P(s|s_t-1, a_t-1)
        prev_tokens -- into RSSM --> approximate current state from inner dynamics
        currently using ShiftTransformer here, but in the long run might be replaced by some sequence Transformer
        """
        h_prior = self.shift_encoder.forward_encoder(prev_state, prev_action) #h_prior should be singled out and used to train for accuracy
        
        h_prior_out = h_prior[:, :, 1:] #self.mae_readout(h_prior[:, :, 1:]) #deprive assistant dim
        
        """Step 3-> get posteriors P(s|o)
        """
        h_posterior_out = embedded #self.mae_readout(embedded)
        #h_out = self.grucell(hidden)

        #print(h_out)
        
        
        """Step 3
        sample the action with highest EFE -> subject to D_KL(Q(x)||P(x)), but in the context of token embeddings
        perhaps it's ok to try vector Euclidean distance instead.
        """

        act_batch = torch.linalg.vector_norm(h_prior_out-h_posterior_out, dim=-1) #calculate euclidean distance between state vectors
        

        '''
        #This is for determined action
        act_batch = torch.argmax(eu_dist, dim=-1) 
        x=act_batch//self.grid_size
        y=act_batch%self.grid_size
        act_xy=torch.stack([x,y],dim=-1)
        '''

        """Step 4
        mae_decoder -> loss(real obs, predicted)-> neg -> state_value?
        """
        
        """Step 5
        get grad_F == KL(expected obs || the real obs received) and optimize it
        """
        

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

        act_probs=F.softmax(act_batch, dim=-1) #softmax on 196 patches

        return act_probs, target_values, act_batch.sum(dim=-1), h_posterior_out
    
    def predict(self, obs, state, deterministic=False, **kwargs):
        raise NotImplementedError
        pi, _, state = self.forward(obs, state, **kwargs)
        
        if deterministic:
            return th.argmax(pi.probs), state
        else:
            return pi.sample(), state