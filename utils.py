from torch.utils.data import DataLoader,Dataset
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torchvision.datasets import ImageFolder
import os
from PIL import Image
from torchvision.transforms import v2, functional
import torch
import torch.nn.functional as F
#from environment import StaticImgEnv
from torch.distributions import Categorical
from torch.distributions import MultivariateNormal
import torch.nn as nn
import cv2
import numpy as np

class COCOSearch18(Dataset):

    def __init__(self, json, root="COCOSearch18/images"):
        self.root = root
        self.json = json
        self.ImageFolder=ImageFolder(root)

    def __len__(self):
        return len(self.json)
    
    def __getitem__(self, idx):
        '''
        Example:
        {'name': '000000478726.jpg', 'subject': 2, 'task': 'bottle', 'condition': 'present', 'bbox': [1063, 68, 95, 334], 'X': [848.2, 799.2, 731.1, 1114.4, 1121.5], 'Y': [517.2, 476.2, 383.4, 271.1, 205.9], 'T': [73, 193, 95, 635, 592], 'length': 5, 'correct': 1, 'RT': 1159, 'split': 'train'}
        Including the initial center fixation
        '''
        idx=2 #this is test
        fix_len=45 #restricted vector size
        transform=v2.Compose([v2.ToImage(), 
                              v2.ToDtype(torch.float32, scale=True)])
        
        json_file=self.json[idx]
        target=json_file['task']
        target_id=self.ImageFolder.class_to_idx[target]
        
        name=json_file['name']
        image_dir=os.path.join(self.root,target,name)
        img=transform(Image.open(image_dir))

        fix_x=torch.concatenate((torch.tensor(json_file['X']),-1*torch.ones(fix_len-json_file['length'])),dim=0)
        fix_y=torch.concatenate((torch.tensor(json_file['Y']),-1*torch.ones(fix_len-json_file['length'])),dim=0)
        fixations=torch.stack((fix_x,fix_y),dim=1)
        
        correct=json_file['correct']

        bbox=torch.tensor(json_file['bbox']) #target pos
        
        return img, target_id, fixations, correct, bbox

def select_action(obs_t, policy, sample_action=True, action_mask=None,
                  softmask=False, eps=1e-12):
    '''return grid index'''
    probs, values , epi_value = policy(*obs_t)
    if sample_action:
        m = Categorical(probs) #grid sample
        if action_mask is not None:
            raise NotImplementedError
            # prevent sample previous actions by re-normalizing probs
            probs_new = probs.clone().detach()
            if softmask:
                probs_new = probs_new * action_mask
            else:
                probs_new[action_mask] = eps
            
            probs_new /= probs_new.sum(dim=1).view(probs_new.size(0), 1)
                
            m_new = Categorical(probs_new)
            actions = m_new.sample()   
        else:
            actions = m.sample()
        log_probs = m.log_prob(actions)
        return actions.view(-1), log_probs, values.view(-1), probs, epi_value
    else:
        raise NotImplementedError
        probs_new = probs.clone().detach()
        probs_new[action_mask.view(probs_new.size(0), -1)] = 0
        actions = torch.argmax(probs_new, dim=1)
        return actions.view(-1), None, None, None

def collect_trajs(env,
                   policy,
                   max_traj_length=10,
                   is_eval=False):
    #collect agent trajectories and then feed to
    #theoritically max_traj_length should be shorter than max_steps!
    
    obs=env.observe()
    act_batch, log_prob, value, prob, epi_value = select_action((obs, env.target_id),
                                                    policy,
                                                    sample_action=True,
                                                    action_mask=None)#not implemented
    
    epi_values=[epi_value]
    rewards=[]
    actions=[]
    values=[value]
    status=None
    log_probs=[]
    states=[]
    
    #done=False
    
    if is_eval:
        trajs=[]
        raise NotImplementedError
        while env.step_id < max_traj_length-1:
            act_batch, log_prob, value, prob, epi_value= select_action((obs, env.target_id),
                                                    policy,
                                                    sample_action=True,
                                                    action_mask=None)#not implemented
            new_obs,reward,_,_,status=env.step(act_batch)
            actions.append(act_batch)

        trajs={
                'actions':torch.stack(actions)
                
        }
    
    else:
        trajs=[]
        while env.step_id < max_traj_length-1:

            act_batch, log_prob, value, prob, epi_value = select_action((obs, env.target_id),
                                                    policy,
                                                    sample_action=True,
                                                    action_mask=None)#not implemented

            new_obs,reward,_,_,status=env.step(act_batch)
            epi_values.append(epi_value)
            actions.append(act_batch)
            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            states.append(new_obs)
            #status.append which I dont know what should status be

        #batch-process so I guess there no need to specify batch idx
        trajs={
                'actions':torch.stack(actions),#[steps,batch]
                'rewards':torch.stack(rewards),#[steps,batch]
                'values':torch.stack(values),#[steps,batch]
                'status':status.T[1:],#[steps,batch] truncate 1 empty step
                'log_probs':torch.stack(log_probs),
                'states':torch.stack(states),
                'target_id':env.target_id, #[batch,]
                'epi_values':torch.stack(epi_values)
        }            
        
    
    return trajs

def process_trajs(trajs, gamma, mtd='CRITIC', tau=0.96, epistemic_coef=2):
    # compute discounted cummulative reward
    # rewards effect Value Loss, advtanges effect Action Loss
    device = trajs['log_probs'].device
    avg_return = 0

    epi_entropy=nn.BCELoss(reduction='none')
    epi_values=trajs['epi_values']

    acc_reward = torch.zeros_like(trajs['rewards'],
                                  dtype=torch.float,
                                  device=device)
    acc_reward[-1] = trajs['rewards'][-1]
    for i in reversed(range(acc_reward.size(0) - 1)):
        acc_reward[i] = trajs['rewards'][i] + gamma * acc_reward[i + 1]

    trajs['acc_rewards'] = acc_reward
    avg_return += acc_reward[0]
    
    values = trajs['values']
    # compute advantages
    if mtd == 'MC':  # Monte-Carlo estimation
        trajs['advantages'] = trajs['acc_rewards'] - values[:-1]

    elif mtd == 'CRITIC':  # critic estimation

        epi_bonus=epi_entropy(epi_values[1:],epi_values[:-1]).mean(-1)-0.5 #0.5 seems to be the crossentropy of two rand vectors
        '''I deprived the - values[:-1] term because the Value implemented here is not accumulated but present'''
        trajs['advantages'] = trajs['rewards'] + gamma * values[1:]  + epistemic_coef*epi_bonus #- values[:-1]
        #in the circumstance, 'rewards' should be externally fed

    elif mtd == 'GAE':  # generalized advantage estimation
        delta = trajs['rewards'] + gamma * values[1:] - values[:-1]
        adv = torch.zeros_like(delta, dtype=torch.float, device=device)
        adv[-1] = delta[-1]
        for i in reversed(range(delta.size(0) - 1)):
            adv[i] = delta[i] + gamma * tau * adv[i + 1]
        trajs['advantages'] = adv
    else:
        raise NotImplementedError

    return avg_return / len(trajs)

class RolloutStorage(object):
    def __init__(self, trajs_all, shuffle=True, norm_adv=False):
        self.obs_fovs = trajs_all["states"].transpose(1,0)
        self.actions = trajs_all["actions"].T
        self.lprobs = trajs_all['log_probs'].T
        self.tids = trajs_all['target_id']
        self.returns = trajs_all['acc_rewards'].T
        self.advs = trajs_all['advantages'].T
        self.status=trajs_all['status'].T
        
        #transpose->batch first
        
        if norm_adv:
            raise NotImplementedError
            self.advs = (self.advs - self.advs.mean()) / (self.advs.std() +
                                                          1e-8)

        self.sample_num = self.obs_fovs.size(0) #used to be 1
        self.shuffle = shuffle

    def get_generator(self, minibatch_size):
        minibatch_size = min(self.sample_num, minibatch_size)
        sampler = BatchSampler(SubsetRandomSampler(range(self.sample_num)),
                               minibatch_size,
                               drop_last=True)
        for ind in sampler:

            obs_fov_batch = self.obs_fovs[ind]
            actions_batch = self.actions[ind]
            tids_batch = self.tids[ind]
            return_batch = self.returns[ind]
            log_probs_batch = self.lprobs[ind]
            advantage_batch = self.advs[ind]
            status_batch=self.status[ind]

            yield (
                obs_fov_batch, tids_batch
            ), actions_batch, return_batch, log_probs_batch, advantage_batch, status_batch

def center_crop(images, xy, patch_size, img_size):
    #xy->coords, patch_size=16
    
    grid_size = img_size//patch_size #20
    b = images.shape[0]
    #xy = xy[:,:,0]
    scaled_xy = xy * patch_size
    shifted_xy=scaled_xy - img_size//2 + patch_size//2 #top, left
    image_folder = []
    masks = [] #mask inside boundary
    for i in range(b):
        image_folder.append(functional.crop(images[i], shifted_xy[i,0], shifted_xy[i,1], img_size, img_size)) #top,left,height,width
        mask = torch.ones((grid_size, grid_size),dtype=bool)

        if (grid_size//2-xy[i,0]) < 0: #from bottom
            mask[(grid_size//2-xy[i,0]):, :] = 0
        else:
            mask[:(grid_size//2-xy[i,0]), :] = 0
        
        if (grid_size//2-xy[i,0]) < 0: #from right
            mask[:, (grid_size//2-xy[i,1]):] = 0
        else:
            mask[:, :(grid_size//2-xy[i,1])] = 0
            
        masks.append(torch.flatten(mask))
    images = torch.stack(image_folder)
    masks = torch.stack(masks)
    return images, masks

def get_fovea_transform(height, width, device, alpha=0.1, beta=-0.005):
    
    center_x, center_y = height // 2, width // 2
    # Create meshgrid and calculate distances in a vectorized way
    x, y = np.ogrid[:height, :width]
    distances = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    factors = 0.2 + alpha * np.exp(-beta * distances)
    new_x = center_x + (x - center_x) * factors
    new_y = center_y + (y - center_y) * factors

    # Normalize coordinates to the range [-1, 1] for grid_sample
    new_x = (new_x / (height - 1)) * 2 - 1
    new_y = (new_y / (width - 1)) * 2 - 1

    # Stack and reshape to form a grid of shape [1, H, W, 2]
    grid = np.stack((new_y, new_x), axis=-1)
    grid = torch.tensor(grid, dtype=torch.float32).unsqueeze(0).to(device)

    return grid, factors
    
    
def foveate_transform_cuda(image, fix_pos, output_size, grid, device):
    # fix_pos -> top,left,height,width
    # Input [C, H, W]
    image=image.to(device)
    image = functional.crop(image,fix_pos[0],fix_pos[1], fix_pos[2], fix_pos[3])
    assert image.shape[0] == 3
    assert len(fix_pos) == 4

    # Convert image to tensor and add batch dimension
    image = image.unsqueeze(0)

    # Apply grid_sample
    trans_image = F.grid_sample(image, grid, mode='bilinear', align_corners=True)

    # Resize the image to the output size
    trans_image = F.interpolate(trans_image, size=(output_size, output_size), mode='bilinear', align_corners=True)

    # Remove batch dimension and convert to [C, H, W]
    trans_image = trans_image.squeeze(0)

    return trans_image