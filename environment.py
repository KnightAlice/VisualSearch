'''https://www.gymlibrary.dev/api/core/'''
import torch
import numpy as np
#from .utils import foveal2mask
import warnings
from torchvision.transforms import v2
from PIL import Image,ImageDraw
from matplotlib import pyplot as plot
import utils
from torchvision.transforms import functional
import time

"""Clarify: act->{x,y} : x is row idx, which is counted from the top; y is column index, counted from the left
            but bbox->{x,y,w,h}, x is column idx, y is row idx, the same coord as ImageDraw by the way
"""

"""action coord:
            ↖ ↑ ↗
            ←  o  →
            ↙ ↓ ↘
"""

class StaticImgEnv:

    def __init__(self,env_args=None,agent=None,device='cuda', show_plot=False):
        self.step_id=0
        self.fixations=None #[horizontal,vertical]
        self.env_args=env_args

        max_steps=self.env_args['max_steps']
        if max_steps==-1: #-1 for unlimited
            raise NotImplementedError
            self.max_steps=50 
        else:
            self.max_steps=max_steps

        self.init='center' #or 'random'/'manual'
        self.action_range=self.env_args['action_range']
        assert self.action_range==self.env_args['vision_radius'] #in this version should be the same
        
        self.set_count=0 #count env num
        self.device=device
        
        #action-vision mapping constant matrix
        self.grid, scale_factors = utils.get_fovea_transform(height=self.action_range, width=self.action_range, device=device)
        self.scale_factors=torch.tensor(scale_factors).to(device)

        self.show_plot = show_plot

    def observe(self):
        #start_time=time.time()
        observations=[]

        #vision_radius including untransformed periphery
        height=self.env_args['vision_radius']
        width=self.env_args['vision_radius']
        
        for i in range(self.batch_size):
            fix_pos=self.fixations[i,self.step_id] #take latest fix
            

            #transform baseline
            #top: int, left: int, height: int, width: int
            #fovea_args
            
            top=int(fix_pos[0]-height/2) #fovea upper abs pos
            left=int(fix_pos[1]-width/2) #fovea left abs pos

            '''should add penalty here, it's just a boundary avoidance branch'''
            if top<= 0-height: #upper boundary, with tolerance of vision_radius
                top= 0-height+1
                self.fixation_reset(i,mode='center')
                self.reaching_boundary[self.step_id] = 1
            if left<= 0-width: #left boundary, with tolerance of vision_radius
                left= 0-width+1
                self.fixation_reset(i,mode='center')
                self.reaching_boundary[self.step_id] = 1
            if top>= self.boundary[2]: #lower boundary, with tolerance of vision_radius
                top= self.boundary[2]-1
                self.fixation_reset(i,mode='center')
                self.reaching_boundary[self.step_id] = 1
            if left>= self.boundary[3]: #right boundary, with tolerance of vision_radius
                left= self.boundary[3]-1
                self.fixation_reset(i,mode='center')
                self.reaching_boundary[self.step_id] = 1
                 
            #observation = v2.functional.crop(self.obs_space[i],top,left,height,width) #self-padding
            
            """TIME COMPLEXITY S.T. (height, width) -> vision_radius"""
            observation=utils.foveate_transform_cuda(self.obs_space[i], [top, left, height, width], output_size=2*self.env_args['radius'], grid=self.grid, device=self.device) #foveate
            #plot.imshow(functional.to_pil_image(observation.detach().cpu()))
            #raise ValueError
            
            observations.append(observation)
            
        observations=torch.stack(observations)
        #end_time=time.time()
        #print("Time:",end_time-start_time)
            
        return observations
    

    def get_reward(self):
        #external reward not assigned yet
         #if fixation is inside bbox, give rewards
        '''Actually in here we can implement some rewards for uncertainty suppression, maybe assign rewards in proportion to the information amount or relevant context?'''
        reward=torch.ones((self.batch_size,)).to(self.device)
        #print(self.bbox)
        #check if fixations hit the bbox
        #the bbox in the json is structured [x,y,w,h], where x is literally X coordinate
        #so we have to transpose our coord system
        hit_bbox_left=self.fixations[:,self.step_id, 1]>self.bbox[:,0]
        hit_bbox_right=self.fixations[:,self.step_id,1]<(self.bbox[:,0] + self.bbox[:,2])
        hit_bbox_top=self.fixations[:,self.step_id,0]>self.bbox[:,1]
        hit_bbox_bottom=self.fixations[:,self.step_id,0]<(self.bbox[:,1] + self.bbox[:,3])
        hit_bbox=torch.logical_and(torch.logical_and(torch.logical_and(hit_bbox_left,hit_bbox_right),hit_bbox_top),hit_bbox_bottom)
            
        with torch.no_grad():
            self.status[:,self.step_id]=hit_bbox.to(self.device)

            distract_penalty=self.status[:,self.step_id]-self.status[:,self.step_id-1]

            reward=reward*hit_bbox*50  # + distract_penalty * 20 - 5 #-5 being time step penalty

            
        return reward

    def step(self, act_batch):
        #fix->[horizontal,vertical]
        #fix step 0 is occupied
        
            
        self.step_id+=1
        
        #assert self.step_id < self.max_steps, "Error: Exceeding maximum step!"

        #act batch being the grid index, so a mapping is applied
        act_batch=self.idx_to_vector(act_batch)
        self.fixations[:,self.step_id,:]+= (act_batch+self.fixations[:,self.step_id-1,:])
        #boundary check
        if torch.any(self.fixations[:,self.step_id,:])<0:
            print("Reaching Upper/Left Boundary!")
            raise ValueError
        elif torch.any(self.fixations[:,self.step_id,1])>self.boundary[3]:
            print("Reaching Right Boundary!")
            raise ValueError
        elif torch.any(self.fixations[:,self.step_id,0])>self.boundary[2]:
            print("Reaching Bottom Boundary!")
            raise ValueError
            
        observations=self.observe()
        reward=self.get_reward() #state_update by the way
        #self.status_update()
        terminated=False
        truncated=False
        status=torch.clone(self.status) #not implemented
        
        #if self.step_id>=self.max_steps-1 and self.set_count%self.env_args.plot_freq==0: #step limit reached
        if self.set_count%self.env_args['plot_freq']==0 and self.show_plot:
            imgshow=self.plot_history()
            imgshow.save(f'img/{self.target_id[0]}_{self.step_id}.jpg')
            plot.imshow(imgshow)
            plot.show()
            if self.step_id>=self.max_steps-1 and self.status[0,self.step_id]==1:
                raise ValueError

        
        return observations,reward,terminated,truncated,status

    def status_update(self):
        '''check if a target is found'''
        '''I wonder if I should differ a miss from a not-found, because doing so could allow separate penalties to be assigned to different agent component (like actor and critic)'''
        
        '''if self.status_update_mtd == 'SOT':  # stop on target
            done = self.label_coding[torch.arange(self.batch_size
                                                  ), 0, act_batch]
        else:
            raise NotImplementedError
      
        done[self.status > 0] = 2
        self.status = done.to(torch.uint8)
        '''
        #Temp
        #assign_state=torch.randn(self.batch_size,)>0.8
        #self.status[:,self.step_id]=torch.logical_or(self.status[:,self.step_id-1],assign_state.to(self.device))
        pass
        
    def step_back(self):
        pass
        
    def reset(self):
        self.step_id = 0  # step id of the environment
        self.fixations = torch.zeros((self.batch_size, self.max_steps, 2),
                                     dtype=torch.long,
                                     device=self.device)
        self.status = torch.zeros((self.batch_size,self.max_steps,),
                                  dtype=torch.int8,
                                  device=self.device)#0 not found, 1 found
        self.reaching_boundary = torch.zeros((self.batch_size,self.max_steps,),
                                  dtype=torch.int8,
                                  device=self.device)#0 not hit, 1 hit
        self.is_active = torch.ones(self.batch_size,
                                    dtype=torch.uint8,
                                    device=self.device)
        self.reset_count=torch.zeros((self.batch_size,),
                                  dtype=torch.int8,
                                  device=self.device) #reset due to reaching obs boundaries

        if self.init=='center':
            h=self.obs_space.shape[2]
            w=self.obs_space.shape[3]
            #[b,3,h,w]

            px=np.round(h/2)
            py=np.round(w/2)
            self.fixations[:, 0, 0]=px
            self.fixations[:, 0, 1]=py
        elif self.init=='random':
            raise NotImplementedError

        #self.observe()
        #print("Init done.")

    def fixation_reset(self,idx,mode='center'):
        if mode=='center':
            h=self.obs_space.shape[2]
            w=self.obs_space.shape[3]
            #[b,3,h,w]

            px=np.round(h/2)
            py=np.round(w/2)
            self.fixations[idx, self.step_id, 0]=px
            self.fixations[idx, self.step_id, 1]=py
            
        self.reset_count[idx]=self.reset_count[idx] + 1

        return 1
            

    def set_data(self, img, target_id, bbox):
        self.obs_space = img
        self.boundary=[0,0,img.shape[2],img.shape[3]] #top, left, bottom, right
        self.bbox=bbox.to(self.device)
        self.batch_size=img.shape[0]
        self.target_id=target_id
        self.set_count+=1

        self.reset()

    def plot_history(self,idx=0):
        fixations=self.fixations[idx].detach().cpu().flip(-1) #[steps,coords], flip to adapt to ImageDraw coord
        fixations=[tuple(item) for item in fixations[0:self.step_id+1]]
        img=self.obs_space[idx].detach().cpu().squeeze(0)
        toimg = v2.ToPILImage()
        img=toimg(img)
        '''
        fovea=self.observe()[idx]
        fovea=toimg(fovea)
        plot.imshow(fovea)
        plot.show()
        '''
        found=torch.any(self.status[idx,:])
        steps='NaN'
        if found:
            steps=torch.where(self.status[idx,:])[0][0].item()
        draw = ImageDraw.Draw(img)
        xy=torch.clone(self.bbox[idx])
        xy[2]=xy[0]+xy[2]
        xy[3]=xy[1]+xy[3]
        draw.text([0,0],str(found.item())+' '+str(steps)+' '+str(self.reset_count[idx].item()),fill='blue',font_size=70)
        draw.rectangle(tuple(xy.cpu().numpy()),outline='red')
        draw.line(fixations,fill="red",width=4)
        #draw.polygon(fixations,fill="blue")
        return img

    def close(self):
        #clear storage
        pass

    def idx_to_vector(self,act_batch):
        #from grid_idx to precise pixel-level displacement
        #manual mapping instead of actor training

        #scaled->non-scaled
        action_range=self.action_range
        grid_size=self.env_args['grid_size']

        idx_x=act_batch/grid_size
        idx_x=torch.floor(idx_x) #row_idx
        
        idx_y=act_batch%(grid_size) #column_idx
        
        coord=torch.stack((idx_x,idx_y),dim=-1)+0.5 #to center
        
        center=self.env_args['radius']*torch.ones(act_batch.shape[0],2)
        center=center.to(self.device)
        
        step=self.env_args['radius']*2/grid_size #224/14=16
        
        #inverse
        target_pos=coord*step #224,224
        target_pos=(target_pos/(self.env_args['radius']*2))*self.env_args['vision_radius']
        target_pos=torch.round(target_pos).to(torch.long)

        grid=(self.grid+1)/2 * (self.env_args['vision_radius']-1) #grid for transformed=grid_sample(origin)
        grid=torch.flip(grid, dims=[-1]) #flip y,x to x,y
        
        true_pos = torch.round(grid[0,target_pos[:,0],target_pos[:,1]] - self.env_args['vision_radius']/2)
        
        return true_pos.to(torch.long)
        
        