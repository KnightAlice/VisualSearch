import torch
from torch.distributions import Categorical
import torch.optim as optim
import utils
#process collected trajectories
'''value loss disabled for optimizer'''

class PPO():
    def __init__(self,
                 agent,
                 lr,
                 betas,
                 clip_param,
                 num_epoch,
                 batch_size,
                 value_coef=1.,
                 entropy_coef=0.1,
                 drop_failed=True,
                 vgg_backbone_fixed=True):

        self.agent = agent
        self.clip_param = clip_param
        self.num_epoch = num_epoch
        self.minibatch_size = batch_size
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        if vgg_backbone_fixed: #vgg not involved in updating
            #acmerge
            #param_dict=[{'params':self.agent.acmerge.actor_read_out.parameters()},{'params':self.agent.acmerge.value_read_out.parameters()}]
            raise NotImplementedError
            
        else:
            param_dict=[{'params':self.agent.parameters()}]
            
        self.optimizer = optim.Adam(param_dict,
                                    lr=lr,
                                    betas=betas)
        
        self.value_loss_fun = torch.nn.SmoothL1Loss()
        self.drop_failed=drop_failed #only those find the target will be effecitive trials

    def evaluate_actions(self, obs_batch, actions_batch):

        probs, values, _ = self.agent(*obs_batch)
        dist = Categorical(probs) #sampling probability of categories
        log_probs = dist.log_prob(actions_batch)

        return values, log_probs, dist.entropy().mean()

    def update(self, rollouts):
        avg_loss = 0

        for e in range(self.num_epoch):
            data_generator = rollouts.get_generator(self.minibatch_size)

            value_loss_epoch = 0
            action_loss_epoch = 0
            dist_entropy_epoch = 0
            loss_epoch = 0

            for i, sample in enumerate(data_generator):
                obs_batch, actions_batch, return_batch, \
                   old_action_log_probs_batch, adv_targ, status = sample
                '''select trials succeeded'''
                found_idx=torch.where(torch.any(status,dim=1))[0].cpu()
                if self.drop_failed:
                    if len(found_idx)==0: #current batch is wasted
                        continue
                    obs_batch, actions_batch, return_batch, \
                   old_action_log_probs_batch, adv_targ=(obs_batch[0][found_idx],obs_batch[1][found_idx]), actions_batch[found_idx], return_batch[found_idx], \
                   old_action_log_probs_batch[found_idx], adv_targ[found_idx]
                    
                self.optimizer.zero_grad()

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy = self.evaluate_actions(
                    obs_batch, actions_batch)

                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
            
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                value_loss = self.value_loss_fun(
                    return_batch.squeeze(), values.squeeze()) * self.value_coef #value approximate acc_rewards

                entropy_loss = -dist_entropy * self.entropy_coef
                if abs(dist_entropy<0.001):
                    print(action_log_probs)
                    raise ValueError
                loss =  action_loss + entropy_loss #+value_loss 
                loss.backward(retain_graph=True)

                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += entropy_loss.item()
                loss_epoch += loss.item()

            value_loss_epoch /= i + 1
            action_loss_epoch /= i + 1
            dist_entropy_epoch /= i + 1
            loss_epoch /= i + 1
            if e % 4 == 0:
                print('[{}/{}]: VLoss: {:.3f}, PLoss: {:.3f}, loss: {:.3f}'.
                      format(e+1, self.num_epoch, value_loss_epoch,
                             action_loss_epoch, loss_epoch))
            avg_loss += loss_epoch

        avg_loss /= self.num_epoch

        return avg_loss