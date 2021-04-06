import copy
from torch.distributions import Categorical
from components.episode_buffer import EpisodeBatch
from modules.critics.maddpg import OffPGCritic
import torch as th
from utils.offpg_utils import build_target_q
from utils.rl_utils import build_td_lambda_targets
from torch.optim import RMSprop
from modules.mixers.qmix import QMixer
from collections import OrderedDict


class CoopMADDPGLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.mac = mac
        self.target_mac = copy.deepcopy(self.mac)
        self.logger = logger

        self.last_target_update_step = 0
        self.critic_training_steps = 0

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.critic = OffPGCritic(scheme, args)
        self.target_critic = copy.deepcopy(self.critic)

        self.agent_params = list(mac.parameters())
        self.critic_params = list(self.critic.parameters())
        self.params = self.agent_params + self.critic_params

        self.agent_optimiser = RMSprop(params=self.agent_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.critic_optimiser = RMSprop(params=self.critic_params, lr=args.critic_lr, alpha=args.optim_alpha, eps=args.optim_eps)

    def train(self, batch: EpisodeBatch, t_env: int, log):
        # Get the relevant quantities
        bs = batch['state'].size(0)
        #build q

        inputs = self.critic._build_inputs(batch)
        actions, raw_action = self.mac.now_forward(batch, bs)
        q_vals = self.critic.forward(inputs, actions.reshape(-1, self.n_agents * self.n_actions)).squeeze()
        dpg_loss = - q_vals.mean()
        reg_loss = 1e-3 * th.sqrt((raw_action*raw_action).sum(dim=-1)).mean()
        self.agent_optimiser.zero_grad()
        loss = dpg_loss + reg_loss
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        self.agent_optimiser.step()

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            ts_logged = len(log["critic_loss"])
            for key in ["critic_loss", "critic_grad_norm", "td_error_abs"]:
                self.logger.log_stat(key, sum(log[key])/ts_logged, t_env)
            self.logger.log_stat("dpg_loss", dpg_loss.item(), t_env)
            self.logger.log_stat("reg_loss", reg_loss.item(), t_env)
            self.logger.log_stat("agent_grad_norm", grad_norm, t_env)
            self.log_stats_t = t_env

    def train_critic(self, batch, log=None):
        bs = batch['state'].size(0)
        rewards = batch["reward"][:]
        actions = batch["actions"][:].reshape(-1, self.n_agents * self.n_actions)
        terminated = batch['ter'].float()

        #build_target_q
        target_inputs = self.target_critic._build_target_inputs(batch)
        target_actions = self.target_mac.target_forward(batch, bs).detach().reshape(-1, self.n_actions * self.n_agents)
        target_q_vals = self.target_critic.forward(target_inputs, target_actions).detach()  # Target actions should be derived from current target policy
        target_q = self.args.gamma * target_q_vals * (1 - terminated) + rewards

        inputs = self.critic._build_inputs(batch)

        # train critic
        q_vals = self.critic.forward(inputs, actions)
        q_err = (q_vals - target_q.detach())
        critic_loss = (q_err ** 2).mean()
        self.critic_optimiser.zero_grad()
        critic_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
        self.critic_optimiser.step()
        self.critic_training_steps += 1

        log["critic_loss"].append(critic_loss.item())
        log["critic_grad_norm"].append(grad_norm)
        log["td_error_abs"].append((q_err.abs().mean().item()))

        #update target network
        if (self.critic_training_steps - self.last_target_update_step) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_step = self.critic_training_steps

    def _update_targets(self):
        temp = OrderedDict([(k, self.args.target_update_rate * v) for (k, v) in self.target_critic.state_dict().items()])
        for (k, v) in temp.items():
            temp[k] = v + (1-self.args.target_update_rate) * self.critic.state_dict()[k]
        self.target_critic.load_state_dict(temp)

        temp = OrderedDict([(k, self.args.target_update_rate * v) for (k, v) in self.target_mac.agent.state_dict().items()])
        for (k, v) in temp.items():
            temp[k] = v + (1-self.args.target_update_rate) * self.mac.agent.state_dict()[k]
        self.target_mac.agent.load_state_dict(temp)

        # self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        self.critic.cuda()
        self.target_critic.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.critic.state_dict(), "{}/critic.th".format(path))
        th.save(self.agent_optimiser.state_dict(), "{}/agent_opt.th".format(path))
        th.save(self.critic_optimiser.state_dict(), "{}/critic_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.critic.load_state_dict(th.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage))
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.agent_optimiser.load_state_dict(th.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.critic_optimiser.load_state_dict(th.load("{}/critic_opt.th".format(path), map_location=lambda storage, loc: storage))