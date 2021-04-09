import copy
from torch.distributions import Categorical
from components.episode_buffer import EpisodeBatch
from modules.critics.offpg import OffPGCritic
import torch as th
from utils.offpg_utils import build_target_q
from utils.rl_utils import build_td_lambda_targets
from torch.optim import RMSprop, Adam
from modules.mixers.qmix import QMixer
from collections import OrderedDict


class ContOffPGLearner:
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
        self.mixer = QMixer(args)
        self.target_critic = copy.deepcopy(self.critic)
        self.target_mixer = copy.deepcopy(self.mixer)

        self.agent_params = list(mac.parameters())
        self.critic_params = list(self.critic.parameters())
        self.mixer_params = list(self.mixer.parameters())
        self.params = self.agent_params + self.critic_params
        self.c_params = self.critic_params + self.mixer_params

        # self.agent_optimiser = RMSprop(params=self.agent_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.agent_optimiser = Adam(params=self.agent_params,lr=args.lr)

        # self.critic_optimiser = RMSprop(params=self.c_params, lr=args.critic_lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.critic_optimiser = Adam(params=self.c_params, lr=args.critic_lr)

    def train(self, batch: EpisodeBatch, t_env: int, log):
        # Get the relevant quantities
        bs = batch['state'].size(0)
        rewards = batch["reward"]
        states = batch["state"]
        #build q
        dpg_loss = 0
        reg_loss = 0
        inputs = self.critic.critic_build_inputs(batch, bs)
        actions, raw_action = self.mac.now_forward(batch, bs)
        q_vals = self.critic.forward(inputs, actions).squeeze()
        # k = self.mixer.k(states).detach().squeeze()
        dpg_loss = -(q_vals).sum(dim=-1).mean()
        # dpg_loss = -q_vals.mean()
        reg_loss = 1e-3 * th.sqrt((raw_action*raw_action).sum(dim=-1)).mean()
        self.agent_optimiser.zero_grad()
        loss = dpg_loss + reg_loss
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        self.agent_optimiser.step()

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            ts_logged = len(log["critic_loss"])
            for key in ["critic_loss", "td_error_abs"]:
                self.logger.log_stat(key, sum(log[key])/ts_logged, t_env)
            self.logger.log_stat("critic_grad_norm", sum(log['critic_grad_norm']).item()/ts_logged, t_env)
            self.logger.log_stat("dpg_loss", dpg_loss.item(), t_env)
            self.logger.log_stat("reg_loss", reg_loss.item(), t_env)
            self.logger.log_stat("agent_grad_norm", grad_norm.item(), t_env)
            self.log_stats_t = t_env

    def eval_q(self, bs, states, obs, actions):
        inputs = self.critic.critic_build_inputs_raw(states, obs, bs)

        # train critic
        q_vals = self.critic.forward(inputs, actions)
        #q_vals = self.mixer.forward(q_vals, states).squeeze(-1)
        q_vals = th.sum(q_vals, dim=-1)

        return q_vals.detach()

    def train_critic(self, batch, log=None):
        bs = batch['state'].size(0)
        rewards = batch["reward"]
        actions = batch["actions"].detach()
        states = batch["state"]
        terminated = batch['ter'].float()

        #build_target_q
        target_inputs = self.target_critic.critic_build_target_inputs(batch, bs)
        target_actions = self.target_mac.target_forward(batch, bs)
        target_q_vals = self.target_critic.forward(target_inputs, target_actions)  # Target actions should be derived from current target policy
        #targets_taken = self.target_mixer(target_q_vals, states).detach().squeeze(-1)
        targets_taken = th.sum(target_q_vals, dim=-1).detach()
        target_q = self.args.gamma * targets_taken * (1 - terminated) + rewards

        inputs = self.critic.critic_build_inputs(batch, bs)

        # train critic
        q_vals = self.critic.forward(inputs, actions)
        # q_vals = self.mixer.forward(q_vals, states).squeeze(-1)
        q_vals = th.sum(q_vals, dim=-1)
        q_err = (target_q.detach() - q_vals)
        critic_loss = (q_err ** 2).mean()
        self.critic_optimiser.zero_grad()
        critic_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.c_params, self.args.grad_norm_clip)
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

        temp.clear()
        temp = OrderedDict([(k, self.args.target_update_rate * v) for (k, v) in self.target_mixer.state_dict().items()])
        for (k, v) in temp.items():
            temp[k] = v + (1-self.args.target_update_rate) * self.mixer.state_dict()[k]
        self.target_mixer.load_state_dict(temp)

        temp.clear()
        temp = OrderedDict([(k, self.args.target_update_rate * v) for (k, v) in self.target_mac.agent.state_dict().items()])
        for (k, v) in temp.items():
            temp[k] = v + (1-self.args.target_update_rate) * self.mac.agent.state_dict()[k]
        self.target_mac.agent.load_state_dict(temp)

        temp.clear()

        # self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        self.critic.cuda()
        self.mixer.cuda()
        self.target_critic.cuda()
        self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.critic.state_dict(), "{}/critic.th".format(path))
        th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.agent_optimiser.state_dict(), "{}/agent_opt.th".format(path))
        th.save(self.critic_optimiser.state_dict(), "{}/critic_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.critic.load_state_dict(th.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage))
        self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.agent_optimiser.load_state_dict(th.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.critic_optimiser.load_state_dict(th.load("{}/critic_opt.th".format(path), map_location=lambda storage, loc: storage))
