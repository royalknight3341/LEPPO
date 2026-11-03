"""Gaussian latent policy, identifiable logistic-normal actions and dual values."""
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, users, algorithm='leppo', hidden=128):
        super().__init__()
        self.algorithm, self.users = algorithm, users
        self.temporal = algorithm == 'leppo'
        self.gru = nn.GRU(3, 16, batch_first=True) if self.temporal else None
        self.trunk = nn.Sequential(nn.Linear(obs_dim+(16*users if self.temporal else 0), hidden),
                                   nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.actor = nn.Linear(hidden, action_dim)
        self.logstd = nn.Parameter(torch.full((action_dim,), -.7))
        self.delay_critic = nn.Linear(hidden, 1)
        self.energy_critic = nn.Linear(hidden, 1) if algorithm != 'ppo' else None
        nn.init.orthogonal_(self.actor.weight, .01)
        nn.init.zeros_(self.actor.bias)

    def forward(self, obs, history):
        if self.temporal:
            b, h, k, _ = history.shape
            _, hidden = self.gru(history.permute(0, 2, 1, 3).reshape(b*k, h, 3))
            obs = torch.cat([obs, hidden[-1].reshape(b, k*16)], -1)
        x = self.trunk(obs)
        dist = Normal(self.actor(x), self.logstd.clamp(-3, .5).exp())
        vd = self.delay_critic(x).squeeze(-1)
        values = torch.stack([vd, self.energy_critic(x).squeeze(-1)], -1) if self.energy_critic else vd[:, None]
        return dist, values

    @torch.no_grad()
    def act(self, state, deterministic=False):
        obs, hist = (torch.as_tensor(x[None], dtype=torch.float32) for x in state)
        dist, values = self(obs, hist)
        z = dist.mean if deterministic else dist.sample()
        return z[0].numpy(), float(dist.log_prob(z).sum(-1)[0]), values[0].numpy()


def gae(rewards, values, bootstrap, gamma=.99, lam=.95):
    """Continuing-task rollout: bootstrap at segment boundaries, no artificial reset."""
    advantages = np.zeros_like(rewards)
    carry = np.zeros(rewards.shape[1], dtype=np.float32)
    nxt = bootstrap
    for t in range(len(rewards)-1, -1, -1):
        delta = rewards[t] + gamma*nxt - values[t]
        carry = delta + gamma*lam*carry
        advantages[t] = carry
        nxt = values[t]
    return advantages, advantages+values


def update(model, optimizer, batch, epochs=4, batch_size=64, clip=.2):
    obs, hist, latent, oldlp, adv, returns = [torch.as_tensor(x, dtype=torch.float32) for x in batch]
    total_adv = adv.sum(-1)
    total_adv = (total_adv-total_adv.mean())/(total_adv.std(unbiased=False)+1e-8)
    stats = []
    for _ in range(epochs):
        for ids in torch.randperm(len(obs)).split(batch_size):
            dist, values = model(obs[ids], hist[ids])
            logp = dist.log_prob(latent[ids]).sum(-1)
            ratio = (logp-oldlp[ids]).exp()
            actor_loss = -torch.minimum(ratio*total_adv[ids], ratio.clamp(1-clip, 1+clip)*total_adv[ids]).mean()
            value_loss = (values-returns[ids]).square().mean()
            # Entropy regularizes the latent Gaussian, not the transformed distribution.
            loss = actor_loss + .5*value_loss - .001*dist.entropy().sum(-1).mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), .5)
            optimizer.step()
            stats.append((float(actor_loss.detach()), float(value_loss.detach()),
                          float((oldlp[ids]-logp).mean().detach())))
    return np.mean(stats, axis=0)
