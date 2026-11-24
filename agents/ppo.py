"""PPO, written out step by step: collect games, score moves, update the network."""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def to_tensor(x, dtype=torch.float32):
    return torch.as_tensor(np.asarray(x), dtype=dtype)


class RolloutCollector:
    """Plays several games at once and records every step."""

    def __init__(self, envs, gamma):
        self.envs = envs
        self.gamma = gamma
        self.masks = to_tensor(np.stack([e.action_mask() for e in envs]), torch.bool)
        self.obs = np.stack([e.reset() for e in envs])
        self.ep_return = np.zeros(len(envs))
        self.finished = []  # (game name, episode return) for logging

    def collect(self, net, steps, on_step=None):
        """Play `steps` moves in every game. `on_step(t, actions, probs)` is an
        optional hook called after each move (the dashboard uses it)."""
        buf = {k: [] for k in ("obs", "mask", "action", "logprob", "value", "reward", "done")}
        for t in range(steps):
            obs_t = to_tensor(self.obs)
            with torch.no_grad():
                logits, value = net(obs_t, self.masks)
                dist = Categorical(logits=logits)
                action = dist.sample()

            rewards = np.zeros(len(self.envs), dtype=np.float32)
            dones = np.zeros(len(self.envs), dtype=np.float32)
            next_obs = np.empty_like(self.obs)
            for i, env in enumerate(self.envs):
                obs, reward, terminated, truncated = env.step(int(action[i]))
                self.ep_return[i] += reward
                if truncated:
                    # Time ran out, but the game didn't really end.
                    # Add the value of where we stopped, so the agent isn't punished for it.
                    with torch.no_grad():
                        _, v = net(to_tensor(obs[None]), self.masks[i:i + 1])
                    reward += self.gamma * float(v)
                if terminated or truncated:
                    self.finished.append((env.name, float(self.ep_return[i])))
                    self.ep_return[i] = 0.0
                    obs = env.reset()
                rewards[i], dones[i], next_obs[i] = reward, float(terminated or truncated), obs

            buf["obs"].append(obs_t)
            buf["mask"].append(self.masks)
            buf["action"].append(action)
            buf["logprob"].append(dist.log_prob(action))
            buf["value"].append(value)
            buf["reward"].append(to_tensor(rewards))
            buf["done"].append(to_tensor(dones))
            self.obs = next_obs
            if on_step is not None:
                on_step(t, action, dist.probs)

        with torch.no_grad():
            _, last_value = net(to_tensor(self.obs), self.masks)
        batch = {k: torch.stack(v) for k, v in buf.items()}  # each is [steps, num_envs, ...]
        return batch, last_value


def compute_gae(rewards, values, dones, last_value, gamma, lam):
    """Advantage = how much better a move was than expected (GAE)."""
    steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros_like(last_value)
    for t in reversed(range(steps)):
        next_value = last_value if t == steps - 1 else values[t + 1]
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * not_done - values[t]
        gae = delta + gamma * lam * not_done * gae
        advantages[t] = gae
    return advantages, advantages + values


def ppo_update(net, optimizer, batch, advantages, returns, cfg, on_minibatch=None):
    """Several passes over the batch, using the clipped PPO loss.
    Returns averaged training statistics."""
    flat = lambda x: x.reshape(-1, *x.shape[2:])  # [steps, envs, ...] -> [steps*envs, ...]
    obs, mask, actions = flat(batch["obs"]), flat(batch["mask"]), flat(batch["action"])
    old_logprob, adv, ret = flat(batch["logprob"]), flat(advantages), flat(returns)
    old_value = flat(batch["value"])

    stats = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": [], "clip_frac": []}
    size = obs.shape[0]
    mb_size = size // cfg["minibatches"]
    for epoch in range(cfg["epochs"]):
        order = torch.randperm(size)
        for start in range(0, mb_size * cfg["minibatches"], mb_size):  # drop any leftover samples
            idx = order[start:start + mb_size]
            logits, value = net(obs[idx], mask[idx])
            dist = Categorical(logits=logits)

            # Ratio: how much more (or less) likely the move is now than before.
            log_ratio = dist.log_prob(actions[idx]) - old_logprob[idx]
            ratio = log_ratio.exp()
            a = adv[idx]
            a = (a - a.mean()) / (a.std() + 1e-8)

            # Clipping stops one update from changing the policy too much.
            policy_loss = -torch.min(ratio * a,
                                     ratio.clamp(1 - cfg["clip"], 1 + cfg["clip"]) * a).mean()
            value_loss = 0.5 * ((value - ret[idx]) ** 2).mean()
            entropy = dist.entropy().mean()  # keeps some exploration
            loss = policy_loss + cfg["vf_coef"] * value_loss - cfg["ent_coef"] * entropy

            optimizer.zero_grad()
            loss.backward()
            if on_minibatch is not None:
                on_minibatch(epoch)  # gradients exist here, before clipping and the step
            nn.utils.clip_grad_norm_(net.parameters(), cfg["max_grad_norm"])
            optimizer.step()

            with torch.no_grad():
                stats["policy_loss"].append(policy_loss.item())
                stats["value_loss"].append(value_loss.item())
                stats["entropy"].append(entropy.item())
                # How far the new policy moved from the old one.
                stats["approx_kl"].append(((ratio - 1) - log_ratio).mean().item())
                stats["clip_frac"].append(((ratio - 1).abs() > cfg["clip"]).float().mean().item())

    out = {k: float(np.mean(v)) for k, v in stats.items()}
    # Explained variance: 1 = the value head predicts returns perfectly, 0 = no better than a constant.
    var_ret = ret.var().item()
    out["explained_var"] = float(1 - (ret - old_value).var().item() / var_ret) if var_ret > 0 else 0.0
    return out
