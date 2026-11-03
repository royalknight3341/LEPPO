"""Training, matched-randomness evaluation and plots. python -m leppo.experiment --help"""
import os
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from .environment import Config, MECEnv
from .policy import ActorCritic, gae, update


def write_csv(path, rows):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def train(c, output, steps, rollout, seed, algorithm):
    torch.manual_seed(seed)
    env = MECEnv(c, seed)
    model = ActorCritic(env.obs_dim, env.action_dim, c.users, algorithm)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-5)
    state = env.observe()
    rows, done = [], 0
    while done < steps:
        ob, hi, zs, lps, vs, rs, infos = [], [], [], [], [], [], []
        for _ in range(min(rollout, steps-done)):
            z, lp, value = model.act(state)
            ob.append(state[0]); hi.append(state[1]); zs.append(z); lps.append(lp); vs.append(value)
            state, reward, info = env.step(z)
            rs.append([reward.sum()] if algorithm == 'ppo' else reward)
            infos.append(info)
        _, _, bootstrap = model.act(state, deterministic=True)
        advantages, returns = gae(np.asarray(rs), np.asarray(vs), bootstrap)
        losses = update(model, optimizer, [np.asarray(ob), np.asarray(hi), np.asarray(zs),
                                          np.asarray(lps), advantages, returns])
        done += len(rs)
        row = dict(step=done, reward=float(np.mean(np.sum(rs, axis=1))),
                   latency_s=float(np.mean([i['latency_s'] for i in infos])),
                   outage_rate=float(np.mean([i['outage'] for i in infos])),
                   low_fraction=float(np.mean([i['low_fraction'] for i in infos])),
                   queue=float(np.mean([i['queue'] for i in infos])),
                   projected_fraction=float(np.mean([i['projected'] for i in infos])),
                   policy_loss=float(losses[0]), value_loss=float(losses[1]), kl=float(losses[2]))
        rows.append(row)
        if len(rows)%10 == 0 or done == steps:
            print(f'{algorithm} step={done} reward={row["reward"]:.3f} delay={row["latency_s"]:.4f}s low={row["low_fraction"]:.3f}', flush=True)
    torch.save(dict(state_dict=model.state_dict(), algorithm=algorithm, config=vars(c), seed=seed, steps=steps), output/f'{algorithm}.pt')
    write_csv(output/f'train_{algorithm}.csv', rows)
    return model


def baseline(env, name, rng):
    c = env.c
    if name == 'random':
        return rng.normal(0, 1, env.action_dim)
    z = np.zeros(env.action_dim)
    # All methods use the same feasibility-repair layer.
    z[0] = np.log((.3-.02)/(.95-.3))
    if name == 'local':
        z[1+c.users+c.aps*(c.users-1):] = -30
    elif name == 'greedy':
        # Adapt harvesting to battery depletion, choose balanced local/offload split.
        alpha = np.clip(.15 + .65*(1-env.battery.mean()/c.battery_j), .03, .94)
        z[0] = np.log((alpha-.02)/(.95-alpha))
        stop = 1+c.users+c.aps*(c.users-1)
        z[stop:] = np.log(.7/(.3*c.aps))
    return z


def evaluate(c, output, models, slots, seeds):
    records, summaries = [], []
    methods = ['random', 'fixed', 'local', 'greedy'] + list(models)
    for seed in seeds:
        for name in methods:
            # Environment draws never depend on actions: matched topology/tasks/mobility.
            env = MECEnv(c, seed)
            rng = np.random.default_rng(seed+90000)
            for t in range(slots):
                z = models[name].act(env.observe(), deterministic=True)[0] if name in models else baseline(env, name, rng)
                _, _, info = env.step(z)
                records.append(dict(seed=seed, method=name, slot=t, **info))
            run = records[-slots:]
            success = [r['latency_s'] for r in run if r['completed']]
            summaries.append(dict(seed=seed, method=name, slots=slots,
                                  mean_latency_s=float(np.mean(success)) if success else None,
                                  mean_service_s=float(np.mean([r['service_s'] for r in run if r['completed']])) if success else None,
                                  worst_user_low_fraction=max(float(np.mean([r[f'low_{k}'] for r in run])) for k in range(c.users)),
                                  p95_latency_s=float(np.percentile(success,95)) if success else None,
                                  outage_rate=float(np.mean([r['outage'] for r in run])),
                                  low_fraction=float(np.mean([r['low_fraction'] for r in run])),
                                  final_mean_queue=run[-1]['queue'],
                                  queue_per_slot=run[-1]['queue']/slots,
                                  projected_fraction=float(np.mean([r['projected'] for r in run])),
                                  mean_battery_j=float(np.mean([r['battery_j'] for r in run]))))
    write_csv(output/'evaluation.csv', records)
    with open(output/'summary.json', 'w') as f:
        json.dump(summaries, f, indent=2, allow_nan=False)
    plot(output, records, summaries, c)
    for name in methods:
        rows = [s for s in summaries if s['method'] == name]
        vals = [r['mean_latency_s'] for r in rows if r['mean_latency_s'] is not None]
        print(name, 'successful-task latency=', np.mean(vals) if vals else 'no completed tasks',
              'low=', np.mean([r['low_fraction'] for r in rows]),
              'outage=', np.mean([r['outage_rate'] for r in rows]), flush=True)


def plot(output, records, summaries, c):
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/leppo-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    methods = list(dict.fromkeys(r['method'] for r in records))
    first = records[0]['seed']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for name in methods:
        rows = [r for r in records if r['method']==name and r['seed']==first]
        t = np.arange(1,len(rows)+1)
        axes[0,0].plot(t, np.asarray([r['battery_j'] for r in rows])*1e3, label=name)
        axes[0,1].plot(t, [r['queue'] for r in rows], label=name)
        axes[1,0].plot(t, np.cumsum([r['low_fraction'] for r in rows])/t, label=name)
    axes[0,0].axhline(c.battery_j*c.safe_battery_fraction*1e3, ls='--', color='k')
    axes[0,0].set_ylabel('Mean battery (mJ)')
    axes[0,1].set_ylabel('Mean safety queue')
    axes[1,0].axhline(c.allowed_low_fraction, ls='--', color='k')
    axes[1,0].set_ylabel('Cumulative low-battery fraction')
    means, errs = [], []
    for name in methods:
        values = [s['mean_latency_s'] for s in summaries if s['method']==name and s['mean_latency_s'] is not None]
        means.append(np.mean(values) if values else np.nan)
        errs.append(np.std(values, ddof=1)/np.sqrt(len(values)) if len(values)>1 else 0)
    axes[1,1].bar(methods, means, yerr=errs, capsize=3)
    axes[1,1].set_ylabel('Completed-task latency (s), mean ± SE across seeds')
    axes[1,1].tick_params(axis='x', rotation=35)
    for ax in axes.flat:
        ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output/'comparison.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, key, label in zip(axes.flat,
            ['mean_service_s','outage_rate','worst_user_low_fraction','projected_fraction'],
            ['Service delay objective (s)','Outage rate','Worst-user low-battery fraction','Feasibility repair fraction']):
        means, errs = [], []
        for name in methods:
            vals = [r[key] for r in summaries if r['method']==name and r[key] is not None]
            means.append(np.mean(vals) if vals else np.nan)
            errs.append(np.std(vals,ddof=1)/np.sqrt(len(vals)) if len(vals)>1 else 0)
        ax.bar(methods,means,yerr=errs,capsize=3)
        ax.set_ylabel(label)
        ax.tick_params(axis='x',rotation=35)
        ax.grid(alpha=.2)
        if key=='worst_user_low_fraction':
            ax.axhline(c.allowed_low_fraction,ls='--',color='k')
    fig.tight_layout(); fig.savefig(output/'metrics.png',dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8,4))
    for path in output.glob('train_*.csv'):
        with open(path) as f:
            rows = list(csv.DictReader(f))
        ax.plot([float(r['step']) for r in rows], [float(r['reward']) for r in rows], label=path.stem)
    ax.set(xlabel='Training steps', ylabel='Mean rollout reward')
    if ax.lines:
        ax.legend()
    fig.tight_layout(); fig.savefig(output/'training.png', dpi=160); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/default.json')
    p.add_argument('--output', default='results/demo')
    p.add_argument('--steps', type=int, default=20000)
    p.add_argument('--rollout', type=int, default=256)
    p.add_argument('--eval-slots', type=int, default=1000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--eval-seeds', type=int, nargs='+', default=[101, 102, 103])
    p.add_argument('--algorithms', nargs='*', choices=['leppo','ppo','no-gru'], default=['leppo','ppo'])
    p.add_argument('--evaluate-only', action='store_true')
    a = p.parse_args()
    if min(a.steps,a.rollout,a.eval_slots) <= 0:
        p.error('steps, rollout and eval-slots must be positive')
    torch.set_num_threads(2)
    c = Config.load(a.config)
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    c.save(out/'config.json')
    models = {}
    for algorithm in a.algorithms:
        if a.evaluate_only:
            ckpt = torch.load(out/f'{algorithm}.pt', map_location='cpu', weights_only=True)
            if ckpt['config'] != vars(c):
                raise ValueError('checkpoint configuration differs from requested evaluation configuration')
            env = MECEnv(c)
            model = ActorCritic(env.obs_dim, env.action_dim,c.users,algorithm)
            model.load_state_dict(ckpt['state_dict'])
        else:
            model = train(c,out,a.steps,a.rollout,a.seed,algorithm)
        model.eval(); models[algorithm] = model
    evaluate(c,out,models,a.eval_slots,a.eval_seeds)


if __name__ == '__main__':
    main()
