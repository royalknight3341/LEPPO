"""Run sensitivity experiments; each setting is retrained, not a mislabeled reused model."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/default.json')
    p.add_argument('--parameter',choices=['aps','users','antennas','ap_power_w','battery_j','area_m','delay_weight','task_bits_max'],required=True)
    p.add_argument('--values',type=float,nargs='+',required=True)
    p.add_argument('--train-seeds',type=int,nargs='+',default=[42,43,44])
    p.add_argument('--steps',type=int,default=50000)
    p.add_argument('--eval-slots',type=int,default=2000)
    p.add_argument('--output',default='results/sweep')
    a=p.parse_args()
    base=json.loads(Path(a.config).read_text())
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    aggregate=[]
    for value in a.values:
        cfg=dict(base)
        if a.parameter in ['aps','users','antennas']:
            if not value.is_integer():
                p.error('count parameters require integer values')
            value=int(value)
        cfg[a.parameter]=value
        for seed in a.train_seeds:
            dest=out/f'{a.parameter}_{value}_seed_{seed}';dest.mkdir(parents=True,exist_ok=True)
            path=dest/'input.json';path.write_text(json.dumps(cfg,indent=2))
            subprocess.run([sys.executable,'-m','leppo.experiment','--config',str(path),'--output',str(dest),
                            '--seed',str(seed),'--steps',str(a.steps),'--eval-slots',str(a.eval_slots),
                            '--algorithms','leppo','ppo','no-gru'],check=True)
            for row in json.loads((dest/'summary.json').read_text()):
                aggregate.append(dict(parameter=a.parameter,value=value,training_seed=seed,**row))
    (out/'aggregate.json').write_text(json.dumps(aggregate,indent=2))


if __name__=='__main__':
    main()
