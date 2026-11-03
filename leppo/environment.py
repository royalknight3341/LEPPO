"""SI-unit statistical CF-mMIMO model; assumptions are listed in README_SIMULATION.md."""
from dataclasses import dataclass, asdict
import json
import numpy as np


@dataclass
class Config:
    aps: int = 8
    users: int = 4
    antennas: int = 8
    area_m: float = 40.
    ap_height_m: float = 3.
    md_height_m: float = 1.
    carrier_mhz: float = 915.
    shadow_std_db: float = 4.
    pilot_symbols: int = 2
    coherence_symbols: int = 200
    slot_s: float = .5
    bandwidth_hz: float = 1e6
    noise_figure_db: float = 7.
    pilot_power_w: float = .001
    ap_power_w: float = 1.
    ul_power_w: float = .001
    harvest_efficiency: float = .5
    task_bits_min: float = 5000.
    task_bits_max: float = 15000.
    cycles_per_bit: float = 300.
    local_cpu_hz: float = 1e8
    edge_cpu_hz: float = 1e9
    capacitance: float = 1e-27
    battery_j: float = .0005
    initial_battery_fraction: float = .6
    safe_battery_fraction: float = .3
    allowed_low_fraction: float = .05
    speed_m_s: float = .5
    history_length: int = 8
    delay_weight: float = 1.
    queue_weight: float = 1.
    battery_weight: float = .1
    violation_weight: float = 5.
    outage_penalty: float = 10.

    def __post_init__(self):
        for name in ('aps', 'users', 'antennas', 'pilot_symbols', 'coherence_symbols', 'history_length'):
            if getattr(self, name) < 1:
                raise ValueError(f'{name} must be positive')
        if self.pilot_symbols >= self.coherence_symbols:
            raise ValueError('pilot_symbols must be smaller than coherence_symbols')
        for name in ('initial_battery_fraction', 'safe_battery_fraction', 'allowed_low_fraction', 'harvest_efficiency'):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f'{name} must be in [0,1]')
        for name in ('slot_s', 'area_m', 'carrier_mhz', 'bandwidth_hz', 'pilot_power_w',
                     'ap_power_w', 'ul_power_w', 'task_bits_min', 'task_bits_max',
                     'cycles_per_bit', 'local_cpu_hz', 'edge_cpu_hz', 'battery_j'):
            if getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive')
        if self.task_bits_min > self.task_bits_max:
            raise ValueError('invalid task size range')

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(**json.load(f))

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)


def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


class MECEnv:
    def __init__(self, cfg=None, seed=0):
        self.c = cfg or Config()
        self.rng = np.random.default_rng(seed)
        m, k = self.c.aps, self.c.users
        # Reference logits remove the non-identifiability of full softmax logits.
        self.action_dim = 1 + k + m * (k - 1) + k * m
        self.obs_dim = 2*m*k + 5*k
        self.reset()

    def reset(self):
        c = self.c
        self.ap = self.rng.uniform(0, c.area_m, (c.aps, 2))
        self.pos = self.rng.uniform(0, c.area_m, (c.users, 2))
        self.shadow = self.rng.normal(0, c.shadow_std_db, (c.aps, c.users))
        self.battery = np.full(c.users, c.battery_j*c.initial_battery_fraction)
        self.queue = np.zeros(c.users)
        self.net = np.zeros(c.users)
        self.history = np.zeros((c.history_length, c.users, 3), dtype=np.float32)
        self._exogenous()
        self._history()
        return self.observe()

    def _exogenous(self):
        c = self.c
        self.bits = self.rng.uniform(c.task_bits_min, c.task_bits_max, c.users)
        d = np.sqrt(((self.ap[:, None] - self.pos[None])**2).sum(-1)
                    + (c.ap_height_m - c.md_height_m)**2)
        # Continuous 3-slope close-in pathloss: exponents 0 / 2 / 3.5 at 1 / 10 m.
        fspl = 20*np.log10(4*np.pi*c.carrier_mhz*1e6/299792458.)
        loss = fspl + 20*np.log10(np.clip(d, 1, 10)) + 35*np.log10(np.maximum(d/10, 1))
        self.beta = 10**((-loss + self.shadow)/10)
        # Greedy distance-based pilot reuse, fixed deterministic reference users.
        self.pilots = np.arange(c.users) % c.pilot_symbols
        refs = min(c.pilot_symbols, c.users)
        for k in range(refs, c.users):
            dist = ((self.pos[:refs] - self.pos[k])**2).sum(-1)
            self.pilots[k] = int(np.argmax(dist))
        self.same = self.pilots[:, None] == self.pilots[None, :]
        self.noise = 10**((-174 + 10*np.log10(c.bandwidth_hz) + c.noise_figure_db - 30)/10)
        denominator = self.noise + c.pilot_symbols*c.pilot_power_w*(self.beta @ self.same)
        self.gamma = c.pilot_symbols*c.pilot_power_w*self.beta**2/denominator

    def decode(self, z):
        c = self.c
        z = np.asarray(z, dtype=float)
        if z.shape != (self.action_dim,) or not np.isfinite(z).all():
            raise ValueError('invalid latent action')
        p = 1 + c.users
        alpha = .02 + .93/(1 + np.exp(-np.clip(z[0], -40, 40)))
        eta = 1/(1 + np.exp(-np.clip(z[1:p], -40, 40)))
        stop = p + c.aps*(c.users-1)
        power = softmax(np.c_[z[p:stop].reshape(c.aps, c.users-1), np.zeros(c.aps)])
        # Local component is the fixed zero reference; each row sums to one.
        task = softmax(np.c_[np.zeros(c.users), z[stop:].reshape(c.users, c.aps)])
        return dict(alpha=float(alpha), eta=eta, power=power, task=task)

    def physics(self, a):
        c = self.c
        p = c.ul_power_w*a['eta']
        sg = self.gamma.sum(0)
        desired = p*(c.antennas*sg)**2
        noncoherent = c.antennas*(self.gamma * (self.beta @ p)[:, None]).sum(0)
        coherent = np.zeros(c.users)
        for k in range(c.users):
            for j in range(c.users):
                if j != k and self.same[k, j]:
                    coherent[k] += p[j]*c.antennas**2*np.sum(
                        self.gamma[:, k]*self.beta[:, j]/self.beta[:, k])**2
        sinr = desired/(noncoherent + coherent + self.noise*c.antennas*sg)
        rate = c.bandwidth_hz*np.log2(1 + sinr)
        # Independent energy symbols per AP and user, unit-instantaneous-norm MRT.
        # Same-pilot beams coincide: E|h_k^H w_j|²=beta_k+(L-1)*gamma_k.
        rf = c.ap_power_w*np.sum(self.beta + (c.antennas-1)*self.gamma*(a['power'] @ self.same), axis=0)
        usable = c.slot_s*(1-c.pilot_symbols/c.coherence_symbols)
        harvested = c.harvest_efficiency*rf*a['alpha']*usable
        available = np.minimum(c.battery_j, self.battery + harvested)
        task = a['task']
        local_time = self.bits*c.cycles_per_bit*task[:, 0]/c.local_cpu_hz
        tx_time = self.bits*(1-task[:, 0])/np.maximum(rate, 1e-30)
        edge_time = np.sum(self.bits[:, None]*c.cycles_per_bit*task[:, 1:], axis=0)/c.edge_cpu_hz
        service = max(float(local_time.max()), float(tx_time.max()+edge_time.max()))
        consumed = c.capacitance*c.local_cpu_hz**2*self.bits*c.cycles_per_bit*task[:, 0] + p*tx_time
        return dict(rate=rate, harvested=harvested, available=available, consumed=consumed,
                    service_s=service, deadline_s=(1-a['alpha'])*usable,
                    latency_s=c.slot_s*c.pilot_symbols/c.coherence_symbols + a['alpha']*usable + service,
                    local_s=float(local_time.max()), tx_s=float(tx_time.max()), edge_s=float(edge_time.max()))

    @staticmethod
    def feasible(v):
        return bool(np.all(v['consumed'] <= v['available'] + 1e-15)
                    and v['service_s'] <= v['deadline_s'] + 1e-12)

    def project(self, proposed):
        """Finite candidate feasibility repair, not a Euclidean/global projection."""
        raw = self.physics(proposed)
        if self.feasible(raw):
            return proposed, raw, False
        c = self.c
        candidates = []
        # Retain the proposed task distribution; also try balanced offload/local fallbacks.
        tasks = [proposed['task'], np.c_[np.ones(c.users), np.zeros((c.users, c.aps))],
                 np.c_[np.zeros(c.users), np.full((c.users, c.aps), 1/c.aps)]]
        priority = (c.battery_j-self.battery)/c.battery_j + .01
        power = np.tile(priority/priority.sum(), (c.aps, 1))
        for alpha in (.1, .3, .5, .7, .85):
            for eta in (.05, .3, 1.):
                for task in tasks:
                    a = dict(alpha=alpha, eta=np.full(c.users, eta), power=power, task=task)
                    v = self.physics(a)
                    if self.feasible(v):
                        low = (v['available']-v['consumed'] < c.safe_battery_fraction*c.battery_j)
                        score = c.delay_weight*v['service_s']/c.slot_s + c.queue_weight*np.mean(self.queue*low)
                        candidates.append((score, a, v))
        if not candidates:
            return proposed, raw, True
        _, a, v = min(candidates, key=lambda x: x[0])
        return a, v, True

    def _history(self):
        c = self.c
        self.history = np.roll(self.history, -1, axis=0)
        self.history[-1] = np.stack([self.battery/c.battery_j, np.log1p(self.queue), self.net/c.battery_j], -1)

    def observe(self):
        c = self.c
        obs = np.concatenate([(np.log10(self.beta).ravel()+6)/3,
                              (np.log10(self.gamma).ravel()+6)/3,
                              self.bits/c.task_bits_max, self.battery/c.battery_j,
                              np.log1p(self.queue), self.net/c.battery_j,
                              self.pilots/max(c.pilot_symbols-1, 1)])
        return obs.astype(np.float32), self.history.copy()

    def step(self, z, repair=True):
        c = self.c
        proposed = self.decode(z)
        raw = self.physics(proposed)
        a, v, projected = self.project(proposed) if repair else (proposed, raw, False)
        outage = not self.feasible(v)
        before = self.battery.copy()
        oldq = self.queue.copy()
        # If no candidate completes every task, admit no tasks, harvest only and record outage.
        # No fractional task completion is reported as successful full-task service.
        energy = np.zeros(c.users) if outage else v['consumed']
        self.battery = v['available'] - energy
        self.net = self.battery - before
        low = self.battery < c.safe_battery_fraction*c.battery_j
        self.queue = np.maximum(0, self.queue + low - c.allowed_low_fraction)
        violation = np.maximum(raw['consumed']-raw['available'], 0).mean()/c.battery_j
        lateness = max(0, raw['service_s']-raw['deadline_s'])/c.slot_s
        latency = c.slot_s if outage else v['latency_s']
        rd = -c.delay_weight*((1. if outage else v['service_s']/c.slot_s) + c.outage_penalty*outage)
        re = (-c.queue_weight*np.mean(oldq*(low-c.allowed_low_fraction))
              + c.battery_weight*np.mean(self.battery/c.battery_j)
              - c.violation_weight*(violation+lateness))
        info = dict(latency_s=latency, service_s=v['service_s'], completed=int(not outage),
                    outage=int(outage), projected=int(projected), alpha=a['alpha'],
                    battery_j=float(self.battery.mean()), queue=float(self.queue.mean()),
                    low_fraction=float(low.mean()), harvested_j=float(v['harvested'].mean()),
                    consumed_j=float(energy.mean()), spill_j=float(np.maximum(before+v['harvested']-c.battery_j,0).mean()),
                    raw_energy_violation=violation, raw_deadline_violation=lateness,
                    offload_fraction=float(1-a['task'][:,0].mean()), rate_bps=float(v['rate'].mean()))
        for k in range(c.users):
            info[f'low_{k}'] = int(low[k])
            info[f'battery_{k}_j'] = float(self.battery[k])
            info[f'queue_{k}'] = float(self.queue[k])
        self.pos = np.mod(self.pos+self.rng.normal(0, c.speed_m_s*c.slot_s, self.pos.shape), c.area_m)
        self._exogenous()
        self._history()
        return self.observe(), np.array([rd, re], dtype=np.float32), info
