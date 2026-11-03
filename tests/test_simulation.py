import unittest
import numpy as np
import torch
from leppo.environment import Config, MECEnv
from leppo.policy import ActorCritic, gae, update


class PhysicsTests(unittest.TestCase):
    def test_simplex_and_mmse(self):
        for m,k in [(1,1),(8,4),(2,6)]:
            env = MECEnv(Config(aps=m,users=k),seed=6)
            a = env.decode(np.linspace(-20,20,env.action_dim))
            np.testing.assert_allclose(a['power'].sum(1),1)
            np.testing.assert_allclose(a['task'].sum(1),1)
            self.assertTrue(np.all(env.gamma<=env.beta))
            self.assertTrue(np.all(env.gamma>0))
            self.assertEqual(len(env.observe()[0]),env.obs_dim)

    def test_single_user_closed_form(self):
        c = Config(aps=1,users=1,antennas=4)
        e = MECEnv(c)
        a = e.decode(np.zeros(e.action_dim))
        b,g,p = e.beta[0,0],e.gamma[0,0],c.ul_power_w*a['eta'][0]
        sinr = p*c.antennas*g/(p*b+e.noise)
        v = e.physics(a)
        self.assertAlmostEqual(v['rate'][0]/(c.bandwidth_hz*np.log2(1+sinr)),1)
        rf = c.ap_power_w*(b+(c.antennas-1)*g)
        expected = c.harvest_efficiency*rf*a['alpha']*c.slot_s*(1-c.pilot_symbols/c.coherence_symbols)
        self.assertAlmostEqual(v['harvested'][0]/expected,1)

    def test_mrt_monte_carlo(self):
        # Independently sample the MMSE estimate/error to verify the WPT expression.
        rng = np.random.default_rng(0)
        n,l,b,g = 30000,4,2.,.7
        est = np.sqrt(g/2)*(rng.normal(size=(n,l))+1j*rng.normal(size=(n,l)))
        err = np.sqrt((b-g)/2)*(rng.normal(size=(n,l))+1j*rng.normal(size=(n,l)))
        beam = est/np.linalg.norm(est,axis=1,keepdims=True)
        measured = np.mean(np.abs(np.sum((est+err).conj()*beam,axis=1))**2)
        self.assertLess(abs(measured/(b+(l-1)*g)-1),.025)

    def test_battery_queue_conservation_and_reproducibility(self):
        a,b = MECEnv(seed=77),MECEnv(seed=77)
        rng = np.random.default_rng(4)
        for _ in range(80):
            old,q = a.battery.copy(),a.queue.copy()
            z = rng.normal(size=a.action_dim)
            _,_,v = a.step(z)
            _,_,w = b.step(z)
            self.assertEqual(v,w)
            self.assertTrue(np.all(a.battery>=-1e-14))
            self.assertTrue(np.all(a.battery<=a.c.battery_j+1e-14))
            self.assertAlmostEqual(a.battery.mean()-old.mean(),v['harvested_j']-v['spill_j']-v['consumed_j'])
            low = a.battery<a.c.safe_battery_fraction*a.c.battery_j
            np.testing.assert_allclose(a.queue,np.maximum(0,q+low-a.c.allowed_low_fraction))

    def test_no_hidden_task_completion_when_infeasible(self):
        e = MECEnv(Config(battery_j=1e-20,ap_power_w=1e-20))
        _,_,info = e.step(np.zeros(e.action_dim))
        self.assertEqual(info['outage'],1)
        self.assertEqual(info['completed'],0)
        self.assertEqual(info['consumed_j'],0)

    def test_orthogonal_pilots_improve_estimation(self):
        orth = MECEnv(Config(users=2,pilot_symbols=2),seed=3)
        reused = MECEnv(Config(users=2,pilot_symbols=1),seed=3)
        np.testing.assert_array_equal(orth.beta,reused.beta)
        self.assertTrue(np.all(orth.gamma>reused.gamma))


class LearningTests(unittest.TestCase):
    def test_gae_reference(self):
        r=np.ones((3,2),dtype=np.float32)
        adv,ret=gae(r,np.zeros_like(r),np.zeros(2),gamma=1,lam=1)
        np.testing.assert_allclose(adv,[[3,3],[2,2],[1,1]])
        np.testing.assert_array_equal(adv,ret)

    def test_policy_replay_and_gradient(self):
        torch.set_num_threads(1)
        for algorithm in ['leppo','ppo','no-gru']:
            torch.manual_seed(5)
            e=MECEnv(); model=ActorCritic(e.obs_dim,e.action_dim,e.c.users,algorithm)
            state=e.observe(); z,lp,v=model.act(state)
            dist,_=model(torch.tensor(state[0][None]),torch.tensor(state[1][None]))
            self.assertAlmostEqual(lp,float(dist.log_prob(torch.tensor(z[None])).sum().detach()),places=4)
            n=8; vals=len(v)
            batch=[np.tile(state[0],(n,1)),np.tile(state[1],(n,1,1,1)),np.tile(z,(n,1)),
                   np.full(n,lp),np.ones((n,vals)),np.ones((n,vals))]
            before=model.delay_critic.weight.detach().clone()
            losses=update(model,torch.optim.Adam(model.parameters(),lr=.001),batch,epochs=1)
            self.assertTrue(np.isfinite(losses).all())
            self.assertFalse(torch.equal(before,model.delay_critic.weight))


if __name__=='__main__':
    unittest.main()
