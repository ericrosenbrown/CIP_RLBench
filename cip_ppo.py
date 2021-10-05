import torch
import torch.nn as nn
import numpy as np
from spinup.exercises.pytorch.problem_set_1 import exercise1_1
from spinup.exercises.pytorch.problem_set_1 import exercise1_2_auxiliary
from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget
from rlbench.tasks import OpenDrawer
from rlbench.tasks import OpenDoor
from rlbench.tasks import OpenWindow
from rlbench.tasks import ToiletSeatUp
from rlbench.tasks import ToiletSeatDown
from rlbench.tasks import PushButton
from rlbench.tasks import ReachAndDrag
from rlbench.tasks import PlayJenga
from rlbench.tasks import PickAndLift
import numpy as np
from pyrep.errors import ConfigurationPathError

from collections import OrderedDict

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def run_cip(task):
    #print("Task waypoints:")

    waypoints = task._task.get_waypoints()
    grasped = False

    task._robot.arm.set_control_loop_enabled(True)

    for i, point in enumerate(waypoints):
        done = False
        #i = 0
        #point = waypoints[0]

        point.start_of_path()

        try:
            path = point.get_path()
        except ConfigurationPathError as e:
            print("=============== COULDN'T FIND PATH, RESET TASK!================")
            return True

        ext = point.get_ext()
        #print("Ext:",ext)

        while not done:
            done = path.step()
            task._task.pyrep.step() 

        point.end_of_path()

        if "close_gripper()" in ext:
            done = False
            while not done:
                done = task._robot.gripper.actuate(0.0,0.04)
                task._task.pyrep.step()

            task._robot.arm.set_control_loop_enabled(False)
            return False
    task._robot.arm.set_control_loop_enabled(False)
    return True


"""

Exercise 1.2: PPO Gaussian Policy

You will implement an MLP diagonal Gaussian policy for PPO by
writing an MLP-builder, and a few other key functions.

Log-likelihoods will be computed using your answer to Exercise 1.1,
so make sure to complete that exercise before beginning this one.

"""

def mlp(sizes, activation, output_activation=nn.Identity):
    """
    Build a multi-layer perceptron in PyTorch.

    Args:
        sizes: Tuple, list, or other iterable giving the number of units
            for each layer of the MLP. 

        activation: Activation function for all layers except last.

        output_activation: Activation function for last layer.

    Returns:
        A PyTorch module that can be called to give the output of the MLP.
        (Use an nn.Sequential module.)

    """
    """
    net_params = []
    for size  in sizes:
        net_params.append(nn.Linear(in_features=size[0],out_features=size[1]))
        net_params.append(activation)
    net_params.append(output_activation)
    print("my params:",net_params)
    return(nn.Sequential(*net_params))
    """

    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


class DiagonalGaussianDistribution:

    def __init__(self, mu, log_std):
        self.mu = mu
        self.log_std = log_std

    def sample(self):
        """
        Returns:
            A PyTorch Tensor of samples from the diagonal Gaussian distribution with
            mean and log_std given by self.mu and self.log_std.
        """
        return self.mu + torch.exp(self.log_std)*torch.randn_like(self.mu)

    #================================(Given, ignore)==========================================#
    def log_prob(self, value):
        return exercise1_1.gaussian_likelihood(value, self.mu, self.log_std)

    def entropy(self):
        return 0.5 + 0.5 * np.log(2 * np.pi) + self.log_std.sum(axis=-1)
    #=========================================================================================#


class MLPGaussianActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        """
        Initialize an MLP Gaussian Actor by making a PyTorch module for computing the
        mean of the distribution given a batch of observations, and a log_std parameter.

        Make log_std a PyTorch Parameter with the same shape as the action vector, 
        independent of observations, initialized to [-0.5, -0.5, ..., -0.5].
        (Make sure it's trainable!)
        """
        #######################
        #                     #
        #   YOUR CODE HERE    #
        #                     #
        #######################
        log_std = -0.5 * np.ones(act_dim,dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim]+list(hidden_sizes) + [act_dim], activation)
        print("print mu net!")
        print(self.mu_net)

    #================================(Given, ignore)==========================================#
    def forward(self, obs, act=None):
        mu = self.mu_net(obs)
        pi = DiagonalGaussianDistribution(mu, self.log_std)
        logp_a = None
        if act is not None:
            logp_a = pi.log_prob(act)
        return pi, logp_a
    #=========================================================================================#

import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import spinup.algos.pytorch.ppo.core as core
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, 78), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, 7), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}



def ppo(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
        target_kl=0.01, logger_kwargs=dict(), save_freq=10):
    """
    Proximal Policy Optimization (by clipping), 

    with early stopping based on approximate KL

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with a 
            ``step`` method, an ``act`` method, a ``pi`` module, and a ``v`` 
            module. The ``step`` method should accept a batch of observations 
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``v``        (batch,)          | Numpy array of value estimates
                                           | for the provided observations.
            ``logp_a``   (batch,)          | Numpy array of log probs for the
                                           | actions in ``a``.
            ===========  ================  ======================================

            The ``act`` method behaves the same as ``step`` but only returns ``a``.

            The ``pi`` module's forward call should accept a batch of 
            observations and optionally a batch of actions, and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       N/A               | Torch Distribution object, containing
                                           | a batch of distributions describing
                                           | the policy for the provided observations.
            ``logp_a``   (batch,)          | Optional (only returned if batch of
                                           | actions is given). Tensor containing 
                                           | the log probability, according to 
                                           | the policy, of the provided actions.
                                           | If actions not given, will contain
                                           | ``None``.
            ===========  ================  ======================================

            The ``v`` module's forward call should accept a batch of observations
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``v``        (batch,)          | Tensor containing the value estimates
                                           | for the provided observations. (Critical: 
                                           | make sure to flatten this!)
            ===========  ================  ======================================


        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to PPO.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        clip_ratio (float): Hyperparameter for clipping in the policy objective.
            Roughly: how far can the new policy go from the old policy while 
            still profiting (improving the objective function)? The new policy 
            can still go farther than the clip_ratio says, but it doesn't help
            on the objective anymore. (Usually small, 0.1 to 0.3.) Typically
            denoted by :math:`\epsilon`. 

        pi_lr (float): Learning rate for policy optimizer.

        vf_lr (float): Learning rate for value function optimizer.

        train_pi_iters (int): Maximum number of gradient descent steps to take 
            on policy loss per epoch. (Early stopping may cause optimizer
            to take fewer than this.)

        train_v_iters (int): Number of gradient descent steps to take on 
            value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used 
            for early stopping. (Usually small, 0.01 or 0.05.)

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = env_fn
    obs_dim = np.zeros([78],dtype=np.int) #7 dimensional EE pose
    act_dim = np.zeros([7],dtype=np.int) #8 dimensional joint control

    # Create actor-critic module
    ac = actor_critic(obs_dim, act_dim, **ac_kwargs)

    # Sync params across processes
    sync_params(ac)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

    # Set up function for computing PPO policy loss
    def compute_loss_pi(data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Policy loss
        pi, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        return ((ac.v(obs) - ret)**2).mean()

    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

    # Set up model saving

    def update():
        data = buf.get()

        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            kl = mpi_avg(pi_info['kl'])
            if kl > 1.5 * target_kl:
                break
            loss_pi.backward()
            mpi_avg_grads(ac.pi)    # average grads across MPI processes
            pi_optimizer.step()


        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            mpi_avg_grads(ac.v)    # average grads across MPI processes
            vf_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']

    # Prepare for interaction with environment
    start_time = time.time()
    keepResetting = True
    if keepResetting:
        env.reset()
        keepResetting = run_cip(env)

    ep_ret, ep_len = 0, 0
    of = env.get_observation()
    o = np.concatenate([of.joint_positions,of.joint_velocities,of.task_low_dim_state])

    # Main loop: collect experience in env and update/log each epoch
    epoch_returns = []
    for epoch in range(epochs):
        epoch_ret = 0
        for t in range(local_steps_per_epoch):
            a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32))
            af = np.concatenate([a, [0]], axis=-1)

            next_o, r, d = env.step(af)
            ep_ret += r
            ep_len += 1

            # save and log
            buf.store(o, a, r, v, logp)
            
            # Update obs (critical!)
            o = np.concatenate([next_o.joint_positions,of.joint_velocities,next_o.task_low_dim_state])

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t==local_steps_per_epoch-1

            if terminal or epoch_ended:
                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    _, v, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
                else:
                    v = 0
                buf.finish_path(v)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    pass
                #print("Ep ret:",ep_ret)
                epoch_ret += ep_ret
                keepResetting = True
                if keepResetting:
                    o, ep_ret, ep_len = env.reset(), 0, 0
                    keepResetting = run_cip(env)
                of = env.get_observation()
                o = np.concatenate([of.joint_positions,of.joint_velocities,of.task_low_dim_state])
        print("EPOCH RET:",epoch_ret)
        epoch_returns.append(epoch_ret)
        if epoch % 100 == 0:
            print(epoch_returns)




        # Perform PPO update!
        update()




if __name__ == '__main__':
    """
    Run this file to verify your solution.
    """

    #from spinup import ppo_pytorch as ppo
    from spinup.exercises.common import print_result
    from functools import partial
    import gym
    import os
    import pandas as pd
    import psutil
    import time

    logdir = "/tmp/experiments/%i"%int(time.time())

    ActorCritic = partial(exercise1_2_auxiliary.ExerciseActorCritic, actor=MLPGaussianActor)
    
    obs_config = ObservationConfig()
    obs_config.set_all_high_dim(False)
    obs_config.set_all_low_dim(True)

    action_mode = ActionMode(ArmActionMode.ABS_JOINT_VELOCITY)
    rlenv = Environment(
        action_mode, obs_config=obs_config, headless=False)

    rlenv.launch()

    env = rlenv.get_task(ToiletSeatDown)


    ppo(env_fn = env,
        actor_critic=ActorCritic,
        ac_kwargs=dict(hidden_sizes=(64,)),
        steps_per_epoch=100, epochs=1000, logger_kwargs=dict(output_dir=logdir))

    # Get scores from last five epochs to evaluate success.
    data = pd.read_table(os.path.join(logdir,'progress.txt'))
    last_scores = data['AverageEpRet'][-5:]

    # Your implementation is probably correct if the agent has a score >500,
    # or if it reaches the top possible score of 1000, in the last five epochs.
    correct = np.mean(last_scores) > 500 or np.max(last_scores)==1e3
    print_result(correct)
