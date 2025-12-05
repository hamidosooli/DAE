"""
VMAS environment wrapper for DAE codebase
Adapted from DISSCv2 VMAS wrapper to work with the DAE environment interface
"""
import numpy as np
import gymnasium as gym
from gym import spaces
import vmas


class VMASEnv:
    """
    Wrapper for VMAS environments to match the MultiAgentEnv interface
    used in the DAE codebase.
    """
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, args):
        """
        Initialize VMAS environment wrapper.
        
        Args:
            args: Arguments object containing:
                - scenario_name: Name of the VMAS scenario
                - num_agents: Number of agents (optional, inferred from scenario if not provided)
                - Other VMAS environment parameters
        """
        self.scenario_name = args.scenario_name
        
        # Create VMAS environment
        # VMAS uses gymnasium interface, so we need to adapt it
        vmas_kwargs = {}
        if hasattr(args, 'num_agents'):
            vmas_kwargs['n_agents'] = args.num_agents
        
        # Create base VMAS environment with gymnasium wrapper
        self._env = vmas.make_env(
            scenario=self.scenario_name,
            num_envs=1,
            continuous_actions=False,
            dict_spaces=False,
            terminated_truncated=True,
            wrapper="gymnasium",
            device="cpu",
            **vmas_kwargs
        )
        
        # Get number of agents
        self.n = self._env.unwrapped.n_agents
        
        # Set up observation and action spaces
        # VMAS returns observations as a list/tuple, we need to convert to our format
        self.observation_space = []
        self.share_observation_space = []
        self.action_space = []
        
        # Get observation space from VMAS
        # VMAS with gymnasium wrapper returns observations as a list/tuple
        # Get shape from first observation
        test_obs, _ = self._env.reset()
        if isinstance(test_obs, (list, tuple)):
            self.obs_dim = len(test_obs[0]) if len(test_obs) > 0 else 0
        elif isinstance(test_obs, np.ndarray):
            if len(test_obs.shape) == 1:
                self.obs_dim = test_obs.shape[0]
            else:
                self.obs_dim = test_obs.shape[1] if test_obs.shape[0] == self.n else test_obs.shape[0]
        else:
            self.obs_dim = len(test_obs) if hasattr(test_obs, '__len__') else 1
        
        # Create observation spaces for each agent
        for _ in range(self.n):
            self.observation_space.append(
                spaces.Box(low=-np.inf, high=+np.inf, shape=(self.obs_dim,), dtype=np.float32)
            )
        
        # Share observation space is concatenation of all observations
        share_obs_dim = self.obs_dim * self.n
        for _ in range(self.n):
            self.share_observation_space.append(
                spaces.Box(low=-np.inf, high=+np.inf, shape=(share_obs_dim,), dtype=np.float32)
            )
        
        # Action space - VMAS uses discrete actions (5 actions: no-op, up, down, left, right)
        for _ in range(self.n):
            self.action_space.append(spaces.Discrete(5))
        
        # Track current step for episode length
        self.current_step = 0
        if hasattr(args, 'episode_length'):
            self.world_length = args.episode_length
        else:
            self.world_length = 100  # default episode length
        
        # Rendering
        self.viewers = [None]
        self.render_mode = None

    def seed(self, seed=None):
        """Set random seed for the environment."""
        if seed is not None:
            np.random.seed(seed)
            # VMAS environments use gymnasium reset with seed
            if hasattr(self._env, 'reset'):
                # Reset with seed to initialize RNG
                self._env.reset(seed=seed)
        return [seed] if seed is not None else [None]

    def reset(self):
        """
        Reset the environment and return initial observations.
        
        Returns:
            obs_n: List of observations for each agent
        """
        self.current_step = 0
        
        # Reset VMAS environment
        # VMAS uses gymnasium interface: reset() returns (obs, info)
        obs, info = self._env.reset()
        
        # Convert to list format expected by DAE
        # VMAS with gymnasium wrapper returns observations as a list/tuple
        if isinstance(obs, (list, tuple)):
            obs_n = [np.array(o, dtype=np.float32).flatten() for o in obs]
        elif isinstance(obs, np.ndarray):
            # If it's a single array, split by agents
            if len(obs.shape) == 1:
                # Single observation, assume it's for one agent or needs splitting
                obs_n = [obs.copy().flatten() for _ in range(self.n)]
            else:
                # Multiple observations stacked
                obs_n = [obs[i].copy().flatten() for i in range(min(self.n, obs.shape[0]))]
                # Pad if needed
                while len(obs_n) < self.n:
                    obs_n.append(obs_n[0].copy() if len(obs_n) > 0 else np.zeros(self.obs_dim, dtype=np.float32))
        else:
            # Fallback: create list from observation
            obs_n = [np.array(obs, dtype=np.float32).flatten() for _ in range(self.n)]
        
        return obs_n

    def step(self, action_n):
        """
        Step the environment with actions for all agents.
        
        Args:
            action_n: List of actions, one per agent
            
        Returns:
            obs_n: List of observations for each agent
            reward_n: List of rewards for each agent (as nested lists for compatibility)
            done_n: List of done flags for each agent
            info_n: List of info dicts for each agent
        """
        self.current_step += 1
        
        # Convert actions to format expected by VMAS
        # VMAS expects actions as a tuple/list of discrete actions
        if isinstance(action_n[0], (list, tuple, np.ndarray)):
            # If actions are already in list format, extract the action value
            actions = [int(a[0]) if len(a) > 0 else int(a) for a in action_n]
        else:
            # Actions are already integers
            actions = [int(a) for a in action_n]
        
        # Step VMAS environment
        # VMAS uses gymnasium interface: step() returns (obs, reward, terminated, truncated, info)
        obs, rewards, terminated, truncated, info = self._env.step(actions)
        
        # Convert observations to list format
        # VMAS with gymnasium wrapper returns observations as a list/tuple
        if isinstance(obs, (list, tuple)):
            obs_n = [np.array(o, dtype=np.float32).flatten() for o in obs]
        elif isinstance(obs, np.ndarray):
            if len(obs.shape) == 1:
                obs_n = [obs.copy().flatten() for _ in range(self.n)]
            else:
                obs_n = [obs[i].copy().flatten() for i in range(min(self.n, obs.shape[0]))]
                # Pad if needed
                while len(obs_n) < self.n:
                    obs_n.append(obs_n[0].copy() if len(obs_n) > 0 else np.zeros(self.obs_dim, dtype=np.float32))
        else:
            obs_n = [np.array(obs, dtype=np.float32).flatten() for _ in range(self.n)]
        
        # Convert rewards to list format (nested lists for compatibility)
        if isinstance(rewards, (list, tuple, np.ndarray)):
            if isinstance(rewards, np.ndarray) and rewards.ndim == 0:
                # Scalar reward
                reward_n = [[float(rewards)]] * self.n
            else:
                reward_n = [[float(r)] for r in rewards]
        else:
            reward_n = [[float(rewards)]] * self.n
        
        # Convert done flags
        # Check if episode is done (terminated or truncated)
        # VMAS returns terminated and truncated as arrays or booleans
        if isinstance(terminated, (list, tuple, np.ndarray)):
            terminated = np.asarray(terminated)
            truncated = np.asarray(truncated) if isinstance(truncated, (list, tuple, np.ndarray)) else truncated
            done = bool(np.any(terminated)) or bool(np.any(truncated) if isinstance(truncated, np.ndarray) else truncated)
        else:
            done = bool(terminated) or bool(truncated)
        
        # Create done_n list (all agents get the same done flag)
        done_n = [done] * self.n
        
        # Also check episode length
        if self.current_step >= self.world_length:
            done_n = [True] * self.n
        
        # Convert info to list format
        info_n = []
        for i in range(self.n):
            agent_info = {}
            if isinstance(info, dict):
                # If info is a dict, try to extract agent-specific info
                if 'agents' in info:
                    agent_info = info['agents'][i] if i < len(info['agents']) else {}
                else:
                    agent_info = info.copy()
            elif isinstance(info, (list, tuple)):
                agent_info = info[i] if i < len(info) else {}
            else:
                agent_info = {}
            
            # Add individual reward if available
            if isinstance(rewards, (list, tuple, np.ndarray)) and len(rewards) > i:
                agent_info['individual_reward'] = float(rewards[i])
            elif not isinstance(rewards, (list, tuple, np.ndarray)):
                agent_info['individual_reward'] = float(rewards)
            
            info_n.append(agent_info)
        
        return obs_n, reward_n, done_n, info_n

    def render(self, mode='human'):
        """
        Render the environment.
        
        Args:
            mode: 'human' or 'rgb_array'
            
        Returns:
            Rendered frame (if mode='rgb_array') or None
        """
        self.render_mode = mode
        
        # VMAS environments have render method
        if hasattr(self._env, 'render'):
            if mode == 'rgb_array':
                frame = self._env.render(mode='rgb_array')
                if frame is not None:
                    return frame
            elif mode == 'human':
                self._env.render(mode='human')
                return None
        
        return None

    def close(self):
        """Close the environment and cleanup resources."""
        if hasattr(self._env, 'close'):
            self._env.close()
        
        # Close viewers if any
        if self.viewers is not None:
            for viewer in self.viewers:
                if viewer is not None:
                    try:
                        viewer.close()
                    except:
                        pass
            self.viewers = [None]

