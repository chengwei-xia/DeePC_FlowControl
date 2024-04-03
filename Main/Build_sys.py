import numpy as np
import scipy.signal as scipysig
from typing import Optional
from Utils.utils import Data
import gym
from Environment.env import resume_env, nb_actuations
import time

class System(object):
    """
    Represents a dynamical system that can be simulated
    """
    def __init__(self, sys: scipysig.StateSpace, x0: Optional[np.ndarray] = None):
        """
        :param sys: a linear system
        :param x0: initial state
        """
        assert x0 is None or sys.A.shape[0] == len(x0), 'Invalid initial condition'
        self.sys = sys
        self.x0 = x0 if x0 is not None else np.zeros(sys.A.shape[0])
        self.u = None
        self.y = None

    def apply_input(self, u: np.ndarray, noise_std: float = 0.5) -> Data:
        """
        Applies an input signal to the system.
        :param u: input signal. Needs to be of shape T x M, where T is the batch size and
                  M is the number of features
        :param noise_std: standard deviation of the measurement noise
        :return: tuple that contains the (input,output) of the system
        """
        T = len(u)
        if T > 1:
            # If u is a signal of length > 1 use dlsim for quicker computation
            t, y, x0 = scipysig.dlsim(self.sys, u, t = np.arange(T) * self.sys.dt, x0 = self.x0)
            self.x0 = x0[-1]
        else:
            y = self.sys.C @ self.x0
            self.x0 = self.sys.A @ self.x0.flatten() + self.sys.B @ u.flatten()
            
        np.random.seed(99999)
        y = y + noise_std * np.random.normal(size = T).reshape(T, 1)

        self.u = np.vstack([self.u, u]) if self.u is not None else u
        self.y = np.vstack([self.y, y]) if self.y is not None else y
        return Data(u, y)

    def get_last_n_samples(self, n: int) -> Data:
        """
        Returns the last n samples
        :param n: integer value
        """
        assert self.u.shape[0] >= n, 'Not enough samples are available'
        return Data(self.u[-n:], self.y[-n:])

    def get_all_samples(self) -> Data:
        """
        Returns all samples
        """
        return Data(self.u, self.y)

    def reset(self, data_ini: Optional[Data] = None, x0: Optional[np.ndarray] = None):
        """
        Reset initial state and collected data
        """
        self.u = None if data_ini is None else data_ini.u
        self.y = None if data_ini is None else data_ini.y
        self.x0 = x0 if x0 is not None else np.zeros(self.sys.A.shape[0])
        
        
class GymSystem(object):
    """
    Set up a gym system to collect i/o data
    """
    def __init__(self, x0: Optional[np.ndarray] = None):
        """
        :param env: a gym system, e.g. inverted pendulum
        :param x0: initial state
        """
        self.env = gym.make("InvertedPendulum-v4")
        # assert x0 is None or sys.A.shape[0] == len(x0), 'Invalid initial condition'
        # self.env = env
        # self.x0 = x0 if x0 is not None else np.zeros(sys.A.shape[0])
        self.u = None # Buffer for u
        self.y = None

    def apply_input(self, u: np.ndarray, noise_std: float = 0.5) -> Data:
        """
        Applies an input signal to the system.
        :param u: input signal. Needs to be of shape T x M, where T is the batch size and
                  M is the number of features
        :param noise_std: standard deviation of the measurement noise
        :return: tuple that contains the (input,output) of the system
        """
        
        T = len(u)
        u_run = None
        y_run = None
        
        for k in range(T):
            y, reward, terminated, truncated, info = self.env.step(u[k])

            #if terminated or truncated:
            #    y, info = self.env.reset()

            y = y + noise_std * np.random.normal(size = 1)
            
            u_run = np.vstack([u_run, u[k]]) if u_run is not None else u[k] # Fill the buffer for single run
            y_run = np.vstack([y_run, y]) if y_run is not None else y
            
        self.u = np.vstack([self.u, u_run]) if self.u is not None else u_run # Fill the buffer for all data
        self.y = np.vstack([self.y, y_run]) if self.y is not None else y_run
            
        return Data(u_run, y_run)

    def get_last_n_samples(self, n: int) -> Data:
        """
        Returns the last n samples
        :param n: integer value
        """
        assert self.u.shape[0] >= n, 'Not enough samples are available'
        return Data(self.u[-n:], self.y[-n:])

    def get_all_samples(self) -> Data:
        """
        Returns all samples
        """
        return Data(self.u, self.y)

    def reset(self, data_ini: Optional[Data] = None, x0: Optional[np.ndarray] = None):
        """
        Reset initial state and collected data
        """
        self.u = None if data_ini is None else data_ini.u
        self.y = None if data_ini is None else data_ini.y
        #obs_ini, info = self.env.reset()
        
    def close(self):
        """
        Close gym environment
        """
        self.env.close()

class FlowSystem(object):
    """
    Set up a gym system to collect i/o data
    """
    def __init__(self, x0: Optional[np.ndarray] = None):
        """
        :param env: a gym system, e.g. inverted pendulum
        :param x0: initial state
        """
        self.env = resume_env(plot=False, dump_CL=False, dump_debug=10, n_env=1)
        # assert x0 is None or sys.A.shape[0] == len(x0), 'Invalid initial condition'
        # self.env = env
        # self.x0 = x0 if x0 is not None else np.zeros(sys.A.shape[0])
        self.u = None # Buffer for u
        self.y = None

    def apply_input(self, u: np.ndarray, noise_std: float = 0.5) -> Data:
        """
        Applies an input signal to the system.
        :param u: input signal. Needs to be of shape T x M, where T is the batch size and
                  M is the number of features
        :param noise_std: standard deviation of the measurement noise
        :return: tuple that contains the (input,output) of the system
        """
        
        T = len(u)
        u_run = None
        y_run = None
        
        for k in range(T):
            
            #print(u[k][0])
            #print(type(u[k]))
            start_time = time.perf_counter()
            y, reward, terminated, info = self.env.step(np.array([u[k][0],u[k][0]]))
            end_time = time.perf_counter()
            print('One step takes :% s' % ((end_time - start_time)))
            #print(f'Simulation one step takes {end_time - start_time} ms.')
            #if terminated or truncated:
            #    y, info = self.env.reset()

            y = y + noise_std * np.random.normal(size = 1)
            
            u_run = np.vstack([u_run, u[k]]) if u_run is not None else u[k] # Fill the buffer for single run
            y_run = np.vstack([y_run, y]) if y_run is not None else y
            print(f'Run simulation for the {k} th step.')
            print(f'The inputs are {u[k]} and outputs are {y}.')
            
        self.u = np.vstack([self.u, u_run]) if self.u is not None else u_run # Fill the buffer for all data
        self.y = np.vstack([self.y, y_run]) if self.y is not None else y_run
            
        return Data(u_run, y_run)

    def get_last_n_samples(self, n: int) -> Data:
        """
        Returns the last n samples
        :param n: integer value
        """
        
        assert self.u.shape[0] >= n, 'Not enough samples are available'
        return Data(self.u[-n:], self.y[-n:])


    def get_all_samples(self) -> Data:
        """
        Returns all samples from the flow system
        """
        #measure_type = 'drag'
        #u, y = self.env.read_buffer_all()
        
        return Data(self.u,self.y)

    def reset(self, data_ini: Optional[Data] = None, x0: Optional[np.ndarray] = None):
        """
        Reset initial state and collected data
        """
        x0 = self.env.reset()
        
        self.u = None if data_ini is None else data_ini.u
        self.y = None if data_ini is None else data_ini.y
        #obs_ini, info = self.env.reset()
        
    def close(self):
        """
        Close gym environment
        """
        self.env.close()

