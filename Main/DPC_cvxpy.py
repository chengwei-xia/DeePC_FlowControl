from cmath import isclose
from copy import deepcopy
import math
import numpy as np
import cvxpy as cp
from typing import Tuple, Callable, List, Optional, Union, Dict
from scipy.linalg import hankel, block_diag

import os
import sys
cwd = os.getcwd()
sys.path.append(cwd + "../Utils/")
sys.path.append(cwd + "../Environment/")

from cvxpy.expressions.expression import Expression
from cvxpy.constraints.constraint import Constraint
from Utils.utils import (
    Data,
    split_data,
    low_rank_matrix_approximation,
    OptimizationProblem,
    OptimizationProblemVariables)




class DeePC(object):
    optimization_problem: OptimizationProblem = None
    _SMALL_NUMBER: float = 1e-32

    def __init__(self, data: Data, Tini: int, horizon: int):
        """
        Solves the DeePC optimization problem
        For more info check alg. 2 in https://arxiv.org/pdf/1811.05890.pdf

        :param data:                A tuple of input/output data. Data should have shape TxM
                                    where T is the batch size and M is the number of features
        :param Tini:                number of samples needed to estimate initial conditions
        :param horizon:             horizon length
        :param explained_variance:  Regularization term in (0,1] used to approximate the Hankel matrices.
                                    By default is None (no low-rank approximation is performed).
        """
        self.Tini = Tini
        self.horizon = horizon
        self.M = data.u.shape[1]
        self.P = data.y.shape[1]
        self.T = data.y.shape[0]
        self.update_data(data)

        self.optimization_problem = None

    def update_data(self, data: Data):
        """
        Update Hankel matrices of DeePC. You need to rebuild the optimization problem
        after calling this funciton.

        :param data:                A tuple of input/output data. Data should have shape TxM
                                    where T is the batch size and M is the number of features
        """
        assert len(data.u.shape) == 2, \
            "Data needs to be shaped as a TxM matrix (T is the number of samples and M is the number of features)"
        assert len(data.y.shape) == 2, \
            "Data needs to be shaped as a TxM matrix (T is the number of samples and M is the number of features)"
        assert data.y.shape[0] == data.u.shape[0], \
            "Input/output data must have the same length"
        assert data.y.shape[0] - self.Tini - self.horizon + 1 >= 1, \
            f"There is not enough data: this value {data.y.shape[0] - self.Tini - self.horizon + 1} needs to be >= 1"
            
        ## Build Hankel matrices
        utr = np.transpose(data.u)
        ytr = np.transpose(data.y)
        
        Up, Uf, Yp, Yf, uini, yini = self.GetHankels(utr,ytr,self.horizon,self.Tini,self.T)

        self.Up = Up
        self.Uf = Uf
        self.Yp = Yp
        self.Yf = Yf
        #self.uini = uini
        #self.yini = yini
        
        self.optimization_problem = None
        
    def Hankelbuilder(self,data,T,Tf,Tini,num):
        
        num_g = T-Tini-Tf+1 #The second dimension of Hankel matrix
        hp = np.zeros([Tini*num,num_g]) # n is the dimension of your state
        hf = np.zeros([Tf*num,num_g])
        
        for j in range(num):
            pcol = data[j,:Tini]
            prow = data[j,Tini-1:T-Tf]
            fcol = data[j,Tini:Tini+Tf]
            frow = data[j,Tini+Tf-1:T]
            pj = hankel(pcol,prow) # get the intermediate hakel for the jjth dimension of the state
            fj = hankel(fcol,frow)
            sizepj = np.shape(pj)[0]
            sizefj = np.shape(fj)[0]
            
            for k in range(sizepj):
                hp[k*num+j,:] = pj[k,:] # allocate the intermediate hakel to the final large hankel
            
            for k in range(sizefj):
                hf[k*num+j,:] = fj[k,:]
            
        return hp,hf

    def GetHankels(self,u,y,Tf,Tini,T):
        
        #Load training data and initial data'
        utr = u[:,:T]
        ytr = y[:,:T]
        uini = u[:,0:Tini] #uini = u[:,T-Tini:T]
        yini = y[:,0:Tini] #yini = y[:,T-Tini:T] 

        """
        Up = np.zeros([Tini,num_g])
        Uf = np.zeros([Tf,num_g])
        Yp = np.zeros([Tini,num_g])
        Yf = np.zeros([Tf,num_g])
        """
        num_u = np.shape(utr)[0] # The number of u in one sample
        num_y = np.shape(ytr)[0]
        Up,Uf = self.Hankelbuilder(utr,T,Tf,Tini,num_u)
        Yp,Yf = self.Hankelbuilder(ytr,T,Tf,Tini,num_y)
        
        return Up,Uf,Yp,Yf,uini,yini

    def opt_setup(self,
            build_loss: Callable[[cp.Variable, cp.Variable], Expression],
            build_constraints: Optional[Callable[[cp.Variable, cp.Variable], Optional[List[Constraint]]]] = None,
            lambda_g: float = 0.,
            lambda_y: float = 0.,
            lambda_u: float= 0.,
            lambda_proj: float = 0.) -> OptimizationProblem:
        """
        Builds the DeePC optimization problem
        For more info check alg. 2 in https://arxiv.org/pdf/1811.05890.pdf

        For info on the projection (least-square) regularizer, see also
        https://arxiv.org/pdf/2101.01273.pdf


        :param build_loss:          Callback function that takes as input an (input,output) tuple of data
                                    of shape (TxM), where T is the horizon length and M is the feature size
                                    The callback should return a scalar value of type Expression
        :param build_constraints:   Callback function that takes as input an (input,output) tuple of data
                                    of shape (TxM), where T is the horizon length and M is the feature size
                                    The callback should return a list of constraints.
        :param lambda_g:            non-negative scalar. Regularization factor for g. Used for
                                    stochastic/non-linear systems.
        :param lambda_y:            non-negative scalar. Regularization factor for y_init. Used for
                                    stochastic/non-linear systems.
        :param lambda_u:            non-negative scalar. Regularization factor for u_init. Used for
                                    stochastic/non-linear systems.
        :param lambda_proj:         Positive term that penalizes the least square solution.
        :return:                    Parameters of the optimization problem
        """
        assert build_loss is not None, "Loss function callback cannot be none"
        assert lambda_g >= 0 and lambda_y >= 0, "Regularizers must be non-negative"
        assert lambda_u >= 0, "Regularizer of u_init must be non-negative"
        assert lambda_proj >= 0, "The projection regularizer must be non-negative"

        self.optimization_problem = None

        # Build variables
        uini = cp.Parameter(shape=(self.M * self.Tini), name='u_ini')
        yini = cp.Parameter(shape=(self.P * self.Tini), name='y_ini')
        u = cp.Variable(shape=(self.M * self.horizon), name='u')
        y = cp.Variable(shape=(self.P * self.horizon), name='y')
        g = cp.Variable(shape=(self.T - self.Tini - self.horizon + 1), name='g')
        slack_u = cp.Variable(shape=(self.Tini * self.M), name='slack_u')
        slack_y = cp.Variable(shape=(self.Tini * self.P), name='slack_y')

        Up, Yp, Uf, Yf = self.Up, self.Yp, self.Uf, self.Yf

        if lambda_proj > DeePC._SMALL_NUMBER:
            # Compute projection matrix (for the least square solution)
            Zp = np.vstack([Up, Yp, Uf])
            ZpInv = np.linalg.pinv(Zp)
            I = np.eye(self.T - self.Tini - self.horizon + 1)
            # Kernel orthogonal projector
            I_min_P = I - (ZpInv@ Zp)

        A = np.vstack([Up, Yp, Uf, Yf])
        b = cp.hstack([uini + slack_u, yini + slack_y, u, y])

        # Build system constraints based on data set
        constraints = [A @ g == b]

        if math.isclose(lambda_y, 0):
            constraints.append(cp.norm(slack_y, 2) <= DeePC._SMALL_NUMBER)
        if math.isclose(lambda_u, 0):
            constraints.append(cp.norm(slack_u, 2) <= DeePC._SMALL_NUMBER)

        # Reshape u and y to Tf * dim_u and Tf * dim_y
        u = cp.reshape(u, (self.horizon, self.M),'C')
        y = cp.reshape(y, (self.horizon, self.P),'C')

        _constraints = build_constraints(u, y) if build_constraints is not None else (None, None)

        for idx, constraint in enumerate(_constraints):
            if constraint is None or not isinstance(constraint, Constraint) or not constraint.is_dcp():
                raise Exception(f'Constraint {idx} is not defined or is not convex.')

        constraints.extend([] if _constraints is None else _constraints)

        # Build loss
        _loss = build_loss(u, y)
        
        if _loss is None or not isinstance(_loss, Expression) or not _loss.is_dcp():
            raise Exception('Loss function is not defined or is not convex!')

        # Add regularizers
        _regularizers = lambda_g * cp.norm(g, p=2)**2 if lambda_g > DeePC._SMALL_NUMBER else 0
        _regularizers += lambda_y * cp.norm(slack_y, p=2)**2 if lambda_y > DeePC._SMALL_NUMBER else 0
        _regularizers += lambda_proj * cp.norm(I_min_P @ g) if lambda_proj > DeePC._SMALL_NUMBER  else 0
        _regularizers += lambda_u * cp.norm(slack_u, p=2)**2 if lambda_u > DeePC._SMALL_NUMBER else 0

        problem_loss = _loss + _regularizers

        # Solve problem
        objective = cp.Minimize(problem_loss)

        try:
            problem = cp.Problem(objective, constraints)
        except cp.SolverError as e:
            raise Exception(f'Error while constructing the DeePC problem. Details: {e}')

        self.optimization_problem = OptimizationProblem(
            variables = OptimizationProblemVariables(
                u_ini = uini, y_ini = yini, u = u, y = y, g = g, slack_y = slack_y, slack_u = slack_u),
            constraints = constraints,
            objective_function = problem_loss,
            problem = problem
        )

        return self.optimization_problem

    def solve(
            self,
            data_ini: Data,
            **cvxpy_kwargs
        ) -> Tuple[np.ndarray, Dict[str, Union[float, np.ndarray, OptimizationProblemVariables]]]:
        """
        Solves the DeePC optimization problem
        For more info check alg. 2 in https://arxiv.org/pdf/1811.05890.pdf

        :param data_ini:            A tuple of input/output data used to estimate initial condition.
                                    Data should have shape Tini x M where Tini is the batch size and
                                    M is the number of features
        :param cvxpy_kwargs:        All arguments that need to be passed to the cvxpy solve method.
        :return u_optimal:          Optimal input signal to be applied to the system, of length `horizon`
        :return info:               A dictionary with 5 keys:
                                    info['variables']: variables of the optimization problem
                                    info['value']: value of the optimization problem
                                    info['u_optimal']: the same as the first value returned by this function
        """
        assert len(data_ini.u.shape) == 2, "Data needs to be shaped as a TxM matrix (T is the number of samples and M is the number of features)"
        assert len(data_ini.y.shape) == 2, "Data needs to be shaped as a TxM matrix (T is the number of samples and M is the number of features)"
        assert data_ini.u.shape[1] == self.M, "Incorrect number of features for the input signal"
        assert data_ini.y.shape[1] == self.P, "Incorrect number of features for the output signal"
        assert data_ini.y.shape[0] == data_ini.u.shape[0], "Input/output data must have the same length"
        assert data_ini.y.shape[0] == self.Tini, "Invalid size"
        assert self.optimization_problem is not None, "Problem was not built"


        # Need to transpose to make sure that time is over the columns, and features over the rows
        uini, yini = data_ini.u[:self.Tini].flatten(), data_ini.y[:self.Tini].flatten()

        self.optimization_problem.variables.u_ini.value = uini
        self.optimization_problem.variables.y_ini.value = yini

        try:
            result = self.optimization_problem.problem.solve(**cvxpy_kwargs)
        except cp.SolverError as e:
            raise Exception(f'Error while solving the DeePC problem. Details: {e}')

        if np.isinf(result):
            raise Exception('Problem is unbounded')

        u_optimal = (self.Uf @ self.optimization_problem.variables.g.value).reshape(self.horizon, self.M)
        info = {
            'value': result, 
            'variables': self.optimization_problem.variables,
            'u_optimal': u_optimal
            }

        return u_optimal, info
    
    def qp_setup(self,
            lambda_g: float = 0.,
            lambda_y: float = 0.,
            lambda_u: float= 0.,
            lambda_proj: float = 0.,
            u_low: float = 0.,
            u_up: float = 0.,
            yref: float = 0.)-> OptimizationProblem:
        """
        Solve QP in DeePC as min (1/2)xTPx + qTx s.t. Ax=b Gx<=h
        For convenience, x is formulated as x = (g,sigmau,sigmay,u,y)' with dimension dim_g + dim_u*Tini + dim_y*Tini + dim_u*Tf + dim_y*Tf
        """
        self.num_g = self.T - self.Tini - self.horizon + 1
        self.uini = cp.Parameter(shape=(self.M * self.Tini), name='u_ini')
        self.yini = cp.Parameter(shape=(self.P * self.Tini), name='y_ini')
        
        self.x_dim = self.num_g + self.M * self.Tini + self.P * self.Tini + self.M * self.horizon + self.P * self.horizon 
        
        self.x = cp.Variable(shape=(self.x_dim), name='x')
        
        G_g = lambda_g * np.eye(self.num_g)
        Su = lambda_u * np.eye(self.M * self.Tini)
        Sy = lambda_y * np.eye(self.P * self.Tini)
        R = 0*np.eye(self.M * self.horizon) #np.zeros([self.M * self.horizon,self.M * self.horizon]) #0.001*np.eye(self.M * self.horizon)
        Q = 100*np.eye(self.P * self.horizon)
        
        
        # Formulate matrices loss function
        P = block_diag(G_g,Su,Sy,R,Q)
        eig_P = np.linalg.eigvals(P)
        assert np.all(eig_P) >= 0,  "Quadratic matrix needs to be positive semidefinite."            
        q = np.hstack([np.zeros(self.x_dim - self.P * self.horizon), -np.dot(yref*np.ones(self.P * self.horizon),Q)])
        
        # TODO: set up P with Kronecker product
        
        # Formulate equality constraints
        Hankel = np.vstack([self.Up, self.Yp, self.Uf, self.Yf])
        Eye_sub = - block_diag(np.eye(self.M * self.Tini),np.eye(self.P * self.Tini),np.eye(self.M * self.horizon),np.eye(self.P * self.horizon)) # A diagnol matrix for subtraction
        A = np.hstack([Hankel, Eye_sub])
        b = cp.hstack([self.uini, self.yini, np.zeros(self.M * self.horizon + self.P * self.horizon)])
        b = b.T
        
        # Formulate inequality constraints
        Eye_u = np.vstack([np.eye(self.M * self.horizon),-np.eye(self.M * self.horizon)])
        G_sub = np.hstack([np.zeros([self.M * self.horizon + self.M * self.horizon, self.num_g + self.M * self.Tini + self.P * self.Tini]),Eye_u])
        G = np.hstack([G_sub,np.zeros([self.M * self.horizon + self.M * self.horizon, self.P * self.horizon])])
        h = np.hstack([u_up*np.ones(self.M * self.horizon),-u_low*np.ones(self.M * self.horizon)])
        h = h.T
        
        self.qp_problem = cp.Problem(cp.Minimize((1/2)*cp.quad_form(self.x, P)+ q.T @ self.x), #+ q.T @ self.x
                 [G @ self.x <= h,
                  A @ self.x == b])
        
    
        return self.qp_problem
    
    def qp_one_norm_setup(self,
            lambda_g: float = 0.,
            lambda_y: float = 0.,
            lambda_u: float= 0.,
            lambda_proj: float = 0.,
            u_low: float = 0.,
            u_up: float = 0.,
            yref: float = 0.)-> OptimizationProblem:
        """
        Solve QP in DeePC as min (1/2)xTPx + qTx s.t. Ax=b Gx<=h with 
        For convenience, x is formulated as x = (g,sigmau,sigmay,u,y,s)' with dimension dim_g + dim_u*Tini + dim_y*Tini + dim_u*Tf + dim_y*Tf
        """
        self.num_g = self.T - self.Tini - self.horizon + 1
        self.uini = cp.Parameter(shape=(self.M * self.Tini), name='u_ini')
        self.yini = cp.Parameter(shape=(self.P * self.Tini), name='y_ini')
        
        self.x_dim = self.num_g + self.M * self.Tini + self.P * self.Tini + self.M * self.horizon + self.P * self.horizon 
        
        self.x = cp.Variable(shape=(self.x_dim), name='x')
        
        Q = np.zeros([self.M * self.horizon,self.M * self.horizon]) #0.001*np.eye(self.M * self.horizon)
        R = np.eye(self.P * self.horizon)
        G = lambda_g * np.eye(self.num_g)
        Su = lambda_u * np.eye(self.M * self.Tini)
        Sy = lambda_y * np.eye(self.P * self.Tini)
        
        # Formulate loss function
        P = block_diag(G,Su,Sy,Q,R)
        eig_P = np.linalg.eigvals(P)
        assert np.all(eig_P) >= 0,  "Quadratic matrix needs to be positive semidefinite."            
        q = np.hstack([np.zeros(self.x_dim - self.P * self.horizon), -np.dot(yref*np.ones(self.P * self.horizon),R)])
        q = q.reshape((self.x_dim,1))
        
        # TODO: set up P with Kronecker product
        
        # Formulate equality constraints
        Hankel = np.vstack([self.Up, self.Yp, self.Uf, self.Yf])
        Eye_sub = - block_diag(np.eye(self.M * self.Tini),np.eye(self.P * self.Tini),np.eye(self.M * self.horizon),np.eye(self.P * self.horizon)) # A diagnol matrix for subtraction
        A = np.hstack([Hankel, Eye_sub])
        b = cp.hstack([self.uini, self.yini, np.zeros(self.M * self.horizon + self.P * self.horizon)])
        b = b.T
        
        # Formulate inequality constraints
        Eye_u = np.vstack([np.eye(self.M * self.horizon),-np.eye(self.M * self.horizon)])
        G_sub = np.hstack([np.zeros([self.M * self.horizon + self.M * self.horizon, self.num_g + self.M * self.Tini + self.P * self.Tini]),Eye_u])
        G = np.hstack([G_sub,np.zeros([self.M * self.horizon + self.M * self.horizon, self.P * self.horizon])])
        h = np.hstack([abs(u_up)*np.ones(self.M * self.horizon),abs(u_low)*np.ones(self.M * self.horizon)])
        h = h.T
        
        self.qp_problem = cp.Problem(cp.Minimize((1/2)*cp.quad_form(self.x, P)+ q @ self.x), #+ q.T @ self.x
                 [G @ self.x <= h,
                  A @ self.x == b])
        
    
        return self.qp_problem
    
    def qp_solve(
            self,
            data_ini: Data,
            **cvxpy_kwargs):
        
        assert self.qp_problem is not None, "Problem was not built"
        
        uini, yini = data_ini.u[:self.Tini].flatten(), data_ini.y[:self.Tini].flatten()

        self.uini.value = uini
        self.yini.value = yini
        
        try:
            result = self.qp_problem.solve(**cvxpy_kwargs)
        except cp.SolverError as e:
            raise Exception(f'Error while solving the DeePC problem. Details: {e}')
        
        if np.isinf(result):
            raise Exception('Problem is unbounded')
            
        g = self.x.value[:self.num_g]
        u_optimal = (self.Uf @ g).reshape(self.horizon, self.M)
        y_optimal = (self.Yf @ g).reshape(self.horizon, self.P)

            
        return u_optimal, y_optimal