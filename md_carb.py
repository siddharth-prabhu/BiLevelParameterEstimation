import os
from functools import partial, reduce
from typing import List, Tuple
import operator
import itertools
import logging
from datetime import datetime
from pprint import pformat
import argparse

import jax
import jax.numpy as jnp
import jax.random as jrandom
jax.config.update("jax_enable_x64", True)
from jax.experimental.ode import odeint
from jax import flatten_util, tree_util

import matplotlib.pyplot as plt
import pysindy as ps
import numpy as np
from cyipopt import minimize_ipopt
from scipy.integrate import odeint as scipy_odeint
from scipy.interpolate import CubicSpline

from utils import differentiable_optimization, odeint_diffrax, plot_coefficients, plot_trajectories


# choose hyperparameters
parser = argparse.ArgumentParser("ModelDiscoveryEsterificationOfCarboxylicAcid")
parser.add_argument("--iters", type = int, default = 100, help = "The maximum number of iterations to be performed in parameter estimation")
parser.add_argument("--siters", type = int, default = 20, help = "The maximum number of thresholding iterations to be performed in parameter estimation")
parser.add_argument("--tol", type = float, default = 1e-4, help = "Ipopt tolerance")
parser.add_argument("--atol", type = float, default = 1e-8, help = "Absolute tolerance of ode solver")
parser.add_argument("--rtol", type = float, default = 1e-6, help = "Relative tolerance of ode solver")
parser.add_argument("--mxstep", type = int, default = 10_000, help = "The maximum number of steps a ode solver takes")
parser.add_argument("--reg", type = float, default = 0.1, help = "L2 regularization penalty")
parser.add_argument("--threshold", type = float, default = 0.1, help = "Thresholding parameter")
parser.add_argument("--msg", type = str, default = "", help = "Sample msg that briefly describes this problem")
parser.add_argument("--method", type = int, default = 0, help = "Formulation type 0 : BiLevelOpt (DFSINDy innter + DFSINDy outer), 2 : FullNLP (DFSINDy), 2 : FullNLP (shooting/sequential)")

parser.add_argument("--id", type = str, default = "", help = "Slurm job id")
parser.add_argument("--partition", type = str, default = "", help = "The partition this job is assigned to")
parser.add_argument("--cpus", type = int, default = 1, help = "Maximum number of cpus availabe per node")

pargs = parser.parse_args()

_dir = os.path.join("log", "Esterification", str(datetime.now()))
divider = "--"*50 # printing separater
if not os.path.exists(_dir) : os.makedirs(_dir)

logger = logging.getLogger("__name__")
logger.setLevel(logging.INFO)
logfile = logging.FileHandler(os.path.join(_dir, "record.txt"))
logger.addHandler(logfile)
logger.info("Model Discovery - Esterificaiton of Carboxylic Acid")
logger.info(pformat(pargs.__dict__))


# Original system
def rate_constant(T, Tref, act) : return jnp.exp(- act * (10**4/T - 10**4/Tref) / 8.314)

stoic = jnp.array([
    0, 0, -1, 0, -1, 0, 0, -1, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0,
    -1, 1, 0, -1, 1, 1, 1, -1, 1, 1, -1, 0, -1, 0, 0, 0, 0, 0, 
    0, 1, -1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, -1, 
    0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, -1
]).reshape(-1, 6)

activation = jnp.array([ 9.48, 6.59, 10.9, 4.88, 7.74, 10.9, 8.58, 2.77, 4.05, 5.59, 1.14, 3.46]) # values divided by 1e4

def kinetic(x, t, p, T) -> jnp.ndarray:
    # species = [A, B, C, D, E, F, G, J, K, L, X] 
        # reactions = [
        #       F + D <k7 --  k1> E + X;
        #       E + B <k8 --  k2> G + D;
        #       A + G <k9 --  k3> E + C;
        #       B + D <k10 -- k4> J + E;
        #       A + E <k11 -- k5> D + K;
        #       X + K <k12 -- k6> L + D ]
    # p = [.3, .4, 1.1, .9, 1.0, .2, 1.2, .6, .7, .1, .5, .8]
    
    k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12 = p * rate_constant(T, 373., activation)
    
    reactions = jnp.array([
            k1 * x[5] * x[3] - k7 * x[4] * x[10],
            k2 * x[1] * x[4] - k8 * x[3] * x[6],
            k3 * x[0] * x[6] - k9 * x[2] * x[4],
            k4 * x[1] * x[3] - k10 * x[4] * x[7],
            k5 * x[0] * x[4] - k11 * x[3] * x[8],
            k6 * x[8] * x[10] - k12 * x[3] * x[9]
        ])
    
    return stoic @ reactions 


# Generate data
nx = 11 # Number of states
nr = 6 # Number of reactions
nexpt = 30
key = jrandom.PRNGKey(20)
key_temp, key_xinit = jrandom.split(key, 2)
_temp = [370, 375, 380., 385, 373]
temperature = jnp.array([i for i, _ in zip(itertools.cycle(_temp), range(nexpt))])
xinit = jrandom.uniform(key_xinit, shape = (nexpt, nx), minval = 4., maxval = 10.)
time_span = jnp.arange(0, 10., 0.05)
p_actual = jnp.array([.3, .4, 1.1, .9, 1.0, .2, 1.2, .6, .7, .1, .5, .8])
solution = jnp.stack([odeint(kinetic, xi, time_span, p_actual, ti) for xi, ti in zip(xinit, temperature)])
interpolations = [[CubicSpline(time_span, _sol[:, i]) for i in range(nx)] for _sol in solution]


def derivatives(xi, ti):

    def _derivatives(xi, ti):
        return ps.FiniteDifference()._differentiate(np.array(xi), np.array(ti))
    
    return jax.pure_callback(_derivatives, jax.ShapeDtypeStruct(xi.shape, xi.dtype), xi, ti)

estimated_derivatives = jax.vmap(lambda xi : derivatives(xi, time_span))(solution)


def data_matrix(p, features, T):
    _p = jax.vmap(rate_constant, in_axes = (0, None, None))(T, jnp.array(373.), p) # shape = (nexpt, F)
    return jnp.vstack(jax.vmap(lambda _p, feat : _p * feat)(_p, features)) # shape = (nexpt, T, F)

def poly2d(x, t):
    return jnp.array([
        x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], # 11
        
        x[0]**2, x[0]*x[1], x[0]*x[2], x[0]*x[3], x[0]*x[4], x[0]*x[5], x[0]*x[6], x[0]*x[7], x[0]*x[8], x[0]*x[9], x[0]*x[10], # 11
        x[1]**2, x[1]*x[2], x[1]*x[3], x[1]*x[4], x[1]*x[5], x[1]*x[6], x[1]*x[7], x[1]*x[8], x[1]*x[9], x[1]*x[10], # 10
        x[2]**2, x[2]*x[3], x[2]*x[4], x[2]*x[5], x[2]*x[6], x[2]*x[7], x[2]*x[8], x[2]*x[9], x[2]*x[10], # 9
        x[3]**2, x[3]*x[4], x[3]*x[5], x[3]*x[6], x[3]*x[7], x[3]*x[8], x[3]*x[9], x[3]*x[10], # 8
        x[4]**2, x[4]*x[5], x[4]*x[6], x[4]*x[7], x[4]*x[8], x[4]*x[9], x[4]*x[10], # 7
        x[5]**2, x[5]*x[6], x[5]*x[7], x[5]*x[8], x[5]*x[9], x[5]*x[10], # 6
        x[6]**2, x[6]*x[7], x[6]*x[8], x[6]*x[9], x[6]*x[10], # 5
        x[7]**2, x[7]*x[8], x[7]*x[9], x[7]*x[10], # 4
        x[8]**2, x[8]*x[9], x[8]*x[10], # 3
        x[9]**2, x[9]*x[10], # 2
        x[10]**2, # 1
    ])

def poly2d_interp(z, t, x):
    return jnp.array([
        x[0](t), x[1](t), x[2](t), x[3](t), x[4](t), x[5](t), x[6](t), x[7](t), x[8](t), x[9](t), x[10](t), 
        
        x[0](t)**2, x[0](t)*x[1](t), x[0](t)*x[2](t), x[0](t)*x[3](t), x[0](t)*x[4](t), x[0](t)*x[5](t), x[0](t)*x[6](t), x[0](t)*x[7](t), x[0](t)*x[8](t), x[0](t)*x[9](t), x[0](t)*x[10](t), 
        x[1](t)**2, x[1](t)*x[2](t), x[1](t)*x[3](t), x[1](t)*x[4](t), x[1](t)*x[5](t), x[1](t)*x[6](t), x[1](t)*x[7](t), x[1](t)*x[8](t), x[1](t)*x[9](t), x[1](t)*x[10](t), 
        x[2](t)**2, x[2](t)*x[3](t), x[2](t)*x[4](t), x[2](t)*x[5](t), x[2](t)*x[6](t), x[2](t)*x[7](t), x[2](t)*x[8](t), x[2](t)*x[9](t), x[2](t)*x[10](t), 
        x[3](t)**2, x[3](t)*x[4](t), x[3](t)*x[5](t), x[3](t)*x[6](t), x[3](t)*x[7](t), x[3](t)*x[8](t), x[3](t)*x[9](t), x[3](t)*x[10](t), 
        x[4](t)**2, x[4](t)*x[5](t), x[4](t)*x[6](t), x[4](t)*x[7](t), x[4](t)*x[8](t), x[4](t)*x[9](t), x[4](t)*x[10](t), 
        x[5](t)**2, x[5](t)*x[6](t), x[5](t)*x[7](t), x[5](t)*x[8](t), x[5](t)*x[9](t), x[5](t)*x[10](t), 
        x[6](t)**2, x[6](t)*x[7](t), x[6](t)*x[8](t), x[6](t)*x[9](t), x[6](t)*x[10](t), 
        x[7](t)**2, x[7](t)*x[8](t), x[7](t)*x[9](t), x[7](t)*x[10](t), 
        x[8](t)**2, x[8](t)*x[9](t), x[8](t)*x[10](t),
        x[9](t)**2, x[9](t)*x[10](t),
        x[10](t)**2, 
    ])

def get_small_ind(include, n, nx):
    # include : include columns 
    # n : upto order n
    # nx : dimensions of states
    ind = jnp.zeros(nx, dtype = int)
    ind = ind.at[include].set(1)

    sol = jnp.concatenate([jnp.stack([*map(
        lambda z : tree_util.tree_reduce(operator.mul, z),
        itertools.combinations_with_replacement(ind, i)
        )]) for i in range(1, n + 1)])
    
    return jnp.where(sol == 0, jnp.arange(len(sol)), len(sol))

F = len(poly2d(xinit[0], 0.))

param_labels = [[[fr"$k^{{{j}}}_{{{i}}}$" for i in range(F)], [fr"$E^{{{j}}}_{{{i}}}$" for i in range(F)]] for j in range(nr) ]

param_actual = [ # reaction rate constant, activation energy
    [
        jnp.zeros(F).at[jnp.array([43, 55])].set([0.3, - 1.2]), 
        jnp.zeros(F).at[jnp.array([43, 55])].set([9.48, 8.58])
    ],
    [
        jnp.zeros(F).at[jnp.array([25, 44])].set([0.4, - 0.6]), 
        jnp.zeros(F).at[jnp.array([25, 44])].set([6.59, 2.77])
    ],
    [
        jnp.zeros(F).at[jnp.array([17, 34])].set([1.1, - 0.7]), 
        jnp.zeros(F).at[jnp.array([17, 34])].set([10.9, 4.05])
    ],
    [
        jnp.zeros(F).at[jnp.array([24, 52])].set([0.9, - 0.1]), 
        jnp.zeros(F).at[jnp.array([24, 52])].set([4.88, 5.59])
    ],
    [
        jnp.zeros(F).at[jnp.array([15, 46])].set([1., - 0.5]), 
        jnp.zeros(F).at[jnp.array([15, 46])].set([7.74, 1.14])
    ],
    [
        jnp.zeros(F).at[jnp.array([73, 47])].set([0.2, - 0.8]), 
        jnp.zeros(F).at[jnp.array([73, 47])].set([10.9, 3.46])
    ]
]


small_ind = jnp.array([
    get_small_ind(jnp.array([3, 4, 5, 10]), 2, nx),
    get_small_ind(jnp.array([1, 3, 4, 6]), 2, nx),
    get_small_ind(jnp.array([0, 2, 4, 6]), 2, nx),
    get_small_ind(jnp.array([1, 3, 4, 7]), 2, nx),
    get_small_ind(jnp.array([0, 3, 4, 8]), 2, nx),
    get_small_ind(jnp.array([3, 8, 9, 10]), 2, nx),
]).reshape(nr, -1)

p_guess = jnp.zeros(shape = (nr, F))

# DF-SINDY features and target
dfsindy_features = jnp.stack([scipy_odeint(poly2d_interp, poly2d(_xi, 0), time_span, args = (_interp, )) - poly2d(_xi, 0) for _xi, _interp in zip(xinit, interpolations)]) # shape = (nexpt, T, F)
dfsindy_target = jnp.vstack(jax.vmap(lambda z : z - z[0])(solution)).T # shape = (nx, nexpt * T)


def g(p, x):
    # The equality constraints can itself be nonlinear but given p, they should be linear with respect to x.
    # Can only add constraints with respect to p and x for the terms that are present in the original model
    # If constraints for all temperatures are specified then jacobian of constraints will not be full rank
    x = x * rate_constant(temperature[0], 373., p)
    return jnp.array([
    ])

def h(p, x):
    # h >= 0 constraints
    return jnp.array([
    ])

def _foo(x, t, p):
    T, p, theta = p
    _theta = theta * rate_constant(T, jnp.array(373.), p)
    return stoic @ (_theta @ poly2d(x, t))

def f(p, x, small_ind, features, target, regularization):
    # The optimization problem is separable with respect to x and p of different reactions

    # update small index columns
    A_stack = jax.vmap(lambda _p, _si : data_matrix(_p, features, temperature).at[:, _si].set(0))(p, small_ind) # shape = (nr, nexpt * T, F)
    pred = stoic @ jnp.einsum("ijk,ik->ij", A_stack, x) # shape (nx, nexpt *T)
    mse = jnp.mean((target - pred)**2)

    big_x = jax.vmap(lambda _x, _si : _x.at[_si].set(0))(x, small_ind)
    return mse + regularization * jnp.mean(big_x**2)


###############################################################################################################################################
# Comparing with BiLevel NLP : DFSINDy inner + DFSINDy outer

def simple_objective_sindy(f, g, p, states, small_ind):
    (x_opt, v_opt), _ = differentiable_optimization(f, g, p, jnp.zeros_like(p), (small_ind, ))
    _loss = f(p, x_opt, small_ind) + v_opt @ g(p, x_opt)
    return _loss, x_opt

def outer_objective_sindy(p_guess, solution, features, target, small_ind, reg, thresholding = 0.1, maxiter = 10):
    
    # implement sequential threshold least square algorithm
    # TODO previous subproblems can have lower tolerance or less maximum iterations to eliminate the terms. How to choose initial tolerance ?
    sindy_logger = logging.getLogger("BiLevelShootingInterp")
    sindy_logger.setLevel(logging.INFO)
    logfile = logging.FileHandler(os.path.join(_dir, "record_bilevelshootinginterp.txt"))
    sindy_logger.addHandler(logfile)
    _output_file = os.path.join(_dir, "ipopt_bilevelshootinginterp_output.txt")

    _, F = small_ind.shape
    _ind = jnp.arange(F)
    _f = partial(f, features = features, target = target, regularization = reg)
    _g = g
    
    iteration = 0
    p_guess, unravel = flatten_util.ravel_pytree(p_guess)

    _simple_obj = jax.jit(lambda p, si : simple_objective_sindy(_f, _g, unravel(p), solution, si)[0])
    _simple_jac = jax.jit(jax.grad(_simple_obj, argnums = 0))
    _simple_hess = jax.jit(jax.jacfwd(_simple_jac))

    def _simple_obj_error(p, si):
        try :
            sol = _simple_obj(p, si)
        except : 
            sol = jnp.inf
        return sol

    while iteration < maxiter : 
        
        sindy_logger.info(f"{divider} \nIteration {iteration}, \nsmall index {small_ind}")

        if iteration > 0 and jnp.allclose(prev_small_ind, small_ind) : 
            sindy_logger.info("Optimal solution found")
            break

        solution_object = minimize_ipopt(
            partial(_simple_obj_error, si = small_ind), 
            x0 = p_guess, # restart from intial guess 
            jac = partial(_simple_jac, si = small_ind),
            hess = partial(_simple_hess, si = small_ind),  
            # bounds = list(zip(jnp.zeros_like(p_guess), jnp.inf * jnp.ones_like(p_guess))), 
            tol = pargs.tol, 
            options = {"maxiter" : pargs.iters, "output_file" : _output_file, "disp" : 0, "file_print_level" : 5, "mu_strategy" : "adaptive"}
            )
        
        sindy_logger.info(f"{divider}")
        with open(_output_file, "r") as file:
            for line in file : sindy_logger.info(line.strip())

        os.remove(_output_file)

        p = jnp.array(solution_object.x)
        loss, x = simple_objective_sindy(_f, _g, unravel(p), solution, small_ind)
        sindy_logger.info(f"{divider} \nIteration {iteration} : loss {loss}")
        sindy_logger.info(f"{divider} \nNonlinear parameters : {unravel(p)}")
        sindy_logger.info(f"{divider} \nLinear parameters : {x}")

        prev_small_ind = small_ind
        small_ind = jnp.minimum(
            jax.vmap(lambda _x : jnp.where(jnp.abs(_x) < thresholding, _ind, F))(x), # new indices
            prev_small_ind # previous indices
        )
        iteration += 1

    p_opt = unravel(p) * jnp.where(small_ind == F, 1., 0.) # set small index coefficients to zero
    return p_opt, x

if pargs.method == 0 : 
    p, x = outer_objective_sindy(p_guess, solution, dfsindy_features, dfsindy_target, small_ind, pargs.reg, pargs.threshold, pargs.siters)
    for i in range(nr) : 
        plot_coefficients(param_actual[i], [x[i], p[i]], param_labels[i], f"BiLevelShootingInterpCoeff{i + 1}", _dir)
    prediction = odeint_diffrax(_foo, xinit[0], time_span, (temperature[0], p, x), atol = pargs.atol, rtol = pargs.rtol, mxstep = pargs.mxstep)
    plot_trajectories(solution[0], prediction, time_span, 4, "BiLevelShootingInterpStates", _dir)


###############################################################################################################################################
# Comparing with Full NLP (DFSINDy)


###############################################################################################################################################
# Comparing with Full NLP shooting / sequential optimization

def simple_objective_nlp(f, p, states, small_ind):
    _p, x, = p
    _x = jax.vmap(lambda z, si : z.at[si].set(0))(x, small_ind)
    solution = jax.vmap(lambda xi, ti : odeint_diffrax(lambda z, t, p : _foo(z, t, p), xi, time_span, (ti, _p, _x), atol = pargs.atol, rtol = pargs.rtol, mxstep = pargs.mxstep))(xinit, temperature)
    _loss = jnp.mean((solution - states)**2)

    return _loss

def outer_objective_nlp(px_guess, solution, features, target, small_ind, reg, thresholding = 0.1, maxiter = 10):
    
    nlp_logger = logging.getLogger("Shooting")
    nlp_logger.setLevel(logging.INFO)
    logfile = logging.FileHandler(os.path.join(_dir, "record_shooting.txt"))
    nlp_logger.addHandler(logfile)
    _output_file = os.path.join(_dir, "ipopt_shooting_output.txt")

    _, F = small_ind.shape
    _ind = jnp.arange(F)
    px_guess, unravel = flatten_util.ravel_pytree(px_guess)

    _f = lambda p, si : f(*unravel(p), si, features = features, target = target, regularization = reg)
    _g = lambda p : g(*unravel(p))
    _h = lambda p : h(*unravel(p))
    
    iteration = 0

    # JIT compiled objective function
    _simple_obj = jax.jit(lambda p, si : simple_objective_nlp(_f, unravel(p), solution, si))
    _simple_jac = jax.jit(jax.grad(_simple_obj, argnums = 0))
    _simple_hess = jax.jit(jax.jacfwd(_simple_jac, argnums = 0))

    def _simple_obj_error(p, si):
        try :
            sol = _simple_obj(p, si)
        except : 
            sol = jnp.inf
        return sol

    # JIT compiled constraints
    _g_jac = jax.jit(jax.jacobian(_g))
    _g_hess = jax.jit(lambda p, v : jax.jacfwd(lambda _p : v @ _g_jac(_p))(p))
    _h_jac = jax.jit(jax.jacfwd(_h))
    _h_hess = jax.jit(lambda p, v : jax.jacfwd(lambda _p : v @ _h_jac(_p))(p))

    while iteration < maxiter : 
        
        nlp_logger.info(f"{divider} \nIteration {iteration}, \nsmall index {small_ind}")

        if iteration > 0 and jnp.allclose(prev_small_ind, small_ind) : 
            nlp_logger.info("Optimal solution found")
            break

        solution_object = minimize_ipopt(
            partial(_simple_obj_error, si = small_ind), 
            x0 = px_guess, # restart from intial guess 
            jac = partial(_simple_jac, si = small_ind),
            hess = partial(_simple_hess, si = small_ind),  
            constraints = [
                {"type" : "eq", "fun" : _g, "jac" : _g_jac, "hess" : _g_hess}, 
                {"type" : "ineq", "fun" : _h, "jac" : _h_jac, "hess" : _h_hess}
                ],
            tol = pargs.tol, 
            options = {"maxiter" : pargs.iters, "output_file" : _output_file, "disp" : 0, "file_print_level" : 5, "mu_strategy" : "adaptive"}
            )
        
        nlp_logger.info(f"{divider}")
        with open(_output_file, "r") as file:
            for line in file : nlp_logger.info(line.strip())

        os.remove(_output_file)

        p, x = unravel(jnp.array(solution_object.x))
        nlp_logger.info(f"{divider} \nIteration {iteration} : loss {solution_object.fun}")
        nlp_logger.info(f"{divider} \nNonlinear parameters : {p}")
        nlp_logger.info(f"{divider} \nLinear parameters : {x}")

        prev_small_ind = small_ind
        small_ind = jnp.minimum(
            jax.vmap(lambda _x : jnp.where(jnp.abs(_x) < thresholding, _ind, F))(x), # new indices
            prev_small_ind # previous indices
        )
        iteration += 1

    return p, x

if pargs.method == 2 : 
    p, x = outer_objective_nlp((p_guess, jnp.zeros_like(p_guess)), solution, dfsindy_features, dfsindy_target, small_ind, pargs.reg, pargs.threshold)
    for i in range(nr) : 
        plot_coefficients(param_actual[i], [x[i], p[i]], param_labels[i], f"ShootingCoeff{i + 1}", _dir)
    prediction = odeint_diffrax(_foo, xinit[0], time_span, (temperature[0], p, x), atol = pargs.atol, rtol = pargs.rtol, mxstep = pargs.mxstep)
    plot_trajectories(solution[0], prediction, time_span, 4, "ShootingStates", _dir)

