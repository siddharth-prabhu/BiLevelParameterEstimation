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

from utils import differentiable_regression, odeint_diffrax


# choose hyperparameters
parser = argparse.ArgumentParser("ParameterEstimationCalciumIon")
parser.add_argument("--iters", type = int, default = 100, help = "The maximum number of iterations to be performed in parameter estimation")
parser.add_argument("--tol", type = int, default = 1e-4, help = "Ipopt tolerance")
parser.add_argument("--atol", type = float, default = 1e-8, help = "Absolute tolerance of ode solver")
parser.add_argument("--rtol", type = float, default = 1e-6, help = "Relative tolerance of ode solver")
parser.add_argument("--mxstep", type = int, default = 10_000, help = "The maximum number of steps a ode solver takes")
parser.add_argument("--reg", type = float, default = 0.1, help = "L2 regularization penalty")
parser.add_argument("--threshold", type = float, default = 0.1, help = "Thresholding parameter")
parser.add_argument("--msg", type = str, default = "", help = "Sample msg that briefly describes this problem")

parser.add_argument("--id", type = str, default = "", help = "Slurm job id")
parser.add_argument("--partition", type = str, default = "", help = "The partition this job is assigned to")
parser.add_argument("--cpus", type = int, default = 1, help = "Maximum number of cpus availabe per node")

pargs = parser.parse_args()

_dir = os.path.join("log", str(datetime.now()))
divider = "--"*50 # printing separater
if not os.path.exists(_dir) : os.makedirs(_dir)

logger = logging.getLogger("__name__")
logger.setLevel(logging.INFO)
logfile = logging.FileHandler(os.path.join(_dir, "record.txt"))
logger.addHandler(logfile)
logger.info("PARAMETER ESTIMATION CALCIUM ION")
logger.info(pformat(pargs.__dict__))


# Original system
def rate_constant(T, Tref, act) : return jnp.exp(- act * (10**4/T - 10**4/Tref) / 8.314)

stoic = jnp.array([
        -1, 0, -1, 1, 1, -1, 0, 1
    ]).reshape(-1, 2)

activation = jnp.array([4, 7.5, 7.5]) # values divided by 1e4

def kinetic(x, t, p, T) -> jnp.ndarray:
    # species = [A, B, C, D] 
    # reactions = [
    #       ]

    k1, k2, k3 = p * rate_constant(T, 373., activation)
    
    reactions = jnp.array([
        k1 * x[0] * x[1] - k2 * x[2],
        k3 * x[2]
    ])
    
    return stoic @ reactions 


# Generate data
nx = 4
nexpt = 2
key = jrandom.PRNGKey(20)
key_temp, key_xinit = jrandom.split(key, 2)
temperature = jrandom.uniform(key_temp, shape = (nexpt, 1), minval = 363., maxval = 383.)
xinit = jrandom.uniform(key_xinit, shape = (nexpt, nx), minval = 4., maxval = 10.)
time_span = jnp.arange(1, 5., 0.01)
p_actual = jnp.array([.3, .4, 1.1])
solution = jnp.stack([odeint(kinetic, xi, time_span, p_actual, ti) for xi, ti in zip(xinit, temperature)])
actual_derivatives = jax.vmap(lambda xi, ti : jax.vmap(lambda _xi : kinetic(_xi, time_span[0], p_actual, T = ti))(xi))(solution, temperature)
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
        x[0], x[1], x[2], x[3],
        
        x[0]**2, x[0]*x[1], x[0]*x[2], x[0]*x[3], 
        x[1]**2, x[1]*x[2], x[1]*x[3], 
        x[2]**2, x[2]*x[3], 
        x[3]**2,
    ])

def poly2d_interp(z, t, x):
    return jnp.array([
        x[0](t), x[1](t), x[2](t), x[3](t),
        
        x[0](t)**2, x[0](t)*x[1](t), x[0](t)*x[2](t), x[0](t)*x[3](t), 
        x[1](t)**2, x[1](t)*x[2](t), x[1](t)*x[3](t),
        x[2](t)**2, x[2](t)*x[3](t),
        x[3](t)**2, 
    ])

def get_small_ind(include, n, nx):
    ind = jnp.zeros(nx, dtype = int)
    ind = ind.at[include].set(1)

    sol = jnp.concatenate([jnp.stack([*map(
        lambda z : tree_util.tree_reduce(operator.mul, z),
        itertools.combinations_with_replacement(ind, i)
        )]) for i in range(1, n + 1)])
    
    return jnp.where(sol == 0, jnp.arange(len(sol)), len(sol))


F = len(poly2d(xinit[0], 0.))

small_ind = jnp.array([
    get_small_ind(jnp.array([0, 1, 2]), 2, nx),
    get_small_ind(jnp.array([0, 1, 2]), 2, nx),
    get_small_ind(jnp.array([0, 1, 2]), 2, nx),
    get_small_ind(jnp.array([2]), 2, nx),
]).reshape(nx, -1)

p_guess = jnp.ones(shape = (nx, F))

# SINDy features
sindy_features = jax.vmap(jax.vmap(lambda xi : poly2d(xi, 0.)))(solution)
sindy_target = jnp.vstack(solution).T

# DF-SINDY features and target
dfsindy_features = jnp.stack([scipy_odeint(poly2d_interp, poly2d(_xi, 0), time_span, args = (_interp, )) - poly2d(_xi, 0) for _xi, _interp in zip(xinit, interpolations)])
dfsindy_target = jnp.vstack(jax.vmap(lambda z : z - z[0])(solution)).T


def g(p, x):
    # The equality constraints can itself be nonlinear but given p, they should be linear with respect to x.
    # Can only add constraints with respect to p and x for the terms that are present in the original model
    # If constraints for all temperatures are specified then jacobian of constraints will not be full rank
    # _x = x * rate_constant(T, 373., p)
    return jnp.array([
        x[1, 2] + x[2, 2],
        x[1, 2] - x[0, 2] - x[3, 2],
        
        x[0, 5] - x[1, 5],
        x[1, 5] + x[2, 5],
    ])

def _foo(x, t, p):
    T, p, theta = p
    _theta = theta * rate_constant(T, jnp.array(373.), p)
    return _theta @ poly2d(x, t)

def f(p, x, small_ind, features, target, regularization):
    # The optimization problem is separable with respect to x and p of different reactions
    # TODO remove mass matrix and add constraints instead

    # update small index columns
    A_stack = jax.vmap(lambda _p, _si : data_matrix(_p, features, temperature).at[:, _si].set(0))(p, small_ind) # shape = (nx, nexpt * T, F)
    pred = jnp.einsum("ijk,ik->ij", A_stack, x) # shape (nx, nexpt *T)
    mse = jnp.mean((target - pred)**2)

    big_x = jax.vmap(lambda _x, _si : _x.at[_si].set(0))(x, small_ind)

    return mse + regularization * jnp.mean(big_x**2)


###############################################################################################################################################
# Comparing with DFSINDy inner + shooting outer

def simple_objective(f, g, p, states, small_ind):
    # states shape = (nexpt, T, nx) # no of experiments, time points, states. Target values for outer optimization
    # features shape = (nexpt, T, F) # no of experiments, time points, no of features. Features for inner optimization 
    # target shape = (nexpt, T, nx) no of experiments, time points, states. Target values for inner optimization
    # p shape = (nx, F) # no of nonlinear decision variables
    # small_ind shape = (nx, F)

    # TODO outer optimization problem can be a DF-SINDy problem or a sequential/simultaneous optimization problem
    (x, _), _ = differentiable_regression(f, g, p, jnp.zeros_like(p), (small_ind, ))
    solution = jax.vmap(lambda xi, ti : odeint_diffrax(_foo, xi, time_span, (ti, p, x), atol = pargs.atol, rtol = pargs.rtol, mxstep = pargs.mxstep))(xinit, temperature)
    _loss = jnp.mean((solution - states)**2)
    return _loss, x

def outer_objective(p_guess, solution, features, target, small_ind, reg, thresholding = 0.1, maxiter = 10):
    
    # implement sequential threshold least square algorithm
    # TODO previous subproblems can have lower tolerance or less maximum iterations to eliminate the terms. How to choose initial tolerance ?
    shooting_logger = logging.getLogger("shooting")
    shooting_logger.setLevel(logging.INFO)
    logfile = logging.FileHandler(os.path.join(_dir, "record_shooting.txt"))
    shooting_logger.addHandler(logfile)
    _output_file = os.path.join(_dir, "ipopt_shooting_output.txt")

    _, F = small_ind.shape
    _ind = jnp.arange(F)
    _f = partial(f, features = features, target = target, regularization = reg)
    _g = g
    
    iteration = 0
    p_guess, unravel = flatten_util.ravel_pytree(p_guess)

    _simple_obj = jax.jit(lambda p, si : simple_objective(_f, _g, unravel(p), solution, si)[0])
    _simple_jac = jax.jit(jax.grad(_simple_obj, argnums = 0))
    _simple_hess = jax.jit(jax.jacrev(_simple_jac, argnums = 0))

    while iteration < maxiter : 
        
        shooting_logger.info(f"{divider} \nIteration {iteration}, \nsmall index {small_ind}")

        if iteration > 0 and jnp.allclose(prev_small_ind, small_ind) : 
            shooting_logger.info("Optimal solution found")
            break

        solution_object = minimize_ipopt(
            partial(_simple_obj, si = small_ind), 
            x0 = p_guess, # restart from intial guess 
            jac = partial(_simple_jac, si = small_ind),
            hess = partial(_simple_hess, si = small_ind),  
            tol = pargs.tol, 
            options = {"maxiter" : pargs.iters, "output_file" : _output_file, "disp" : 0, "file_print_level" : 5}
            )
        
        shooting_logger.info(f"{divider}")
        with open(_output_file, "r") as file:
            for line in file : shooting_logger.info(line.strip())

        os.remove(_output_file)

        p = jnp.array(solution_object.x)
        loss, x = simple_objective(_f, _g, unravel(p), solution, small_ind)
        shooting_logger.info(f"{divider} \nIteration {iteration} : loss {loss}")
        shooting_logger.info(f"{divider} \nNonlinear parameters : {unravel(p)}")
        shooting_logger.info(f"{divider} \nLinear parameters : {x}")

        prev_small_ind = small_ind
        small_ind = jnp.minimum(
            jax.vmap(lambda _x : jnp.where(jnp.abs(_x) < thresholding, _ind, F))(x), # new indices
            prev_small_ind # previous indices
        )
        iteration += 1

    return solution_object, x

# solution_object, xopt = outer_objective(p_guess, solution, dfsindy_features, dfsindy_target, small_ind, pargs.reg, pargs.threshold)


###############################################################################################################################################
# Comparing with DFSINDy inner + DFSINDy outer

def simple_objective_sindy(f, g, p, states, small_ind):
    # states shape = (nexpt, T, nx) # no of experiments, time points, states. Target values for outer optimization
    # features shape = (nexpt, T, F) # no of experiments, time points, no of features. Features for inner optimization 
    # target shape = (nexpt, T, nx) no of experiments, time points, states. Target values for inner optimization
    # p shape = (nx * F, ) # no of nonlinear decision variables
    # small_ind shape = (nx, F)

    (x, _), _ = differentiable_regression(f, g, p, jnp.zeros_like(p), (small_ind, ))
    _loss = f(p, x, small_ind)
    return _loss, x

def outer_objective_sindy(p_guess, solution, features, target, small_ind, reg, thresholding = 0.1, maxiter = 10):
    
    # implement sequential threshold least square algorithm
    # TODO previous subproblems can have lower tolerance or less maximum iterations to eliminate the terms. How to choose initial tolerance ?
    sindy_logger = logging.getLogger("sindy")
    sindy_logger.setLevel(logging.INFO)
    logfile = logging.FileHandler(os.path.join(_dir, "record_sindy.txt"))
    sindy_logger.addHandler(logfile)
    _output_file = os.path.join(_dir, "ipopt_sindy_output.txt")

    _, F = small_ind.shape
    _ind = jnp.arange(F)
    _f = partial(f, features = features, target = target, regularization = reg)
    _g = g
    
    iteration = 0
    p_guess, unravel = flatten_util.ravel_pytree(p_guess)

    _simple_obj = jax.jit(lambda p, si : simple_objective_sindy(_f, _g, unravel(p), solution, si)[0])
    _simple_jac = jax.jit(jax.grad(_simple_obj, argnums = 0))
    _simple_hess = jax.jit(jax.jacrev(_simple_jac, argnums = 0))

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
            tol = pargs.tol, 
            options = {"maxiter" : pargs.iters, "output_file" : _output_file, "disp" : 0, "file_print_level" : 5}
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

    return solution_object, x

solution_object, xopt = outer_objective_sindy(p_guess, solution, dfsindy_features, dfsindy_target, small_ind, pargs.reg, pargs.threshold)


###############################################################################################################################################
# Comparing with full NLP 

def simple_objective_nlp(f, p, states, small_ind):
    # states shape = (nexpt, T, nx) # no of experiments, time points, states. Target values for outer optimization
    # features shape = (nexpt, T, F) # no of experiments, time points, no of features. Features for inner optimization 
    # target shape = (nexpt, T, nx) no of experiments, time points, states. Target values for inner optimization
    # p shape = (nx * F, ) # no of nonlinear decision variables
    # small_ind shape = (nx, F)

    # TODO outer optimization problem can be a DF-SINDy problem or a sequential/simultaneous optimization problem
    # _loss = f(p, small_ind) # dfsindy based loss function
    
    _p, x, = p
    _x = jax.vmap(lambda z, si : z.at[si].set(0))(x, small_ind)
    solution = jax.vmap(lambda xi, ti : odeint_diffrax(lambda z, t, p : _foo(z, t, p), xi, time_span, (ti, _p, _x), atol = pargs.atol, rtol = pargs.rtol, mxstep = pargs.mxstep))(xinit, temperature)
    _loss = jnp.mean((solution - states)**2)

    return _loss

def outer_objective_nlp(px_guess, solution, features, target, small_ind, reg, thresholding = 0.1, maxiter = 10):
    
    nlp_logger = logging.getLogger("nlp")
    nlp_logger.setLevel(logging.INFO)
    logfile = logging.FileHandler(os.path.join(_dir, "record_nlp.txt"))
    nlp_logger.addHandler(logfile)
    _output_file = os.path.join(_dir, "ipopt_nlp_output.txt")

    _, F = small_ind.shape
    _ind = jnp.arange(F)
    px_guess, unravel = flatten_util.ravel_pytree(px_guess)

    _f = lambda p, si : f(*unravel(p), si, features = features, target = target, regularization = reg)
    _g = lambda p : g(*unravel(p))
    
    iteration = 0

    # JIT compiled objective function
    _simple_obj = jax.jit(lambda p, si : simple_objective_nlp(_f, unravel(p), solution, si))
    _simple_jac = jax.jit(jax.grad(_simple_obj, argnums = 0))
    _simple_hess = jax.jit(jax.jacrev(_simple_jac, argnums = 0))

    def _simple_obj_error(p, si):
        try :
            sol = _simple_obj(p, si)
        except : 
            sol = jnp.inf
        return sol

    # JIT compiled constraints
    _g_jac = jax.jit(jax.jacobian(_g))
    _g_hess = jax.jit(lambda p, v : jax.hessian(lambda _p : v @ _g(_p))(p))

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
            constraints = {"type" : "eq", "fun" : _g, "jac" : _g_jac, "hess" : _g_hess},
            tol = pargs.tol, 
            options = {"maxiter" : pargs.iters, "output_file" : _output_file, "disp" : 0, "file_print_level" : 5}
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

    return solution_object, x

solution_object, xopt = outer_objective_nlp((p_guess, jnp.zeros_like(p_guess)), solution, dfsindy_features, dfsindy_target, small_ind, pargs.reg, pargs.threshold)
