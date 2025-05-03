import os
import logging
from datetime import datetime
from pprint import pformat
import argparse

import jax
import jax.numpy as jnp
import jax.random as jrandom
jax.config.update("jax_enable_x64", True)
from jax import flatten_util, tree_util
import matplotlib.pyplot as plt
from cyipopt import minimize_ipopt
from scipy.interpolate import CubicSpline
from ddeint import ddeint

from utils import differentiable_optimization, odeint_diffrax, plot_coefficients, plot_trajectories

# Choose Hyperparameters
parser = argparse.ArgumentParser("ParameterEstimationKermackMcKendrick")
parser.add_argument("--iters", type = int, default = 1000, help = "The maximum number of iterations to be performed in parameter estimation")
parser.add_argument("--tol", type = float, default = 1e-4, help = "Ipopt tolerance")
parser.add_argument("--atol", type = float, default = 1e-8, help = "Absolute tolerance of ode solver")
parser.add_argument("--rtol", type = float, default = 1e-6, help = "Relative tolerance of ode solver")
parser.add_argument("--mxstep", type = int, default = 10_000, help = "The maximum number of steps a ode solver takes")
parser.add_argument("--msg", type = str, default = "", help = "Sample msg that briefly describes this problem")
parser.add_argument("--method", type = int, default = 0, help = "Formulation type 0 : BiLevelOpt (DFSINDy innter + DFSINDy outer), 1 : FullNLP (DFSINDy)")

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
logger.info("PARAMETER ESTIMATION (DDE) KERMACK-MCKENDRICK")
logger.info(pformat(pargs.__dict__))


def km(_x, t, p) -> jnp.ndarray :
    # https://utoronto.scholaris.ca/server/api/core/bitstreams/f557ed43-1f11-4949-854d-feb192355263/content

    (k1, k2, k3, k4, k5, k6), (tau1, tau2) = p 

    x = _x(t)
    x1 = _x(t - tau1)
    x2 = _x(t - tau2)

    return jnp.array([
        - k1 * x[0] * x1[1] + k2 * x2[1],
        k3 * x[0] * x1[1] - k4 * x[1],
        k5 * x[1] - k6 * x2[1]
    ])

# Generate data
nx = 3
key = jrandom.PRNGKey(20)
param_labels = [[r"$k_1$", r"$k_2$", r"$k_3$", r"$k_4$", r"$k_5$", r"$k_6$"], [r"$\tau _1$", r"$\tau _2$"]]

xinit = jnp.array([5, 0.1, 1])
time_span = jnp.arange(0, 60, 0.1)
p_actual = ((0.1, 0.2, 0.3, 0.4, 0.5, 0.6), (1, 10))
solution = ddeint(km, lambda t : xinit, time_span, fargs = (p_actual, ))


class Interpolation():

    def __init__(self, solution, time_span):
        self.time_span = time_span
        self.interpolations = [CubicSpline(time_span, sol) for sol in solution.T]

    def __call__(self, t):
        return jnp.stack([interp(jnp.clip(t, min = self.time_span[0], max = None)) for interp in self.interpolations])

class InterpolationDerivative():

    def __init__(self, interpolation : Interpolation, order : int = 1):
        self.derivatives = [interp.derivative(order) for interp in interpolation.interpolations]
        self.time_span = interpolation.time_span

    def __call__(self, t):
        derivative = jnp.stack([interp(t) for interp in self.derivatives])
        return jnp.where(t < time_span[0], jnp.zeros_like(derivative[0]), derivative)
    

interpolations = Interpolation(solution, time_span)
interpolations_derivative = lambda t, order : InterpolationDerivative(interpolations, order)(t)

# Plotting states and interpolations 
solution_interp = interpolations(time_span).T
with plt.style.context(["science", "notebook", "bright"]) :
    fig, ax = plt.subplots(1, 3, figsize = (20, 10))
    
    for i, _ax in enumerate(ax.ravel()):
        _ax.plot(time_span, solution[:, i], label = "Solution")
        _ax.plot(time_span, solution_interp[:, i], label = "Interpolation")
        _ax.legend()
    
    plt.savefig(os.path.join(_dir, "states.png"))
    plt.close()

# https://github.com/jax-ml/jax/discussions/13282
@jax.custom_jvp
def _interp(t):
    return jax.pure_callback(interpolations, jax.ShapeDtypeStruct((len(xinit), *t.shape), xinit.dtype), t)

@jax.custom_jvp
def _interp_der(t):
    return jax.pure_callback(interpolations_derivative, jax.ShapeDtypeStruct((len(xinit), *t.shape), xinit.dtype), t, 1)

@_interp.defjvp
def _interp_fwd(primals, tangents):
    
    dt, = tangents
    output = _interp(*primals)
    output_tangent = _interp_der(*primals) * dt
    return output, output_tangent

@_interp_der.defjvp
def _interp_der_fwd(primals, tangents):

    t, = primals
    dt, = tangents 
    output = _interp_der(t)
    output_tangent = jax.pure_callback(interpolations_derivative, jax.ShapeDtypeStruct((len(xinit), *t.shape), xinit.dtype), t, 2) * dt
    return output, output_tangent

def _foo_interp(z, t, px):

    (tau1, tau2), (k1, k2, k3, k4, k5, k6) = px 
    z = _interp(t)
    z1 = _interp(t - tau1)
    z2 = _interp(t - tau2)

    return jnp.array([
        - k1 * z[0] * z1[1] + k2 * z2[1],
        k3 * z[0] * z1[1] - k4 * z[1],
        k5 * z[1] - k6 * z2[1]
    ])

p_guess, x_guess = jnp.ones(2), jnp.ones(6)
px_guess, unravel = flatten_util.ravel_pytree((p_guess, x_guess))
dfsindy_target = solution - solution[0]

###############################################################################################################################################
# Comparing with BiLevel NLP : DFSINDy (shooting + interpolation) inner + DFSINDy (shooting + interpolation) outer 

def f(p, x, target):
    solution = odeint_diffrax(_foo_interp, xinit, time_span, (p, x), pargs.rtol, pargs.atol, pargs.mxstep) - xinit
    return jnp.mean((solution - target)**2)

def g(p, x) : return jnp.array([ ])

def simple_objective_shooting(f, g, p, states, target):
    # states shape = (nexpt, T, nx) # no of experiments, time points, states. Target values for outer optimization
    # target shape = (nexpt, T, nx) no of experiments, time points, states. Target values for inner optimization
    # p shape = (nx * F, ) # no of nonlinear decision variables
    (x, _), _ = differentiable_optimization(f, g, p, x_guess, (target, ))
    _loss = f(p, x, target)
    return _loss, x

def outer_objective_shooting(p_guess, solution, target):
    
    shooting_logger = logging.getLogger("BiLevelShootingInterp")
    shooting_logger.setLevel(logging.INFO)
    logfile = logging.FileHandler(os.path.join(_dir, "record_bilevelshootinginterp.txt"))
    shooting_logger.addHandler(logfile)
    _output_file = os.path.join(_dir, "ipopt_bilevelshootinginterp_output.txt")

    # JIT compiled objective function
    _simple_obj = jax.jit(lambda p : simple_objective_shooting(f, g, p, solution, target)[0])
    _simple_jac = jax.jit(jax.grad(_simple_obj))
    _simple_hess = jax.jit(jax.jacfwd(_simple_jac))

    def _simple_obj_error(p):
        try :
            sol = _simple_obj(p)
        except : 
            sol = jnp.inf
        
        return sol

    solution_object = minimize_ipopt(
        _simple_obj_error, 
        x0 = p_guess, # restart from intial guess 
        jac = _simple_jac,
        hess = _simple_hess,  
        tol = pargs.tol, 
        options = {"maxiter" : pargs.iters, "output_file" : _output_file, "disp" : 0, "file_print_level" : 5, "mu_strategy" : "adaptive"}
        )
        
    shooting_logger.info(f"{divider}")
    with open(_output_file, "r") as file:
        for line in file : shooting_logger.info(line.strip())

    os.remove(_output_file)

    p = jnp.array(solution_object.x)
    loss, x = simple_objective_shooting(f, g, p, solution, target)
    shooting_logger.info(f"{divider} \nLoss {loss}")
    shooting_logger.info(f"{divider} \nNonlinear parameters : {p}")
    shooting_logger.info(f"{divider} \nLinear parameters : {x}")

    return p.flatten(), x.flatten()

if pargs.method == 0 : 
    p, x = outer_objective_shooting(p_guess, solution, dfsindy_target)
    plot_coefficients([*p_actual], [x, p], param_labels, "BiLevelShootingInterpCoeff", _dir)
    prediction = ddeint(km, lambda t : xinit, time_span, fargs = ((x, p), ))
    plot_trajectories(solution, prediction, time_span, 3, "BiLevelShootingInterpStates", _dir)

###############################################################################################################################################
# Comparing with Full NLP : DFSINDy shooting + interpolation 

def simple_objective(px, target):
    solution = odeint_diffrax(_foo_interp, xinit, time_span, unravel(px), pargs.rtol, pargs.atol, pargs.mxstep) - xinit
    _loss = jnp.mean((solution - target)**2)
    return _loss

def outer_objective(px_guess, target):
    
    interp_logger = logging.getLogger("ShootingInterp")
    interp_logger.setLevel(logging.INFO)
    logfile = logging.FileHandler(os.path.join(_dir, "record_shootinginterp.txt"))
    interp_logger.addHandler(logfile)
    _output_file = os.path.join(_dir, "ipopt_shootinginterp_output.txt")

    # JIT compiled objective function
    _simple_obj = jax.jit(lambda p : simple_objective(p, target))
    _simple_jac = jax.jit(jax.grad(_simple_obj))
    _simple_hess = jax.jit(jax.jacfwd(_simple_jac))

    def _simple_obj_error(p):
        try :
            sol = _simple_obj(p)
        except : 
            sol = jnp.inf
        
        return sol

    solution_object = minimize_ipopt(
        _simple_obj_error, 
        x0 = px_guess, # restart from intial guess 
        jac = _simple_jac,
        hess = _simple_hess,  
        tol = pargs.tol, 
        options = {"maxiter" : pargs.iters, "output_file" : _output_file, "disp" : 0, "file_print_level" : 5, "mu_strategy" : "adaptive"}
        )
        
    interp_logger.info(f"{divider}")
    with open(_output_file, "r") as file:
        for line in file : interp_logger.info(line.strip())

    os.remove(_output_file)

    p, x = unravel(jnp.array(solution_object.x))
    interp_logger.info(f"{divider} \nLoss {solution_object.fun}")
    interp_logger.info(f"{divider} \nNonlinear parameters : {p}")
    interp_logger.info(f"{divider} \nLinear parameters : {x}")

    return p.flatten(), x.flatten()

if pargs.method == 1 : 
    p, x = outer_objective(px_guess, dfsindy_target)
    plot_coefficients([*p_actual], [x, p], param_labels, "ShootingInterpCoeff", _dir)
    prediction = ddeint(km, lambda t : xinit, time_span, fargs = ((x, p), ))
    plot_trajectories(solution, prediction, time_span, 3, "ShootingInterpStates", _dir)

###############################################################################################################################################
# Comparing with Full NLP : shooting / sequential optimization 
# Will need sensitivities across DDE to perform these calculations