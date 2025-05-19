import os
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
from cyipopt import minimize_ipopt
from scipy.interpolate import CubicSpline

from utils import differentiable_optimization, odeint_diffrax, plot_coefficients, plot_trajectories

# Choose Hyperparameters
parser = argparse.ArgumentParser("ParameterEstimationCalciumIon")
parser.add_argument("--iters", type = int, default = 1000, help = "The maximum number of iterations to be performed in parameter estimation")
parser.add_argument("--tol", type = float, default = 1e-4, help = "Ipopt tolerance")
parser.add_argument("--atol", type = float, default = 1e-8, help = "Absolute tolerance of ode solver")
parser.add_argument("--rtol", type = float, default = 1e-6, help = "Relative tolerance of ode solver")
parser.add_argument("--mxstep", type = int, default = 10_000, help = "The maximum number of steps a ode solver takes")
parser.add_argument("--msg", type = str, default = "", help = "Sample msg that briefly describes this problem")
parser.add_argument("--method", type = int, default = 0, help = "Formulation type 0 : BiLevelOpt (DFSINDy innter + DFSINDy outer), 1 : FullNLP (DFSINDy), 2 : FullNLP (shooting/sequential)")

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
logger.info("PARAMETER ESTIMATION (ODE) CALCIUM ION")
logger.info(pformat(pargs.__dict__))


def calcium_ion(x, t, p) -> jnp.ndarray :
    # https://utoronto.scholaris.ca/server/api/core/bitstreams/f557ed43-1f11-4949-854d-feb192355263/content

    (
        k1, k2, k3, k4, k5, k6, 
        k7, k8, k9, k10, k11, km1, 
        km2, km3, km4, km5, km6
    ) = p # default values (0.09, 2, 1.27, 3.73, 1.27, 32.24, 2, 0.05, 13.58, 153, 4.85, 0.19, 0.73, 29.09, 2.67, 0.16, 0.05)
    
    return jnp.array([
        k1 + k2 * x[0] - k3 * x[1] * x[0] / (x[0] + km1) - k4 * x[2] * x[0] / (x[0] + km2),
        k5 * x[0] - k6 * x[1] / (x[1] + km3),
        k7 * x[1] * x[2] * x[3] / (x[3] + km4) + k8 * x[1] + k9 * x[0] - k10 * x[2] / (x[2] + km5) - k11 * x[2] / (x[2] + km6),
        -k7 * x[1] * x[2] * x[3] / (x[3] + km4) + k11 * x[2] / (x[2] + km6)
    ])


# Generate data
nx = 4
key = jrandom.PRNGKey(20)
param_labels = [
    [r"$k_1$", r"$k_2$", r"$k_3$", r"$k_4$", r"$k_5$", r"$k_6$", r"$k_7$", r"$k_8$", r"$k_9$", r"$k_{10}$", r"$k_{11}$"], 
    [r"$Km _1$", r"$Km _2$", r"$Km _3$", r"$Km _4$", r"$Km _5$", r"$Km _6$"]
]

xinit = jnp.array([0.12, 0.31, 0.0058, 4.3])
time_span = jnp.arange(1, 20., 0.1)
p_actual = jnp.array([0.09, 2, 1.27, 3.73, 1.27, 32.24, 2, 0.05, 13.58, 153, 4.85, 0.19, 0.73, 29.09, 2.67, 0.16, 0.05])
solution = odeint(calcium_ion, xinit, time_span, p_actual)
actual_derivatives = jax.vmap(calcium_ion, in_axes = (0, 0, None))(solution, time_span, p_actual)


class Interpolation():

    def __init__(self, solution, time_span):
        self.interpolations = [CubicSpline(time_span, sol) for sol in solution.T]

    def __call__(self, t):
        return jnp.stack([interp(t) for interp in self.interpolations])

class InterpolationDerivative():

    def __init__(self, interpolation : Interpolation, order : int = 1):
        self.derivatives = [interp.derivative(order) for interp in interpolation.interpolations]

    def __call__(self, t):
        return jnp.stack([interp(t) for interp in self.derivatives])


interpolations = Interpolation(solution, time_span)
interpolation_derivative = InterpolationDerivative(interpolations)

# Plotting states and interpolations 
solution_interp = interpolations(time_span).T
with plt.style.context(["science", "notebook", "bright"]) :
    fig, ax = plt.subplots(2, 2, figsize = (20, 10))
    
    for i, _ax in enumerate(ax.ravel()):
        _ax.plot(time_span, solution[:, i], label = "Solution")
        _ax.plot(time_span, solution_interp[:, i], label = "Interpolation")
        _ax.legend()
    
    plt.savefig(os.path.join(_dir, "states.png"))
    plt.close()


def _foo(z, t, px):
    p, x = px
    (
        k1, k2, k3, k4, k5, k6, 
        k7, k8, k9, k10, k11
    ) = x 

    (km1, km2, km3, km4, km5, km6) = p
    
    return jnp.array([
        k1 + k2 * z[0] - k3 * z[1] * z[0] / (z[0] + km1) - k4 * z[2] * z[0] / (z[0] + km2),
        k5 * z[0] - k6 * z[1] / (z[1] + km3),
        k7 * z[1] * z[2] * z[3] / (z[3] + km4) + k8 * z[1] + k9 * z[0] - k10 * z[2] / (z[2] + km5) - k11 * z[2] / (z[2] + km6),
        -k7 * z[1] * z[2] * z[3] / (z[3] + km4) + k11 * z[2] / (z[2] + km6)
    ])


@jax.custom_jvp
def _interp(t) : return jax.pure_callback(interpolations, jax.ShapeDtypeStruct(xinit.shape, xinit.dtype), t)
_interp.defjvp(lambda primals, tangents : (_interp(*primals), None))


p_guess, x_guess = jnp.ones(6), jnp.ones(11)
px_guess, unravel = flatten_util.ravel_pytree((p_guess, x_guess))

def _foo_interp(z, t, px):
    p, x = px
    (
        k1, k2, k3, k4, k5, k6, 
        k7, k8, k9, k10, k11
    ) = x 

    (km1, km2, km3, km4, km5, km6) = p
    z = _interp(t)

    return jnp.array([
        k1 + k2 * z[0] - k3 * z[1] * z[0] / (z[0] + km1) - k4 * z[2] * z[0] / (z[0] + km2),
        k5 * z[0] - k6 * z[1] / (z[1] + km3),
        k7 * z[1] * z[2] * z[3] / (z[3] + km4) + k8 * z[1] + k9 * z[0] - k10 * z[2] / (z[2] + km5) - k11 * z[2] / (z[2] + km6),
        -k7 * z[1] * z[2] * z[3] / (z[3] + km4) + k11 * z[2] / (z[2] + km6)
    ])


###############################################################################################################################################
# Comparing with Full NLP : DFSINDy shooting + interpolation

dfsindy_target = solution - solution[0]

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

    return p, x.flatten()

if pargs.method == 1 : 
    p, x = outer_objective(px_guess, dfsindy_target)
    plot_coefficients(jnp.array_split(p_actual, [11]), [x, p], param_labels, "ShootingInterpCoeff", _dir)
    prediction = odeint(calcium_ion, xinit, time_span, jnp.concatenate((x, p)))
    plot_trajectories(solution, prediction, time_span, 4, "ShootingInterpStates", _dir)

###############################################################################################################################################
# Comparing with BiLevel NLP : DFSINDy (shooting-interp) inner + DFSINDy (shooting-interp) outer 

def f(p, x, target):
    solution = odeint_diffrax(_foo_interp, xinit, time_span, (p.flatten(), x.flatten()), pargs.rtol, pargs.atol, pargs.mxstep) - xinit
    return jnp.mean((solution - target)**2)

def g(p, x) : return jnp.array([ ])

def simple_objective_shooting(f, g, p, states, target):
    (x_opt, v_opt), _ = differentiable_optimization(f, g, p, x_guess, (target, ))
    _loss = f(p, x_opt, target) + v_opt @ g(p, x_opt)
    return _loss, x_opt

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

    return p, x.flatten()

if pargs.method == 0 :
    p, x = outer_objective_shooting(p_guess, solution, dfsindy_target)
    plot_coefficients(jnp.array_split(p_actual, [11]), [x, p], param_labels, "BiLevelShootingInterpCoeff", _dir)
    prediction = odeint(calcium_ion, xinit, time_span, jnp.concatenate((x, p)))
    plot_trajectories(solution, prediction, time_span, 4, "BiLevelShootingInterpStates", _dir)

###############################################################################################################################################
# Comparing with Full NLP : shooting / sequential optimization 

def simple_objective_nlp(px, states):
    # states shape = (nexpt, T, nx) # no of experiments, time points, states. Target values for outer optimization
    # features shape = (nexpt, T, F) # no of experiments, time points, no of features. Features for inner optimization 
    # target shape = (nexpt, T, nx) no of experiments, time points, states. Target values for inner optimization
    # p shape = (nx * F, ) # no of nonlinear decision variables
    # small_ind shape = (nx, F)
    
    solution = odeint_diffrax(_foo, xinit, time_span, unravel(px), pargs.rtol, pargs.atol, pargs.mxstep)
    _loss = jnp.mean((solution - states)**2)
    return _loss

def outer_objective_nlp(px_guess, solution):
    
    nlp_logger = logging.getLogger("Shooting")
    nlp_logger.setLevel(logging.INFO)
    logfile = logging.FileHandler(os.path.join(_dir, "record_shooting.txt"))
    nlp_logger.addHandler(logfile)
    _output_file = os.path.join(_dir, "ipopt_shooting_output.txt")

    # JIT compiled objective function
    _simple_obj = jax.jit(lambda p : simple_objective_nlp(p, solution))
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
        # constraints = {"type" : "eq", "fun" : _g, "jac" : _g_jac, "hess" : _g_hess},
        tol = pargs.tol, 
        options = {"maxiter" : pargs.iters, "output_file" : _output_file, "disp" : 0, "file_print_level" : 5, "mu_strategy" : "adaptive"}
        )
        
    nlp_logger.info(f"{divider}")
    with open(_output_file, "r") as file:
        for line in file : nlp_logger.info(line.strip())

    os.remove(_output_file)

    p, x = unravel(jnp.array(solution_object.x))
    nlp_logger.info(f"{divider} \nLoss {solution_object.fun}")
    nlp_logger.info(f"{divider} \nNonlinear parameters : {p}")
    nlp_logger.info(f"{divider} \nLinear parameters : {x}")

    return p, x.flatten()

if pargs.method == 2 : 
    p, x = outer_objective_nlp(px_guess, solution)
    plot_coefficients(jnp.array_split(p_actual, [11]), [x, p], param_labels, "ShootingCoeff", _dir)
    prediction = odeint(calcium_ion, xinit, time_span, jnp.concatenate((x, p)))
    plot_trajectories(solution, prediction, time_span, 4, "ShootingStates", _dir)

###############################################################################################################################################
