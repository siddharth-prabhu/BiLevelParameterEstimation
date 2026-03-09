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

from utils import constraint_differentiable_optimization, differentiable_optimization, odeint_diffrax, plot_coefficients, plot_trajectories

# Choose Hyperparameters
parser = argparse.ArgumentParser("ParameterEstimationYeastGlycolysis")
parser.add_argument("--iters", type = int, default = 1000, help = "The maximum number of iterations to be performed in parameter estimation")
parser.add_argument("--tol", type = float, default = 1e-5, help = "Ipopt tolerance")
parser.add_argument("--atol", type = float, default = 1e-8, help = "Absolute tolerance of ode solver")
parser.add_argument("--rtol", type = float, default = 1e-6, help = "Relative tolerance of ode solver")
parser.add_argument("--mxstep", type = int, default = 10_000, help = "The maximum number of steps a ode solver takes")
parser.add_argument("--msg", type = str, default = "", help = "Sample msg that briefly describes this problem")
parser.add_argument("--method", type = int, default = 0, help = "Formulation type 0 : BiLevelOpt (DFSINDy innter + DFSINDy outer), 1 : FullNLP (DFSINDy), 2 : FullNLP (shooting/sequential)")

parser.add_argument("--id", type = str, default = "", help = "Slurm job id")
parser.add_argument("--partition", type = str, default = "", help = "The partition this job is assigned to")
parser.add_argument("--cpus", type = int, default = 1, help = "Maximum number of cpus availabe per node")

pargs = parser.parse_args()

_dir = os.path.join("log", "YeastGlycolysis", str(datetime.now()))
divider = "--"*50 # printing separater
if not os.path.exists(_dir) : os.makedirs(_dir)

logger = logging.getLogger("__name__")
logger.setLevel(logging.INFO)
logfile = logging.FileHandler(os.path.join(_dir, "record.txt"))
logger.addHandler(logfile)
logger.info("PARAMETER ESTIMATION (ODE) Yeast Glycolysis")
logger.info(pformat(pargs.__dict__))


def yeast_glycolysis(x, t, p):
    # https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007575
    
    (
        J0, k1, k2, k3, k4, k5, k6, k, psi, q, K1, kappa, N, A
    ) = p # 2.5, 100, 6, 16, 100, 1.28, 12, 1.8, 13, 4, 0.52, 0.1, 1, 4 (linear parameters, nonlinear parameters)

    S1, S2, S3, S4, S5, S6, S7 = x
    phi = k1 * S1 * S6 / (1 + (S6 / K1)**q)
    
    return jnp.array([
        J0 - phi,
        2 * phi - k2 * S2 * (N - S5) - k6 * S2 * S5,
        k2 * S2 * (N - S5) - k3 * S3 * (A - S6),
        k3 * S3 * (A - S6) - k4 * S4 * S5 - kappa * (S4 - S7),
        k2 * S2 * (N - S5) - k4 * S4 * S5 - k6 * S2 * S5,
        -2 * phi + 2 * k3 * S3 * (A - S6) - k5 * S6,
        psi * kappa * (S4 - S7) - k * S7,
    ])


# Generate data
nx = 7
param_labels = [
    [r"$J_0$", r"$k_1$", r"$k_2$", r"$k_3$", r"$k_4$", r"$k_5$", r"$k_6$", r"$k$", r"$\psi$"], 
    [r"$q$", r"$K_1$", r"$\kappa$", r"$N$", r"$A$"]
]

xinit = jnp.array([0.501, 1.955, 0.198, 0.148, 0.161, 0.161, 0.064])
time_span = jnp.arange(0, 10, 0.1)
p_actual = jnp.array([2.5, 100, 6, 16, 100, 1.28, 12, 1.8, 0.1, 4, 0.52, 13, 1, 4])
solution = odeint(yeast_glycolysis, xinit, time_span, p_actual)


class Interpolation():

    def __init__(self, solution, time_span):
        self.interpolations = CubicSpline(time_span, solution)

    def __call__(self, t):
        return jnp.asarray(self.interpolations(t))

class InterpolationDerivative():

    def __init__(self, interpolation : Interpolation, order : int = 1):
        self.derivatives = interpolation.interpolations.derivative(order)

    def __call__(self, t):
        return jnp.asarray(self.derivatives(t))


interpolations = Interpolation(solution, time_span)
interpolation_derivative = InterpolationDerivative(interpolations)

# Plotting states and interpolations 
solution_interp = interpolations(time_span)
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
    J0, k1, k2, k3, k4, k5, k6, k, psi= x 
    q, K1, kappa, N, A = p
    
    S1, S2, S3, S4, S5, S6, S7 = z
    phi = k1 * S1 * S6 / (1 + (S6 / K1)**q)
    
    return jnp.array([
        J0 - phi,
        2 * phi - k2 * S2 * (N - S5) - k6 * S2 * S5,
        k2 * S2 * (N - S5) - k3 * S3 * (A - S6),
        k3 * S3 * (A - S6) - k4 * S4 * S5 - kappa * (S4 - S7),
        k2 * S2 * (N - S5) - k4 * S4 * S5 - k6 * S2 * S5,
        -2 * phi + 2 * k3 * S3 * (A - S6) - k5 * S6,
        psi * kappa * (S4 - S7) - k * S7,
    ])


@jax.custom_jvp
def _interp(t) : return jax.pure_callback(interpolations, jax.ShapeDtypeStruct(xinit.shape, xinit.dtype), t)
_interp.defjvp(lambda primals, tangents : (_interp(*primals), None))


p_guess, x_guess = 5 * jnp.ones(5), jnp.ones(9)
px_guess, unravel = flatten_util.ravel_pytree((p_guess, x_guess))

def _foo_interp(z, t, px):
    p, x = px
    J0, k1, k2, k3, k4, k5, k6, k, psi = x 
    q, K1, kappa, N, A = p
    
    S1, S2, S3, S4, S5, S6, S7 = _interp(t)
    phi = k1 * S1 * S6 / (1 + (S6 / K1)**q)
    
    return jnp.array([
        J0 - phi,
        2 * phi - k2 * S2 * (N - S5) - k6 * S2 * S5,
        k2 * S2 * (N - S5) - k3 * S3 * (A - S6),
        k3 * S3 * (A - S6) - k4 * S4 * S5 - kappa * (S4 - S7),
        k2 * S2 * (N - S5) - k4 * S4 * S5 - k6 * S2 * S5,
        -2 * phi + 2 * k3 * S3 * (A - S6) - k5 * S6,
        psi * kappa * (S4 - S7) - k * S7,
    ])

def g(p, x) : return jnp.array([ ])

def h(p, x) : return x # geq 0

dfsindy_target = solution - xinit

###############################################################################################################################################
# Comparing with BiLevel NLP : DFSINDy (shooting-interp) inner + DFSINDy (shooting-interp) outer 

def f(p, x, target):
    solution = odeint_diffrax(_foo_interp, xinit, time_span, (p.flatten(), x.flatten()), pargs.rtol, pargs.atol, pargs.mxstep) - xinit
    return jnp.mean((solution - target)**2)

def simple_objective_shooting(f, g, p, states, target):
    (x_opt, v_opt, m_opt), _ = constraint_differentiable_optimization(f, g, h, p, x_guess, (target, ))
    _loss = f(p, x_opt, target) + v_opt @ g(p, x_opt) + m_opt @ h(p, x_opt)
    # (x_opt, v_opt), _ = differentiable_optimization(f, g, p, x_guess, (target, ))
    # _loss = f(p, x_opt, target) + v_opt @ g(p, x_opt)
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
        x0 = p_guess,
        jac = _simple_jac,
        hess = _simple_hess,  
        bounds = list(zip(jnp.zeros_like(p_guess), jnp.inf * jnp.ones_like(p_guess))), 
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
    plot_coefficients(jnp.array_split(p_actual, [len(x_guess)]), [x, p], param_labels, "BiLevelShootingInterpCoeff", _dir)
    prediction = odeint(yeast_glycolysis, xinit, time_span, jnp.concatenate((x, p)))
    plot_trajectories(solution, prediction, time_span, 4, "BiLevelShootingInterpStates", _dir)

###############################################################################################################################################
# Comparing with Full NLP : shooting / sequential optimization 

def simple_objective_nlp(px, states):
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
        bounds = list(zip(jnp.concatenate((jnp.zeros_like(p_guess), -jnp.inf * jnp.ones_like(x_guess))), jnp.inf * jnp.ones_like(px_guess))),
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
    plot_coefficients(jnp.array_split(p_actual, [len(x_guess)]), [x, p], param_labels, "ShootingCoeff", _dir)
    prediction = odeint(yeast_glycolysis, xinit, time_span, jnp.concatenate((x, p)))
    plot_trajectories(solution, prediction, time_span, 4, "ShootingStates", _dir)

###############################################################################################################################################
