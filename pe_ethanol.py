import os
import logging
from typing import List, Tuple
from datetime import datetime
from pprint import pformat
import argparse
import operator

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
parser = argparse.ArgumentParser("ParameterEstimationEthanolFermentation")
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

_dir = os.path.join("log", "EthanolFermentation", str(datetime.now()))
divider = "--"*50 # printing separater
if not os.path.exists(_dir) : os.makedirs(_dir)

logger = logging.getLogger("__name__")
logger.setLevel(logging.INFO)
logfile = logging.FileHandler(os.path.join(_dir, "record.txt"))
logger.addHandler(logfile)
logger.info("PARAMETER ESTIMATION (ODE) ETHANOL FERMENTATION")
logger.info(pformat(pargs.__dict__))


def ethanol_fermentation(x, t, p) -> jnp.ndarray :
    # https://pubs.acs.org/doi/epdf/10.1021/ie000544%2B?ref=article_openPDF
    
    # [x, s, p1, p2] = x
    (
        um, vp1, vp2, kp1, kp1_d, kp2_d, ks, kp2, ks_d, ks_dd, ks1, kp11, kp21, ks1_d, kp11_d, ks1_dd, kp21_d, yp1, yp2
    ) = p # default values (0.6397, 3.429, 1.748, 90.35, 460.4, 32.26, 4.895, 10, 17.45, 499.4, 1115.1, 12.89, 12.45, 110.3, 3.71, 27.78, 0.3797, 0.505, 0.1955)
    
    mu = um * x[1] * kp1 * kp2 / (ks + x[1] + x[1]**2 / ks1) / (kp1 + x[2] + x[2]**2 / kp11) / (kp2 + x[3] + x[3]**2 / kp21)
    qp1 = vp1 * x[1] * kp1_d / (ks_d + x[1] + x[1]**2 / ks1_d) / (kp1_d + x[2] + x[2]**2 / kp11_d)
    qp2 = vp2 * x[1] * kp2_d / (ks_dd + x[1] + x[1]**2 / ks1_dd) / (kp2_d + x[3] + x[3]**2 / kp21_d)

    return jnp.array([
        mu * x[0],
        - x[0] * (qp1 / yp1 - qp2 / yp2),
        qp1 * x[0],
        qp2 * x[0]
    ])


# Generate data
nexpt = 15
nx = 4
key = jrandom.PRNGKey(20)
param_labels = [
    [r"$\mu _m$", r"$\nu _{p_1}$", r"$\nu _{p_2}$"], # linear parameters
    [
        r"$K_{p_1}$", r"$K_{p_1}^{\prime}$", r"$K_{p_2}^{\prime}$", r"$K_s$", r"$K_{p_2}$", r"$K_s^{\prime}$", r"$K_s^{\prime \prime}$", r"$K_{s_1}$", r"$K_{p_{11}}$", r"$K_{p_{21}}$", r"$K_{s_1}^{\prime}$", 
        r"$K_{p_{11}}^{\prime}$", r"$K_{s_1}^{\prime \prime}$", r"$K_{p_{21}}^{\prime}$", r"$Y_{p_1/s}$", r"$Y_{p_2/s}$"
    ] # non linear parameters
]

xinit = jnp.column_stack((
    jrandom.uniform(key, shape = (nexpt, ), minval = 2., maxval = 10), 
    jrandom.uniform(key, shape = (nexpt, ), minval = 50., maxval = 100),
    jrandom.uniform(key, shape = (nexpt, 2), minval = 0., maxval = 10)
))
time_span = jnp.arange(1, 15., 0.1)
p_actual = jnp.array([0.6397, 3.429, 1.748, 90.35, 460.4, 32.26, 4.895, 10, 17.45, 499.4, 1115.1, 12.89, 12.45, 110.3, 3.71, 27.78, 0.3797, 0.505, 0.1955])
solution = jax.vmap(lambda xi : odeint(ethanol_fermentation, xi, time_span, p_actual))(xinit)


class Interpolation():

    def __init__(self, solution : List[jnp.ndarray], time_span : jnp.ndarray):
        self.interpolations = [[CubicSpline(time_span, sol) for sol in _solution.T] for _solution in solution]

    def __call__(self, t):
        return jnp.stack([jnp.stack([interp(t) for interp in interpolation]) for interpolation in self.interpolations])

class InterpolationDerivative():

    def __init__(self, interpolation : Interpolation, order : int = 1):
        self.derivatives = [[interp.derivative(order) for interp in _interpolation] for _interpolation in interpolation.interpolations]

    def __call__(self, t):
        return jnp.stack([jnp.stack([interp(t) for interp in derivative]) for derivative in self.derivatives])


interpolations = Interpolation(solution, time_span)
interpolation_derivative = InterpolationDerivative(interpolations)

# Plotting states and interpolations  
solution_interp = interpolations(time_span)[0].T
with plt.style.context(["science", "notebook", "bright"]) :
    ncols = min(4, nx)
    nrows = (nx + ncols - 1) // ncols
    fig, ax = plt.subplots(nrows, ncols, figsize = (20, 5 * nrows))
    
    for i, _ax in enumerate(ax.ravel()):
        _ax.plot(time_span, solution[0][:, i], label = "Solution")
        _ax.plot(time_span, solution_interp[:, i], label = "Interpolation")
        _ax.legend()
    
    plt.savefig(os.path.join(_dir, "states.png"))
    plt.close()


@jax.custom_jvp
def _interp(t) : return jax.pure_callback(interpolations, jax.ShapeDtypeStruct((nexpt, nx), xinit.dtype), t)
_interp.defjvp(lambda primals, tangents : (_interp(*primals), None))

p_guess, x_guess = jnp.ones(16), jnp.ones(3) # nonlinear, linear parameters
px_guess, unravel = flatten_util.ravel_pytree((p_guess, x_guess))

def _foo_interp_intermediate(z, t, px):

    p, x = px # (nonlinear, linear) parameters
    um, vp1, vp2 = x
    (
        kp1, kp1_d, kp2_d, ks, kp2, ks_d, ks_dd, ks1, kp11, kp21, ks1_d, kp11_d, ks1_dd, kp21_d, yp1, yp2
    ) = p 

    mu = um * z[1] * kp1 * kp2 / (ks + z[1] + z[1]**2 / ks1) / (kp1 + z[2] + z[2]**2 / kp11) / (kp2 + z[3] + z[3]**2 / kp21)
    qp1 = vp1 * z[1] * kp1_d / (ks_d + z[1] + z[1]**2 / ks1_d) / (kp1_d + z[2] + z[2]**2 / kp11_d)
    qp2 = vp2 * z[1] * kp2_d / (ks_dd + z[1] + z[1]**2 / ks1_dd) / (kp2_d + z[3] + z[3]**2 / kp21_d)

    return jnp.array([
        mu * z[0],
        - z[0] * (qp1 / yp1 - qp2 / yp2),
        qp1 * z[0],
        qp2 * z[0]
    ])

def _foo_interp(z, t, px):
    z = _interp(t)
    return jax.vmap(_foo_interp_intermediate, in_axes = (0, None, None))(z, t, px).flatten()

def _foo(z, t, px):
    return jax.vmap(_foo_interp_intermediate, in_axes = (0, None, None))(z, t, px)

dfsindy_target = jax.vmap(operator.sub)(solution, xinit)

def g(p, x) : return jnp.array([ ])


###############################################################################################################################################
# Comparing with Full NLP : DFSINDy shooting + interpolation 

def simple_objective(px, target):
    
    _xinit = xinit.flatten()
    solution = jnp.stack(jnp.array_split(
        odeint_diffrax(_foo_interp, _xinit, time_span, unravel(px), pargs.rtol, pargs.atol, pargs.mxstep) - _xinit,
        nexpt, axis = 1
    ))
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
        bounds = list(zip(jnp.concatenate((jnp.zeros_like(p_guess), -jnp.inf * jnp.ones_like(x_guess))), jnp.inf * jnp.ones_like(px_guess))),
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

    return p, x

if pargs.method == 1 : 
    p, x = outer_objective(px_guess, dfsindy_target)
    plot_coefficients(jnp.array_split(p_actual, [len(x_guess)]), [x, p], param_labels, "ShootingInterpCoeff", _dir)
    prediction = odeint(ethanol_fermentation, xinit[0], time_span, jnp.concatenate((x, p)))
    plot_trajectories(solution[0], prediction, time_span, 4, "ShootingInterpStates", _dir)

###############################################################################################################################################
# Comparing with BiLevel NLP : DFSINDy (shooting + interpolation) inner + DFSINDy (shooting + interpolation) outer 

def f(p, x, target):
    # p = nonlinear parameters shape = (1, p)
    # x = linear parameters shape = (1, x)
    
    _xinit = xinit.flatten()
    solution = jnp.stack(jnp.array_split(
        odeint_diffrax(_foo_interp, _xinit, time_span, (p.flatten(), x.flatten()), pargs.rtol, pargs.atol, pargs.mxstep) - _xinit,
        nexpt, axis = 1
    ))

    return jnp.mean((solution - target)**2)

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

    return p, x

if pargs.method == 0 : 
    p, x = outer_objective_shooting(p_guess, solution, dfsindy_target)
    plot_coefficients(jnp.array_split(p_actual, [len(x_guess)]), [x, p], param_labels, "BiLevelShootingCoeff", _dir)
    prediction = odeint(ethanol_fermentation, xinit[0], time_span, jnp.concatenate((x, p)))
    plot_trajectories(solution[0], prediction, time_span, 4, "BiLevelShootingStates", _dir)


###############################################################################################################################################
# Comparing with Full NLP : shooting / sequential optimization 

def simple_objective_nlp(px, states):

    solution = jax.vmap(operator.sub, in_axes = (1, 0))(odeint_diffrax(_foo, xinit, time_span, unravel(px), pargs.rtol, pargs.atol, pargs.mxstep), xinit)
    _loss = jnp.mean((solution - states)**2)
    return _loss

def outer_objective_nlp(px_guess, solution):
    
    nlp_logger = logging.getLogger("shooting")
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
        x0 = px_guess,
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

    return p, x

if pargs.method == 2 : 
    p, x = outer_objective_nlp(px_guess, solution)
    plot_coefficients(jnp.array_split(p_actual, [len(x_guess)]), [x, p], param_labels, "ShootingCoeff", _dir)
    prediction = odeint(ethanol_fermentation, xinit[0], time_span, jnp.concatenate((x, p)))
    plot_trajectories(solution[0], prediction, time_span, 4, "ShootingStates", _dir)
