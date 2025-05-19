import os
from typing import List, Tuple
from functools import partial
import itertools
import logging
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

from utils import constraint_differentiable_optimization, odeint_diffrax, plot_coefficients, plot_trajectories


# Choose Hyperparameters
parser = argparse.ArgumentParser("ParameterEstimationMendes")
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
logger.info("PARAMETER ESTIMATION (ODE) MENDES")
logger.info(pformat(pargs.__dict__))


def mendes(x, t, p) -> jnp.ndarray :
    # https://utoronto.scholaris.ca/server/api/core/bitstreams/f557ed43-1f11-4949-854d-feb192355263/content
    (
        k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, # 1, 1, 1, 1, 1, 1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1, 1, 1, 
        q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, q15, # 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1, 1,
        q16, q17, q18, q19, q20, q21 # 1, 1, 1, 1, 1, 1,
    ), (P, S) = p  

    return jnp.array([
        k1 / (1 + (P / q1)**q2 + (q3 / S)**q4) - k2 * x[0],
        k3 / (1 + (P / q5)**q6 + (q7 / x[6])**q8) - k4 * x[1],
        k5 / (1 + (P / q9)**q10 + (q11 / x[7])**q12) - k6 * x[2],
        k7 * x[0] / (x[0] + q13) - k8 * x[3],
        k9 * x[1] / (x[1] + q14) - k10 * x[4],
        k11 * x[2] / (x[2] + q15) - k12 * x[5],
        k13 * x[3] * (S - x[6]) / q16 / (1 + S / q16 + x[6] / q17) - k14 * x[4] * (x[6] - x[7]) / q18 / (1 + x[6] / q18 + x[7] / q19), 
        k14 * x[4] * (x[6] - x[7]) / q18 / (1 + x[6] / q18 + x[7] / q19) - k15 * x[5] * (x[7] - P) / q20 / (1 + x[7] / q20 + P / q21), 
    ]) 


# Generate data
nx = 8
xinit = jnp.array([2/3, 0.57254, 0.41758, 0.4, 0.36409, 0.29457, 1.419, 0.93464])
time_span = jnp.arange(1, 120., 0.1)
p_actual = jnp.array([1, 1, 1, 1, 1, 1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1])
q_actual = jnp.stack([*map(jnp.array, itertools.product(jnp.array([0.05, 0.13572, 0.3684, 1]), jnp.array([0.1, 0.46416, 2.1544, 10])))])
solution = jax.vmap(lambda qi : odeint(mendes, xinit, time_span, (p_actual, qi)))(q_actual)
nlen = q_actual.shape[0]
param_labels = [
    [
        r"$k_1$", r"$k_2$", r"$k_3$", r"$k_4$", r"$k_5$", r"$k_6$", r"$k_7$", r"$k_8$", r"$k_9$", r"$k_{10}$", 
        r"$k_{11}$", r"$k_{12}$", r"$k_{13}$", r"$k_{14}$", r"$k_{15}$"
    ], 
    [
        r"$q_1$", r"$q_2$", r"$q_3$", r"$q_4$", r"$q_5$", r"$q_6$", r"$q_7$", r"$q_8$", r"$q_9$", 
        r"$q_{10}$", r"$q_{11}$", r"$q_{12}$", r"$q_{13}$", r"$q_{14}$", r"$q_{15}$",
        r"$q_{16}$", r"$q_{17}$", r"$q_{18}$", r"$q_{19}$", r"$q_{20}$", r"$q_{21}$"
    ]
]

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
    fig, ax = plt.subplots(2, 4, figsize = (20, 10))
    
    for i, _ax in enumerate(ax.ravel()):
        _ax.plot(time_span, solution[0][:, i], label = "Solution")
        _ax.plot(time_span, solution_interp[:, i], label = "Interpolation")
        _ax.legend()
    
    plt.savefig(os.path.join(_dir, "states.png"))
    plt.close()


@jax.custom_jvp
def _interp(t) : return jax.pure_callback(interpolations, jax.ShapeDtypeStruct((len(q_actual), len(xinit)), xinit.dtype), t)
_interp.defjvp(lambda primals, tangents : (_interp(*primals), None))

p_guess, x_guess = 2 * jnp.ones(21), jnp.zeros(15)
px_guess, unravel = flatten_util.ravel_pytree((p_guess, x_guess))

def _foo_interp_intermediate(z, t, px, ps):

    p, x = px
    P, S = ps
    
    k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15 = x # linear parameters

    (
        q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, q15,
        q16, q17, q18, q19, q20, q21 
    ) = p  # nonlinear parameters

    return jnp.array([
        k1 / (1 + (P / q1)**q2 + (q3 / S)**q4) - k2 * z[0],
        k3 / (1 + (P / q5)**q6 + (q7 / z[6])**q8) - k4 * z[1],
        k5 / (1 + (P / q9)**q10 + (q11 / z[7])**q12) - k6 * z[2],
        k7 * z[0] / (z[0] + q13) - k8 * z[3],
        k9 * z[1] / (z[1] + q14) - k10 * z[4],
        k11 * z[2] / (z[2] + q15) - k12 * z[5],
        k13 * z[3] * (S - z[6]) / q16 / (1 + S / q16 + z[6] / q17) - k14 * z[4] * (z[6] - z[7]) / q18 / (1 + z[6] / q18 + z[7] / q19), 
        k14 * z[4] * (z[6] - z[7]) / q18 / (1 + z[6] / q18 + z[7] / q19) - k15 * z[5] * (z[7] - P) / q20 / (1 + z[7] / q20 + P / q21), 
    ])

def _foo_interp(z, t, px):
    z = _interp(t)
    return jax.vmap(_foo_interp_intermediate, in_axes = (0, None, None, 0))(z, t, px, q_actual).flatten()

def _foo(z, t, px):
    return jax.vmap(_foo_interp_intermediate, in_axes = (0, None, None, 0))(z, t, px, q_actual)

dfsindy_target = solution - xinit

def g(p, x) : return jnp.array([ ])

def h(p, x) : return x[0:6]


###############################################################################################################################################
# Comparing with Full NLP : DFSINDy shooting + interpolation 

def simple_objective(px, target):
    
    _xinit = jnp.tile(xinit, (nlen, ))
    solution = jnp.stack(jnp.array_split(
        odeint_diffrax(_foo_interp, _xinit, time_span, unravel(px), pargs.rtol, pargs.atol, pargs.mxstep) - _xinit,
        nlen, axis = 1
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

    _h = lambda px : h(*unravel(px))
    _h_jac = jax.jit(jax.jacfwd(_h))
    _h_hess = jax.jit(lambda p, v : jax.jacfwd(lambda _p : v @ _h_jac(_p))(p))

    _g = lambda px : g(*unravel(px))
    _g_jac = jax.jit(jax.jacfwd(_g))
    _g_hess = jax.jit(lambda p, v : jax.jacfwd(lambda _p : v @ _g_jac(_p))(p))

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
        constraints = [
            {"type" : "ineq", "fun" : _h, "jac" : _h_jac, "hess" : _h_hess}, 
            {"type" : "eq", "fun" : _g, "jac" : _g_jac, "hess" : _g_hess}
            ], 
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
    prediction = odeint(mendes, xinit, time_span, (jnp.concatenate((x, p)), q_actual[0]))
    plot_trajectories(solution[0], prediction, time_span, 4, "ShootingInterpStates", _dir)

###############################################################################################################################################
# Comparing with BiLevel NLP : DFSINDy (shooting + interpolation) inner + DFSINDy (shooting + interpolation) outer 

def f(p, x, target):
    # p = nonlinear parameters shape = (1, p)
    # x = linear parameters shape = (1, x)
    
    _xinit = jnp.tile(xinit, (nlen, ))
    solution = jnp.stack(jnp.array_split(
        odeint_diffrax(_foo_interp, _xinit, time_span, (p.flatten(), x.flatten()), pargs.rtol, pargs.atol, pargs.mxstep) - _xinit,
        nlen, axis = 1
    ))

    return jnp.mean((solution - target)**2)

def simple_objective_shooting(f, g, p, states, target):
    (x_opt, v_opt, m_opt), _ = constraint_differentiable_optimization(f, g, h, p, x_guess, (target, ))
    _loss = f(p, x_opt, target) + v_opt @ g(p, x_opt) + m_opt @ h(p, x_opt)
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

    return p, x

if pargs.method == 0 : 
    p, x = outer_objective_shooting(p_guess, solution, dfsindy_target)
    plot_coefficients(jnp.array_split(p_actual, [len(x_guess)]), [x, p], param_labels, "BiLevelShootingCoeff", _dir)
    prediction = odeint(mendes, xinit, time_span, (jnp.concatenate((x, p)), q_actual[0]))
    plot_trajectories(solution[0], prediction, time_span, 4, "BiLevelShootingStates", _dir)


###############################################################################################################################################
# Comparing with Full NLP : shooting / sequential optimization 

def simple_objective_nlp(px, states):

    _xinit = jnp.tile(xinit, (nlen, 1))
    solution = jax.vmap(operator.sub, in_axes = (1, 0))(odeint_diffrax(_foo, _xinit, time_span, unravel(px), pargs.rtol, pargs.atol, pargs.mxstep), _xinit)
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

    _h = lambda px : h(*unravel(px))
    _h_jac = jax.jit(jax.jacfwd(_h))
    _h_hess = jax.jit(lambda p, v : jax.jacfwd(lambda _p : v @ _h_jac(_p))(p))

    _g = lambda px : g(*unravel(px))
    _g_jac = jax.jit(jax.jacfwd(_g))
    _g_hess = jax.jit(lambda p, v : jax.jacfwd(lambda _p : v @ _g_jac(_p))(p))

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
        constraints = [
            {"type" : "ineq", "fun" : _h, "jac" : _h_jac, "hess" : _h_hess}, 
            {"type" : "eq", "fun" : _g, "jac" : _g_jac, "hess" : _g_hess}
            ],
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
    prediction = odeint(mendes, xinit, time_span, (jnp.concatenate((x, p)), q_actual[0]))
    plot_trajectories(solution[0], prediction, time_span, 4, "ShootingStates", _dir)
