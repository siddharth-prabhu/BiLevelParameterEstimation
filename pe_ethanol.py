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

from utils import differentiable_optimization, odeint_diffrax, plot_coefficients, plot_trajectories, OrthogonalCollocationFormulation

# Choose Hyperparameters
parser = argparse.ArgumentParser("ParameterEstimationEthanolFermentation")
parser.add_argument("--iters", type = int, default = 1000, help = "The maximum number of iterations to be performed in parameter estimation")
parser.add_argument("--tol", type = float, default = 1e-5, help = "Ipopt tolerance")
parser.add_argument("--atol", type = float, default = 1e-8, help = "Absolute tolerance of ode solver")
parser.add_argument("--rtol", type = float, default = 1e-6, help = "Relative tolerance of ode solver")
parser.add_argument("--mxstep", type = int, default = 10_000, help = "The maximum number of steps a ode solver takes")
parser.add_argument("--msg", type = str, default = "", help = "Sample msg that briefly describes this problem")
parser.add_argument("--method", type = int, default = 0, help = "Formulation type 0 : BiLevelOpt (DFSINDy innter + DFSINDy outer), 1 : FullNLP (DFSINDy), 2 : FullNLP (shooting/sequential)")

# Orthogonal collocation hyperparameters
parser.add_argument("--ncp", type = int, default = 2, help = "The number of collocation points")
parser.add_argument("--nfe", type = int, default = 1, help = "The number of finite elements")
parser.add_argument("--scheme", type = str, choices = ["LAGRANGE-RADAU", "LAGRANGE-LEGENDRE"], default = "LAGRANGE-RADAU", help = "The interpolation polynomial type")

parser.add_argument("--id", type = str, default = "", help = "Slurm job id")
parser.add_argument("--partition", type = str, default = "", help = "The partition this job is assigned to")
parser.add_argument("--cpus", type = int, default = 1, help = "Maximum number of cpus availabe per node")

pargs = parser.parse_args()

_dir = os.path.join("log", "EthanolFermentation", str(datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")))
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

t0, tf, dt = 1, 15., 0.1 # measurement interval
xinit = jnp.column_stack((
    jrandom.uniform(key, shape = (nexpt, ), minval = 2., maxval = 10), 
    jrandom.uniform(key, shape = (nexpt, ), minval = 50., maxval = 100),
    jrandom.uniform(key, shape = (nexpt, 2), minval = 0., maxval = 10)
))
time_span = jnp.arange(t0, tf, dt)
p_actual = jnp.array([0.6397, 3.429, 1.748, 90.35, 460.4, 32.26, 4.895, 10, 17.45, 499.4, 1115.1, 12.89, 12.45, 110.3, 3.71, 27.78, 0.3797, 0.505, 0.1955])
solution = jax.vmap(lambda xi : odeint(ethanol_fermentation, xi, time_span, p_actual))(xinit)


class Interpolation():

    def __init__(self, solution : List[jnp.ndarray], time_span : jnp.ndarray):
        self.interpolations = [CubicSpline(time_span, _solution) for _solution in solution]

    def __call__(self, t):
        return jnp.stack([jnp.asarray(interpolation(t)) for interpolation in self.interpolations])

class InterpolationDerivative():

    def __init__(self, interpolation : Interpolation, order : int = 1):
        self.derivatives = [interp.derivative(order) for interp in interpolation.interpolations]

    def __call__(self, t):
        return jnp.stack([jnp.asarray(derivative(t)) for derivative in self.derivatives])


interpolations = Interpolation(solution, time_span)
interpolation_derivative = InterpolationDerivative(interpolations)

# Plotting states and interpolations  
solution_interp = interpolations(time_span)[0]
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
        options = {
            "maxiter" : pargs.iters, 
            "output_file" : _output_file, 
            "disp" : 0, 
            "file_print_level" : 5, 
            "print_timing_statistics" : "yes",
            "mu_strategy" : "adaptive"
            }
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
        options = {
            "maxiter" : pargs.iters, 
            "output_file" : _output_file, 
            "disp" : 0, 
            "file_print_level" : 5, 
            "print_timing_statistics" : "yes",
            "mu_strategy" : "adaptive"
            }
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
    plot_coefficients(jnp.array_split(p_actual, [len(x_guess)]), [x, p], param_labels, "BiLevelShootingCoeff", _dir, separate = False)
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
        options = {
            "maxiter" : pargs.iters, 
            "output_file" : _output_file, 
            "disp" : 0, 
            "file_print_level" : 5, 
            "print_timing_statistics" : "yes",
            "mu_strategy" : "adaptive"
            }
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

###############################################################################################################################################
# Comparing with Full NLP : multiple shooting / sequential optimization 

if pargs.method == 3 : 

    window = 20 # number of points in one interval
    ntotal = len(time_span)
    intervals = (ntotal + window - 1 ) // window
    splits = jnp.array_split(time_span, jnp.arange(window, ntotal, window)) # divide the array into equal parts except the last one
    splits[-1] = jnp.pad(splits[-1], (0, window - len(splits[-1])), "maximum") # pad the last array with last time point
    time_intervals = jnp.vstack(splits)
    time_intervals = jnp.column_stack((
        jnp.concatenate((time_intervals[0, :1], time_intervals[:-1, -1])), 
        time_intervals))

    y_guess, unravel_y = flatten_util.ravel_pytree(jnp.zeros(shape = (intervals, nexpt, nx)))
    pxy_guess, unravel = flatten_util.ravel_pytree((p_guess, x_guess, y_guess))

    def equality_constraints(pxy, states, time_intervals):
        p, x, y = unravel(pxy)
        y = unravel_y(y)
        y = jnp.vstack((states[:, 0, :][jnp.newaxis], y))
        solution = jax.vmap(lambda y0, ts : odeint_diffrax(_foo, y0, ts, (p, x)))(y[:-1], time_intervals)
        return jnp.vstack([*(y[1:] - solution[:, -1, :])]).flatten() # jnp.vstack([y[0] - states[:, 0, :], *(y[1:] - solution[:, -1, :])]).flatten()

    def simple_objective_ms(pxy, states, time_intervals):
        p, x, y = unravel(pxy)
        y = unravel_y(y)
        y = jnp.vstack((states[:, 0, :][jnp.newaxis], y))
        solution = jax.vmap(lambda y0, ts : odeint_diffrax(_foo, y0, ts, (p, x), pargs.rtol, pargs.atol, pargs.mxstep))(y[:-1], time_intervals)
        return jnp.mean((jnp.vstack(solution[:, 1:, :])[:ntotal, :] - jnp.einsum("ijk->jik", states))**2)

    def outer_objective_nlp(pxy_guess, states, time_intervals):
        
        nlp_logger = logging.getLogger("MultipleShooting")
        nlp_logger.setLevel(logging.INFO)
        logfile = logging.FileHandler(os.path.join(_dir, "record_mshooting.txt"))
        nlp_logger.addHandler(logfile)
        _output_file = os.path.join(_dir, "ipopt_mshooting_output.txt")

        # JIT compiled objective function
        _simple_obj = jax.jit(lambda pxy : simple_objective_ms(pxy, states, time_intervals))
        _simple_jac = jax.jit(jax.grad(_simple_obj))
        _simple_hess = jax.jit(jax.jacfwd(_simple_jac))

        # JIT compiled equality constraints
        _eq = jax.jit(lambda pxy : equality_constraints(pxy, states, time_intervals))
        _eq_jac = jax.jit(jax.jacfwd(_eq))
        _eq_hvp = jax.jit(lambda pxy, v : jax.jacfwd(lambda _pxy : v @ _eq(_pxy))(pxy))

        def _simple_obj_error(pxy):
            try :
                sol = _simple_obj(pxy)
            except : 
                sol = jnp.inf
            return sol

        solution_object = minimize_ipopt(
            _simple_obj_error, 
            x0 = pxy_guess,
            jac = _simple_jac,
            hess = _simple_hess,  
            constraints = [{"type" : "eq", "fun" : _eq, "jac" : _eq_jac, "hess" : _eq_hvp}], # continuity constraints
            bounds = list(zip(
                jnp.concatenate((jnp.zeros_like(p_guess), -jnp.inf * jnp.ones_like(x_guess), -jnp.inf * jnp.ones_like(y_guess))), 
                jnp.inf * jnp.ones_like(pxy_guess)
            )),
            tol = pargs.tol, 
            options = {"maxiter" : pargs.iters, "output_file" : _output_file, "disp" : 0, "file_print_level" : 5, "mu_strategy" : "adaptive"}
            )
            
        nlp_logger.info(f"{divider}")
        with open(_output_file, "r") as file:
            for line in file : nlp_logger.info(line.strip())

        os.remove(_output_file)

        p, x, y = unravel(jnp.array(solution_object.x))
        nlp_logger.info(f"{divider} \nLoss {solution_object.fun}")
        nlp_logger.info(f"{divider} \nNonlinear parameters : {p}")
        nlp_logger.info(f"{divider} \nLinear parameters : {x}")
        nlp_logger.info(f"{divider} \nContinuity variables : {unravel_y(y)}")

        return p, x

    p, x = outer_objective_nlp(pxy_guess, solution, time_intervals)
    plot_coefficients(jnp.array_split(p_actual, [len(x_guess)]), [x, p], param_labels, "MShootingCoeff", _dir)

###############################################################################################################################################
# Comparing with Full NLP : simultaneous optimization / orthogonal collocation

if pargs.method == 4 : 
    # https://www.do-mpc.com/en/latest/theory_orthogonal_collocation.html
    # https://github.com/casadi/casadi/blob/main/docs/examples/python/direct_collocation.py

    import numpy as np
    import pyomo.environ as pmo
    from pyomo.dae import ContinuousSet, DerivativeVar
    
    nlp_logger = logging.getLogger("Collocation")
    nlp_logger.setLevel(logging.INFO)
    logfile = logging.FileHandler(os.path.join(_dir, "record_collocation.txt"))
    nlp_logger.addHandler(logfile)
    _output_file = os.path.join(_dir, "ipopt_collocation_output.txt")
    
    time_span = np.asarray(time_span)
    xinit = np.asarray(xinit)
    solution = np.asarray(solution)

    model = pmo.ConcreteModel()
    model.t = ContinuousSet(initialize = time_span, bounds = (t0, tf))
    model.nexpt = pmo.RangeSet(0, nexpt - 1)
    model.nx = pmo.RangeSet(0, nx - 1)

    _init = 1
    model.um = pmo.Var(initialize = _init)
    model.vp1 = pmo.Var(initialize = _init)
    model.vp2 = pmo.Var(initialize = _init)
    
    model.kp1 = pmo.Var(initialize = _init, bounds = (0., np.inf))
    model.kp1_d = pmo.Var(initialize = _init, bounds = (0., np.inf))
    model.kp2_d = pmo.Var(initialize = _init, bounds = (0., np.inf))
    model.ks = pmo.Var(initialize = _init, bounds = (0., np.inf))
    model.kp2 = pmo.Var(initialize = _init, bounds = (0., np.inf))
    model.ks_d = pmo.Var(initialize = _init, bounds = (0., np.inf))
    model.ks_dd = pmo.Var(initialize = _init, bounds = (0., np.inf))
    model.ks1 = pmo.Var(initialize = _init, bounds = (0., np.inf))
    model.kp11 = pmo.Var(initialize = _init, bounds = (0., np.inf))
    model.kp21 = pmo.Var(initialize = _init, bounds = (0., np.inf))
    model.ks1_d = pmo.Var(initialize = _init, bounds = (0., np.inf))
    model.kp11_d = pmo.Var(initialize = _init, bounds = (0., np.inf))
    model.ks1_dd = pmo.Var(initialize = _init, bounds = (0., np.inf))
    model.kp21_d = pmo.Var(initialize = _init, bounds = (0., np.inf))
    model.yp1 = pmo.Var(initialize = _init, bounds = (0., np.inf))
    model.yp2 = pmo.Var(initialize = _init, bounds = (0., np.inf))

    model.x = pmo.Var(model.nexpt, model.nx, model.t, rule = lambda m, i, j, t : xinit[i, j])
    model.dxdt = DerivativeVar(model.x, wrt = model.t)

    # Initial conditions
    for i in model.nexpt :
        for j in model.nx : 
            model.x[i, j, model.t.first()].fix(xinit[i, j])

    # Differential equations
    @model.Constraint(model.nexpt, model.nx, model.t)
    def _dxdt_rule(m, i, j, t):
        
        if t == m.t.first()  : return pmo.Constraint.Skip

        mu = m.um * m.x[i, 1, t] * m.kp1 * m.kp2 / (m.ks + m.x[i, 1, t] + m.x[i, 1, t]**2 / m.ks1) / (m.kp1 + m.x[i, 2, t] + m.x[i, 2, t]**2 / m.kp11) / (m.kp2 + m.x[i, 3, t] + m.x[i, 3, t]**2 / m.kp21)
        qp1 = m.vp1 * m.x[i, 1, t] * m.kp1_d / (m.ks_d + m.x[i, 1, t] + m.x[i, 1, t]**2 / m.ks1_d) / (m.kp1_d + m.x[i, 2, t] + m.x[i, 2, t]**2 / m.kp11_d)
        qp2 = m.vp2 * m.x[i, 1, t] * m.kp2_d / (m.ks_dd + m.x[i, 1, t] + m.x[i, 1, t]**2 / m.ks1_dd) / (m.kp2_d + m.x[i, 3, t] + m.x[i, 3, t]**2 / m.kp21_d)

        if j == 0 : 
            return m.dxdt[i, j, t] == mu * m.x[i, 0, t]
        elif j == 1 :
            return m.dxdt[i, j, t] == - m.x[i, 0, t] * (qp1 / m.yp1 - qp2 / m.yp2)
        elif j == 2 : 
            return m.dxdt[i, j, t] == qp1 * m.x[i, 0, t]
        else :
            return m.dxdt[i, j, t] == qp2 * m.x[i, 0, t]    


    @model.Objective(sense = pmo.minimize)
    def objective_function(m):
        
        asum = 0
        count = 0
        for i in model.nexpt : 
            for j in model.nx : 
                for k, t in enumerate(time_span) :
                    asum += (m.x[i, j, t] - solution[i, k, j])**2
                    count += 1

        return asum / count # MSE

    discretizer = pmo.TransformationFactory('dae.collocation')
    discretizer.apply_to(model, wrt = model.t, nfe = pargs.nfe, ncp = pargs.ncp, scheme = pargs.scheme)

    solver = pmo.SolverFactory('ipopt')
    solver.options['tol'] = pargs.tol # Tolerance
    solver.options['max_iter'] = pargs.iters # Max iterations
    solver.options['print_level'] = 5
    solver.options["file_print_level"] = 5
    solver.options["output_file"] = _output_file

    try : 
        results = solver.solve(model, tee = True)
    except Exception as error :
        print(error) 
    finally : 
        x = np.array([*map(pmo.value, [model.um, model.vp1, model.vp2])])
        p = np.array([*map(pmo.value, [model.kp1, model.kp1_d, model.kp2_d, model.ks, model.kp2, model.ks_d, model.ks_dd, model.ks1, model.kp11, model.kp21, model.ks1_d, model.kp11_d, model.ks1_dd, model.kp21_d, model.yp1, model.yp2])])

    # Print IPOPT stats into the file
    nlp_logger.info(f"{divider}")
    with open(_output_file, "r") as file:
        for line in file : nlp_logger.info(line.strip())
    os.remove(_output_file)

    # plot solution
    nlp_logger.info(f"{divider} \nNonlinear parameters : {p}")
    nlp_logger.info(f"{divider} \nLinear parameters : {x}")
    
    # plot solution
    plot_coefficients(jnp.array_split(p_actual, [len(x_guess)]), [x, p], param_labels, "OCollocationCoeff", _dir)
    # prediction = odeint(ethanol_fermentation, xinit[0], time_span, jnp.concatenate((x, p)))
    # plot_trajectories(solution[:, : nx], prediction, time_span, 4, "OCollocationStates", _dir)