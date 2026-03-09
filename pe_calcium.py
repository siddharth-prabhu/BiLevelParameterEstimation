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

from utils import differentiable_optimization, odeint_diffrax, OrthogonalCollocationFormulation, plot_coefficients, plot_trajectories, constraint_differentiable_optimization

# Choose Hyperparameters
parser = argparse.ArgumentParser("ParameterEstimationCalciumIon")
parser.add_argument("--iters", type = int, default = 1000, help = "The maximum number of iterations to be performed in parameter estimation")
parser.add_argument("--tol", type = float, default = 1e-4, help = "Ipopt tolerance")
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

_dir = os.path.join("log", "CalciumIon", str(datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")))
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
    [fr"$k_{{{i}}}$" for i in range(1, 12)], [fr"$Km_{{{i}}}$" for i in range(1, 7)]
    ]

t0, tf, dt = 1, 20., 0.1 # measurement interval
xinit = jnp.array([0.12, 0.31, 0.0058, 4.3])
time_span = jnp.arange(t0, tf, dt)
p_actual = jnp.array([0.09, 2, 1.27, 3.73, 1.27, 32.24, 2, 0.05, 13.58, 153, 4.85, 0.19, 0.73, 29.09, 2.67, 0.16, 0.05])
solution = odeint(calcium_ion, xinit, time_span, p_actual)
actual_derivatives = jax.vmap(calcium_ion, in_axes = (0, 0, None))(solution, time_span, p_actual)


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

p_guess, x_guess = jnp.ones(6), jnp.ones(11) # nonlinear, linear parameters
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
        options = {
            "maxiter" : pargs.iters, 
            "output_file" : _output_file, 
            "disp" : 0, 
            "file_print_level" : 5, 
            "mu_strategy" : "adaptive", 
            "print_timing_statistics" : "yes"
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

    return p, x.flatten()

if pargs.method == 0 :
    p, x = outer_objective_shooting(p_guess, solution, dfsindy_target)
    plot_coefficients(jnp.array_split(p_actual, [11]), [x, p], param_labels, "BiLevelShootingInterpCoeff", _dir, separate = False)
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

    return p, x.flatten()

if pargs.method == 2 : 
    p, x = outer_objective_nlp(px_guess, solution)
    plot_coefficients(jnp.array_split(p_actual, [11]), [x, p], param_labels, "ShootingCoeff", _dir)
    prediction = odeint(calcium_ion, xinit, time_span, jnp.concatenate((x, p)))
    plot_trajectories(solution, prediction, time_span, 4, "ShootingStates", _dir)

###############################################################################################################################################
# Comparing with Full NLP : multiple shooting / sequential optimization 

if pargs.method == 3 : 

    window = 5 # number of points in one interval
    ntotal = len(time_span)
    intervals = (ntotal + window - 1 ) // window
    splits = jnp.array_split(time_span, jnp.arange(window, ntotal, window)) # divide the array into equal parts except the last one
    splits[-1] = jnp.pad(splits[-1], (0, window - len(splits[-1])), "maximum") # pad the last array with last time point
    time_intervals = jnp.vstack(splits)
    time_intervals = jnp.column_stack((
        jnp.concatenate((time_intervals[0, :1], time_intervals[:-1, -1])), 
        time_intervals))

    y_guess, unravel_y = flatten_util.ravel_pytree(jnp.ones(shape = (intervals, nx)))
    pxy_guess, unravel = flatten_util.ravel_pytree((p_guess, x_guess, y_guess))

    def equality_constraints(pxy, states, time_intervals):
        p, x, y = unravel(pxy)
        y = unravel_y(y)
        y = jnp.vstack((states[0][jnp.newaxis], y))
        solution = jax.vmap(lambda y0, ts : odeint_diffrax(_foo, y0, ts, (p, x)))(y[:-1], time_intervals)
        return jnp.vstack([*(y[1:] - solution[:, -1, :])]).flatten() # jnp.vstack([y[0] - states[:, 0, :], *(y[1:] - solution[:, -1, :])]).flatten()

    def simple_objective_ms(pxy, states, time_intervals):
        p, x, y = unravel(pxy)
        y = unravel_y(y)
        y = jnp.vstack((states[0][jnp.newaxis], y))
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
            options = {
                "maxiter" : pargs.iters, 
                "output_file" : _output_file, 
                "disp" : 0, 
                "file_print_level" : 5, 
                "mu_strategy" : "adaptive",
                "print_timing_statistics" : "yes",
                }
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

    _init = 1
    model.k1 = pmo.Var(initialize = _init)
    model.k2 = pmo.Var(initialize = _init)
    model.k3 = pmo.Var(initialize = _init)
    model.k4 = pmo.Var(initialize = _init)
    model.k5 = pmo.Var(initialize = _init)
    model.k6 = pmo.Var(initialize = _init)
    model.k7 = pmo.Var(initialize = _init)
    model.k8 = pmo.Var(initialize = _init)
    model.k9 = pmo.Var(initialize = _init)
    model.k10 = pmo.Var(initialize = _init)
    model.k11 = pmo.Var(initialize = _init)
    model.km1 = pmo.Var(initialize = _init)
    model.km2 = pmo.Var(initialize = _init)
    model.km3 = pmo.Var(initialize = _init)
    model.km4 = pmo.Var(initialize = _init)
    model.km5 = pmo.Var(initialize = _init)
    model.km6 = pmo.Var(initialize = _init)

    model.x = pmo.Var(model.t, initialize = xinit[0])
    model.y = pmo.Var(model.t, initialize = xinit[1])
    model.z = pmo.Var(model.t, initialize = xinit[2])
    model.w = pmo.Var(model.t, initialize = xinit[3])
    model.dxdt = DerivativeVar(model.x, wrt = model.t)
    model.dydt = DerivativeVar(model.y, wrt = model.t)
    model.dzdt = DerivativeVar(model.z, wrt = model.t)
    model.dwdt = DerivativeVar(model.w, wrt = model.t)

    # Initial conditions
    model.x[model.t.first()].fix(xinit[0])
    model.y[model.t.first()].fix(xinit[1])
    model.z[model.t.first()].fix(xinit[2])
    model.w[model.t.first()].fix(xinit[3])

    # Differential equations
    @model.Constraint(model.t)
    def _dxdt_rule(m, t):
        if t == m.t.first()  : return pmo.Constraint.Skip
        return m.dxdt[t] == m.k1 + m.k2 * m.x[t] - m.k3 * m.y[t] * m.x[t] / (m.x[t] + m.km1) - m.k4 * m.z[t] * m.x[t] / (m.x[t] + m.km2)

    @model.Constraint(model.t)
    def _dydt_rule(m, t):
        if t == m.t.first() : return pmo.Constraint.Skip
        return m.dydt[t] == m.k5 * m.x[t] - m.k6 * m.y[t] / (m.y[t] + m.km3)

    @model.Constraint(model.t)
    def _dzdt_rule(m, t):
        if t == m.t.first() : return pmo.Constraint.Skip
        return m.dzdt[t] == m.k7 * m.y[t] * m.z[t] * m.w[t] / (m.w[t] + m.km4) + m.k8 * m.y[t] + m.k9 * m.x[t] - m.k10 * m.z[t] / (m.z[t] + m.km5) - m.k11 * m.z[t] / (m.z[t] + m.km6)

    @model.Constraint(model.t)
    def _dwdt_rule(m, t):
        if t == m.t.first() : return pmo.Constraint.Skip
        return m.dwdt[t] == -m.k7 * m.y[t] * m.z[t] * m.w[t] / (m.w[t] + m.km4) + m.k11 * m.z[t] / (m.z[t] + m.km6)

    @model.Objective(sense = pmo.minimize)
    def objective_function(m):
        return np.mean(
            [(m.x[t] - solution[i, 0])**2 + (m.y[t] - solution[i, 1])**2 + (m.z[t] - solution[i, 2])**2 + (m.w[t] - solution[i, 3])**2
            for i, t in enumerate(time_span)]
        )

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
        x = np.array([*map(pmo.value, [model.k1, model.k2, model.k3, model.k4, model.k5, model.k6, model.k7, model.k8, model.k9, model.k10, model.k11])])
        p = np.array([*map(pmo.value, [model.km1, model.km2, model.km3, model.km4, model.km5, model.km6])])

    # Print IPOPT stats into the file
    nlp_logger.info(f"{divider}")
    with open(_output_file, "r") as file:
        for line in file : nlp_logger.info(line.strip())
    os.remove(_output_file)

    # plot solution
    nlp_logger.info(f"{divider} \nNonlinear parameters : {p}")
    nlp_logger.info(f"{divider} \nLinear parameters : {x}")

    plot_coefficients(jnp.array_split(p_actual, [len(x_guess)]), [x, p], param_labels, "OCollocationCoeff", _dir)
    # prediction = odeint(calcium_ion, xinit, time_span, jnp.concatenate((x, p)))
    # plot_trajectories(solution, prediction, time_span, 4, "OCollocationStates", _dir)