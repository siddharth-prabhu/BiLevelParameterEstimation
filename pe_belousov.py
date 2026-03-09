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
from jax import flatten_util, tree_util

import matplotlib.pyplot as plt
from cyipopt import minimize_ipopt
from scipy.interpolate import CubicSpline
from scipy.integrate import odeint

from utils import differentiable_optimization, odeint_diffrax, plot_coefficients, plot_trajectories


# Choose Hyperparameters
parser = argparse.ArgumentParser("ParameterEstimationBelousovReaction")
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

_dir = os.path.join("log", "BelousovReaction", str(datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")))
divider = "--"*50 # printing separater
if not os.path.exists(_dir) : os.makedirs(_dir)

logger = logging.getLogger("__name__")
logger.setLevel(logging.INFO)
logfile = logging.FileHandler(os.path.join(_dir, "record.txt"))
logger.addHandler(logfile)
logger.info("PARAMETER ESTIMATION (ODE) BELOUSOV REACTION")
logger.info(pformat(pargs.__dict__))


def belousov(x, t, p) -> jnp.ndarray :
    # https://www.sciencedirect.com/science/article/pii/S0098135498002336

    k2, k3, k4, k5, k1 = p # (linear, nonlinear) (1, 0.161, 1, 8.375e-6, 77.27)
    
    return jnp.array([
        k1 * (x[1] + k4 * x[0] - x[1] * x[0] - k5 * x[0]**2), 
        (- k2 * x[1] - x[0] * x[1] + x[2]) / k1, 
        k3 * (x[0] - x[2])
    ])

# Generate data with multiple initial conditions
nexpt = 10
nx = 3

key = jrandom.PRNGKey(10)
t0, tf, dt = 1, 500., 0.01
xinit = 0.1 * jrandom.uniform(key, shape = (nexpt, nx), minval = 1., maxval = 5.)
time_span = jnp.arange(t0, tf, dt)
p_actual = jnp.array([1, 0.161, 1, 8.375e-6, 77.27]) # linear and nonlinear parameters 
solution = jnp.stack([*map(lambda xi : odeint(belousov, xi, time_span, args = (p_actual, )), xinit)])
param_labels = [[fr"$k_{{{i}}}$" for i in range(2, 6)], [r"$k_1$"]]


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

p_guess, x_guess = jnp.ones(1), jnp.ones(4) # nonlinear, linear parameters
px_guess, unravel = flatten_util.ravel_pytree((p_guess, x_guess))

def _foo_interp_intermediate(z, t, px):

    p, x = px # (nonlinear, linear) parameters
    k2, k3, k4, k5 = x
    k1, = p
    
    return jnp.array([
        k1 * (z[1] + z[0] * (k4 - z[1] - k5 * z[0])), 
        (-z[1] * (k2 + z[0]) + z[2]) / k1, 
        k3 * (z[0] - z[2])
    ]) 

def _foo_interp(z, t, px):
    z = _interp(t)
    return jax.vmap(_foo_interp_intermediate, in_axes = (0, None, None))(z, t, px).flatten()

def _foo(z, t, px):
    return jax.vmap(_foo_interp_intermediate, in_axes = (0, None, None))(z, t, px)

dfsindy_target = jax.vmap(operator.sub)(solution, xinit)

def g(p, x) : return jnp.array([ ])

# Data matrix - perfrom integration beforehand

def foo_terms(z, t):

    def _foo_vmap(z) : 
        return jnp.array([
        z[0], z[1], z[2], z[1] * z[0], z[0]**2, 
    ]) 

    return jax.vmap(_foo_vmap)(_interp(t)).flatten()

data_matrix = jnp.stack(jnp.array_split(odeint(foo_terms, foo_terms(0, time_span[0]), time_span), nexpt, axis = 1))

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
    plot_coefficients(jnp.array_split(p_actual, [len(x_guess)]), [x, p], param_labels, "ShootingInterpCoeff", _dir, separate = False)
    prediction = odeint(belousov, xinit[0], time_span, args = (jnp.concatenate((x, p)), ))
    plot_trajectories(solution[0], prediction, time_span, 3, "ShootingInterpStates", _dir)

###############################################################################################################################################
# Comparing with BiLevel NLP : DFSINDy (shooting + interpolation) inner + DFSINDy (shooting + interpolation) outer 

def f(p, x, target):
    # p = nonlinear parameters shape = (1, p)
    # x = linear parameters shape = (1, x)

    # Because all the parameters are linearly separable, precomputing the integration makes the optimization problem cheaper
    k1 = p[0]
    k2, k3, k4, k5 = x
    mat = jnp.array([
        k1*k4, k1, 0, -k1, -k1*k5, 
        0, -k2/k1, 1/k1, -1/k1, 0,
        k3, 0, -k3, 0, 0
        ]).reshape(3, -1)

    solution = jax.vmap(lambda _data, _xi : _data @ mat.T - _xi)(data_matrix, xinit)
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
    prediction = odeint(belousov, xinit[0], time_span, args = (jnp.concatenate((x, p)), ))
    plot_trajectories(solution[0], prediction, time_span, 3, "BiLevelShootingStates", _dir)


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
    plot_coefficients(jnp.array_split(p_actual, [len(x_guess)]), [x, p], param_labels, "ShootingCoeff", _dir, separate = False)
    prediction = odeint(belousov, xinit[0], time_span, args = (jnp.concatenate((x, p)), ))
    plot_trajectories(solution[0], prediction, time_span, 3, "ShootingStates", _dir)

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
    model.nexpt = pmo.RangeSet(0, nexpt - 1) # independant experiments
    model.nx = pmo.RangeSet(0, nx - 1) # Dimension of x

    model.k1 = pmo.Var(initialize = 1., bounds = (0, np.inf))
    model.k2 = pmo.Var(initialize = 1.)
    model.k3 = pmo.Var(initialize = 1.)
    model.k4 = pmo.Var(initialize = 1.)
    model.k5 = pmo.Var(initialize = 1.)

    model.x = pmo.Var(model.nexpt, model.nx, model.t, rule = lambda m, i, j, t : xinit[i, j])
    model.dxdt = DerivativeVar(model.x, wrt = model.t)

    # fix initial condition
    for i in model.nexpt : 
        for j in model.nx : 
            model.x[i, j, model.t.first()].fix(xinit[i, j])

    # Differential equations
    @model.Constraint(model.nexpt, model.nx, model.t)
    def _dxdt_rule(m, i, j, t):

        if t == m.t.first()  : return pmo.Constraint.Skip

        if j == 0 :
            return m.dxdt[i, j, t] == m.k1 * (m.x[i, 1, t] + m.k4 * m.x[i, 0, t] - m.x[i, 1, t] * m.x[i, 0, t] - m.k5 * m.x[i, 0, t]**2)
        elif j == 1 :
            return m.dxdt[i, j, t] == (- m.k2 * m.x[i, 1, t] - m.x[i, 0, t] * m.x[i, 1, t] + m.x[i, 2, t]) / m.k1
        else :
            return m.dxdt[i, j, t] == m.k3 * (m.x[i, 0, t] - m.x[i, 2, t])
        
    @model.Objective(sense = pmo.minimize)
    def objective_function(m):
        # Loop over each state and nq
        asum = 0
        count = 0
        for i in range(nexpt):
            for j in range(nx):
                for k, t in enumerate(time_span) :
                    count += 1
                    asum += (solution[i, k, j] - m.x[i, j, t])**2

        return asum / count # mean squared error

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
        x = np.array([*map(pmo.value, [model.k2, model.k3, model.k4, model.k5])])
        p = np.array([*map(pmo.value, [model.k1])])

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
    # prediction = odeint(belousov, xinit[0], time_span, args = (jnp.concatenate((x, p)), ))
    # plot_trajectories(solution[0], prediction, time_span, 4, "OCollocationStates", _dir)