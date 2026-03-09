import os
import logging
from datetime import datetime
from pprint import pformat
import argparse
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jrandom
jax.config.update("jax_enable_x64", True)
from jax.experimental.ode import odeint
from jax import flatten_util, tree_util

import matplotlib.pyplot as plt
from cyipopt import minimize_ipopt
from scipy.interpolate import CubicSpline as SCubicSpline

from utils import differentiable_optimization, odeint_diffrax, plot_coefficients, plot_trajectories, CubicSpline


# Choose Hyperparameters
parser = argparse.ArgumentParser("ParameterEstimationCSTR")
parser.add_argument("--iters", type = int, default = 1000, help = "The maximum number of iterations to be performed in parameter estimation")
parser.add_argument("--tol", type = float, default = 1e-4, help = "Ipopt tolerance")
parser.add_argument("--atol", type = float, default = 1e-8, help = "Absolute tolerance of ode solver")
parser.add_argument("--rtol", type = float, default = 1e-6, help = "Relative tolerance of ode solver")
parser.add_argument("--mxstep", type = int, default = 10_000, help = "The maximum number of steps a ode solver takes")
parser.add_argument("--msg", type = str, default = "", help = "Sample msg that briefly describes this problem")
parser.add_argument("--method", type = int, default = 1, help = "Formulation type 0 : BiLevelOpt (1), 1 : BiLevelOpt (2), 2 : FullNLP (shooting/sequential)")

parser.add_argument("--id", type = str, default = "", help = "Slurm job id")
parser.add_argument("--partition", type = str, default = "", help = "The partition this job is assigned to")
parser.add_argument("--cpus", type = int, default = 1, help = "Maximum number of cpus availabe per node")

pargs = parser.parse_args()

_dir = os.path.join("log", "CSTR", str(datetime.now()))
if not os.path.exists(_dir) : os.makedirs(_dir)
divider = "--"*50

logger = logging.getLogger("__name__")
logger.setLevel(logging.INFO)
logfile = logging.FileHandler(os.path.join(_dir, "record.txt"))
logger.addHandler(logfile)
logger.info("PARAMETER ESTIMATION (ODE) CALCIUM ION")
logger.info(pformat(pargs.__dict__))


def cstr(x, t, p):
    # https://www.proquest.com/docview/2280493554?%20Theses&fromopenview=true&pq-origsite=gscholar&sourcetype=Dissertations%20
    Fin = 1 # Inflow
    Cin = 2 # Concentration of inflow
    Tin = 323 
    Tc = 340
    U, kref, EdivR = p # 5.3417, 0.461, 0.833
    b = kref * jnp.exp(- 10**4 * EdivR * (1 / x[1] - 1 / 350))
    return jnp.array([
        - b * x[0] + Fin * (Cin - x[0]), 
        130 * b * x[0] + Fin * (Tin - x[1]) + U * (Tc - x[1])
    ])

param_labels = [
    [r"$U$"], # Linear parameters
    [r"$k_{ref}$", r"$EdivR$"], # NonLinear parameters
    ]

xinit = jnp.array([1.6, 340])
p_actual = jnp.array([5.3417, 0.461, 0.833])
time_span = jnp.arange(0, 10, 0.1)
solution = odeint_diffrax(cstr, xinit, time_span, p_actual)

##################################################################################################
# Unobserved states (initial conditions are observed) are handled using differentiable cubic spline interpolation. 
# The convexity of the parameters in the equation of unobserved states can be exploited

if pargs.method == 0 : 

    p_guess, x_guess = jnp.concatenate((jnp.ones(1), jnp.ones_like(solution[:, 0]))), jnp.ones(2)

    with plt.style.context(["science", "notebook", "bright"]) :
        fig, ax = plt.subplots(1, 2, figsize = (20, 10))
        
        for i, _ax in enumerate(ax.ravel()):
            _ax.plot(time_span, solution[:, i], label = "Solution")
            _ax.legend()
        
        plt.savefig("states.png")
        plt.close()

    def f(p, x, states, target):
        
        p, y = jnp.array_split(p, [1])
        # TODO Faster cubic spline inverse (exploit sparsity pattern). 
        # Compute CubicSpline of unobserved states (only). Precompute for observed states
        interp = partial(CubicSpline, t = time_span, y = jnp.column_stack((y, states[:, 1])))

        def cstr_interp(x, t, p):
            Fin = 1 # Inflow
            Cin = 2 # Concentration of inflow
            Tin = 323 # Temperature of inflow
            Tc = 340 # Temperature of coolant
            (EdivR, ), (kref, U) = p

            z = interp(t)[0]
            b = kref * jnp.exp(- 10**4 * EdivR * (1 / z[1] - 1 / 350))
            return jnp.array([
                - b * z[0] + Fin * (Cin - z[0]), 
                130 * b * z[0] + Fin * (Tin - z[1]) + U * (Tc - z[1])
            ])

        solution = odeint_diffrax(cstr_interp, xinit, time_span, (p.flatten(), x.flatten()), atol = pargs.atol, rtol = pargs.rtol, mxstep = pargs.mxstep) - xinit
        return jnp.mean((solution - jnp.column_stack((interp(time_span)[:, 0] - xinit[0], target)))**2)

    def g(p, x) : return jnp.array([ ])

    def simple_objective_shooting(f, g, p, states, target):
        (x_opt, v_opt), _ = differentiable_optimization(f, g, p, x_guess, (states, target))
        _loss = f(p, x_opt, states, target) + v_opt @ g(p, x_opt)
        return _loss, x_opt

    def outer_objective_shooting(p_guess, states, target):
        
        shooting_logger = logging.getLogger("BiLevelInterp")
        shooting_logger.setLevel(logging.INFO)
        logfile = logging.FileHandler(os.path.join(_dir, "record_bilevelinterp.txt"))
        shooting_logger.addHandler(logfile)
        _output_file = os.path.join(_dir, "ipopt_bilevelinterp_output.txt")

        # JIT compiled objective function
        _simple_obj = jax.jit(lambda p : simple_objective_shooting(f, g, p, states, target)[0])
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

    p, x = outer_objective_shooting(p_guess, solution, solution[:, 1] - xinit[1])
    plot_coefficients(jnp.array_split(p_actual, [1]), [x, p], param_labels, "BiLevelInterpCoeff", _dir)
    prediction = odeint(cstr, xinit, time_span, jnp.concatenate((x, p)))
    plot_trajectories(solution, prediction, time_span, 2, "BiLevelInterpStates", _dir)


##################################################################################################
# Unobserved states (initial conditions are observed) are handled using single shooting. 
# The convexity of the parameters in the equation of unobserved states cannot be exploited

if pargs.method == 1 : 

    p_guess, x_guess = jnp.ones(2), jnp.ones(1)


    class Interpolation():

        def __init__(self, solution, time_span):
            self.interpolations = SCubicSpline(time_span, solution)

        def __call__(self, t):
            return self.interpolations(t)

    class InterpolationDerivative():

        def __init__(self, interpolation : Interpolation, order : int = 1):
            self.derivatives = interpolation.interpolations.derivative(order)

        def __call__(self, t):
            return self.derivatives(t)

    interpolations = Interpolation(solution, time_span)
    interpolation_derivative = InterpolationDerivative(interpolations)

    @jax.custom_jvp
    def _interp(t) : return jax.pure_callback(interpolations, jax.ShapeDtypeStruct(xinit.shape, xinit.dtype), t)
    _interp.defjvp(lambda primals, tangents : (_interp(*primals), None))


    # x1 is unobserved and x2 is observed
    def f(p, x, states, target):

        def cstr_interp(x, t, p):
            Fin = 1 # Inflow
            Cin = 2 # Concentration of inflow
            Tin = 323 # Temperature of inflow
            Tc = 340 # Temperature of coolant
            (kref, EdivR), (U, ) = p

            z = _interp(t)
            b = kref * jnp.exp(- 10**4 * EdivR * (1 / z[1] - 1 / 350))
            return jnp.array([
                - b * x[0] + Fin * (Cin - x[0]), 
                130 * b * x[0] + Fin * (Tin - z[1]) + U * (Tc - z[1])
            ])

        solution = odeint_diffrax(cstr_interp, xinit, time_span, (p.flatten(), x.flatten()), atol = pargs.atol, rtol = pargs.rtol, mxstep = pargs.mxstep) - xinit
        return jnp.mean((solution[:, 1] - target)**2)

    def g(p, x) : return jnp.array([ ])

    def simple_objective_shooting(f, g, p, states, target):
        (x_opt, v_opt), _ = differentiable_optimization(f, g, p, x_guess, (states, target))
        _loss = f(p, x_opt, states, target) + v_opt @ g(p, x_opt)
        return _loss, x_opt

    def outer_objective_shooting(p_guess, states, target):
        
        shooting_logger = logging.getLogger("BiLevelShootingInterp")
        shooting_logger.setLevel(logging.INFO)
        logfile = logging.FileHandler(os.path.join(_dir, "record_bilevelshootinginterp.txt"))
        shooting_logger.addHandler(logfile)
        _output_file = os.path.join(_dir, "ipopt_bilevelshootinginterp_output.txt")

        # JIT compiled objective function
        _simple_obj = jax.jit(lambda p : simple_objective_shooting(f, g, p, states, target)[0])
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

    p, x = outer_objective_shooting(p_guess, solution, solution[:, 1] - xinit[1])
    plot_coefficients(jnp.array_split(p_actual, [1]), [x, p], param_labels, "BiLevelShootingInterpCoeff", _dir, separate = False)
    prediction = odeint(cstr, xinit, time_span, jnp.concatenate((x, p)))
    plot_trajectories(solution, prediction, time_span, 2, "BiLevelShootingInterpStates", _dir)


##################################################################################################
# All states (initial conditions are observed) are handled using single shooting. 
# The convexity of the parameters in the equations is not exploited

if pargs.method == 2 :

    px_guess = jnp.ones(3)

    def simple_objective_shooting(p, target):
        solution = odeint_diffrax(cstr, xinit, time_span, p)
        return jnp.mean((solution[:, 1] - target)**2)

    def outer_objective_shooting(p_guess, target):
        
        shooting_logger = logging.getLogger("SingleShootingInterp")
        shooting_logger.setLevel(logging.INFO)
        logfile = logging.FileHandler(os.path.join(_dir, "record_singleshooting.txt"))
        shooting_logger.addHandler(logfile)
        _output_file = os.path.join(_dir, "ipopt_singleshooting_output.txt")

        # JIT compiled objective function
        _simple_obj = jax.jit(lambda p : simple_objective_shooting(p, target))
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
        shooting_logger.info(f"{divider} \nParameters : {p}")

        return p

    p = outer_objective_shooting(px_guess, solution[:, 1])
    plot_coefficients(p_actual, p, param_labels, "ShootingCoeff", _dir, separate = False)
    prediction = odeint(cstr, xinit, time_span, p)
    plot_trajectories(solution, prediction, time_span, 2, "ShootingStates", _dir)
