import os
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jrandom
jax.config.update("jax_enable_x64", True)
from jax.experimental.ode import odeint

import matplotlib.pyplot as plt
import pysindy as ps
import numpy as np
from cyipopt import minimize_ipopt
from scipy.integrate import odeint as scipy_odeint
from scipy.interpolate import CubicSpline


def rate_constant(T, Tref, act):
    return jnp.exp(- act * (10**4/T - 10**4/Tref) / 8.314)

def foo(x, t, p, T):
    # https://chemical-kinetics.readthedocs.io/en/latest/simple_example.html
    # forward reaction A -> B with rate constant k1f
    # backward reaction B -> A with rate constant k1b
    # forward reaction C -> D with rate constant k2
    # simple example with stoichiometry as identity matrix

    k1f, k1b, k2 = p * rate_constant(T, jnp.array(373.), jnp.array([3., 4., 5.]))
    return jnp.array([
        - k1f * x[0]**2 + k1b * x[1],
        k1f * x[0]**2 - k1b * x[1],
        -k2 * x[2],
        k2 * x[2]
    ])

nx = 4
nexpt = 10
key = jrandom.PRNGKey(20)
key_temp, key_xinit = jrandom.split(key, 2)
temperature = jrandom.uniform(key_temp, shape = (nexpt, 1), minval = 363., maxval = 383.)
xinit = jrandom.uniform(key_xinit, shape = (nexpt, nx), minval = 4., maxval = 10.)
time_span = jnp.arange(1, 5., 0.1)
p_actual = jnp.array([1.03, 0.21, 1.5])
solution = jnp.stack([odeint(foo, xi, time_span, p_actual, ti) for xi, ti in zip(xinit, temperature)])
actual_derivatives = jax.vmap(lambda xi, ti : jax.vmap(lambda _xi : foo(_xi, time_span[0], p_actual, T = ti))(xi))(solution, temperature)
interpolations = [[CubicSpline(time_span, _sol[:, i]) for i in range(nx)] for _sol in solution]

fig, ax = plt.subplots(1, nexpt, figsize = (30, 15))
for i in range(nexpt) :
    ax[i].plot(time_span, solution[i], "--")

# plt.savefig("kinetic")
# plt.close()

def derivatives(xi, ti):

    def _derivatives(xi, ti):
        return ps.FiniteDifference()._differentiate(np.array(xi), np.array(ti))
    
    return jax.pure_callback(_derivatives, jax.ShapeDtypeStruct(xi.shape, xi.dtype), xi, ti)

estimated_derivatives = jax.vmap(lambda xi : derivatives(xi, time_span))(solution)

def data_matrix(p, features, T):
    _p = jax.vmap(rate_constant, in_axes = (0, None, None))(T, jnp.array(373.), p)
    return jnp.vstack(jax.vmap(lambda _p, feat : _p * feat)(_p, features))


# permute the nonzero rows and take inverse using dynamic slicing
# record permutation array and number of non zero elements
# TODO keep on increasing tolerance as you prune parameters
# TODO instead of SINDY use DF-SINDY to get theta

def _data_matrix(p, features, T):
    _p = jax.vmap(rate_constant, in_axes = (0, None, None))(T, jnp.array(373.), p)
    return jnp.vstack(jax.vmap(lambda _p, feat : _p * feat)(_p, features))


@partial(jax.custom_vjp, nondiff_argnums = (6, 7))
def _optimal_parameters(p, target, features, temperature, permute, inverse_permute, nonzero_cols, regularization):
    
    A = _data_matrix(p, features, temperature)
    m, n = A.shape
    A = A[:, permute]
    
    theta = jnp.zeros(n)
    _A = jax.lax.slice(A, (0, 0), (m, nonzero_cols))
    u, s, vh = jnp.linalg.svd(_A.T @ _A + regularization * jnp.eye(nonzero_cols))
    _theta = vh.T @ (jnp.diag(1/s) @ (u.T @ (_A.T @ target)))
    # _theta = jnp.linalg.solve(_A.T @ _A + regularization * jnp.eye(nonzero_cols), _A.T @ target)
    theta = theta.at[jnp.arange(nonzero_cols, dtype = int)].set(_theta)

    return theta[inverse_permute], (u, s, vh)

def _optimal_parameters_fwd(p, target, features, temperature, permute, inverse_permute, nonzero_cols, regularization):
    theta, (u, s, vh) = _optimal_parameters(p, target, features, temperature, permute, inverse_permute, nonzero_cols, regularization)
    return (theta, (u, s, vh)), (p, theta, (u, s, vh), target, features, temperature, permute)

def _optimal_parameters_bwd(nonzero_cols, regularization, res, g_dot):
    # save inverse during the forward pass and reuse in reverse pass
    p, theta, (u, s, vh), target, features, temperature, permute = res

    def _g(p):
        A = _data_matrix(p, features, temperature)
        A = A[:, permute]
        _theta = theta[permute]

        m, _ = A.shape
        _A = jax.lax.slice(A, (0, 0), (m, nonzero_cols))
        _theta = jax.lax.slice(_theta, (0, ), (nonzero_cols, ))
        return _A.T @ target - _A.T @ (_A @ _theta) - regularization*_theta, _A

    _, f_vjp, _A = jax.vjp(_g, p, has_aux = True)
    cotangent = vh.T @ (jnp.diag(1/s) @ (u.T @ jax.lax.slice(g_dot[0][permute], (0, ), (nonzero_cols, ))))
    # cotangent = jnp.linalg.solve(_A.T @ _A + regularization * jnp.eye(nonzero_cols), jax.lax.slice(g_dot[0][permute], (0, ), (nonzero_cols, )))
    return f_vjp(cotangent)[0], None, None, None, None, None


_optimal_parameters.defvjp(_optimal_parameters_fwd, _optimal_parameters_bwd)


def poly(x, t):
    return jnp.array([
        x[0], x[1], x[2], x[3], 
        x[0]**2, x[0]*x[1], x[0]*x[2], x[0]*x[3], 
        x[1]**2, x[1]*x[2], x[1]*x[3],
        x[2]**2, x[2]*x[3],
        x[3]**2,
        x[0]**3, x[0]**2*x[1], x[0]**2*x[2], x[0]**2*x[3],
        x[1]**3, x[1]**2*x[0], x[1]**2*x[2], x[1]**2*x[3],
        x[2]**3, x[2]**2*x[0], x[2]**2*x[1], x[2]**2*x[3],
        x[3]**3, x[3]**2*x[0], x[3]**2*x[1], x[3]**2*x[2],
        x[0]*x[1]*x[2], x[0]*x[1]*x[3], x[1]*x[2]*x[3], x[0]*x[2]*x[3]
    ])

def poly_interp(z, t, x):
    return jnp.array([
        x[0](t), x[1](t), x[2](t), x[3](t), 
        x[0](t)**2, x[0](t)*x[1](t), x[0](t)*x[2](t), x[0](t)*x[3](t), 
        x[1](t)**2, x[1](t)*x[2](t), x[1](t)*x[3](t),
        x[2](t)**2, x[2](t)*x[3](t),
        x[3](t)**2,
        x[0](t)**3, x[0](t)**2*x[1](t), x[0](t)**2*x[2](t), x[0](t)**2*x[3](t),
        x[1](t)**3, x[1](t)**2*x[0](t), x[1](t)**2*x[2](t), x[1](t)**2*x[3](t),
        x[2](t)**3, x[2](t)**2*x[0](t), x[2](t)**2*x[1](t), x[2](t)**2*x[3](t),
        x[3](t)**3, x[3](t)**2*x[0](t), x[3](t)**2*x[1](t), x[3](t)**2*x[2](t),
        x[0](t)*x[1](t)*x[2](t), x[0](t)*x[1](t)*x[3](t), x[1](t)*x[2](t)*x[3](t), x[0](t)*x[2](t)*x[3](t)
    ])

def _foo(x, t, T, theta, p):
    theta = jax.vmap(lambda _theta : _theta * rate_constant(T, jnp.array(373.), p))(theta)
    return theta @ poly(x, t)

def simple_objective(p, states, features, derivatives, permute, inverse_permute, nonzero_cols, regularization = 0.01):
    # states shape = (E, T, n) # no of experiments X time points X states. Target values for outer optimization
    # derivatives shape = (E, T, n) # no of experiments X time points X states. Target values for inner optimization
    # features shape = (E, T, F) # no of experiments X time points X no of features. Features for inner optimization 
    # p shape = (F) # no of features
    
    # cannot vmap because nonzero_cols cannot be converted to a matrix using vmap
    # cannot scan over the list nonzero_cols either.
    # unrolling the loop is the only option
    """
    theta = jnp.vstack([
        _optimal_parameters(p, deri_iter, features, temperature, perm_iter, inv_perm_iter, cols_iter, regularization)[0]
        for deri_iter, perm_iter, inv_perm_iter, cols_iter in zip(jnp.vstack(derivatives).T, permute, inverse_permute, nonzero_cols)
    ]) """
    # TODO get theta from an optimization problem. Equality constraint convex problem
    theta = jnp.vstack(
        jax.tree_util.tree_map(
            lambda deri_iter, perm_iter, inv_perm_iter, cols_iter : _optimal_parameters(p, deri_iter, features, temperature, perm_iter, inv_perm_iter, cols_iter, regularization)[0], 
            [*jnp.vstack(derivatives).T], [*permute], [*inverse_permute], [*nonzero_cols]
        )
    )

    solution = jax.vmap(lambda xi, ti : odeint(_foo, xi, time_span, ti, theta, p))(xinit, temperature)
    _loss = jax.vmap(lambda pred, meas : jnp.mean((pred - meas)**2))(solution, states)
    return jnp.mean(_loss), theta

len_features = len(poly(xinit[0], 0.))
nonzero_cols = jnp.array([2, 2, 1, 1], dtype = int) # number of nonzero columns
permute = jnp.array([0, 1, 2, 3, 0, 1, 2, 3, 2, 0, 1, 3, 2, 0, 1, 3], dtype = int).reshape(nx, -1) # big indexes followed by small indexes
inverse_permute = jnp.array([0, 1, 2, 3, 0, 1, 2, 3, 1, 2, 0, 3, 1, 2, 0, 3], dtype = int).reshape(nx, -1) # reverse the permutation
p_guess = jnp.ones(len_features)
theta_guess = jnp.ones((nx, len_features))

# SINDy features
sindy_features = jax.vmap(jax.vmap(lambda xi : poly(xi, 0.)))(solution)

# DF-SINDY features
dfsindy_features = jnp.stack([scipy_odeint(poly_interp, poly(_xi, 0), time_span, args = (_interp, )) - poly(_xi, 0) for _xi, _interp in zip(xinit, interpolations)])

# loss, linear_params = simple_objective(p_guess, solution, features, estimated_derivatives, permute, inverse_permute, nonzero_cols)
# gradients, aux = jax.grad(simple_objective, has_aux = True)(p_guess, solution, features, estimated_derivatives, permute, inverse_permute, nonzero_cols)

# _simple_obj = jax.jit(lambda p : simple_objective(p, solution, features, estimated_derivatives, permute, inverse_permute, nonzero_cols)[0])
# _simple_jac = jax.jit(jax.grad(_simple_obj))
# _simple_hess = jax.jit(jax.jacrev(_simple_jac))


def outer_objective(p_guess, theta_guess, solution, features, estimated_derivatives, thresholding = 0.1, maxiter = 10):
    # implement sequential threshold least square algorithm
    # TODO previous subproblems can have lower tolerance or less maximum iterations to eliminate the terms. How to choose initial tolerance ?

    def get_permutations(theta):

        ind = jnp.arange(len(theta), dtype = int)
        big_ind = jnp.abs(theta) >= thresholding
        permute = jnp.concatenate((ind[big_ind], ind[jnp.logical_not(big_ind)]))
        mask = jnp.zeros(len(theta), dtype = int) # used to stop iterations
        mask = mask.at[ind[big_ind]].set(1)

        # x[permute][inverse_permute] == x
        inverse_permute = jnp.zeros(len(theta), dtype = int)
        for i, value in enumerate(permute):
            inverse_permute = inverse_permute.at[value].set(i)
        
        nonzero_cols = [jnp.sum(big_ind)]
        
        return mask, permute, inverse_permute, nonzero_cols

    iteration = 0
    p, theta = p_guess, theta_guess

    while iteration < maxiter : 

        mask, permute, inverse_permute, nonzero_cols = map(jnp.vstack, zip(*[get_permutations(_theta) for _theta in theta]))
        print(f"Iteration {iteration}, linear parameters {theta}")
        nonzero_cols = nonzero_cols.flatten().tolist()

        if iteration == 0 : prev_mask = mask
        if iteration > 0 and jnp.allclose(mask, prev_mask) : 
            print("Optimal solution found")
            break

        _simple_obj = jax.jit(lambda p : simple_objective(p, solution, features, estimated_derivatives, permute, inverse_permute, nonzero_cols)[0])
        _simple_jac = jax.jit(jax.grad(_simple_obj))
        _simple_hess = jax.jit(jax.jacrev(_simple_jac))

        print("Starting parameteric optimization")
        solution_object = minimize_ipopt(
            _simple_obj, 
            x0 = p_guess, 
            jac = _simple_jac,
            hess = _simple_hess,  
            tol = 1e-7, 
            options = {"maxiter" : 100, "print_level" : 5}
            )
        
        print(solution_object)
        loss, theta = simple_objective(p, solution, features, estimated_derivatives, permute, inverse_permute, nonzero_cols)
        p = jnp.array(solution_object.x)
        print(f"Iteration {iteration} : loss {loss}")

        prev_mask = mask
        iteration += 1

    return solution_object, theta


solution_object, theta = outer_objective(p_guess, theta_guess, solution, dfsindy_features, jax.vmap(lambda z : z - z[0])(solution), thresholding = 0.1, maxiter = 20)


# compare with sequential optimization for the same system
"""
def sequential_objective(p, states, features, mask, regularization = 0.01):
    # states shape = (E, T, n) # no of experiments X time points X states 
    # derivatives shape = (E, T, n) # no of experiments X time points X states 
    # features shape = (E, T, F) # no of experiments X time points X no of features
    # p shape = (F) # no of features

    theta = p[: -len_features].reshape(nx, -1)
    theta = theta * mask
    p = p[-len_features :]
    
    solution = jax.vmap(lambda xi, ti : odeint(_foo, xi, time_span, ti, theta, p))(xinit, temperature)
    _loss = jax.vmap(lambda pred, meas : jnp.mean((pred - meas)**2))(solution, states)
    return jnp.mean(_loss) 

p_guess = jnp.zeros((nx + 1)*len_features)
mask = jnp.array([1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0]).reshape(nx, -1)
seq_loss = sequential_objective(p_guess, solution, features, mask)


print("Starting sequential optimization")
sequential_object = minimize_ipopt(
    lambda p : sequential_objective(p, solution, features, mask),
    x0 = p_guess, 
    jac = jax.grad(lambda p : sequential_objective(p, solution, features, mask)),
    tol = 1e-7, 
    options = {"maxiter" : 100, "print_level" : 5}
)

print(sequential_object)
"""