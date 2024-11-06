import os
from functools import partial, reduce
from typing import List, Tuple
import operator
import itertools

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

def equality_constraints(p, theta, nx):
    # The equality constraints can itself be nonlinear but given p, they should be linear with respect to theta.
    # flattened inputs. Reshape theta and then specify constraints for simplicity
    # Can only add constraints with respect to p and theta for the terms that are present in the original model
    # TODO add args i.e. temperature and other variables
    theta = theta.reshape(nx, -1)
    return jnp.array([
        theta[0, 1] + theta[1, 1],
        theta[0, 4] + theta[1, 4],
        theta[2, 2] + theta[3, 2], 
        # theta[0, 2] - 4 # spurious constraints (cannot add)
    ])

@partial(jax.custom_vjp, nondiff_argnums = (6, 7))
def _optimal_parameters(p, target, features, temperature, permute, inverse_permute, nonzero_cols, regularization):
    
    regularization = jnp.maximum(regularization, 1e-12)
    nx, _ = permute.shape
    A = _data_matrix(p, features, temperature)
    m, n = A.shape
    theta_guess = jnp.zeros(n * nx)
    
    g = equality_constraints(p, theta_guess, nx)
    g_x = jax.jacobian(lambda _theta : equality_constraints(p, _theta, nx))(theta_guess) # primals can be zero since constraints are linear

    def unconstrained(_A, _permute, _inverse_permute, _nonzero_cols, _target):
        # returns the unconstrained solution and the inverse matrix 
        theta = jnp.zeros(n)
        _zero_cols = n - _nonzero_cols
        _A = jax.lax.slice(_A[:, _permute], (0, 0), (m, _nonzero_cols))
        _inv = jnp.linalg.pinv(_A.T @ _A + regularization * jnp.eye(_nonzero_cols))
        _theta = _inv @ (_A.T @ _target)
        
        theta = theta.at[jnp.arange(_nonzero_cols, dtype = int)].set(_theta)
        inv = jnp.block([
            [_inv, jnp.zeros((_nonzero_cols, _zero_cols))], 
            [jnp.zeros((_zero_cols, _nonzero_cols)), (1/regularization) * jnp.eye(_zero_cols)]
        ])
        return theta[_inverse_permute], inv[:, _inverse_permute][_inverse_permute, :]

    # cannot vmap because nonzero_cols cannot be converted to a matrix form
    _theta_and_inv = jax.tree_util.tree_map(
        lambda perm_iter, inv_perm_iter, cols_iter, target_iter : unconstrained(A, perm_iter, inv_perm_iter, cols_iter, target_iter), 
        [*permute], [*inverse_permute], [*nonzero_cols], [*target]
    )
    x_hat, inverses = jnp.stack([_theta for _theta, _ in _theta_and_inv]), jnp.stack([_inv for _, _inv in _theta_and_inv]) # unconstrained solution and pseudo inverses.

    # calculate the constraints part 
    # g_x.T @ (g_x @ (A.T @ A)-1 g_x.T)-1 @ (g - g_x @ x + g_x @ x_hat)
    lam_star = jnp.linalg.solve(
        jnp.sum(jax.vmap(lambda _g_x, _inv : _g_x @ _inv @ _g_x.T)(jnp.stack(jnp.array_split(g_x, nx, axis = 1)), inverses), axis = 0), 
        g - g_x @ theta_guess + g_x @ x_hat.flatten()
    )

    x_star = jax.vmap(lambda _x_hat, _inv, _con : _x_hat - _inv @ _con)(x_hat, inverses, jnp.stack(jnp.array_split(g_x.T @ lam_star, nx)))
    return x_star, (lam_star, inverses)

def _optimal_parameters_fwd(p, target, features, temperature, permute, inverse_permute, nonzero_cols, regularization):
    theta, (lam_star, inv) = _optimal_parameters(p, target, features, temperature, permute, inverse_permute, nonzero_cols, regularization)
    return (theta, (lam_star, inv)), (p, theta, lam_star, inv, target, features, temperature, permute, inverse_permute)

def _optimal_parameters_bwd(nonzero_cols, regularization, res, g_dot):
    # save inverse during the forward pass and reuse in reverse pass
    p, theta, lam_star, inverses, target, features, temperature, permute, inverse_permute = res
    theta_dot, _ = g_dot
    nx, _ = theta.shape

    g_x = jax.jacobian(lambda _theta : equality_constraints(p, _theta, nx))(theta.flatten()) # theta primals can be zero since constraints are linear wrt theta

    def L_xp(cotangent : List[jnp.ndarray]):

        def L_x(p, _theta, _permute, _inverse_permute, _nonzero_cols, _target):
            # gradient of lagrangian wrt to theta
            A = _data_matrix(p, features, temperature)
            A = A[:, _permute]
            
            m, n = A.shape
            _A = jax.lax.slice(A, (0, 0), (m, _nonzero_cols))
            _A = jnp.block([_A, jnp.zeros((m, n - _nonzero_cols))])[:, _inverse_permute]
            return - _A.T @ _target + _A.T @ (_A @ _theta)
        
        Lxp_vjp = jax.tree_util.tree_map(
            lambda _theta, _perm, _inv_perm, _nzero, _tar : jax.vjp(lambda _p : L_x(_p, _theta, _perm, _inv_perm, _nzero, _tar), p)[-1], 
            [*theta], [*permute], [*inverse_permute], [*nonzero_cols], [*target]
        )

        lxp = jax.tree_util.tree_reduce(
            operator.add,
            [_Lxp_vjp(_cotan)[0] for _Lxp_vjp, _cotan in zip(Lxp_vjp, cotangent)]
        )

        _, gxp_vjp = jax.vjp(lambda p : jax.vjp(lambda _theta : equality_constraints(p, _theta, nx), theta.flatten())[-1](lam_star)[0], p)
        return lxp + gxp_vjp(jnp.concatenate(cotangent))[0]

    _, gp_vjp = jax.vjp(lambda _p : equality_constraints(_p, theta.flatten(), nx), p)
    
    cotangent_Lxp = jax.vmap(lambda _g_dot, _inv : _inv @ _g_dot)(theta_dot, inverses) # left inverse == right inverse for a symmetric matrix
    dx_dp = - L_xp([*cotangent_Lxp])

    cotangent_gp = jnp.linalg.solve(
        jnp.sum(jax.vmap(lambda _g_x, _inv : _g_x @ _inv @ _g_x.T)(jnp.stack(jnp.array_split(g_x, nx, axis = 1)), inverses), axis = 0), 
        g_x @ cotangent_Lxp.flatten()
    )
    
    dx_dp -= gp_vjp(cotangent_gp)[0]

    cotangent_Lxp = cotangent_gp @ jnp.column_stack(jax.vmap(lambda _g_x, _inv : _g_x @ _inv)(jnp.stack(jnp.array_split(g_x, nx, axis = 1)), inverses))
    dx_dp += L_xp(jnp.array_split(cotangent_Lxp, nx))

    return dx_dp, None, None, None, None, None


_optimal_parameters.defvjp(_optimal_parameters_fwd, _optimal_parameters_bwd)

degree = 2

def poly(x, t):
    """
    return jnp.concatenate([
        jnp.array([*map(lambda _x : reduce(operator.mul, _x), itertools.combinations_with_replacement(x, i))]) for i in range(1, degree + 1)
    ])"""
    return jnp.array([
        x[0], x[1], x[2], x[3], 
        
        x[0]**2, x[0]*x[1], x[0]*x[2], x[0]*x[3], 
        x[1]**2, x[1]*x[2], x[1]*x[3],
        x[2]**2, x[2]*x[3],
        x[3]**2,
        
        # x[0]**3, x[0]**2*x[1], x[0]**2*x[2], x[0]**2*x[3],
        # x[1]**3, x[1]**2*x[0], x[1]**2*x[2], x[1]**2*x[3],
        # x[2]**3, x[2]**2*x[0], x[2]**2*x[1], x[2]**2*x[3],
        # x[3]**3, x[3]**2*x[0], x[3]**2*x[1], x[3]**2*x[2],
        # x[0]*x[1]*x[2], x[0]*x[1]*x[3], x[1]*x[2]*x[3], x[0]*x[2]*x[3], 

        # x[0]**4, x[0]**3*x[1], x[0]**3*x[2], x[0]**3*x[3], 
        # x[0]**2*x[1]**2, x[0]**2*x[1]*x[2], x[0]**2*x[1]*x[3], x[0]**2*x[2]**2, x[0]**2*x[2]*x[3], x[0]**2*x[3]**2,
        # x[0]*x[1]**3, x[0]*x[1]**2*x[2], x[0]*x[1]**2*x[3], x[0]*x[1]*x[2]**2, x[0]*x[1]*x[2]*x[3], x[0]*x[1]*x[3]**2, x[0]*x[2]**3,
        # x[0]*x[2]**2*x[3], x[0]*x[2]*x[3]**2, x[0]*x[3]**3, x[1]**4, x[1]**3*x[2], x[1]**3*x[3], x[1]**2*x[2]**2, x[1]**2*x[2]*x[3], 
        # x[1]**2*x[3]**2, x[1]*x[2]**3, x[1]*x[2]**2*x[3], x[1]*x[2]*x[3]**2, x[1]*x[3]**3, x[2]**4, x[2]**3*x[3], x[2]**2*x[3]**2, x[2]*x[3]**3, x[3]**4
        
    ])

def poly_interp(z, t, x):
    """
    return jnp.concatenate([
        jnp.array([*map(lambda _x : reduce(lambda accum, value : accum * value(t), _x, 1.), itertools.combinations_with_replacement(x, i))]) for i in range(1, degree + 1)
    ])"""
    return jnp.array([
        x[0](t), x[1](t), x[2](t), x[3](t), 

        x[0](t)**2, x[0](t)*x[1](t), x[0](t)*x[2](t), x[0](t)*x[3](t), 
        x[1](t)**2, x[1](t)*x[2](t), x[1](t)*x[3](t),
        x[2](t)**2, x[2](t)*x[3](t),
        x[3](t)**2,
        
        # x[0](t)**3, x[0](t)**2*x[1](t), x[0](t)**2*x[2](t), x[0](t)**2*x[3](t),
        # x[1](t)**3, x[1](t)**2*x[0](t), x[1](t)**2*x[2](t), x[1](t)**2*x[3](t),
        # x[2](t)**3, x[2](t)**2*x[0](t), x[2](t)**2*x[1](t), x[2](t)**2*x[3](t),
        # x[3](t)**3, x[3](t)**2*x[0](t), x[3](t)**2*x[1](t), x[3](t)**2*x[2](t),
        # x[0](t)*x[1](t)*x[2](t), x[0](t)*x[1](t)*x[3](t), x[1](t)*x[2](t)*x[3](t), x[0](t)*x[2](t)*x[3](t),

        # x[0](t)**4, x[0](t)**3*x[1](t), x[0](t)**3*x[2](t), x[0](t)**3*x[3](t), 
        # x[0](t)**2*x[1](t)**2, x[0](t)**2*x[1](t)*x[2](t), x[0](t)**2*x[1](t)*x[3](t), x[0](t)**2*x[2](t)**2, x[0](t)**2*x[2](t)*x[3](t), x[0](t)**2*x[3](t)**2,
        # x[0](t)*x[1](t)**3, x[0](t)*x[1](t)**2*x[2](t), x[0](t)*x[1](t)**2*x[3](t), x[0](t)*x[1](t)*x[2](t)**2, x[0](t)*x[1](t)*x[2](t)*x[3](t), x[0](t)*x[1](t)*x[3](t)**2, x[0](t)*x[2](t)**3,
        # x[0](t)*x[2](t)**2*x[3](t), x[0](t)*x[2](t)*x[3](t)**2, x[0](t)*x[3](t)**3, x[1](t)**4, x[1](t)**3*x[2](t), x[1](t)**3*x[3](t), x[1](t)**2*x[2](t)**2, x[1](t)**2*x[2](t)*x[3](t), 
        # x[1](t)**2*x[3](t)**2, x[1](t)*x[2](t)**3, x[1](t)*x[2](t)**2*x[3](t), x[1](t)*x[2](t)*x[3](t)**2, x[1](t)*x[3](t)**3, x[2](t)**4, x[2](t)**3*x[3](t), x[2](t)**2*x[3](t)**2, x[2](t)*x[3](t)**3, x[3](t)**4
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

    # TODO get theta from an optimization problem. Equality constraint convex problem
    """
    theta = jnp.vstack(
        jax.tree_util.tree_map(
            lambda deri_iter, perm_iter, inv_perm_iter, cols_iter : _optimal_parameters(p, deri_iter, features, temperature, perm_iter, inv_perm_iter, cols_iter, regularization)[0], 
            [*jnp.vstack(derivatives).T], [*permute], [*inverse_permute], [*nonzero_cols]
        )
    )"""

    theta, _ = _optimal_parameters(p, jnp.vstack(derivatives).T, features, temperature, permute, inverse_permute, nonzero_cols, regularization)

    solution = jax.vmap(lambda xi, ti : odeint(_foo, xi, time_span, ti, theta, p))(xinit, temperature)
    _loss = jax.vmap(lambda pred, meas : jnp.mean((pred - meas)**2))(solution, states)
    return jnp.mean(_loss), theta

len_features = len(poly(xinit[0], 0.))
nonzero_cols = jnp.array([2, 2, 1, 1], dtype = int).tolist() # number of nonzero columns
permute = jnp.array([0, 1, 2, 3, 0, 1, 2, 3, 2, 0, 1, 3, 2, 0, 1, 3], dtype = int).reshape(nx, -1) # big indexes followed by small indexes
inverse_permute = jnp.array([0, 1, 2, 3, 0, 1, 2, 3, 1, 2, 0, 3, 1, 2, 0, 3], dtype = int).reshape(nx, -1) # reverse the permutation
p_guess = jnp.ones(len_features)
theta_guess = jnp.ones((nx, len_features))

# SINDy features
sindy_features = jax.vmap(jax.vmap(lambda xi : poly(xi, 0.)))(solution)

# DF-SINDY features
dfsindy_features = jnp.stack([scipy_odeint(poly_interp, poly(_xi, 0), time_span, args = (_interp, )) - poly(_xi, 0) for _xi, _interp in zip(xinit, interpolations)])

# loss, linear_params = simple_objective(p_guess, solution, sindy_features, estimated_derivatives, permute, inverse_permute, nonzero_cols)
# gradients, aux = jax.grad(simple_objective, has_aux = True)(p_guess, solution, sindy_features, estimated_derivatives, permute, inverse_permute, nonzero_cols)

# _simple_obj = jax.jit(lambda p : simple_objective(p, solution, sindy_features, estimated_derivatives, permute, inverse_permute, nonzero_cols)[0])
# _simple_jac = jax.jit(jax.grad(_simple_obj))
# _simple_hess = jax.jit(jax.jacrev(_simple_jac))

"""
### checking gradients with finite difference
eps = 1e-5 
fd0 = (_simple_obj(p_guess + jnp.array([eps, 0., 0., 0.])) - loss) / eps
fd1 = (_simple_obj(p_guess + jnp.array([0., eps, 0., 0.])) - loss) / eps
fd2 = (_simple_obj(p_guess + jnp.array([0., 0., eps, 0.])) - loss) / eps
fd3 = (_simple_obj(p_guess + jnp.array([0., 0., 0., eps])) - loss) / eps

print("gradients fd", fd0, fd1, fd2, fd3)
print("gradients autodiff", gradients)
"""

def outer_objective(p_guess, theta_guess, solution, features, estimated_derivatives, thresholding = 0.1, maxiter = 10):
    # implement sequential threshold least square algorithm
    # TODO previous subproblems can have lower tolerance or less maximum iterations to eliminate the terms. How to choose initial tolerance ?
    # TODO how to eliminate coefficients that have equality constraints dependant on others coefficients that are kept
    # eg : constraint x1 + x2 = 0. If x1 has to be kept and x2 is neglected then how to remove x2 if the constraint forces x2 to be -x1

    def get_permutations(theta) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, List[jnp.ndarray]]:

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

# compare with nonlinear optimization on derivatives

def derivative_objective(p, state_derivatives, states, mask, regularization = 0.01):
    # states shape = (E, T, n) # no of experiments X time points X states 
    # derivatives shape = (E, T, n) # no of experiments X time points X states 
    # features shape = (E, T, F) # no of experiments X time points X no of features
    # p shape = (F) # no of features

    theta = p[: -len_features].reshape(nx, -1)
    theta = theta * mask
    p = p[-len_features :]
    
    solution = jax.vmap(lambda xi, ti : jax.vmap(lambda _xi : _foo(_xi, 0., ti, theta, p))(xi))(states, temperature)
    _loss = jax.vmap(lambda pred, meas : jnp.mean((pred - meas)**2))(solution, state_derivatives)
    return jnp.mean(_loss) + regularization * jnp.linalg.norm(theta), theta

p_guess = jnp.ones((nx + 1)*len_features)
mask = jnp.array([
    1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
    1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
    0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
    0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
]).reshape(nx, -1)
derivative_loss = derivative_objective(p_guess, estimated_derivatives, solution, mask)


def outer_derivative_objective(p_guess, solution, estimated_derivatives, thresholding = 0.1, maxiter = 10):
    # implement sequential threshold least square algorithm
    # TODO previous subproblems can have lower tolerance or less maximum iterations to eliminate the terms. How to choose initial tolerance ?
    # TODO how to eliminate coefficients that have equality constraints dependant on others coefficients that are kept
    # eg : constraint x1 + x2 = 0. If x1 has to be kept and x2 is neglected then how to remove x2 if the constraint forces x2 to be -x1

    def get_mask(theta) -> jnp.ndarray :

        _theta = theta.reshape(-1)
        ind = jnp.arange(len(_theta), dtype = int)
        big_ind = jnp.abs(_theta) >= thresholding
        mask = jnp.zeros_like(_theta, dtype = int) # used to stop iterations
        mask = mask.at[ind[big_ind]].set(1)
        
        return mask.reshape(*theta.shape)

    iteration = 0
    theta = p_guess[: -len_features].reshape(nx, -1)

    while iteration < maxiter : 

        mask = get_mask(theta)
        print(f"Iteration {iteration}, linear parameters {theta}")
        
        if iteration == 0 : prev_mask = mask
        if iteration > 0 and jnp.allclose(mask, prev_mask) : 
            print("Optimal solution found")
            break
        
        _simple_obj = jax.jit(lambda p : derivative_objective(p, estimated_derivatives, solution, mask)[0])
        _simple_jac = jax.jit(jax.grad(_simple_obj))
        _simple_hess = jax.jit(jax.jacrev(_simple_jac))

        print("Starting optimization")
        solution_object = minimize_ipopt(
            _simple_obj, 
            x0 = p_guess, 
            jac = _simple_jac,
            hess = _simple_hess,  
            tol = 1e-7, 
            options = {"maxiter" : 100, "print_level" : 5}
            )
        
        print(solution_object)
        p = jnp.array(solution_object.x)
        loss, theta = derivative_objective(p, estimated_derivatives, solution, mask)
        print(f"Iteration {iteration} : loss {loss}")

        prev_mask = mask
        iteration += 1

    return solution_object

# solution_object = outer_derivative_objective(p_guess, solution, estimated_derivatives, thresholding = 0.1, maxiter = 20)