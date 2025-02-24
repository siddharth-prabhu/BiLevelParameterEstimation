from typing import Callable, Tuple
from functools import partial

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import jax.scipy as jsp
from jax import flatten_util
from cyipopt import minimize_ipopt

# Multiplies the svd decomposition of inverse of a matrix with a vector (v)
def inv_vp(u, sinv, vh, v) : return vh.T @ ((u.T @ v) * sinv)

# Differentiable regression with only equality constraints
@partial(jax.custom_vjp, nondiff_argnums = (0, 1))
def differentiable_regression(f : Callable, g : Callable, p : jnp.ndarray, x_guess : jnp.ndarray, args : Tuple[jnp.ndarray]) -> Tuple[jnp.ndarray] :

    # f = objective function arguments (p, x) -> scalar output
    # g = equality constraints arguments (p, x) -> vector output
    # p = nonlinear decision variables shape (nx, F) # atleast 2d
    # x_guess = linear decision variables shape (nx, x) # atleast 2d 

    # F = number of features (columns) in the library
    # g = number of equality constraints

    # p and x may not have same last dimensions. However they should have the same first dimension 
    nx, F = p.shape
    _, xF = x_guess.shape

    v_guess = jnp.zeros_like(g(p, x_guess))
    eps = 10. * max(nx, F, 1) * jnp.array(jnp.finfo(p.dtype).eps)

    # p : shape (nx * F, 1), x : shape = (nx, F), v : shape = (g, )
    def L(p, x, v) : return f(p, x, *args) + v @ g(p, x) # Lagrangian of equality constraint optimization problem

    L_x, L_v = jax.grad(L, argnums = (1, 2))(p, x_guess, v_guess) # shape = (nx, F), (g, )
    
    # Since equality constraints are linear in x, the Hessian of Lagrangian is the Hessian of the objective function. 
    # Therefore, parallel implementation of Block diagonal hessians can be exploited
    Lxx_jvp = lambda t : jax.jvp(jax.grad(lambda x : L(p, x, v_guess)), (x_guess, ), (t, ))[-1]
    T = jax.vmap(lambda t : jnp.tile(t, (nx, 1)))(jnp.eye(xF)) # create tangent vector. shape = (F, nx, F)
    L_xx = jnp.einsum("ijk->jik", jax.vmap(Lxx_jvp)(T)) # shape = (nx, F, F)

    # jvp / vjp of equality constraints
    gx_jvp = lambda t : jax.jvp(lambda x : g(p, x.reshape(nx, -1)), (x_guess.flatten(), ), (t, ))[-1]
    gx_vjp = lambda ct : jax.vjp(lambda x : g(p, x), x_guess)[-1](ct)[0]

    # Take pinv of block diagonal matrices in hessian of Lagrangian
    (u, s, vh) = jax.vmap(partial(jnp.linalg.svd, hermitian = True))(L_xx) # shape = (nx, F, F), (nx, F), (nx, F, F)
    sinv = jnp.where(s <= eps * jnp.max(s, initial = -jnp.inf), 0., 1/s) # initial value is provided to deal with zero dimensional arrays

    (gu, gs, gvh) = jnp.linalg.svd(jax.vmap(gx_jvp)(jsp.linalg.block_diag(*vh)).T @ jnp.diag(sinv.flatten()) @ jax.vmap(gx_jvp)(jsp.linalg.block_diag(*u).T), hermitian = True)
    gsinv = jnp.where(gs <= eps * jnp.max(gs, initial = -jnp.inf), 0., 1/gs) # initial value is provided to deal with zero dimensional arrays

    dv = inv_vp(gu, gsinv, gvh, L_v - gx_jvp(jax.vmap(inv_vp)(u, sinv, vh, L_x).flatten())) # shape = (g, )
    x_opt = x_guess - jax.vmap(inv_vp)(u, sinv, vh, L_x + gx_vjp(dv)) # shape = (nx, F)

    return (x_opt, v_guess + dv), (u, sinv, vh, gu, gsinv, gvh) # (optimal x, optimal Lagrange variables), inverse of hessian of Lagrangian

def differentiable_regression_fwd(f : Callable, g : Callable, p : jnp.ndarray, x_guess : jnp.ndarray, args : Tuple[jnp.ndarray]) -> Tuple[jnp.ndarray] :
    optimal_solution, inverse = differentiable_regression(f, g, p, x_guess, args)
    return (optimal_solution, inverse), (optimal_solution, inverse, p, args)

def differentiable_regression_bwd(f : Callable, g : Callable, res : Tuple[jnp.ndarray], g_dot : Tuple[jnp.ndarray]) -> Tuple[jnp.ndarray] :
    
    (x_dot, v_dot), _ = g_dot
    (x_opt, v_opt), (u, sinv, vh, gu, gsinv, gvh), p, args = res

    gx_jvp = lambda t : jax.jvp(lambda x : g(p, x), (x_opt, ), (t, ))[-1]
    _, gx_vjp = jax.vjp(lambda x : g(p, x), x_opt)
    
    def L(p, x, v) : return f(p, x, *args) + v @ g(p, x) # Lagrangian of equality constraint optimization problem

    mu_v = inv_vp(gu, gsinv, gvh, v_dot - gx_jvp(jax.vmap(inv_vp)(u, sinv, vh, x_dot))) # shape = (g, )
    mu_x = - jax.vmap(inv_vp)(u, sinv, vh, x_dot + gx_vjp(mu_v)[0]) # shape = (nx * F, )

    # L_zp = d([Lx, Lv])/dp
    _, f_Lzp = jax.vjp(lambda _p : jax.grad(L, argnums = (1, 2))(_p, x_opt, v_opt), p)

    return f_Lzp((mu_x, mu_v))[0], None, None

differentiable_regression.defvjp(differentiable_regression_fwd, differentiable_regression_bwd)


# Differentiable regression with equality and inequality constraints
@partial(jax.custom_vjp, nondiff_argnums = (0, 1, 2))
def constraint_differentiable_regression(f : Callable, g : Callable, h : Callable, p : jnp.ndarray, x_guess : jnp.ndarray, args : Tuple[jnp.ndarray]) -> Tuple[jnp.ndarray] :
    # f = objective function arguments (p, x) -> scalar output (assumed as only reverse mode compatible) shape = ()
    # g = equality constraints arguments (p, x) -> vector output (should be forward and reverse mode compatible) shape = (ng, )
    # h = inequality constraints (h >= 0) arguments (p, x) -> vector output (should be forward and reverse mode compatible) shape = (h, )

    # p = nonlinear decision variables
    # x_guess = linear decision variables 
    x_flat, unravel = flatten_util.ravel_pytree(x_guess)

    ng = len(g(p, x_guess)) # number of equality constraints
    nh = len(h(p, x_guess)) # number of inequality constraints

    def _minimize(x_flat, p, args):

        obj = jax.jit(lambda x : f(p, unravel(x), *args))
        jac = jax.jit(jax.grad(obj))
        hess = jax.jit(jax.hessian(obj)) # objective function is only reverse mode compatible ## TODO change

        _g = jax.jit(lambda x : g(p, unravel(x)))
        _h = jax.jit(lambda x : h(p, unravel(x)))

        res = minimize_ipopt(
            obj,
            jac = jac,
            hess = hess,
            x0 = x_flat, 
            constraints = [
                {"type" : "eq", "fun" : _g, "jac" : jax.jacobian(_g), "hess" : lambda x, lam : jax.hessian(lambda _x : lam @ _g(_x))(x)},
                {"type" : "ineq", "fun" : _h, "jac" : jax.jacobian(_h), "hess" : lambda x, lam : jax.hessian(lambda _x : lam @ _h(_x))(x)}
            ],
            tol = 1e-5, 
            options = {"maxiter" : 1000, "disp" : 0, "sb" : "yes"}
        )
        print("Inner optimization problem success : ", res.success)
        x = jnp.array(res.x) # optimal primal variables
        v, m = map(jnp.array, (res.info["mult_g"][:ng], res.info["mult_g"][ng:])) # optimal dual variables corresponding to equality constraints and inequality constraints respectively
        return x, v, m

    x_opt, v, m = jax.pure_callback(
        _minimize, 
        (jax.ShapeDtypeStruct(x_flat.shape, x_flat.dtype), jax.ShapeDtypeStruct((ng, ), x_flat.dtype), jax.ShapeDtypeStruct((nh, ), x_flat.dtype)), 
        x_flat, p, args
    )

    return unravel(x_opt), v, m # (optimal primal variables, optimal dual variables)

def constraint_differentiable_regression_fwd(f : Callable, g : Callable, h : Callable, p : jnp.ndarray, x_guess : jnp.ndarray, args : Tuple[jnp.ndarray]) -> Tuple[jnp.ndarray] :
    optimal_solution = constraint_differentiable_regression(f, g, h, p, x_guess, args)
    return optimal_solution, (optimal_solution, p, args)

def constraint_differentiable_regression_bwd(f : Callable, g : Callable, h : Callable, res : Tuple[jnp.ndarray], g_dot : Tuple[jnp.ndarray]) -> Tuple[jnp.ndarray] :
    
    x_dot, v_dot, m_dot = g_dot
    (x_opt, v_opt, m_opt), p, args = res
    eps = 10. * max(*x_opt.shape, 1) * jnp.array(jnp.finfo(x_opt.dtype).eps)
    
    x_opt, unravel = flatten_util.ravel_pytree(x_opt)
    x_dot, _ = flatten_util.ravel_pytree(x_dot)

    _f = lambda p, x : f(p, unravel(x), *args)
    _g = lambda p, x : g(p, unravel(x))
    _h = lambda p, x : h(p, unravel(x))
    
    gx_jvp = lambda t : jax.jvp(lambda x : _g(p, x), (x_opt, ), (t, ))[-1]
    gx_vjp = lambda ct : jax.vjp(lambda x : _g(p, x), x_opt)[-1](ct)[0]
    hx_vjp = lambda ct, p : jax.vjp(lambda x : _h(p, x), x_opt)[-1](ct)[0]

    def L(p, x, v, m) : return _f(p, x) + v @ _g(p, x) + m @ _h(p, x) # Lagrange function
    L_x = jax.grad(L, argnums = 1)

    # leverage block diagonal structure of hessian
    # B_xx = jax.jacrev(L_x, argnums = 1)(p, x_opt, v_opt, m_opt) - jax.vmap(hx_vjp, in_axes = (0, None))(jax.vmap(hx_vjp, in_axes = (0, None))(jnp.diag(m_opt / _h(p, x_opt)), p).T, p)
    B_xx = jax.hessian(L, argnums = 1)(p, x_opt, v_opt, m_opt) - jax.vmap(hx_vjp, in_axes = (0, None))(jax.vmap(hx_vjp, in_axes = (0, None))(jnp.diag(m_opt / _h(p, x_opt)), p).T, p)
    (u, s, vh) = jnp.linalg.svd(B_xx, hermitian = True) # shape = (nx, nx), (nx, nx), (nx, nx)
    sinv = jnp.where(s <= eps * jnp.max(s, initial = -jnp.inf), 0., 1/s) # initial value is provided to deal with zero dimensional arrays

    (gu, gs, gvh) = jnp.linalg.svd(jax.vmap(gx_jvp)(vh).T @ jnp.diag(sinv) @ jax.vmap(gx_jvp)(u.T), hermitian = True)
    gsinv = jnp.where(gs <= eps * jnp.max(gs, initial = -jnp.inf), 0., 1/gs) # initial value is provided to deal with zero dimensional arrays

    # Find the cotangents
    mu_v = inv_vp(gu, gsinv, gvh, v_dot - gx_jvp(inv_vp(u, sinv, vh, x_dot))) # shape = (g, )
    mu_x = - inv_vp(u, sinv, vh, x_dot + gx_vjp(mu_v)) # shape = (nx, )

    # L_zp = d([Lx, Lv])/dp
    _, f_Lzp = jax.vjp(lambda _p : jax.grad(L, argnums = (1, 2))(_p, x_opt, v_opt, m_opt), p)
    _, f_rxp = jax.vjp(lambda _p : hx_vjp(m_opt, _p), p) # residual vjp wrt p

    # vjp of (v + m) wrt p is zero
    return f_Lzp((mu_x, mu_v))[0] - f_rxp(mu_x)[0], None, None

constraint_differentiable_regression.defvjp(constraint_differentiable_regression_fwd, constraint_differentiable_regression_bwd)

