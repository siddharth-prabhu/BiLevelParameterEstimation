import os
from typing import Callable, Tuple, List, Any
from functools import partial

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from jax import flatten_util, tree_util
from cyipopt import minimize_ipopt
import diffrax
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

# Multiplies the svd decomposition of inverse of a matrix with a vector (v)
def inv_vp(u, sinv, vh, v) : return vh.T @ ((u.T @ v) * sinv)

##########################################################################################################################################################
# Differentiable optimization with equality constraints
# For convex quadratic inner optimization problem
def _differentiable_optimization(f : Callable, g : Callable, p : jnp.ndarray, x_guess : jnp.ndarray, args : Tuple[jnp.ndarray]) -> Tuple[jnp.ndarray] : 
    # f = objective function arguments (p, x) -> scalar output
    # g = equality constraints arguments (p, x) -> vector output
    # p = nonlinear decision variables shape (np, ) # atleast 1d
    # x_guess = linear decision variables shape (nx, ) # atleast 1d 

    # F = number of features (columns) in the library
    # g = number of equality constraints

    x_guess, unravel = flatten_util.ravel_pytree(x_guess)
    _f = lambda p, x : f(p, unravel(x), *args)
    _g = lambda p, x : g(p, unravel(x))

    v_guess = jnp.zeros_like(_g(p, x_guess))
    eps = 10. * max(*x_guess.shape, 1) * jnp.array(jnp.finfo(x_guess.dtype).eps)

    def L(p, x, v) : return _f(p, x) + v @ _g(p, x) # Lagrangian of equality constraint optimization problem

    # jvp / vjp of equality constraints
    gx_jvp = jax.linearize(lambda x : _g(p, x), x_guess)[-1]
    gx_vjp = lambda ct : jax.vjp(lambda x : _g(p, x), x_guess)[-1](ct)[0]

    L_x, L_v = jax.grad(L, argnums = (1, 2))(p, x_guess, v_guess)
    
    L_xx = jax.hessian(L, argnums = 1)(p, x_guess, v_guess)
    (u, s, vh) = jnp.linalg.svd(L_xx, hermitian = True, full_matrices = False) 
    sinv = jnp.where(s <= eps * jnp.max(s, initial = -jnp.inf), 0., 1/s) # initial value is provided to deal with zero dimensional arrays

    (gu, gs, gvh) = jnp.linalg.svd(jax.vmap(gx_jvp)(vh).T @ jnp.diag(sinv) @ jax.vmap(gx_jvp)(u.T), hermitian = True, full_matrices = False)
    gsinv = jnp.where(gs <= eps * jnp.max(gs, initial = -jnp.inf), 0., 1/gs) # initial value is provided to deal with zero dimensional arrays

    dv = inv_vp(gu, gsinv, gvh, L_v + gx_jvp(inv_vp(u, sinv, vh, - L_x))) # shape = (g, )
    x_opt = x_guess - inv_vp(u, sinv, vh, L_x + gx_vjp(dv)) # shape = (nx, )

    return (unravel(x_opt), v_guess + dv), (u, sinv, vh, gu, gsinv, gvh) # (optimal x, optimal Lagrange variables), inverse of hessian of Lagrangian


# Forward- and reverse-mode autodiff compatible differentiable optimization with equality constraints
@partial(jax.custom_jvp, nondiff_argnums = (0, 1))
def differentiable_optimization(f : Callable, g : Callable, p : jnp.ndarray, x_guess : jnp.ndarray, args : Tuple[jnp.ndarray]) -> Tuple[jnp.ndarray] :
    return _differentiable_optimization(f, g, p, x_guess, args)

@differentiable_optimization.defjvp
def differentiable_optimization_fwd(f : Callable, g : Callable, primals : Tuple[jnp.ndarray], tangents : Tuple[jnp.ndarray]):
    
    p, x_guess, args = primals
    p_dot, *_ = tangents
    
    # Note that reusing the inverses from forward pass gives incorrect higher order derivatives (> 1). 
    # However, in BiLevel optimization, terms containing higer order derivatives are canceled out.
    # Therefore, such a scheme works and also saves on recomputing the inverses again
    (x_opt, v_opt), (u, sinv, vh, gu, gsinv, gvh) = differentiable_optimization(f, g, p, x_guess, args)

    x_opt, unravel = flatten_util.ravel_pytree(x_opt)
    _f = lambda p, x : f(p, unravel(x), *args)
    _g = lambda p, x : g(p, unravel(x))

    gx_jvp = jax.linearize(lambda x : _g(p, x), x_opt)[-1]
    gx_vjp = lambda ct : jax.vjp(lambda x : _g(p, x), x_opt)[-1](ct)[0]
    
    def L(p, x, v) : return _f(p, x) + v @ _g(p, x) # Lagrangian of equality constraint optimization problem

    v = jax.tree_util.tree_map(
        jnp.negative, 
        [
            jax.jvp(lambda _p : jax.grad(L, argnums = 1)(_p, x_opt, v_opt), (p, ), (p_dot, ))[-1], 
            jax.jvp(lambda _p : _g(_p, x_opt), (p, ), (p_dot, ))[-1],
        ])
    
    mu_v = inv_vp(gu, gsinv, gvh, gx_jvp(inv_vp(u, sinv, vh, v[0])) - v[1]) # shape = (g, )
    mu_x = inv_vp(u, sinv, vh, v[0] - gx_vjp(mu_v)) # shape = (nx, )
    
    return ((unravel(x_opt), v_opt), (u, sinv, vh, gu, gsinv, gvh)), ((unravel(mu_x), mu_v), jax.tree_util.tree_map(jnp.zeros_like, (u, sinv, vh, gu, gsinv, gvh)))


##########################################################################################################################################################
# For general convex inner optimization problem
# Differentiable optimization with equality and inequality constraints
def _constraint_differentiable_optimization(f : Callable, g : Callable, h : Callable, p : jnp.ndarray, x_guess : jnp.ndarray, args : Tuple[jnp.ndarray]) -> Tuple[jnp.ndarray] :
    # f = objective function arguments (p, x) -> scalar output (assumed as only reverse mode compatible) shape = ()
    # g = equality constraints arguments (p, x) -> vector output (should be forward and reverse mode compatible) shape = (ng, )
    # h = inequality constraints (h >= 0) arguments (p, x) -> vector output (should be forward and reverse mode compatible) shape = (h, )

    # p = nonlinear decision variables
    # x_guess = linear decision variables 
    x_flat, unravel = flatten_util.ravel_pytree(x_guess)
    eps = 10. * max(*x_flat.shape, 1) * jnp.array(jnp.finfo(x_flat.dtype).eps)

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
        if not res.success : print("Inner optimization problem failed with message : ", res.message)
        x = jnp.array(res.x) # optimal primal variables
        v, m = map(jnp.array, (res.info["mult_g"][:ng], res.info["mult_g"][ng:])) # optimal dual variables corresponding to equality constraints and inequality constraints respectively
        return x, v, m

    x_opt, v_opt, m_opt = jax.pure_callback(
        _minimize, 
        (jax.ShapeDtypeStruct(x_flat.shape, x_flat.dtype), jax.ShapeDtypeStruct((ng, ), x_flat.dtype), jax.ShapeDtypeStruct((nh, ), x_flat.dtype)), 
        x_flat, p, args
    )

    def L(p, x, v, m) : return f(p, unravel(x), *args) + v @ g(p, unravel(x)) + m @ h(p, unravel(x)) # Lagrange function

    gx_jvp = jax.linearize(lambda x : g(p, unravel(x)), x_opt)[-1]
    hx_vjp = lambda ct, p : jax.vjp(lambda x : h(p, unravel(x)), x_opt)[-1](ct)[0]

    B_xx = jax.hessian(L, argnums = 1)(p, x_opt, v_opt, m_opt) - jax.vmap(hx_vjp, in_axes = (0, None))(jax.vmap(hx_vjp, in_axes = (0, None))(jnp.diag(m_opt / h(p, unravel(x_opt))), p).T, p)
    (u, s, vh) = jnp.linalg.svd(B_xx, hermitian = True, full_matrices = False) # shape = (nx, nx), (nx, nx), (nx, nx)
    sinv = jnp.where(s <= eps * jnp.max(s, initial = -jnp.inf), 0., 1/s) # initial value is provided to deal with zero dimensional arrays

    (gu, gs, gvh) = jnp.linalg.svd(jax.vmap(gx_jvp)(vh).T @ jnp.diag(sinv) @ jax.vmap(gx_jvp)(u.T), hermitian = True, full_matrices = False)
    gsinv = jnp.where(gs <= eps * jnp.max(gs, initial = -jnp.inf), 0., 1/gs) # initial value is provided to deal with zero dimensional arrays

    return (unravel(x_opt), v_opt, m_opt), (u, sinv, vh, gu, gsinv, gvh) # (optimal primal variables, optimal dual variables)

"""
# Reverse-mode autdiff compatible differentiable optimization with equality and inequality constraints
@partial(jax.custom_vjp, nondiff_argnums = (0, 1, 2))
def constraint_differentiable_optimization_rev(f : Callable, g : Callable, h : Callable, p : jnp.ndarray, x_guess : jnp.ndarray, args : Tuple[jnp.ndarray]) -> Tuple[jnp.ndarray] :
    return _constraint_differentiable_optimization(f, g, h, p, x_guess, args)

def constraint_differentiable_optimization_rev_fwd(f : Callable, g : Callable, h : Callable, p : jnp.ndarray, x_guess : jnp.ndarray, args : Tuple[jnp.ndarray]) -> Tuple[jnp.ndarray] :
    optimal_solution = constraint_differentiable_optimization_rev(f, g, h, p, x_guess, args)
    return optimal_solution, (optimal_solution, p, args)

def constraint_differentiable_optimization_rev_bwd(f : Callable, g : Callable, h : Callable, res : Tuple[jnp.ndarray], g_dot : Tuple[jnp.ndarray]) -> Tuple[jnp.ndarray] :
    
    (x_dot, v_dot, m_dot), _ = g_dot
    ((x_opt, v_opt, m_opt), (u, sinv, vh, gu, gsinv, gvh)), p, args = res
    
    x_opt, unravel = flatten_util.ravel_pytree(x_opt)
    x_dot, _ = flatten_util.ravel_pytree(x_dot)

    _f = lambda p, x : f(p, unravel(x), *args)
    _g = lambda p, x : g(p, unravel(x))
    _h = lambda p, x : h(p, unravel(x))
    
    gx_jvp = jax.linearize(lambda x : _g(p, x), x_opt)[-1]
    gx_vjp = lambda ct : jax.vjp(lambda x : _g(p, x), x_opt)[-1](ct)[0]
    gp_vjp = lambda ct : jax.vjp(lambda _p : _g(_p, x_opt), p)[-1](ct)[0]

    hx_vjp = lambda ct, p : jax.vjp(lambda x : _h(p, x), x_opt)[-1](ct)[0]
    hp_vjp = lambda ct : jax.vjp(lambda _p : m_opt * _h(_p, x_opt), p)[-1](ct)[0]

    def L(p, x, v, m) : return _f(p, x) + v @ _g(p, x) + m @ _h(p, x) # Lagrange function

    # Find the cotangents
    x_hat_dot = x_dot - hx_vjp(m_dot * m_opt / _h(p, x_opt), p)
    mu_v = inv_vp(gu, gsinv, gvh, v_dot - gx_jvp(inv_vp(u, sinv, vh, x_hat_dot))) # shape = (g, )
    mu_x = - inv_vp(u, sinv, vh, x_hat_dot + gx_vjp(mu_v)) # shape = (nx, )
    mu_m = - (m_dot + jax.jvp(lambda x : _h(p, x), (x_opt, ), (mu_x, ))[-1]) / _h(p, x_opt) # shape = (h, )

    # L_zp = d([Lx, Lv])/dp
    _, f_Lzp = jax.vjp(lambda _p : jax.grad(L, argnums = (1, 2))(_p, x_opt, v_opt, m_opt), p)
    _, f_rxp = jax.vjp(lambda _p : hx_vjp(m_opt, _p), p) # residual vjp wrt p

    return f_Lzp((mu_x, mu_v))[0] - f_rxp(mu_x)[0] + gp_vjp(mu_v) + hp_vjp(mu_m), None, None

constraint_differentiable_optimization_rev.defvjp(constraint_differentiable_optimization_rev_fwd, constraint_differentiable_optimization_rev_bwd)
"""

# Forward- and reverse-mode autodiff compatible differentiable optimization with equality and inequality constriants
@partial(jax.custom_jvp, nondiff_argnums = (0, 1, 2))
def constraint_differentiable_optimization(f : Callable, g : Callable, h : Callable, p : jnp.ndarray, x_guess : jnp.ndarray, args : Tuple[jnp.ndarray]) -> Tuple[jnp.ndarray] : 
    return _constraint_differentiable_optimization(f, g, h, p, x_guess, args)

@constraint_differentiable_optimization.defjvp
def constraint_differentiable_optimization_fwd(f : Callable, g : Callable, h : Callable, primals : Tuple[jnp.ndarray], tangents : Tuple[jnp.ndarray]) -> Tuple[jnp.ndarray] :
    p, x_guess, args = primals
    p_dot, _, _ = tangents

    _, aux = (x_opt, v_opt, m_opt), (u, sinv, vh, gu, gsinv, gvh) = constraint_differentiable_optimization(f, g, h, p, x_guess, args)
    x_opt, unravel = flatten_util.ravel_pytree(x_opt)

    _f = lambda p, x : f(p, unravel(x), *args)
    _g = lambda p, x : g(p, unravel(x))
    _h = lambda p, x : h(p, unravel(x))
    
    gx_jvp = jax.linearize(lambda x : _g(p, x), x_opt)[-1]
    gx_vjp = lambda ct : jax.vjp(lambda x : _g(p, x), x_opt)[-1](ct)[0]
    hx_vjp = lambda ct, p : jax.vjp(lambda x : _h(p, x), x_opt)[-1](ct)[0]

    def L(p, x, v, m) : return _f(p, x) + v @ _g(p, x) + m @ _h(p, x) # Lagrange function

    v = jax.tree_util.tree_map(
        jnp.negative, 
        [
            jax.jvp(lambda _p : jax.grad(L, argnums = 1)(_p, x_opt, v_opt, m_opt), (p, ), (p_dot, ))[-1], 
            jax.jvp(lambda _p : _g(_p, x_opt), (p, ), (p_dot, ))[-1],
            jax.jvp(lambda _p : m_opt * _h(_p, x_opt), (p, ), (p_dot, ))[-1]
        ])

    v_hat = v[0] - hx_vjp(v[2] / _h(p, x_opt), p)
    mu_v = inv_vp(gu, gsinv, gvh, gx_jvp(inv_vp(u, sinv, vh, v_hat)) - v[1]) # shape = (g, )
    mu_x = inv_vp(u, sinv, vh, v_hat - gx_vjp(mu_v)) # shape = (nx, )
    mu_m = (v[2] - m_opt * jax.jvp(lambda x : _h(p, x), (x_opt, ), (mu_x, ))[-1]) / _h(p, x_opt) # shape = (h, )

    return ((unravel(x_opt), v_opt, m_opt), aux), ((unravel(mu_x), mu_v, mu_m), tree_util.tree_map(jnp.zeros_like, aux))


##########################################################################################################################################################
def odeint_diffrax(afunc : Callable, xinit : jnp.ndarray, time_span : jnp.ndarray, parameters : Any, rtol = 1e-6, atol = 1e-8, mxstep = 10_000) -> jnp.ndarray :
    # Forward and reverse mode autodiff compatible ode solver
    _afunc = lambda t, x, p : afunc(x, t, p)
    return diffrax.diffeqsolve(
                diffrax.ODETerm(_afunc), 
                diffrax.Tsit5(),
                t0 = time_span[0], # make sure that initial conditions are at time_span[0]
                t1 = time_span[-1],
                dt0 = None, 
                saveat = diffrax.SaveAt(ts = time_span), 
                y0 = xinit, 
                args = parameters,
                stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol, pcoeff = 0.4, icoeff = 0.3, dcoeff = 0.),
                adjoint = diffrax.DirectAdjoint(), 
                max_steps = mxstep
        ).ys


def plot_coefficients(params_actual : List[jnp.ndarray], params_obtained : List[jnp.ndarray], labels : List[List[str]], filename : str, path : str = ".") -> None :

    assert len(params_actual) == len(params_obtained) == len(labels), "Should have same number of sets of parameters"
    
    np = max(map(len, params_actual))
    alen = len(params_actual)
    width = 0.4

    with plt.style.context(["science", "notebook", "bright"]):
        
        # figsize = (width, height)
        fig, ax =  plt.subplots(alen, 1, figsize = (1.5 * np, 4 * alen), gridspec_kw = {"wspace" : 0.3})
        
        for i, (p_actual, p_obtain, label) in enumerate(zip(params_actual, params_obtained, labels)) :
            
            x = jnp.arange(len(p_actual))
            rects = ax[i].bar(x, p_actual, label = "Actual", width = width)
            ax[i].bar_label(rects, [f"{round(rect.get_height(), 2)}" if rect.get_height() != 0 else "" for rect in rects], padding = 3)
            
            rects = ax[i].bar(x + width, p_obtain, label = "Predicted", width = width)
            ax[i].bar_label(rects, [f"{round(rect.get_height(), 2)}" if rect.get_height() != 0 else "" for rect in rects], padding = 3)

            ax[i].margins(y = 0.2)
            ax[i].set_xticks(x + width / 2, label)
            ax[i].legend()

        fig.tight_layout()
        plt.savefig(os.path.join(path, filename), bbox_inches = "tight")
        plt.close()

def plot_trajectories(solution : jnp.ndarray, prediction : jnp.ndarray, time_span : jnp.ndarray, ncols : int, filename : str, path : str = ".") -> None :

    assert solution.shape == prediction.shape, "List of solutions should have the same shape"
    _, nx = solution.shape

    with plt.style.context(["science", "notebook", "bright"]):

        nrows = (nx + ncols - 1) // ncols
        fig, ax =  plt.subplots(nrows, ncols, figsize = (7 * ncols, 5 * nrows), gridspec_kw = {"wspace" : 0.2})
        ax = ax.ravel()

        for i in range(nx) :
            ax[i].plot(time_span, solution[:, i], "o", label = "Actual")
            ax[i].plot(time_span, prediction[:, i], label = "Predicted")                    
            ax[i].set(ylabel = r"$x_" + f"{i}$", xlabel = "Time")
            ax[i].legend()

        fig.tight_layout()
        plt.savefig(os.path.join(path, filename))
        plt.close()