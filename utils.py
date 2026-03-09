import os
from typing import Callable, Tuple, List, Any, Iterable, Optional
import functools
import itertools as it

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from jax import flatten_util, tree_util
from cyipopt import minimize_ipopt
from scipy.interpolate import CubicSpline as SCubicSpline
import diffrax
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True


##########################################################################################################################################################
# Multiplies the svd decomposition of inverse of a matrix with a vector (v)
def inv_vp(u, sinv, vh, v) : return vh.T @ ((u.T @ v) * sinv)

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
    # sinv = jnp.where(s <= eps * jnp.max(s, initial = -jnp.inf), 0., 1/s) # initial value is provided to deal with zero dimensional arrays
    sinv = jnp.where(s <= eps, 0., 1/s)

    (gu, gs, gvh) = jnp.linalg.svd(jax.vmap(gx_jvp)(vh).T @ jnp.diag(sinv) @ jax.vmap(gx_jvp)(u.T), hermitian = True, full_matrices = False)
    gsinv = jnp.where(gs <= eps * jnp.max(gs, initial = -jnp.inf), 0., 1/gs) # initial value is provided to deal with zero dimensional arrays

    dv = inv_vp(gu, gsinv, gvh, L_v + gx_jvp(inv_vp(u, sinv, vh, - L_x))) # shape = (g, )
    x_opt = x_guess - inv_vp(u, sinv, vh, L_x + gx_vjp(dv)) # shape = (nx, )

    return (unravel(x_opt), v_guess + dv), (u, sinv, vh, gu, gsinv, gvh) # (optimal x, optimal Lagrange variables), inverse of hessian of Lagrangian


# Forward- and reverse-mode autodiff compatible differentiable optimization with equality constraints
@functools.partial(jax.custom_jvp, nondiff_argnums = (0, 1))
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
    # sinv = jnp.where(s <= eps * jnp.max(s, initial = -jnp.inf), 0., 1/s) # initial value is provided to deal with zero dimensional arrays
    sinv = jnp.where(s <= eps, 0., 1/s) # works for stiff problems

    (gu, gs, gvh) = jnp.linalg.svd(jax.vmap(gx_jvp)(vh).T @ jnp.diag(sinv) @ jax.vmap(gx_jvp)(u.T), hermitian = True, full_matrices = False)
    gsinv = jnp.where(gs <= eps * jnp.max(gs, initial = -jnp.inf), 0., 1/gs) # initial value is provided to deal with zero dimensional arrays

    return (unravel(x_opt), v_opt, m_opt), (u, sinv, vh, gu, gsinv, gvh) # (optimal primal variables, optimal dual variables)

"""
# Reverse-mode autdiff compatible differentiable optimization with equality and inequality constraints
@functools.partial(jax.custom_vjp, nondiff_argnums = (0, 1, 2))
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
@functools.partial(jax.custom_jvp, nondiff_argnums = (0, 1, 2))
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
# Forward and reverse mode autodiff compatible ordinary differential equation solver
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


##########################################################################################################################################################
# Differentiable root-finding 

Pytree = Any

def flatten_output(afunc, unravel_first_arg):
    @functools.wraps(afunc)
    def _afunc(*args):
        x, *args = args
        return jax.flatten_util.ravel_pytree(afunc(unravel_first_arg(x), *args))[0]
    return _afunc

def newton_method(f : Callable, z_guess : jnp.ndarray) -> jnp.ndarray :
    # solves root finding problem : f(z) = 0 using newtons method 
    # Every linear solve uses explicit hessian 

    # function is only forward and reverse mode autodiff compatible
    grad_f = jax.jacfwd(f)
    
    def body_fun(val):
        dval = jnp.linalg.solve(grad_f(val), f(val))
        return val - dval
    
    def cond_fun(val):
        return jnp.linalg.norm(f(val)) > 1e-8
    
    def scan_fun(carry, xs):
        carry = jax.lax.cond(cond_fun(carry), body_fun, lambda carry : carry, carry)
        return carry, None

    z, _ = jax.lax.scan(scan_fun, z_guess, xs = None, length = 20.)
    return z

@functools.partial(jax.custom_jvp, nondiff_argnums = (0, 1))
def _root_finding_fwd(solver : Callable, f : Callable, z : jnp.ndarray, p : Pytree) -> Pytree :
    # Forward-mode auto diff compatible root finding problem with f : Rn -> Rn
    return solver(lambda z : f(z, p), z)

@_root_finding_fwd.defjvp
def _root_finding_fwd_fwd(solver, f, primals, tangents):
    z, p = primals
    _, pdot = tangents
    
    zstar = _root_finding_fwd(solver, f, z, p)
    # computing the jacobian is cheaper (in this case) than solving another root-finding problem
    tangents_out = jnp.linalg.solve(jax.jacfwd(f)(zstar, p), - jax.jvp(lambda p : f(zstar, p), (p, ), (pdot, ))[-1])
    return zstar, tangents_out

def root_finding_fwd(f : Callable, z : Pytree, p : Pytree) -> Pytree : 
    # Forward- and reverse-mode autodiff compatible root finding problem. 
    # Note that reusing inverse incorrectly predicts higher order derivatives (> 1)

    z_flat, unravel = flatten_util.ravel_pytree(z)
    _f = flatten_output(f, unravel_first_arg = unravel)
    z_opt = _root_finding_fwd(newton_method, _f, z_flat, p)
    return unravel(z_opt)


##########################################################################################################################################################
def plot_coefficients(
        params_actual : List[jnp.ndarray], params_obtained : List[jnp.ndarray], labels : List[List[str]], 
        filename : str, path : str = ".", separate : bool = True
    ) -> None :

    assert len(params_actual) == len(params_obtained) == len(labels), "Should have same number of sets of parameters"
    
    if not separate : 
        params_actual, params_obtained, labels = map(lambda x : [list(it.chain(*x))], (params_actual, params_obtained, labels))

    np = max(map(len, params_actual))
    alen = len(params_actual)
    width = 0.4

    with plt.style.context(["science", "notebook", "bright"]):
        
        # figsize = (width, height)
        fig, ax =  plt.subplots(alen, 1, figsize = (max(1.5 * np, 10), max(4 * alen, 5)), gridspec_kw = {"wspace" : 0.5})
        if not separate : ax = [ax] # wrap in a list for compatibility

        for i, (p_actual, p_obtain, label) in enumerate(zip(params_actual, params_obtained, labels)) :
            
            x = jnp.arange(len(p_actual))
            rects = ax[i].bar(x, p_actual, label = "Actual", width = width)
            ax[i].bar_label(rects, [f"{rect.get_height():.2e}" if rect.get_height() != 0 else "" for rect in rects], padding = 3, rotation = "vertical")
            
            rects = ax[i].bar(x + width, p_obtain, label = "Predicted", width = width)
            ax[i].bar_label(rects, [f"{rect.get_height():.2e}" if rect.get_height() != 0 else "" for rect in rects], padding = 3, rotation = "vertical")

            ax[i].set_xticks(x + width / 2, label)
            ax[i].set_yscale("symlog", linthresh = 1)
            ax[i].margins(y = 0.3)
            ax[i].legend() # ax[i].legend(loc = "upper left")

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
            ax[i].set(ylabel = fr"$x_{{{i}}}$", xlabel = "Time")
            ax[i].legend()

        fig.tight_layout()
        plt.savefig(os.path.join(path, filename))
        plt.close()


##########################################################################################################################################################
# Differentiable Cubic Spline Interpolation

def CubicSplineParameters(t, y) : 
    # Gives the optimal values of parameters of cubic polynomial given time range t and function values y
    
    npoints = len(t)
    cubic_poly = lambda t, tj, p : jnp.dot(p, jnp.array([(t - tj)**3, (t - tj)**2, (t - tj), 1.]))
    _f = jax.vmap(cubic_poly) # polynomial evaluation
    _jac = jax.vmap(lambda t, tj, p : jnp.dot(p, jnp.array([3*(t - tj)**2, 2*(t - tj), 1, 0]))) # first-order derivative w.r.t time
    _hess = jax.vmap(lambda t, tj, p : jnp.dot(p, jnp.array([6*(t - tj), 2, 0, 0.]))) # second-order derivative w.r.t time
    _ghess = jax.vmap(lambda t, tj, p : jnp.dot(p, jnp.array([6, 0, 0, 0.]))) # third-order derivative w.r.t time

    # _jac = jax.vmap(jax.grad(cubic_poly)) # first-order derivative w.r.t time
    # _hess = jax.vmap(jax.hessian(cubic_poly)) # second-order derivative w.r.t time
    # _ghess = jax.vmap(jax.grad(jax.hessian(cubic_poly))) # third-order derivative w.r.t time

    def hvp(v, t) : 
        # v is vector of all the parameters (4 * (n - 1))
        _v = v.reshape(-1, 4) # shape (n - 1, 4)
        
        return jnp.concatenate([
            _f(t[:-1], t[:-1], _v), # (n - 1) equations
            _f(t[1:], t[:-1], _v), # (n - 1) equations
            _jac(t[1:-1], t[:-2], _v[:-1]) - _jac(t[1:-1], t[1:-1], _v[1:]), # (n - 2) equations
            _hess(t[1:-1], t[:-2], _v[:-1]) - _hess(t[1:-1], t[1:-1], _v[1:]), # (n - 2) equations
            _ghess(t[1:2], t[:1], _v[:1]) - _ghess(t[1:2], t[1:2], _v[1:2]), # 1 equation. Not-a-Knot spline
            _ghess(t[-2:-1], t[-3:-2], _v[-2:-1]) - _ghess(t[-2:-1], t[-2:-1], _v[-1:]) # 1 equation. Not-a-Knot spline
        ])

    y = jnp.atleast_2d(y) # shape (n, ny)
    return jnp.linalg.solve(
        jax.vmap(hvp, in_axes = (0, None))(jnp.eye(4 * (npoints - 1)), t).T, 
        jnp.concatenate((y[:-1], y[1:], jnp.zeros(shape = (2 * npoints - 2, y.shape[-1]))))
    ) # shape (4 * (n - 1), ny)

def CubicSplineSimulate(ti, t, p) :    
    # Get values at time points ti, for given time range t and parameters p
    
    cubic_poly = lambda t, tj, p : jnp.dot(p, jnp.array([(t - tj)**3, (t - tj)**2, (t - tj), 1.]))
    p = p.reshape(-1, 4) # shape (n - 1, 4)

    # Append duplicates of first and last set of parameters to account for edge cases (ti < t0) & (ti > tf)
    _p = jnp.vstack((p[:1, :], p, p[-1:, :]))
    _t = jnp.array([-jnp.inf, *t, jnp.inf])
    _tj = jnp.array([t[0], *t[:-1], t[-2]])
    return jnp.sum(
        jnp.where(
            (ti > _t[:-1]) & (ti <= _t[1:]),
            jax.vmap(cubic_poly, in_axes = (None, 0, 0))(ti, _tj, _p),
            jnp.zeros_like(ti)
        )
    )

@jax.custom_jvp
def CubicSplineParametersScipy(t, y) :
    
    def _scipy_interp_params(t, y) : 
        return jnp.vstack(jnp.einsum("ijk->jik", SCubicSpline(t, y).c))
    
    return jax.pure_callback(_scipy_interp_params, jax.ShapeDtypeStruct((4 * (y.shape[0] - 1), y.shape[1]), y.dtype), t, y)

@CubicSplineParametersScipy.defjvp
def CubicSplineParametersScipy_fwd(primals, tangents):
    t, y = primals
    _, ydot = tangents
    n, ny = y.shape

    # Because pure_callback is not linear in tangent space, custom jvp would not be sufficient for vjp
    # p = CubicSplineParametersScipy(t, y)
    # p_out = CubicSplineParametersScipy(t, ydot)
    
    # Computing explicit inverse makes the jvp linear in tangent space and therefore reverse-mode differentiable
    # We need CubicSpline to be forward and reverse mode differentiable because we compute Hessian
    p, AinvI = jnp.array_split(CubicSplineParametersScipy(t, jnp.concatenate((y, jnp.eye(n)), axis = 1)), [ny], axis = 1)
    p_out = AinvI @ ydot
    return p, p_out

@functools.partial(jax.jit, static_argnums = (3, ))
def CubicSpline(ti : jnp.ndarray, t : jnp.ndarray, y : jnp.ndarray, method : str = "jax") -> jnp.ndarray :
    # https://sites.millersville.edu/rbuchanan/math375/CubicSpline.pdf
    # Fully differentiable Cubic Spline Interpolation
    # Given measurements y at time points t. The time arguments are ti
    _y = y if y.ndim == 2 else y[:, jnp.newaxis] # makes sure that array is 2D. 
    popt = CubicSplineParameters(t, _y) if method == "jax" else CubicSplineParametersScipy(t, _y)
    return jax.vmap(
        jax.vmap(CubicSplineSimulate, in_axes = (None, None, 1)), 
        in_axes = (0, None, None)
    )(jnp.atleast_1d(ti), t, popt) 


##########################################################################################################################################################
# Implementation of Orthogonal Collocation 

def OrthogonalCollocationFormulation(
        dynamics : Callable, xinit : jnp.ndarray, time_span : jnp.ndarray, solution : jnp.ndarray, nstates : int, nparams : int,
        d : int = 3, nint : int = 1, nexpt : int = 1, method : str = "legendre", args : Optional[jnp.ndarray] = None
    ):
    
    # https://www.do-mpc.com/en/latest/theory_orthogonal_collocation.html
    # https://github.com/casadi/casadi/blob/main/docs/examples/python/direct_collocation.py

    # dynamics = A function representing the dynamics 
    # xinit = Initial states (Assumed to be known)
    # time_span = Equally spaced time interval
    # solution = The values of the states at time time_span. Multiple experiments are stacked as columns
    # nstates = Dimensions of the states
    # nparams = Number of unknown parameters
    # d = Degree of interpolating polynomial
    # nint = Number of collocation intervals between measurements
    # nexpt = Number of independant experiments
    # method = collocation points chosen using either legendre or radau scheme
    # args = any additional arguments passed to the dynamic function

    import numpy as np
    import casadi as cd

    opti = cd.Opti() # Casadi optimization stack

    tau_root = np.append(0, cd.collocation_points(d, method)) # Get collocation points
    C = np.zeros((d + 1, d + 1)) # Coefficients of the collocation equation
    D = np.zeros(d + 1) # Coefficients of the continuity equation
    B = np.zeros(d + 1) # Coefficients of the quadrature function

    # Construct polynomial basis
    for j in range(d + 1):
        # Construct Lagrange polynomials to get the polynomial basis at the collocation point
        p = np.poly1d([1])
        for r in range(d + 1):
            if r != j:
                p *= np.poly1d([1, -tau_root[r]]) / (tau_root[j]-tau_root[r])

        # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
        D[j] = p(1.0)

        # Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the continuity equation
        pder = np.polyder(p)
        for r in range(d + 1) :
            C[j, r] = pder(tau_root[r])

        # Evaluate the integral of the polynomial to get the coefficients of the quadrature function
        pint = np.polyint(p)
        B[j] = pint(1)

    # Declare model variables
    x_sym = cd.MX.sym("x", nstates)
    p_sym = cd.MX.sym("p", nparams)
    args = np.zeros(shape = (nexpt, 1)) if args is None else args.reshape(nexpt, -1)
    args_sym = cd.MX.sym("args", args.shape[-1])

    # Time horizon
    dt = time_span[1] - time_span[0] # Equally spaced
    N = nint * (len(time_span) - 1) # number of intervals
    h = dt / nint

    # Continuous time dynamics
    f = cd.Function('f', [x_sym, args_sym, p_sym], [h * dynamics(x_sym, 0, (p_sym, args_sym))], ['x', 'args', 'p'], ['xdot'])
    
    # Start with an empty NLP
    p_var = opti.variable(nparams)

    # Formulate the NLP for all finite elements
    x_var = opti.variable(nstates * nexpt, N * (d + 1)) # can only initialize matrix, not tensors
    xk_end = cd.MX(xinit.flatten()) # Given initial conditions
    x_aux = []
    cost = 0

    for k in range(N):
        
        # states at collocation points
        xk = x_var[:, k * (d + 1) : (k + 1) * (d + 1)]

        # Add equality constraint
        opti.subject_to(cd.vec(xk[:, 0] - xk_end) == 0)

        # Expression for the state derivative at the collocation point
        xp = cd.mtimes(xk, cd.MX(C[:, 1:]))

        # Dynamic equation constraint at the collocation point
        # split -> map -> concat
        opti.subject_to(
            cd.vec(cd.vertcat(
                *map(
                    lambda z : f(*z, p_var), zip(cd.vertsplit(xk[:, 1:], list(range(0, nstates * (nexpt + 1), nstates))), args)
                )
            ) - xp) == 0
        )

        # Expression for the end state
        xk_end = cd.mtimes(xk, cd.MX(D))

        # add objective function
        if k % nint == 0 : cost += cd.sumsqr(xk[:, 0] - solution[k // nint])

    opti.minimize(cost / np.prod(solution.shape))
    opti.set_initial(cd.reshape(x_var, 1, -1), np.repeat(solution[:-1].T, nint * (d + 1), axis = 1).flatten())

    # Dont solve the optimization problem. There might be other problem specific instances such as constraints. 
    return opti, p_var
