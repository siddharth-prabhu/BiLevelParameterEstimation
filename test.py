import unittest

import jax
import jax.numpy as jnp
from jax import flatten_util

from utils import differentiable_optimization, constraint_differentiable_optimization, CubicSpline

eps = 1e-5
p = jnp.array([2, 0.1, 5])


class Test(unittest.TestCase):

    @unittest.skipIf(True, "Skipped Equality Constrained Differential Optimizaiton Test")
    def test_EConDiffOpt(self):

        # Constraint (equality) differential optimization    
        def f(p, x) : return jnp.mean((jnp.array([p[1], p[0] * p[1], p[1]**2 * p[2]]) * x - jnp.array([1, 2, 3]))**2)
        def g(p, x) : return jnp.log(p) * (jnp.array([1, 2, 3, 0, 1, 4, 5, 6, 0]).reshape(3, -1) @ x) - jnp.array([0.6931472 * 10, -2.3025851 * 6, 1.609438 * 27]) # Linear constraints to be convex

        # constaint (equality) differential optimization forward-mode
        def obj_fwd(p) : 
            (x_opt, v_opt), *_ = differentiable_optimization(f, g, p, jnp.zeros(3), ())
            return f(p, x_opt) + v_opt @ g(p, x_opt)

        value, gradient = jax.value_and_grad(obj_fwd)(p)
        gradient_fd = jax.vmap(lambda v : (obj_fwd(p + v * eps) - value) / eps)(jnp.eye(len(p)))
        # assert jnp.allclose(gradient, gradient_fd, atol = 100 * eps), "Gradients (forward-mode) for constraint (equality) differential optimization does not match"
        print("Gradients (autodiff)", gradient)
        print("Gradients (finitediff)", gradient_fd)

        # Hessian calculations
        hessian = jax.hessian(obj_fwd)(p)

        _grad = jax.grad(obj_fwd)
        hessian_fd = jax.vmap(lambda v : (_grad(p + v * eps) - _grad(p - v * eps)) / eps / 2)(jnp.eye(len(p)))
        self.assertTrue(jnp.allclose(hessian_fd, hessian, atol = 100 * eps))

    @unittest.skipIf(True, "Skipped Equality and Inequality Constrained Differential Optimiation Test")
    def test_EIConDiffOpt(self):

        # Constraint (equality + inequality) differential optimization    
        # solution = [3, 2, 1]
        def f(p, x) : return jnp.mean((jnp.array([p[1], p[0] * p[1], p[0] * p[1]**2 * p[2]]) * x - jnp.array([1.3, 1.2, 1.05]))**2)
        def g(p, x) : return jnp.log(p) * (jnp.array([1, 2, 3, 0, 1, 4, 5, 6, 0]).reshape(3, -1) @ x) - jnp.array([0.6931472 * 10, -2.3025851 * 6, 1.609438 * 27]) # Linear constraints to be convex
        def h(p, x) : return p * x**2 # geq ; Convex constraints

        def objcon_fwd(p) : 
            (x_opt, v_opt, m_opt), _ = constraint_differentiable_optimization(f, g, h, p, jnp.zeros(3), ())
            return f(p, x_opt) + v_opt @ g(p, x_opt) + m_opt @ h(p, x_opt)

        value, gradient = jax.value_and_grad(objcon_fwd)(p)
        gradient_fd = jax.vmap(lambda v : (objcon_fwd(p + v * eps) - value) / eps)(jnp.eye(len(p)))
        # assert jnp.allclose(gradient, gradient_fd, atol = 100 * eps), "Gradients (forward-mode) for constraint differential optimization does not match"

        # Hessian calculations
        hessian = jax.hessian(objcon_fwd)(p)
        _grad = jax.grad(objcon_fwd)
        hessian_fd = jax.vmap(lambda v : (_grad(p + v * eps) - _grad(p - v * eps)) / eps / 2)(jnp.eye(len(p)))
        self.assertTrue(jnp.allclose(hessian_fd, hessian, atol = 100 * eps))

    @unittest.skipIf(False, "Skipped Differential Cubic Spline Interpolation Test")
    def test_DiffCSInterp(self):
        
        # Differentiable Cubic Spline Interpolation  

        from scipy.interpolate import CubicSpline as SCubicSpline

        npoints = 5
        t = jnp.arange(npoints, dtype = jnp.float64)
        y = jnp.column_stack((2 * jnp.sin(t), 2 * jnp.cos(t), 2 * jnp.tan(t)))
        targ = jnp.concatenate((t, t + 0.2, t - 0.2))
        jinterp = CubicSpline(targ, t, y)
        sinterp = SCubicSpline(t, y)(targ)
        self.assertTrue(jnp.allclose(jinterp, sinterp))

        def obj(ti, t, y, method = "jax"):
            sol = CubicSpline(ti, t, y, method)
            return jnp.mean((sol - jnp.ones_like(sol))**2)
        
        # Reverse mode autodiff gradients
        loss, (tidot, tdot, ydot) = jax.value_and_grad(obj, argnums = (0, 1, 2))(targ, t, y)

        def fd(eps):
            vars, unravel = flatten_util.ravel_pytree((targ, t, y))
            grads = jax.vmap(lambda v : (obj(*unravel(vars + eps * v)) - loss) / eps)(jnp.eye(len(vars)))
            return unravel(grads)

        # Gradients using finite difference
        tifd, tfd, yfd = fd(1e-5)

        self.assertTrue(jnp.allclose(tifd, tidot, 1e-3))
        self.assertTrue(jnp.allclose(tfd, tdot, 1e-3))
        self.assertTrue(jnp.allclose(yfd, ydot, 1e-3))

        # Checking gradients using calback function
        loss, (stidot, stdot, sydot) = jax.value_and_grad(obj, argnums = (0, 1, 2))(targ, t, y, "scipy")

        self.assertTrue(jnp.allclose(stidot, tidot)) # Should match
        self.assertFalse(jnp.allclose(stdot, tdot)) # Should not match
        self.assertTrue(jnp.allclose(sydot, ydot)) # Should match