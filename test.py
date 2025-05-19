import operator

import jax
import jax.numpy as jnp

from utils import differentiable_optimization, constraint_differentiable_optimization

##########################################################################################################################################################
# Constraint (equality) differential optimization    
def f(p, x) : return jnp.mean((jnp.array([p[1], p[0] * p[1], p[1]**2 * p[2]]) * x - jnp.array([1, 2, 3]))**2)
def g(p, x) : return jnp.log(p) * (jnp.array([1, 2, 3, 0, 1, 4, 5, 6, 0]).reshape(3, -1) @ x) - jnp.array([0.6931472 * 10, -2.3025851 * 6, 1.609438 * 27]) # Linear constraints to be convex
p = jnp.array([2, 0.1, 5])
eps = 1e-5

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
assert jnp.allclose(hessian_fd, hessian, atol = 100 * eps), "Hessian (fwd-over-rev) for constraint (equality) differential optimization does not match"


##########################################################################################################################################################
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
assert jnp.allclose(hessian_fd, hessian, atol = 100 * eps), "Hessian (fwd-over-rev) for constraint differential optimization does not match"
