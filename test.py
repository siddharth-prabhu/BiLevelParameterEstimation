import operator

import jax
import jax.numpy as jnp

from utils import differentiable_optimization, constraint_differentiable_optimization, constraint_differentiable_optimization_rev


##########################################################################################################################################################
# Constraint (equality) differential optimization    
def f(p, x) : return jnp.mean((jnp.array([p[1], p[0] * p[1], p[1]**2 * p[2]]) * x - jnp.array([1, 2, 3]))**2)
def g(p, x) : return x - jnp.array([1, 2, 3.]) # return jnp.array([ ])
p = jnp.array([1, 0.1, 5])
eps = 1e-5

# constaint (equality) differential optimization forward-mode
def obj_fwd(p) : 
    (x_opt, _), *_ = differentiable_optimization(f, g, p, jnp.zeros(3), ())
    return f(p, x_opt)

value, gradient = jax.value_and_grad(obj_fwd)(p)
gradient_fd = jax.vmap(lambda v : (obj_fwd(p + v * eps) - value) / eps)(jnp.eye(len(p)))
assert jnp.allclose(gradient, gradient_fd, atol = 100 * eps), "Gradients (forward-mode) for constraint (equality) differential optimization does not match"

# Hessian calculations
hessian = jax.hessian(obj_fwd)(p)

_grad = jax.grad(obj_fwd)
hessian_fd = jax.vmap(lambda v : (_grad(p + v * eps) - _grad(p - v * eps)) / eps / 2)(jnp.eye(len(p)))
assert jnp.allclose(hessian_fd, hessian, atol = 100 * eps), "Hessian (fwd-over-rev) for constraint (equality) differential optimization does not match"


##########################################################################################################################################################
# Constraint (equality + inequality) differential optimization    
def f(p, x) : return jnp.mean((jnp.array([p[1], p[0] * p[1], p[1]**2 * p[2]]) * x - jnp.array([1, 2, 3]))**2)
def g(p, x) : return x[:1] - 5 # return jnp.array([ ])
def h(p, x) : return p[1:] - x[1:] # geq # jnp.array([ ]) #  

def objcon_rev(p) : 
    x_opt, *_ = constraint_differentiable_optimization_rev(f, g, h, p, jnp.zeros(3), ())
    return f(p, x_opt)

value, gradient = jax.value_and_grad(objcon_rev)(p)
gradient_fd = jax.vmap(lambda v : (objcon_rev(p + v * eps) - value) / eps)(jnp.eye(len(p)))
assert jnp.allclose(gradient, gradient_fd, atol = 100 * eps), "Gradients (reverse-mode) for constraint differential optimization does not match"

def objcon_fwd(p) : 
    x_opt, *_ = constraint_differentiable_optimization(f, g, h, p, jnp.zeros(3), ())
    return f(p, x_opt)

value, gradient = jax.value_and_grad(objcon_fwd)(p)
gradient_fd = jax.vmap(lambda v : (objcon_fwd(p + v * eps) - value) / eps)(jnp.eye(len(p)))
assert jnp.allclose(gradient, gradient_fd, atol = 100 * eps), "Gradients (forward-mode) for constraint differential optimization does not match"

# Hessian calculations
hessian = jax.hessian(objcon_fwd)(p)
hessian_rev = jax.jacrev(jax.jacrev(objcon_rev))(p)

_grad = jax.grad(objcon_fwd)
hessian_fd = jax.vmap(lambda v : (_grad(p + v * eps) - _grad(p - v * eps)) / eps / 2)(jnp.eye(len(p)))
assert jnp.allclose(hessian_fd, hessian, atol = 100 * eps), "Hessian (fwd-over-rev) for constraint differential optimization does not match"
assert jnp.allclose(hessian_fd, hessian_rev, atol = 100 * eps), "Hessian (rev-over-rev) for constraint differential optimization does not match"
