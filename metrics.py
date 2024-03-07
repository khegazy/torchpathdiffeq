import jax
import jax.numpy as jnp


def E_vre_pvre(geo_val, geo_grad, pes_val, pes_grad):
    return jnp.linalg.norm(pes_grad)*jnp.linalg.norm(geo_grad),\
        jnp.abs(jnp.sum(geo_grad*pes_grad))#/jnp.norm(geo_grad)

def E_vre(geo_val, geo_grad, pes_val, pes_grad):
    e_vre, _ = E_vre_pvre(geo_val, geo_grad, pes_val, pes_grad)
    return e_vre

def E_pvre(geo_val, geo_grad, pes_val, pes_grad):
    _, e_pvre = E_vre_pvre(geo_val, geo_grad, pes_val, pes_grad)
    return e_pvre

def E_pvre_mag(geo_val, geo_grad, pes_val, pes_grad):
    return jnp.linalg.norm(geo_grad*pes_grad)#/jnp.linalg.norm(geo_grad)

def vre_residual(geo_val, geo_grad, pes_val, pes_grad):
    e_vre, e_pvre = E_vre_pvre(geo_val, geo_grad, pes_val, pes_grad)
    return e_vre - e_pvre