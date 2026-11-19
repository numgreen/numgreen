"""Proxy-source Method of Fundamental Solutions (MFS) solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.linalg import lstsq

from .pde import PDE


@dataclass
class MFSSolution:
    """Linear combination ``u(x) = sum_j w_j G(x, y_j)`` of fundamental solutions."""

    pde: PDE
    green: Any
    sources: np.ndarray
    weights: np.ndarray
    residual: float

    def eval(self, X: np.ndarray) -> np.ndarray:
        return self.green.eval(np.atleast_2d(X), self.sources) @ self.weights

    __call__ = eval

    def grad(self, X: np.ndarray) -> np.ndarray:
        _, gx = self.green.eval(np.atleast_2d(X), self.sources, grad_x=True)
        return np.einsum("mnk,n->mk", gx, self.weights)


def solve_mfs(
    pde: PDE,
    green: Any,
    boundary_pts: np.ndarray,
    rhs: np.ndarray,
    *,
    normals: np.ndarray | None = None,
    sources: np.ndarray | None = None,
    offset: float = 0.25,
    interior: bool = True,
    alpha: complex | float | np.ndarray = 1.0,
    beta: complex | float | np.ndarray = 0.0,
    conormal_flux: bool = True,
    rcond: float = 1e-12,
) -> MFSSolution:
    """Solve ``alpha u + beta d_nu u = rhs`` on ``boundary_pts`` via proxy sources.

    Here ``normals`` are outward unit normals to the bounded domain (for interior
    BVPs, ``interior=True`` places proxy sources outside at ``+offset * normals``;
    for exterior obstacle problems, ``interior=False`` places them inside the
    obstacle at ``-offset * normals``).
    """
    X_b = np.atleast_2d(np.asarray(boundary_pts, dtype=float))
    N_b = None if normals is None else np.atleast_2d(np.asarray(normals, dtype=float))
    if sources is None:
        if N_b is None:
            raise ValueError("Provide `normals` when `sources` are generated automatically.")
        sources = X_b + (1.0 if interior else -1.0) * offset * N_b
    Y_s = np.atleast_2d(np.asarray(sources, dtype=float))

    a_col = np.broadcast_to(alpha, (X_b.shape[0],))[:, None]
    if np.all(np.asarray(beta) == 0):
        K = a_col * green.eval(X_b, Y_s)
    else:
        if N_b is None:
            raise ValueError("`normals` required when `beta != 0`.")
        G, gx = green.eval(X_b, Y_s, grad_x=True)
        dG = np.einsum("mnk,mk->mn", gx, pde.conormal(X_b, N_b) if conormal_flux else N_b)
        K = a_col * G + np.broadcast_to(beta, (X_b.shape[0],))[:, None] * dG

    rhs_vec = np.asarray(rhs)
    weights, *_ = lstsq(K, rhs_vec, cond=rcond)
    res = float(np.linalg.norm(K @ weights - rhs_vec) / max(np.linalg.norm(rhs_vec), 1e-15))
    return MFSSolution(pde=pde, green=green, sources=Y_s, weights=weights, residual=res)
