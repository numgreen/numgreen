"""3D surface operator wrapper coupling SplitGreen with bempp-cl DP0 spaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .pde import PDE
from .split import SplitGreen


@dataclass
class CavitySolution:
    """Solution of a coupled interior-Laplace / exterior-dielectric interface problem."""

    energy: float
    u_rf: np.ndarray
    du_rf_dn: np.ndarray
    u_total: np.ndarray
    sigma_ext: np.ndarray


class BemppOperators:
    """Dense piecewise-constant (``DP0``) 3D boundary operators ``S`` and ``D``.

    Assembles singular Laplace or constant-``k0`` Helmholtz operators on ``grid``
    with ``bempp-cl`` (outward unit normals) and adds the Liouville scaling and
    centroid-quadrature remainder matrices ``(S_smooth, D_smooth)`` evaluated
    from ``green`` for arbitrary rigid translations ``grid + shift``.
    """

    def __init__(self, pde: PDE, green: SplitGreen, grid) -> None:
        if pde.dim != 3 or pde.has_advection or pde.is_anisotropic:
            raise ValueError("BemppOperators requires a 3D isotropic advection-free PDE.")
        import bempp_cl.api as bempp

        self.pde, self.green, self.grid = pde, green, grid
        self.space = bempp.function_space(grid, "DP", 0)
        self.centroids = np.asarray(grid.centroids, dtype=float)
        self.normals = np.asarray(grid.normals, dtype=float)
        self.areas = np.asarray(grid.volumes, dtype=float)

        ops = bempp.operators.boundary.laplace if pde.k0 == 0 else bempp.operators.boundary.helmholtz
        args = (self.space, self.space, self.space) if pde.k0 == 0 else (self.space, self.space, self.space, pde.k0)
        self.S_bempp = np.asarray(bempp.as_matrix(ops.single_layer(*args).weak_form())) / self.areas[:, None]
        self.D_bempp = np.asarray(bempp.as_matrix(ops.double_layer(*args).weak_form())) / self.areas[:, None]

    def operators(self, shift: Sequence[float] = (0.0, 0.0, 0.0)):
        """Return ``(S, D, S_smooth, D_smooth)`` on ``grid + shift``."""
        C = self.centroids + np.asarray(shift, dtype=float)[None, :]
        H, gy_H = self.green.eval_H(C, C, grad_y=True)
        eps_c = self.pde.eval_a(C)
        s_c = np.sqrt(eps_c)
        ds_dn = np.sum(self.pde.eval_grad_a(C) * self.normals, axis=1) / (2.0 * s_c)

        S_sing = (1.0 / s_c)[:, None] * self.S_bempp * (1.0 / s_c)[None, :]
        D_sing = (s_c[None, :] / s_c[:, None]) * self.D_bempp - (ds_dn[None, :] / s_c[:, None]) * self.S_bempp
        S_smooth = H * self.areas[None, :]
        D_smooth = (eps_c[None, :] * np.einsum("mnk,nk->mn", gy_H, self.normals)) * self.areas[None, :]
        return S_sing + S_smooth, D_sing + D_smooth, S_smooth, D_smooth

    @property
    def S(self) -> np.ndarray:
        return self.operators()[0]

    @property
    def D(self) -> np.ndarray:
        return self.operators()[1]

    def solve_exterior_dirichlet(self, u_bnd: np.ndarray, shift: Sequence[float] = (0.0, 0.0, 0.0)) -> np.ndarray:
        """Solve exterior Dirichlet BIE ``S sigma = (-1/2 I + D) u`` for ``sigma = a(x) du/dn``."""
        S, D, _, _ = self.operators(shift=shift)
        eye = np.eye(len(self.areas), dtype=S.dtype)
        return np.linalg.solve(S, (-0.5 * eye + D) @ np.asarray(u_bnd))

    def eval_exterior(
        self,
        X: np.ndarray,
        u_bnd: np.ndarray,
        sigma_ext: np.ndarray,
        shift: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> np.ndarray:
        """Evaluate exterior representation ``u(x) = -S_ext sigma + D_ext u`` at points ``X``."""
        C = self.centroids + np.asarray(shift, dtype=float)[None, :]
        G, gy_G = self.green.eval(np.atleast_2d(X), C, grad_y=True)
        dG_dn = self.pde.eval_a(C)[None, :] * np.einsum("mnk,nk->mn", gy_G, self.normals)
        return -(G * self.areas[None, :]) @ sigma_ext + (dG_dn * self.areas[None, :]) @ u_bnd

    def solve_dielectric_cavity(
        self,
        charges: Sequence[float],
        charge_positions: np.ndarray,
        eps_in: float = 2.0,
        shift: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> CavitySolution:
        """Solve coupled cavity-solvent BIE on ``grid + shift`` in exterior ``eps(x)``."""
        sh = np.asarray(shift, dtype=float)
        C = self.centroids + sh[None, :]
        q = np.asarray(charges, dtype=float)
        rq = np.atleast_2d(np.asarray(charge_positions, dtype=float)) + sh[None, :]
        diff = C[:, None, :] - rq[None, :, :]
        r = np.maximum(np.linalg.norm(diff, axis=-1), 1e-12)
        u_coul = np.sum(q[None, :] / (4.0 * np.pi * eps_in * r), axis=1)
        du_coul = -np.sum((q[None, :] / (4.0 * np.pi * eps_in * r**3)) * np.sum(diff * self.normals[:, None, :], axis=-1), axis=1)

        S_ext, D_ext, _, _ = self.operators(shift=sh)
        N = len(self.areas)
        I = np.eye(N)
        lhs = np.block([[0.5 * I + self.D_bempp, -self.S_bempp], [0.5 * I - D_ext, eps_in * S_ext]])
        rhs = np.concatenate([np.zeros(N), (-0.5 * I + D_ext) @ u_coul - eps_in * (S_ext @ du_coul)])
        sol = np.linalg.solve(lhs, rhs)
        u_rf, psi = sol[:N], sol[N:]

        diff_q = rq[:, None, :] - C[None, :, :]
        rq_c = np.maximum(np.linalg.norm(diff_q, axis=-1), 1e-12)
        G_qc = (1.0 / (4.0 * np.pi * rq_c)) * self.areas[None, :]
        dG_qc = (np.sum(diff_q * self.normals[None, :, :], axis=-1) / (4.0 * np.pi * rq_c**3)) * self.areas[None, :]
        phi_rf_q = G_qc @ psi - dG_qc @ u_rf
        return CavitySolution(
            energy=0.5 * float(np.dot(q, phi_rf_q)),
            u_rf=u_rf,
            du_rf_dn=psi,
            u_total=u_coul + u_rf,
            sigma_ext=eps_in * (du_coul + psi),
        )
