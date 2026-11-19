"""Tests for 2D/3D PDE specification and singularity-split Green's functions."""

from __future__ import annotations

import numpy as np
from scipy.special import hankel1
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from numgreen import PDE, SplitGreen, solve_mfs


def _exact_3d_exp_dielectric(X: np.ndarray, Y: np.ndarray, alpha: float):
    """Closed-form Green's function for -div(exp(2 alpha z) grad G) = delta in R^3."""
    diff = X[:, None, :] - Y[None, :, :]
    r = np.linalg.norm(diff, axis=-1)
    sx = np.exp(alpha * X[:, 2])[:, None]
    sy = np.exp(alpha * Y[:, 2])[None, :]
    g = np.exp(-abs(alpha) * r) / (4.0 * np.pi * r)
    G = g / (sx * sy)
    dg_dr = -(abs(alpha) * r + 1.0) * np.exp(-abs(alpha) * r) / (4.0 * np.pi * r**3)
    gx = (dg_dr[:, :, None] * diff) / (sx * sy)[:, :, None]
    gx[:, :, 2] -= alpha * G
    gy = -(dg_dr[:, :, None] * diff) / (sx * sy)[:, :, None]
    gy[:, :, 2] -= alpha * G
    return G, gx, gy


def test_3d_variable_dielectric_split_convergence() -> None:
    alpha = 0.65
    pde = PDE.dielectric(eps=lambda X: np.exp(2.0 * alpha * X[:, 2]), dim=3)
    Y = np.array([[0.15, -0.2, 0.25], [-0.25, 0.1, -0.3]])
    ang = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    X = np.column_stack([0.85 * np.cos(ang), 0.85 * np.sin(ang), np.linspace(-0.7, 0.7, 24)])
    G_ex, gx_ex, gy_ex = _exact_3d_exp_dielectric(X, Y, alpha)

    errs = []
    for n in (19, 31):
        green = SplitGreen(pde, bounds=[(-4.8, 4.8)] * 3, shape=n, bc="absorbing")
        assert green.num_factorizations == 1
        G_num, gx_num, gy_num = green.eval(X, Y, grad_x=True, grad_y=True)
        errs.append(np.linalg.norm(G_num - G_ex) / np.linalg.norm(G_ex))

    assert errs[1] < errs[0]
    assert errs[1] < 1.2e-2
    assert np.linalg.norm(gx_num - gx_ex) / np.linalg.norm(gx_ex) < 2.0e-2
    assert np.linalg.norm(gy_num - gy_ex) / np.linalg.norm(gy_ex) < 2.0e-2


def test_source_tabulation_matches_direct_and_keeps_single_factorization() -> None:
    alpha = 0.4
    pde = PDE.dielectric(eps=lambda X: np.exp(2.0 * alpha * X[:, 2]), dim=3)
    green_dir = SplitGreen(pde, bounds=[(-3.8, 3.8)] * 3, shape=25, bc="absorbing")
    green_tab = SplitGreen(pde, bounds=[(-3.8, 3.8)] * 3, shape=25, bc="absorbing").tabulate_sources(
        bounds=[(-0.8, 0.8), (-0.8, 0.8), (-1.2, 1.2)], shape=(11, 11, 13)
    )
    assert green_tab.num_factorizations == 1

    X = np.array([[0.3, -0.2, 0.45], [-0.25, 0.3, 0.55], [0.1, 0.25, 0.35]])
    Y = np.array([[-0.2, 0.15, -0.45], [0.25, -0.2, -0.55]])
    H_d, gx_d, gy_d = green_dir.eval_H(X, Y, grad_x=True, grad_y=True)
    H_t, gx_t, gy_t = green_tab.eval_H(X, Y, grad_x=True, grad_y=True)

    assert np.linalg.norm(H_t - H_d) / np.linalg.norm(H_d) < 4e-3
    assert np.linalg.norm(gx_t - gx_d) / np.linalg.norm(gx_d) < 1.5e-2
    assert np.linalg.norm(gy_t - gy_d) / np.linalg.norm(gy_d) < 4.0e-2


def test_2d_luneburg_helmholtz_green_vs_radial_reference() -> None:
    k0, R, tau = 5.5, 0.7, 0.09
    n_fn = lambda X: np.sqrt(1.0 + tau * np.logaddexp(0.0, (1.0 - (X[:, 0] ** 2 + X[:, 1] ** 2) / R**2) / tau))
    pde = PDE.helmholtz(k=lambda X: k0 * n_fn(X), k0=k0, dim=2)
    green = SplitGreen(pde, bounds=[(-2.2, 2.2), (-2.2, 2.2)], shape=141, bc="absorbing")

    # For a source at the origin, G(r, 0) = (i/4) H_0^(1)(k0 r) + h(r) solves a 1D radial ODE.
    R_out = 1.35
    r_ode = np.linspace(1e-4, R_out, 1200)
    dr = r_ode[1] - r_ode[0]
    n2 = n_fn(np.column_stack([r_ode, np.zeros_like(r_ode)])) ** 2
    G0_r = 0.25j * hankel1(0, k0 * r_ode)
    main = -2.0 / dr**2 + k0**2 * n2
    up = 1.0 / dr**2 + 1.0 / (2.0 * r_ode[:-1] * dr)
    lo = 1.0 / dr**2 - 1.0 / (2.0 * r_ode[1:] * dr)
    A1 = sp.diags([lo, main, up], [-1, 0, 1], format="lil", dtype=complex)
    rhs1 = -k0**2 * (n2 - 1.0) * G0_r
    A1[0, 0], A1[0, 1] = -2.0 / dr**2 + k0**2 * n2[0], 2.0 / dr**2
    ratio = -k0 * hankel1(1, k0 * R_out) / hankel1(0, k0 * R_out)
    cg = 1.0 / dr**2 + 1.0 / (2.0 * R_out * dr)
    A1[-1, :] = 0
    A1[-1, -2] = 2.0 / dr**2
    A1[-1, -1] = -2.0 / dr**2 + k0**2 * n2[-1] + cg * (2.0 * dr * ratio)
    h_ode = spla.spsolve(A1.tocsc(), rhs1)

    rs = np.linspace(0.18, 1.15, 25)
    th = np.linspace(0.0, 2.0 * np.pi, 25, endpoint=False)
    X = np.column_stack([rs * np.cos(th), rs * np.sin(th)])
    G_num = green.eval(X, np.array([[0.0, 0.0]]))[:, 0]
    G_ref = 0.25j * hankel1(0, k0 * rs) + np.interp(rs, r_ode, h_ode)
    assert np.linalg.norm(G_num - G_ref) / np.linalg.norm(G_ref) < 1.5e-2


def test_2d_anisotropic_tensor_pde_mfs() -> None:
    # Separable exact solution u(x, y) = arctan(x) arctan(y) of -((1 + x^2) u_x)_x - ((1 + y^2) u_y)_y = 0.
    def A_fn(X: np.ndarray) -> np.ndarray:
        out = np.zeros((X.shape[0], 2, 2))
        out[:, 0, 0] = 1.0 + X[:, 0] ** 2
        out[:, 1, 1] = 1.0 + X[:, 1] ** 2
        return out

    u_exact = lambda X: np.arctan(X[:, 0]) * np.arctan(X[:, 1])
    pde = PDE(dim=2, A=A_fn, name="AnisotropicSeparable")
    green = SplitGreen(pde, bounds=[(-1.4, 1.4), (-1.4, 1.4)], shape=101, bc="dirichlet")

    th = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
    X_b = np.column_stack([0.75 * np.cos(th), 0.65 * np.sin(th)])
    nrm = np.column_stack([np.cos(th) / 0.75, np.sin(th) / 0.65])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    sol = solve_mfs(pde, green, X_b, u_exact(X_b), normals=nrm, offset=0.28)

    X_in = np.column_stack([0.35 * np.cos(th), 0.3 * np.sin(th)])
    assert np.linalg.norm(sol(X_in) - u_exact(X_in)) / np.linalg.norm(u_exact(X_in)) < 8e-3
