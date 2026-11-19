"""MFS and Nyström convergence tests against exact and polar finite-difference references."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from numgreen import PDE, SplitGreen, solve_mfs


def _star_curve(n_pts: int, r0: float = 0.8, amp: float = 0.14, lobes: int = 3):
    th = np.linspace(0.0, 2.0 * np.pi, n_pts, endpoint=False)
    r = r0 * (1.0 + amp * np.cos(lobes * th))
    dr = -r0 * amp * lobes * np.sin(lobes * th)
    X = np.column_stack([r * np.cos(th), r * np.sin(th)])
    tx = dr * np.cos(th) - r * np.sin(th)
    ty = dr * np.sin(th) + r * np.cos(th)
    ds = np.hypot(tx, ty)
    normals = np.column_stack([ty / ds, -tx / ds])
    return th, X, normals, ds


def test_mfs_convergence_variable_coeff_dirichlet_and_robin() -> None:
    # For a(x, y) = exp(alpha x + beta y) and u(x, y) = exp(p x + q y),
    # choosing c(x, y) = ((alpha + p) p + (beta + q) q) a(x, y) gives -div(a grad u) + c u = 0.
    alpha_a, beta_a = 0.5, -0.35
    p, q = 0.7, 0.45
    lam = (alpha_a + p) * p + (beta_a + q) * q
    a_fn = lambda X: np.exp(alpha_a * X[:, 0] + beta_a * X[:, 1])
    c_fn = lambda X: lam * a_fn(X)
    u_ex = lambda X: np.exp(p * X[:, 0] + q * X[:, 1])
    grad_u_ex = lambda X: np.column_stack([p * u_ex(X), q * u_ex(X)])

    pde = PDE(dim=2, a=a_fn, c=c_fn)
    green = SplitGreen(pde, bounds=[(-1.45, 1.45), (-1.45, 1.45)], shape=131, bc="dirichlet")

    _, X_test, _, _ = _star_curve(60, r0=0.42, amp=0.1, lobes=3)
    u_ref = u_ex(X_test)

    errs = []
    for n_b in (16, 32, 64):
        _, X_b, nrm, _ = _star_curve(n_b)
        sol = solve_mfs(pde, green, X_b, u_ex(X_b), normals=nrm, offset=0.28)
        errs.append(np.linalg.norm(sol(X_test) - u_ref) / np.linalg.norm(u_ref))

    assert errs[0] > errs[1] > errs[2]
    assert errs[2] < 1.5e-3

    # Robin BC: u + 0.5 * (a du/dn) = g_rob
    _, X_b, nrm, _ = _star_curve(72)
    conorm_flux = a_fn(X_b) * np.sum(grad_u_ex(X_b) * nrm, axis=1)
    g_rob = u_ex(X_b) + 0.5 * conorm_flux
    sol_rob = solve_mfs(pde, green, X_b, g_rob, normals=nrm, offset=0.28, alpha=1.0, beta=0.5, conormal_flux=True)
    assert np.linalg.norm(sol_rob(X_test) - u_ref) / np.linalg.norm(u_ref) < 2.5e-3


def test_high_peclet_taylor_green_mfs_vs_fine_fd() -> None:
    # Solve -D Delta u + v_TG(x) . grad u + c u = 0 on disk R = 0.8 at Pe = 100.
    D_diff = 0.016
    c_val = 0.25
    vel_fn = lambda X: np.column_stack([
        np.sin(1.4 * X[:, 0]) * np.cos(1.4 * X[:, 1]),
        -np.cos(1.4 * X[:, 0]) * np.sin(1.4 * X[:, 1]),
    ])
    pde = PDE.advection_diffusion(diffusivity=D_diff, velocity=vel_fn, reaction=c_val, dim=2)
    green = SplitGreen(pde, bounds=[(-1.35, 1.35), (-1.35, 1.35)], shape=151, bc="dirichlet")

    R = 0.8
    th = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
    X_b = np.column_stack([R * np.cos(th), R * np.sin(th)])
    nrm = np.column_stack([np.cos(th), np.sin(th)])
    g_bc = lambda th_ang: np.exp(-4.0 * (1.0 + np.cos(th_ang - 0.2))) + 0.2 * np.sin(2.0 * th_ang)

    sol = solve_mfs(pde, green, X_b, g_bc(th), normals=nrm, offset=0.22)
    assert sol.residual < 1e-4

    # Polar (r, theta) finite-difference reference on the disk.
    nr, nth = 90, 180
    rs = np.linspace(R / nr, R, nr)
    dr = rs[1] - rs[0]
    ths = np.linspace(0.0, 2.0 * np.pi, nth, endpoint=False)
    dth = ths[1] - ths[0]
    Rg, Tg = np.meshgrid(rs, ths, indexing="ij")
    pts = np.column_stack([Rg.ravel() * np.cos(Tg.ravel()), Rg.ravel() * np.sin(Tg.ravel())])
    vel = vel_fn(pts)
    vr = vel[:, 0] * np.cos(Tg.ravel()) + vel[:, 1] * np.sin(Tg.ravel())
    vt = -vel[:, 0] * np.sin(Tg.ravel()) + vel[:, 1] * np.cos(Tg.ravel())

    N_tot = nr * nth
    A_pol = sp.lil_matrix((N_tot, N_tot), dtype=float)
    rhs_pol = np.zeros(N_tot, dtype=float)
    for i in range(nr):
        r_i = rs[i]
        for j in range(nth):
            row = i * nth + j
            if i == nr - 1:
                A_pol[row, row] = 1.0
                rhs_pol[row] = g_bc(ths[j])
                continue
            jp, jm = (j + 1) % nth, (j - 1) % nth
            A_pol[row, row] = D_diff * (2.0 / dr**2 + 2.0 / (r_i * dth) ** 2) + c_val
            A_pol[row, i * nth + jp] = -D_diff / (r_i * dth) ** 2 + vt[row] / (2.0 * r_i * dth)
            A_pol[row, i * nth + jm] = -D_diff / (r_i * dth) ** 2 - vt[row] / (2.0 * r_i * dth)
            c_up = -D_diff * (1.0 / dr**2 + 1.0 / (2.0 * r_i * dr)) + vr[row] / (2.0 * dr)
            c_lo = -D_diff * (1.0 / dr**2 - 1.0 / (2.0 * r_i * dr)) - vr[row] / (2.0 * dr)
            A_pol[row, (i + 1) * nth + j] = c_up
            if i > 0:
                A_pol[row, (i - 1) * nth + j] = c_lo
            else:
                A_pol[row, (j + nth // 2) % nth] += c_lo

    u_pol = spla.spsolve(A_pol.tocsc(), rhs_pol).reshape(nr, nth)
    mask_r = (rs >= 0.15) & (rs <= 0.70)
    pts_cmp = np.column_stack([(Rg[mask_r] * np.cos(Tg[mask_r])).ravel(), (Rg[mask_r] * np.sin(Tg[mask_r])).ravel()])
    rel_l2 = np.linalg.norm(sol(pts_cmp) - u_pol[mask_r].ravel()) / np.linalg.norm(u_pol[mask_r].ravel())
    assert rel_l2 < 6e-3


def test_kress_nystrom_single_layer_convergence() -> None:
    """Check Martensen-Kussmaul / Kress log-splitting Nyström quadrature with SplitGreen."""
    # Exact solution u(x, y) = exp(0.5 * (x^2 - y^2)) of -Delta u + (x^2 + y^2) u = 0.
    u_ex = lambda X: np.exp(0.5 * (X[:, 0] ** 2 - X[:, 1] ** 2))
    pde = PDE(dim=2, a=1.0, c=lambda X: X[:, 0] ** 2 + X[:, 1] ** 2)
    green = SplitGreen(pde, bounds=[(-1.4, 1.4), (-1.4, 1.4)], shape=151, bc="dirichlet")

    _, X_in, _, _ = _star_curve(50, r0=0.38, amp=0.08, lobes=3)
    u_ref = u_ex(X_in)

    errs = []
    for N in (12, 20, 48):
        M_half = N // 2
        th, X_b, _, ds = _star_curve(N, r0=0.78, amp=0.12, lobes=3)
        dth = 2.0 * np.pi / N

        idx = np.arange(N)
        diff_idx = (idx[:, None] - idx[None, :]) % N
        R_mat = sum(np.cos(m * diff_idx * dth) / m for m in range(1, M_half)) + np.cos(M_half * diff_idx * dth) / (2.0 * M_half)
        W_log = R_mat / N

        diff_X = np.linalg.norm(X_b[:, None, :] - X_b[None, :], axis=-1)
        sin_half = np.maximum(2.0 * np.abs(np.sin(0.5 * (th[:, None] - th[None, :]))), 1e-15)
        ratio = np.where(np.eye(N, dtype=bool), ds[:, None], diff_X / sin_half)
        K_reg = -np.log(ratio) / (2.0 * np.pi) + green.eval_H(X_b, X_b)

        S_nys = (W_log + dth * K_reg) * ds[None, :]
        sigma = np.linalg.solve(S_nys, u_ex(X_b))
        u_nys = (green.eval(X_in, X_b) * (dth * ds)[None, :]) @ sigma
        errs.append(np.linalg.norm(u_nys - u_ref) / np.linalg.norm(u_ref))

    assert errs[0] > errs[1] > errs[2]
    assert errs[2] < 1e-4
