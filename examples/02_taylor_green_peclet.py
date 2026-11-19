"""Steady advection-diffusion in a swirling Taylor-Green cellular vortex.

Solves the variable-velocity equation

    -D * Delta u(x) + v(x) . grad u(x) + c * u(x) = 0

using a frozen-drift Bessel parametrix G_0(x, y) plus the box remainder H(x, y)
coupled to proxy-source MFS on the domain boundary (compare Dual Reciprocity
domain-node formulations in Wrobel & DeFigueiredo 1991 and Rap, Elliott et al.
2007, and variable-coefficient BEM discussions in Guiggiani 2020).

Generates:
    assets/02_taylor_green_peclet.png
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.path import Path as MplPath

from numgreen import PDE, SplitGreen, solve_mfs


def taylor_green_velocity(X: np.ndarray, k_vort: float = 1.35, swirl: float = 1.0) -> np.ndarray:
    """Incompressible cellular Taylor-Green vortex v = (d_y psi, -d_x psi), psi = cos(k x) cos(k y)."""
    vx = -swirl * np.cos(k_vort * X[:, 0]) * np.sin(k_vort * X[:, 1])
    vy = +swirl * np.sin(k_vort * X[:, 0]) * np.cos(k_vort * X[:, 1])
    return np.column_stack([vx, vy])


def lobed_chamber(n_pts: int, r0: float = 0.86, amp: float = 0.12, lobes: int = 3):
    th = np.linspace(0.0, 2.0 * np.pi, n_pts, endpoint=False)
    r = r0 * (1.0 + amp * np.cos(lobes * th))
    dr = -r0 * amp * lobes * np.sin(lobes * th)
    pts = np.column_stack([r * np.cos(th), r * np.sin(th)])
    tx = dr * np.cos(th) - r * np.sin(th)
    ty = dr * np.sin(th) + r * np.cos(th)
    ds = np.hypot(tx, ty)
    normals = np.column_stack([ty / ds, -tx / ds])
    return th, pts, normals


def solve_disk_polar_reference(D_diff: float, c_val: float, R: float, bc_fn, nr: int = 110, nth: int = 220):
    """Polar (r, theta) finite-difference reference solver on a disk of radius R."""
    rs = np.linspace(R / nr, R, nr)
    dr = rs[1] - rs[0]
    ths = np.linspace(0.0, 2.0 * np.pi, nth, endpoint=False)
    dth = ths[1] - ths[0]
    Rg, Tg = np.meshgrid(rs, ths, indexing="ij")
    pts = np.column_stack([Rg.ravel() * np.cos(Tg.ravel()), Rg.ravel() * np.sin(Tg.ravel())])
    vel = taylor_green_velocity(pts)
    vr = vel[:, 0] * np.cos(Tg.ravel()) + vel[:, 1] * np.sin(Tg.ravel())
    vt = -vel[:, 0] * np.sin(Tg.ravel()) + vel[:, 1] * np.cos(Tg.ravel())

    N_tot = nr * nth
    A_pol = sp.lil_matrix((N_tot, N_tot), dtype=float)
    rhs_pol = np.zeros(N_tot, dtype=float)
    X_rim = np.column_stack([R * np.cos(ths), R * np.sin(ths)])
    g_rim = bc_fn(X_rim)
    for i in range(nr):
        r_i = rs[i]
        for j in range(nth):
            row = i * nth + j
            if i == nr - 1:
                A_pol[row, row] = 1.0
                rhs_pol[row] = g_rim[j]
                continue
            jp, jm = (j + 1) % nth, (j - 1) % nth
            A_pol[row, row] = D_diff * (2.0 / dr**2 + 2.0 / (r_i * dth) ** 2) + c_val
            A_pol[row, i * nth + jp] = -D_diff / (r_i * dth) ** 2 + vt[row] / (2.0 * r_i * dth)
            A_pol[row, i * nth + jm] = -D_diff / (r_i * dth) ** 2 - vt[row] / (2.0 * r_i * dth)
            A_pol[row, (i + 1) * nth + j] = -D_diff * (1.0 / dr**2 + 1.0 / (2.0 * r_i * dr)) + vr[row] / (2.0 * dr)
            c_lo = -D_diff * (1.0 / dr**2 - 1.0 / (2.0 * r_i * dr)) - vr[row] / (2.0 * dr)
            if i > 0:
                A_pol[row, (i - 1) * nth + j] = c_lo
            else:
                A_pol[row, (j + nth // 2) % nth] += c_lo
    return rs, ths, Rg, Tg, spla.spsolve(A_pol.tocsc(), rhs_pol).reshape(nr, nth)


def _clip_streamplot(st, clip_poly: Polygon) -> None:
    st.lines.set_clip_path(clip_poly)
    arrows = getattr(st, "arrows", None)
    if hasattr(arrows, "set_clip_path"):
        arrows.set_clip_path(clip_poly)
    elif arrows is not None:
        for art in arrows:
            art.set_clip_path(clip_poly)


def main() -> None:
    out_dir = Path("assets")
    out_dir.mkdir(exist_ok=True)

    bc_fn = lambda X: 0.5 * (1.0 - np.tanh(2.2 * X[:, 0])) + 0.35 * np.exp(-5.0 * ((X[:, 0] + 0.35) ** 2 + (X[:, 1] + 0.72) ** 2))
    c_react = 0.15
    L_char = 1.72

    # Evaluate slightly past the smooth boundary (safe since proxy sources sit at +0.20)
    # and clip artist geometry with a vector Polygon for gap-free boundary rendering.
    _, bnd_dense, _ = lobed_chamber(360)
    xg = np.linspace(-1.05, 1.05, 210)
    yg = np.linspace(-1.05, 1.05, 210)
    XX, YY = np.meshgrid(xg, yg, indexing="xy")
    grid_pts = np.column_stack([XX.ravel(), YY.ravel()])
    eval_mask = MplPath(bnd_dense * 1.04).contains_points(grid_pts).reshape(XX.shape)
    strict_in = MplPath(bnd_dense).contains_points(grid_pts).reshape(XX.shape)
    eval_pts = grid_pts[eval_mask.ravel()]

    vel_grid = taylor_green_velocity(grid_pts)
    st_mask = eval_mask & ((XX**2 + YY**2) > 0.08**2)
    VX = np.where(st_mask, vel_grid[:, 0].reshape(XX.shape), np.nan)
    VY = np.where(st_mask, vel_grid[:, 1].reshape(XX.shape), np.nan)

    _, X_b, nrm_b = lobed_chamber(144)
    g_b = bc_fn(X_b)
    fields = {}
    green_hi = None
    for Pe, n_box in [(18.0, 131), (150.0, 175)]:
        t0 = time.perf_counter()
        D_val = L_char / Pe
        pde = PDE.advection_diffusion(diffusivity=D_val, velocity=taylor_green_velocity, reaction=c_react, dim=2)
        green = SplitGreen(pde, bounds=[(-1.42, 1.42), (-1.42, 1.42)], shape=n_box, bc="dirichlet")
        sol = solve_mfs(pde, green, X_b, g_b, normals=nrm_b, offset=0.20)
        u_in = np.full(XX.shape, np.nan)
        u_in[eval_mask] = sol(eval_pts)
        fields[Pe] = u_in
        if Pe == 150.0:
            green_hi = green
        print(f"[Chamber] Pe={Pe:5.1f}: solved in {time.perf_counter() - t0:.2f}s (BC rel residual = {sol.residual:.2e})")

    # Box-Dirichlet Green's function G(x, y_0) restricted to the chamber at Pe = 150
    y0 = np.array([[-0.38, 0.22]])
    G_sample = np.full(XX.shape, np.nan)
    G_sample[eval_mask] = green_hi.eval(eval_pts, y0)[:, 0]  # type: ignore[union-attr]

    # Validation across Péclet numbers on R = 0.80 disk against polar FD solver
    R_disk = 0.80
    th_d = np.linspace(0.0, 2.0 * np.pi, 136, endpoint=False)
    X_bd = np.column_stack([R_disk * np.cos(th_d), R_disk * np.sin(th_d)])
    nrm_d = np.column_stack([np.cos(th_d), np.sin(th_d)])
    pe_list = [15.0, 45.0, 95.0, 150.0]
    rel_errs = []
    line_cmp = None
    for Pe in pe_list:
        D_val = (2.0 * R_disk) / Pe
        pde_d = PDE.advection_diffusion(diffusivity=D_val, velocity=taylor_green_velocity, reaction=c_react, dim=2)
        green_d = SplitGreen(pde_d, bounds=[(-1.35, 1.35), (-1.35, 1.35)], shape=165, bc="dirichlet")
        sol_d = solve_mfs(pde_d, green_d, X_bd, bc_fn(X_bd), normals=nrm_d, offset=0.20)
        rs, ths, Rg, Tg, u_pol = solve_disk_polar_reference(D_val, c_react, R_disk, bc_fn)
        mask_r = (rs >= 0.14) & (rs <= 0.72)
        pts_cmp = np.column_stack([(Rg[mask_r] * np.cos(Tg[mask_r])).ravel(), (Rg[mask_r] * np.sin(Tg[mask_r])).ravel()])
        err = float(np.linalg.norm(sol_d(pts_cmp) - u_pol[mask_r].ravel()) / np.linalg.norm(u_pol[mask_r].ravel()))
        rel_errs.append(err)
        if Pe == 150.0:
            x_sec = np.concatenate([-rs[::-1], rs])
            u_ref_sec = np.concatenate([u_pol[::-1, len(ths) // 2], u_pol[:, 0]])
            u_mfs_sec = sol_d(np.column_stack([x_sec, np.zeros_like(x_sec)]))
            line_cmp = (x_sec, u_ref_sec, u_mfs_sec)
        print(f"[Disk Validation] Pe={Pe:5.1f}: interior rel L2 error = {err:.4e}")

    # Plot
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10, "figure.facecolor": "white"})
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 10.2))
    plt.subplots_adjust(wspace=0.27, hspace=0.30)

    shared_vmax = float(max(np.nanmax(fields[18.0][strict_in]), np.nanmax(fields[150.0][strict_in])))
    for ax, Pe, label in [
        (axes[0, 0], 18.0, r"(a) Moderate-Péclet Transport ($\mathrm{Pe} = 18$)"),
        (axes[0, 1], 150.0, r"(b) High-Péclet Swirling Plume ($\mathrm{Pe} = 150$)"),
    ]:
        clip_poly = Polygon(bnd_dense, closed=True, transform=ax.transData)
        im = ax.pcolormesh(XX, YY, np.clip(fields[Pe], 0.0, shared_vmax), cmap="magma", vmin=0.0, vmax=shared_vmax, shading="gouraud")
        im.set_clip_path(clip_poly)
        st = ax.streamplot(xg, yg, VX, VY, color=(1.0, 1.0, 1.0, 0.36), density=0.92, linewidth=0.75, arrowsize=0.8, maxlength=1.35)
        _clip_streamplot(st, clip_poly)
        ax.plot(np.append(bnd_dense[:, 0], bnd_dense[0, 0]), np.append(bnd_dense[:, 1], bnd_dense[0, 1]), "k-", lw=1.7)
        ax.set_aspect("equal")
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_title(label, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.044, pad=0.02, label=r"$u(x, y)$")

    # (c) Box Green's function G(x, y_0) restricted to chamber
    ax_g = axes[1, 0]
    clip_g = Polygon(bnd_dense, closed=True, transform=ax_g.transData)
    g_vmax = float(np.nanpercentile(G_sample[strict_in], 97.5))
    im_g = ax_g.pcolormesh(XX, YY, np.clip(G_sample, 0.0, g_vmax), cmap="viridis", vmin=0.0, vmax=g_vmax, shading="gouraud")
    im_g.set_clip_path(clip_g)
    st_g = ax_g.streamplot(xg, yg, VX, VY, color=(1.0, 1.0, 1.0, 0.36), density=0.90, linewidth=0.75, arrowsize=0.8, maxlength=1.35)
    _clip_streamplot(st_g, clip_g)
    ax_g.plot(y0[0, 0], y0[0, 1], "o", linestyle="none", ms=7.2, color="#e63946", mec="white", mew=1.3, zorder=6, label=r"Source $y_0$")
    ax_g.plot(np.append(bnd_dense[:, 0], bnd_dense[0, 0]), np.append(bnd_dense[:, 1], bnd_dense[0, 1]), "k-", lw=1.7)
    ax_g.set_aspect("equal")
    ax_g.set_xlim(-1.05, 1.05)
    ax_g.set_ylim(-1.05, 1.05)
    ax_g.set_xlabel(r"$x$")
    ax_g.set_ylabel(r"$y$")
    ax_g.set_title(r"(c) Box Green's Function $G(x, y_0) = G_0 + H$", fontweight="bold")
    ax_g.legend(loc="lower right", fontsize=9, framealpha=0.92)
    fig.colorbar(im_g, ax=ax_g, fraction=0.044, pad=0.02, label=r"$G(x, y_0)$")

    # (d) Validation disk cross-section & Péclet error sweep
    ax_v = axes[1, 1]
    x_sec, u_ref_sec, u_mfs_sec = line_cmp  # type: ignore[misc]
    ax_v.plot(x_sec, u_ref_sec, "k-", lw=2.1, label=r"Polar interior FD ($\mathrm{Pe} = 150$)")
    ax_v.plot(x_sec[::4], u_mfs_sec[::4], "o", linestyle="none", color="#d95f02", ms=4.5, label="Boundary MFS")
    ax_v.set_ylim(-0.02, 1.08)
    ax_v.set_xlabel(r"Disk cross-section $x$ ($y = 0$)")
    ax_v.set_ylabel(r"$u(x, 0)$")
    ax_v.set_title(r"(d) Disk Cross-Section ($\mathrm{Pe} = 150$) & Error vs. $\mathrm{Pe}$", fontweight="bold")
    ax_v.grid(True, alpha=0.25)
    ax_v.legend(loc="upper right", fontsize=8.5, framealpha=0.92)

    ax_in = ax_v.inset_axes([0.36, 0.47, 0.57, 0.31])
    ax_in.semilogy(pe_list, rel_errs, "-s", color="#0b559f", ms=4.2, lw=1.5)
    ax_in.set_xticks(pe_list)
    ax_in.set_ylim(4e-5, 2.5e-3)
    ax_in.set_xlabel(r"Péclet number $\mathrm{Pe}$", fontsize=8, labelpad=2)
    ax_in.set_ylabel(r"Interior rel. $L^2$ err", fontsize=8, labelpad=2)
    ax_in.tick_params(labelsize=7.5)
    ax_in.grid(True, alpha=0.3)

    out_path = out_dir / "02_taylor_green_peclet.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
