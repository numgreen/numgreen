"""2D acoustic scattering in a Munk ocean waveguide and a smooth Luneburg lens.

Demonstrates obstacle scattering in inhomogeneous media ``-Delta u - k(x)^2 u = 0``
using a singularity-split background Green's function on a PML-truncated box
coupled to proxy-source MFS on the obstacle boundary:

1. Deep-ocean SOFAR channel with Munk's (1974) sound-speed profile c(z),
   sweeping a sound-hard kite scatterer across 25 depths using a single
   background LU factorization.
2. Smooth graded-index Luneburg lens n(r) = sqrt(2 - (r/R)^2), comparing the
   background focal wavefield against a 1D radial Fourier-Bessel finite-difference
   reference and checking MFS boundary convergence for a near-focus scatterer.

Generates:
    assets/01_sofar_grin_scattering.png
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib
import numpy as np
from scipy.special import hankel1, jv
import scipy.sparse as sp
import scipy.sparse.linalg as spla

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from numgreen import PDE, SplitGreen, solve_mfs
from numgreen.split import _interp_cubic


def munk_sound_speed(z: np.ndarray, c0: float = 1.0, z_axis: float = 0.0, scale: float = 0.85, eps: float = 0.22) -> np.ndarray:
    """Munk (1974) ocean sound-speed profile (z < 0 deeper ocean, z > 0 thermocline)."""
    zt = -2.0 * (z - z_axis) / scale
    return c0 * (1.0 + eps * (zt - 1.0 + np.exp(-np.clip(zt, -3.2, 8.0))))


def luneburg_index(X: np.ndarray, R: float = 0.78, tau: float = 0.07) -> np.ndarray:
    """Smooth C^infinity regularization of the Luneburg lens profile."""
    r2 = X[:, 0] ** 2 + X[:, 1] ** 2
    return np.sqrt(1.0 + tau * np.logaddexp(0.0, (1.0 - r2 / R**2) / tau))


def kite_boundary(n_pts: int, center: tuple[float, float], scale: float = 0.17):
    t = np.linspace(0.0, 2.0 * np.pi, n_pts, endpoint=False)
    x = scale * (np.cos(t) + 0.35 * np.cos(2.0 * t) - 0.35)
    y = scale * (1.25 * np.sin(t))
    dx = scale * (-np.sin(t) - 0.70 * np.sin(2.0 * t))
    dy = scale * (1.25 * np.cos(t))
    ds = np.hypot(dx, dy)
    pts = np.column_stack([x + center[0], y + center[1]])
    normals = np.column_stack([dy / ds, -dx / ds])
    return pts, normals


def radial_luneburg_reference(xs: np.ndarray, k0: float, R: float = 0.78, n_modes: int = 20) -> np.ndarray:
    """1D radial Fourier-Bessel finite-difference reference along y = 0."""
    R_out = 1.45
    r_ode = np.linspace(1e-5, R_out, 1200)
    dr = r_ode[1] - r_ode[0]
    n2 = luneburg_index(np.column_stack([r_ode, np.zeros_like(r_ode)]), R=R) ** 2
    modes = []
    for m in range(n_modes):
        z = k0 * R_out
        Hm = hankel1(m, z)
        dHm = 0.5 * (hankel1(m - 1, z) - hankel1(m + 1, z)) if m > 0 else -hankel1(1, z)
        Jm = jv(m, z)
        dJm = 0.5 * (jv(m - 1, z) - jv(m + 1, z)) if m > 0 else -jv(1, z)
        ratio = k0 * dHm / Hm
        rhs_bc = (1j**m) * (k0 * dJm - ratio * Jm)

        main = -2.0 / dr**2 + (k0**2 * n2 - m**2 / r_ode**2)
        up = 1.0 / dr**2 + 1.0 / (2.0 * r_ode[:-1] * dr)
        lo = 1.0 / dr**2 - 1.0 / (2.0 * r_ode[1:] * dr)
        A1 = sp.diags([lo, main, up], [-1, 0, 1], format="lil", dtype=complex)
        b1 = np.zeros(len(r_ode), dtype=complex)
        if m == 0:
            A1[0, 0], A1[0, 1] = -2.0 / dr**2 + k0**2 * n2[0], 2.0 / dr**2
        else:
            A1[0, :] = 0
            A1[0, 0] = 1.0
        cg = 1.0 / dr**2 + 1.0 / (2.0 * R_out * dr)
        A1[-1, :] = 0
        A1[-1, -2] = 2.0 / dr**2
        A1[-1, -1] = -2.0 / dr**2 + (k0**2 * n2[-1] - m**2 / R_out**2) + cg * (2.0 * dr * ratio)
        b1[-1] = -cg * (2.0 * dr * rhs_bc)
        modes.append(spla.spsolve(A1.tocsc(), b1))

    out = np.zeros(len(xs), dtype=complex)
    for i, xv in enumerate(xs):
        rv = abs(xv)
        sgn = 1.0 if xv >= 0 else -1.0
        val = np.interp(max(rv, 1e-5), r_ode, modes[0])
        if rv >= 1e-5:
            for m in range(1, n_modes):
                val += 2.0 * np.interp(rv, r_ode, modes[m]) * (sgn**m)
        out[i] = val
    return out


def main() -> None:
    out_dir = Path("assets")
    out_dir.mkdir(exist_ok=True)

    # 1. SOFAR Munk waveguide + 25-depth obstacle sweep
    omega = 9.5
    c_far = float(munk_sound_speed(np.array([0.8])).item())
    k0_sofar = omega / c_far
    k_munk = lambda X: omega / munk_sound_speed(X[:, 1])

    t0 = time.perf_counter()
    pde_sofar = PDE.helmholtz(k=k_munk, k0=k0_sofar, dim=2, name="SOFARMunkWaveguide")
    green_sofar = SplitGreen(pde_sofar, bounds=[(-3.4, 3.4), (-2.1, 2.1)], shape=(211, 141), bc="absorbing", pml_width=0.65)
    print(f"[SOFAR] Factorized waveguide operator in {time.perf_counter() - t0:.2f}s")

    x_src = np.array([[-2.15, 0.05]])
    z_sweep = np.linspace(-1.05, 1.05, 25)
    sensors = np.column_stack([np.full(21, 1.85), np.linspace(-0.9, 0.9, 21)])

    # Evaluate incident field from x_src before tabulating the obstacle sweep region
    xg_s = np.linspace(-2.55, 2.35, 240)
    zg_s = np.linspace(-1.35, 1.35, 145)
    Xs, Zs = np.meshgrid(xg_s, zg_s, indexing="xy")
    pts_s = np.column_stack([Xs.ravel(), Zs.ravel()])
    u_inc_s = green_sofar.eval(pts_s, x_src)[:, 0]
    inc_beam_sweep = np.abs(green_sofar.eval(np.column_stack([np.full_like(z_sweep, 0.15), z_sweep]), x_src)[:, 0])
    all_bnds = [kite_boundary(72, center=(0.15, float(z0)), scale=0.17) for z0 in z_sweep]
    all_bnd_pts = np.vstack([b[0] for b in all_bnds])
    _, gx_inc_all = green_sofar.eval(all_bnd_pts, x_src, grad_x=True)
    gx_inc_blocks = gx_inc_all[:, 0, :].reshape(len(z_sweep), 72, 2)

    green_sofar.tabulate_sources(bounds=[(-0.15, 0.42), (-1.32, 1.32)], shape=(10, 35))

    t_sw0 = time.perf_counter()
    scat_power = np.zeros(len(z_sweep))
    residuals = np.zeros(len(z_sweep))
    featured_idx = 8  # lower-channel peak (z_0 = -0.35)
    featured_sol = None

    for idx, z0 in enumerate(z_sweep):
        bnd_pts, bnd_nrm = all_bnds[idx]
        du_inc_dn = np.sum(gx_inc_blocks[idx] * bnd_nrm, axis=1)
        sol = solve_mfs(
            pde_sofar,
            green_sofar,
            bnd_pts,
            -du_inc_dn,
            normals=bnd_nrm,
            offset=0.055,
            interior=False,
            alpha=0.0,
            beta=1.0,
            conormal_flux=False,
        )
        scat_power[idx] = float(np.sqrt(np.mean(np.abs(sol(sensors)) ** 2)))
        residuals[idx] = sol.residual
        if idx == featured_idx:
            featured_sol = (sol, bnd_pts, float(z0))

    print(
        f"[SOFAR] 25-depth sweep solved in {time.perf_counter() - t_sw0:.2f}s "
        f"(max boundary residual = {np.max(residuals):.2e})"
    )

    sol_feat, bnd_feat, z_feat = featured_sol  # type: ignore[misc]
    u_tot_s = (u_inc_s + sol_feat(pts_s)).reshape(Xs.shape)
    from matplotlib.path import Path as MplPath
    ext_vals_s = u_tot_s[~MplPath(bnd_feat).contains_points(pts_s).reshape(Xs.shape)]
    vlim = float(np.percentile(np.abs(np.real(ext_vals_s)), 98.5))

    # 2. Smooth GRIN Luneburg lens focusing + near-focus scatterer
    k0_lun = 7.0
    R_lun = 0.78
    pde_lun = PDE.helmholtz(k=lambda X: k0_lun * luneburg_index(X, R=R_lun), k0=k0_lun, dim=2, name="LuneburgGRIN")
    green_lun = SplitGreen(pde_lun, bounds=[(-2.25, 2.25), (-2.25, 2.25)], shape=171, bc="absorbing")

    gp = green_lun.grid_points
    k2_g = (k0_lun * luneburg_index(gp, R=R_lun)) ** 2
    rhs_bg = (k2_g - k0_lun**2) * green_lun._chi * np.exp(1j * k0_lun * gp[:, 0])
    rhs_bg[green_lun._bnd_mask] = 0.0
    u_bg_scat_grid = green_lun._lu.solve(rhs_bg).reshape(*green_lun.shape, 1)

    def eval_luneburg_bg(X_q: np.ndarray) -> np.ndarray:
        val, _ = _interp_cubic(green_lun.axes, u_bg_scat_grid, X_q, want_grad=False)
        return np.exp(1j * k0_lun * X_q[:, 0]) + val[:, 0]

    x_chk = np.linspace(-1.3, 1.3, 180)
    u_bg_line = eval_luneburg_bg(np.column_stack([x_chk, np.zeros_like(x_chk)]))
    u_fb_line = radial_luneburg_reference(x_chk, k0=k0_lun, R=R_lun)
    lun_err = float(np.linalg.norm(u_bg_line - u_fb_line) / np.linalg.norm(u_fb_line))
    print(f"[Luneburg] Axis rel L2 error vs radial FD reference = {lun_err:.4e}")

    obs_center = np.array([0.92, 0.14])
    obs_rad = 0.13
    n_conv = [16, 24, 36, 52, 72, 96]
    th_fine = np.linspace(0.007, 2.0 * np.pi + 0.007, 240, endpoint=False)
    X_fine = obs_center + obs_rad * np.column_stack([np.cos(th_fine), np.sin(th_fine)])
    u_bg_fine = eval_luneburg_bg(X_fine)
    conv_errs = []
    sol_lun_obs = None
    for nb in n_conv:
        th_b = np.linspace(0.0, 2.0 * np.pi, nb, endpoint=False)
        nrm_b = np.column_stack([np.cos(th_b), np.sin(th_b)])
        X_b = obs_center + obs_rad * nrm_b
        sol_nb = solve_mfs(pde_lun, green_lun, X_b, -eval_luneburg_bg(X_b), normals=nrm_b, offset=0.055, interior=False)
        conv_errs.append(float(np.linalg.norm(sol_nb(X_fine) + u_bg_fine) / np.linalg.norm(u_bg_fine)))
        if nb == 72:
            sol_lun_obs = sol_nb
    print(f"[Luneburg] MFS boundary error N=16 -> N=96: {conv_errs[0]:.2e} -> {conv_errs[-1]:.2e}")

    xg_l = np.linspace(-1.35, 1.55, 200)
    yg_l = np.linspace(-1.20, 1.20, 180)
    Xl, Yl = np.meshgrid(xg_l, yg_l, indexing="xy")
    pts_l = np.column_stack([Xl.ravel(), Yl.ravel()])
    u_lun_tot = (eval_luneburg_bg(pts_l) + sol_lun_obs(pts_l)).reshape(Xl.shape)  # type: ignore[misc]
    in_cyl = ((Xl - obs_center[0]) ** 2 + (Yl - obs_center[1]) ** 2) < obs_rad**2
    vmax_lun = float(np.max(np.abs(u_lun_tot[~in_cyl])))

    # Plot
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10, "figure.facecolor": "white"})
    fig = plt.figure(figsize=(15.4, 9.2))
    outer = GridSpec(2, 1, height_ratios=[1.0, 1.05], hspace=0.33)
    gs_top = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0], width_ratios=[0.38, 1.38, 1.12], wspace=0.45)
    gs_bot = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1], width_ratios=[1.18, 1.15], wspace=0.34)

    # (a) Munk profile
    ax_munk = fig.add_subplot(gs_top[0, 0])
    z_prof = np.linspace(-1.35, 1.35, 300)
    ax_munk.plot(munk_sound_speed(z_prof), z_prof, color="#0b559f", lw=2.2)
    ax_munk.axhline(0.0, color="#d95f02", ls="--", lw=1.1, label="Channel axis")
    ax_munk.set_xlim(0.85, 5.65)
    ax_munk.set_xticks([1.0, 2.5, 4.0, 5.5])
    ax_munk.set_ylim(-1.35, 1.35)
    ax_munk.set_xlabel(r"Sound speed $c(z)/c_0$")
    ax_munk.set_ylabel(r"Depth $z$")
    ax_munk.set_title("(a) Munk Profile", fontweight="bold")
    ax_munk.grid(True, alpha=0.25)
    ax_munk.legend(loc="lower right", fontsize=8, frameon=True)

    # (b) SOFAR waveguide field (unmasked pcolormesh + vector polygon overlay avoids sawtooth teeth)
    ax_sofar = fig.add_subplot(gs_top[0, 1])
    im_s = ax_sofar.pcolormesh(Xs, Zs, np.clip(np.real(u_tot_s), -vlim, vlim), cmap="RdBu_r", vmin=-vlim, vmax=vlim, shading="gouraud")
    ax_sofar.fill(bnd_feat[:, 0], bnd_feat[:, 1], facecolor="#262626", edgecolor="white", lw=1.1, zorder=5)
    ax_sofar.plot(x_src[0, 0], x_src[0, 1], marker="*", linestyle="none", ms=11.5, color="#ffcc00", mec="black", zorder=6, label="Source")
    ax_sofar.plot(sensors[:, 0], sensors[:, 1], "o", linestyle="none", ms=3.2, color="#1b9e77", label="Sensor array")
    ax_sofar.axhline(0.0, color="#d95f02", ls="--", lw=0.95, alpha=0.75)
    ax_sofar.set_xlim(-2.55, 2.35)
    ax_sofar.set_ylim(-1.35, 1.35)
    ax_sofar.set_xlabel(r"Range $x$")
    ax_sofar.set_ylabel(r"Depth $z$")
    ax_sofar.set_title(r"(b) SOFAR Waveguide Field $\mathrm{Re}\,u_{\mathrm{tot}}$", fontweight="bold")
    ax_sofar.legend(loc="upper right", fontsize=8.5, framealpha=0.92)
    fig.colorbar(im_s, ax=ax_sofar, fraction=0.042, pad=0.02, label=r"$\mathrm{Re}\,u_{\mathrm{tot}}$")

    # (c) Depth sweep
    ax_sw = fig.add_subplot(gs_top[0, 2])
    ax_sw.plot(z_sweep, scat_power / np.max(scat_power), "-o", color="#d95f02", ms=4.3, lw=1.9, label="Scattered RMS at array")
    ax_sw.plot(z_sweep, inc_beam_sweep / np.max(inc_beam_sweep), "--", color="#0b559f", lw=1.5, label=r"Incident $|u_{\mathrm{inc}}(0.15, z_0)|$")
    ax_sw.axvline(z_feat, color="#444444", ls=":", lw=1.2, label=f"Panel (b) depth $z_0={z_feat:.2f}$")
    ax_sw.set_ylim(-0.04, 1.38)
    ax_sw.set_xlabel(r"Obstacle depth $z_0$")
    ax_sw.set_ylabel("Normalized amplitude")
    ax_sw.set_title("(c) Array Response vs. Obstacle Depth", fontweight="bold")
    ax_sw.grid(True, alpha=0.25)
    ax_sw.legend(loc="upper right", fontsize=8, framealpha=0.92)

    # (d) Luneburg lens focusing + scatterer
    ax_lun = fig.add_subplot(gs_bot[0, 0])
    im_l = ax_lun.pcolormesh(Xl, Yl, np.clip(np.abs(u_lun_tot), 0.0, vmax_lun), cmap="inferno", vmin=0.0, vmax=vmax_lun, shading="gouraud")
    th_ring = np.linspace(0.0, 2.0 * np.pi, 220)
    ax_lun.plot(R_lun * np.cos(th_ring), R_lun * np.sin(th_ring), "--", color="cyan", lw=1.35, label=r"Lens rim $r = R$")
    ax_lun.fill(obs_center[0] + obs_rad * np.cos(th_ring), obs_center[1] + obs_rad * np.sin(th_ring), facecolor="#eaeaea", edgecolor="cyan", lw=1.3, zorder=5, label="Sound-soft scatterer")
    ax_lun.set_aspect("equal")
    ax_lun.set_xlim(-1.35, 1.55)
    ax_lun.set_ylim(-1.20, 1.20)
    ax_lun.set_xlabel(r"$x$")
    ax_lun.set_ylabel(r"$y$")
    ax_lun.set_title(r"(d) Luneburg Lens Focusing & Scattering $|u_{\mathrm{tot}}|$", fontweight="bold")
    ax_lun.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
    fig.colorbar(im_l, ax=ax_lun, fraction=0.040, pad=0.02, label=r"$|u_{\mathrm{tot}}(x, y)|$")

    # (e) Luneburg axial validation & MFS convergence
    ax_val = fig.add_subplot(gs_bot[0, 1])
    ax_val.plot(x_chk, np.abs(u_fb_line), "k-", lw=2.0, label="1D radial FD reference")
    ax_val.plot(x_chk[::3], np.abs(u_bg_line[::3]), "o", color="#e7298a", ms=3.8, label=f"2D box PML (rel. err {lun_err:.1%})")
    ax_val.set_ylim(0.82, 2.34)
    ax_val.set_xlabel(r"Optical axis $x$ ($y = 0$)")
    ax_val.set_ylabel(r"Axial amplitude $|u_{\mathrm{bg}}(x, 0)|$")
    ax_val.set_title("(e) Axial Profile & MFS Convergence", fontweight="bold")
    ax_val.grid(True, alpha=0.25)
    ax_val.legend(loc="upper left", fontsize=8.3, framealpha=0.92)

    ax_in = ax_val.inset_axes([0.12, 0.44, 0.43, 0.34])
    ax_in.semilogy(n_conv, conv_errs, "-s", color="#1b9e77", ms=3.6, lw=1.4)
    ax_in.set_xlabel(r"$N_{\mathrm{DOF}}$", fontsize=8, labelpad=2.5)
    ax_in.set_ylabel("Rel. BC err", fontsize=8, labelpad=2.5)
    ax_in.tick_params(labelsize=7.5)
    ax_in.grid(True, alpha=0.3)

    out_path = out_dir / "01_sofar_grin_scattering.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
