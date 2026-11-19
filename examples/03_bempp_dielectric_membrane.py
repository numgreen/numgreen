"""3D electrostatic solvation across a smooth dielectric bilayer with bempp-cl.

Solves coupled interior-Laplace / exterior-variable-permittivity boundary
integral equations across a smooth lipid bilayer profile eps(z) in [2, 80] by
combining Liouville-scaled bempp-cl DP0 operators on a reference surface mesh
with a tabulated 3D smooth remainder (compare Cheng 1984, Sawhney et al. 2022,
Fisicaro et al. 2016, and Parsegian 1969).

Generates:
    assets/03_bempp_dielectric_membrane.png
"""

from __future__ import annotations

import time
from pathlib import Path

import bempp_cl.api as bempp
import matplotlib
import numpy as np
from scipy.interpolate import griddata

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from numgreen import BemppOperators, PDE, SplitGreen


def bilayer_eps(X: np.ndarray, eps_mem: float = 2.0, eps_wat: float = 80.0, L_m: float = 1.9, w: float = 0.55) -> np.ndarray:
    """Smooth symmetric bilayer permittivity profile eps(z) in [2, 80]."""
    z = X[:, 2]
    return eps_mem + 0.5 * (eps_wat - eps_mem) * (2.0 - np.tanh((z + L_m) / w) + np.tanh((z - L_m) / w))


def sphere_grid(level: int, radius: float = 0.85):
    g0 = bempp.shapes.regular_sphere(level)
    verts = np.asfortranarray(radius * g0.vertices / np.linalg.norm(g0.vertices, axis=0, keepdims=True))
    return bempp.Grid(verts, np.asfortranarray(g0.elements))


def peanut_molecule_grid(level: int = 3, r0: float = 0.76):
    """Prolate diatomic 3D surface mesh deformed from a regular sphere."""
    g0 = bempp.shapes.regular_sphere(level)
    u = g0.vertices / np.linalg.norm(g0.vertices, axis=0, keepdims=True)
    rad = r0 * (1.0 + 0.26 * (1.5 * u[2, :] ** 2 - 0.5))
    verts = u * rad[None, :]
    verts[0, :] *= 0.88
    verts[1, :] *= 0.88
    return bempp.Grid(np.asfortranarray(verts), np.asfortranarray(g0.elements))


def main() -> None:
    out_dir = Path("assets")
    out_dir.mkdir(exist_ok=True)

    # 1. Background 3D sparse factorization and source tabulation
    t0 = time.perf_counter()
    pde_mem = PDE.dielectric(eps=bilayer_eps, dim=3)
    green_mem = SplitGreen(pde_mem, bounds=[(-4.5, 4.5), (-4.5, 4.5), (-7.8, 7.8)], shape=(25, 25, 43), bc="absorbing")
    green_mem.tabulate_sources(bounds=[(-1.15, 1.15), (-1.15, 1.15), (-5.6, 5.6)], shape=(8, 8, 21))
    print(f"[Bilayer] Factorized and tabulated background Green's function in {time.perf_counter() - t0:.2f}s")

    # 2. Translocation sweep for a spherical ion and a polar diatomic cavity
    R_ion = 0.85
    eps_in = 2.0
    grid_ion = sphere_grid(level=3, radius=R_ion)
    grid_mol = peanut_molecule_grid(level=3, r0=0.76)

    t_bem0 = time.perf_counter()
    bop_ion = BemppOperators(pde_mem, green_mem, grid_ion)
    bop_mol = BemppOperators(pde_mem, green_mem, grid_mol)
    print(f"[bempp-cl] Assembled reference DP0 operators in {time.perf_counter() - t_bem0:.2f}s")

    z_sweep = np.linspace(-4.8, 4.8, 25)
    q_ion, rq_ion = [1.0], np.array([[0.0, 0.0, 0.0]])
    q_mol, rq_mol = [0.85, -0.85], np.array([[0.0, 0.0, 0.36], [0.0, 0.0, -0.36]])

    t_sw0 = time.perf_counter()
    dG_ion = np.zeros(len(z_sweep))
    dG_mol = np.zeros(len(z_sweep))
    interface_sol = None

    for i, zc in enumerate(z_sweep):
        sh = (0.0, 0.0, float(zc))
        s_i = bop_ion.solve_dielectric_cavity(q_ion, rq_ion, eps_in=eps_in, shift=sh)
        s_m = bop_mol.solve_dielectric_cavity(q_mol, rq_mol, eps_in=eps_in, shift=sh)
        dG_ion[i] = s_i.energy
        dG_mol[i] = s_m.energy
        if abs(zc - 2.0) < 0.15:
            interface_sol = (float(zc), s_m)

    print(f"[Sweep] 25 bilayer positions solved in {time.perf_counter() - t_sw0:.2f}s")

    born_wat = -(1.0**2) / (8.0 * np.pi * R_ion) * (1.0 / eps_in - 1.0 / 80.0)
    print(f"[Solvation] z=+4.8: numgreen={dG_ion[-1]:.5f} vs bulk Born={born_wat:.5f}; midplane z=0: {dG_ion[12]:.5f}")

    # Cross-sectional reaction potential phi_rf(x, 0, z) = phi_tot - phi_Coulomb at z_c = 2.0
    # (continuous across Gamma and free of interior point-charge singularities)
    zc_if, sol_if = interface_sol  # type: ignore[misc]
    xs_sl = np.linspace(-2.1, 2.1, 155)
    zs_sl = np.linspace(-0.35, 4.55, 175)
    Xsl, Zsl = np.meshgrid(xs_sl, zs_sl, indexing="xy")
    pts_sl = np.column_stack([Xsl.ravel(), np.zeros(Xsl.size), Zsl.ravel()])

    rel = pts_sl - np.array([[0.0, 0.0, zc_if]])
    rel_sc = rel.copy()
    rel_sc[:, 0] /= 0.88
    rel_sc[:, 1] /= 0.88
    r_sl = np.maximum(np.linalg.norm(rel_sc, axis=1), 1e-12)
    cos_sl = rel_sc[:, 2] / r_sl
    r_surf = 0.76 * (1.0 + 0.26 * (1.5 * cos_sl**2 - 0.5))
    ext_mask = r_sl > r_surf * 1.045
    in_mask = r_sl < r_surf * 0.955
    shell_mask = ~(ext_mask | in_mask)

    C_if = bop_mol.centroids + np.array([[0.0, 0.0, zc_if]])
    rq_if = rq_mol + np.array([[0.0, 0.0, zc_if]])

    def coulomb_in_ref(P: np.ndarray) -> np.ndarray:
        d = P[:, None, :] - rq_if[None, :, :]
        r = np.maximum(np.linalg.norm(d, axis=-1), 1e-12)
        return np.sum(np.array(q_mol)[None, :] / (4.0 * np.pi * eps_in * r), axis=1)

    phi_rf_sl = np.empty(pts_sl.shape[0], dtype=float)
    # Exterior reaction potential: u_ext - u_Coulomb(eps_in)
    u_ext_pts = bop_mol.eval_exterior(pts_sl[ext_mask], sol_if.u_total, sol_if.sigma_ext, shift=(0.0, 0.0, zc_if))
    phi_rf_sl[ext_mask] = u_ext_pts - coulomb_in_ref(pts_sl[ext_mask])
    # Interior reaction potential: smooth harmonic representation
    pts_in = pts_sl[in_mask]
    d_in_c = pts_in[:, None, :] - C_if[None, :, :]
    r_in_c = np.linalg.norm(d_in_c, axis=-1)
    phi_rf_sl[in_mask] = ((1.0 / (4.0 * np.pi * r_in_c)) * bop_mol.areas[None, :]) @ sol_if.du_rf_dn - (
        (np.sum(d_in_c * bop_mol.normals[None, :, :], axis=-1) / (4.0 * np.pi * r_in_c**3)) * bop_mol.areas[None, :]
    ) @ sol_if.u_rf
    # Bridge the narrow near-boundary shell smoothly using the exact boundary trace u_rf on Gamma
    th_b_dense = np.linspace(0.0, 2.0 * np.pi, 180, endpoint=False)
    rb_dense = 0.76 * (1.0 + 0.26 * (1.5 * np.cos(th_b_dense) ** 2 - 0.5))
    surf_xz = np.column_stack([0.88 * rb_dense * np.sin(th_b_dense), zc_if + rb_dense * np.cos(th_b_dense)])
    # Project C_if near y = 0 + interior/exterior sample ring for smooth interpolation across shell
    known_mask = ~shell_mask
    known_xz = np.vstack([pts_sl[known_mask][:, [0, 2]], C_if[np.abs(C_if[:, 1]) < 0.18][:, [0, 2]]])
    known_vals = np.concatenate([phi_rf_sl[known_mask], sol_if.u_rf[np.abs(C_if[:, 1]) < 0.18]])
    phi_rf_sl[shell_mask] = griddata(known_xz, known_vals, pts_sl[shell_mask][:, [0, 2]], method="linear")
    phi_rf_2d = phi_rf_sl.reshape(Xsl.shape)

    # 3. 3D mesh-refinement convergence against closed-form exponential permittivity
    alpha_ex = 0.55
    eps_ex_fn = lambda X: np.exp(2.0 * alpha_ex * X[:, 2])
    pde_ex = PDE.dielectric(eps=eps_ex_fn, dim=3)
    green_ex = SplitGreen(pde_ex, bounds=[(-4.8, 4.8)] * 3, shape=29, bc="absorbing").tabulate_sources(
        bounds=[(-1.05, 1.05)] * 3, shape=9
    )
    y0_ex = np.array([0.12, -0.15, 0.22])
    ang = np.linspace(0.0, 2.0 * np.pi, 20, endpoint=False)
    X_ext_chk = np.column_stack([1.55 * np.cos(ang), 1.55 * np.sin(ang), np.linspace(-0.85, 0.85, 20)])

    def exact_exp_sol(X: np.ndarray, normals: np.ndarray | None = None):
        d = X - y0_ex[None, :]
        r = np.linalg.norm(d, axis=1)
        sx, s0 = np.exp(alpha_ex * X[:, 2]), np.exp(alpha_ex * y0_ex[2])
        u = np.exp(-alpha_ex * r) / (4.0 * np.pi * r * sx * s0)
        if normals is None:
            return u
        dg = -(alpha_ex * r + 1.0) * np.exp(-alpha_ex * r) / (4.0 * np.pi * r**3 * sx * s0)
        grad = dg[:, None] * d
        grad[:, 2] -= alpha_ex * u
        return u, eps_ex_fn(X) * np.sum(grad * normals, axis=1)

    u_ext_ref = exact_exp_sol(X_ext_chk)
    panel_counts, flux_errs, pot_errs = [], [], []
    for lv in (1, 2, 3):
        gr = sphere_grid(lv, radius=0.9)
        bop = BemppOperators(pde_ex, green_ex, gr)
        ub, sig_ref = exact_exp_sol(bop.centroids, bop.normals)
        sig_num = bop.solve_exterior_dirichlet(ub)
        u_num = bop.eval_exterior(X_ext_chk, ub, sig_num)
        panel_counts.append(len(bop.areas))
        flux_errs.append(float(np.linalg.norm(sig_num - sig_ref) / np.linalg.norm(sig_ref)))
        pot_errs.append(float(np.linalg.norm(u_num - u_ext_ref) / np.linalg.norm(u_ext_ref)))
    print(f"[Convergence] Panels {panel_counts}: flux err = {[round(e, 4) for e in flux_errs]}, pot err = {[round(e, 4) for e in pot_errs]}")

    # Plot
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10, "figure.facecolor": "white"})
    fig = plt.figure(figsize=(15.2, 9.8))
    gs = GridSpec(2, 2, width_ratios=[1.05, 1.15], height_ratios=[1.0, 1.0], wspace=0.36, hspace=0.31)

    # (a) 3D triangulated cavity colored by surface reaction potential u_rf
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    verts_if = grid_mol.vertices.T + np.array([[0.0, 0.0, zc_if]])
    elems = grid_mol.elements.T
    tris = [verts_if[tri] for tri in elems]
    urf_vals = sol_if.u_rf
    vmax_rf = float(np.max(np.abs(urf_vals)))
    norm_rf = Normalize(vmin=-vmax_rf, vmax=vmax_rf)

    # Annular bilayer interface plane at z = +1.9 outside the cavity cross-section (avoids z-fighting)
    r_ann = np.linspace(0.68, 1.22, 10)
    th_ann = np.linspace(0.0, 2.0 * np.pi, 48)
    Ra, Ta = np.meshgrid(r_ann, th_ann)
    ax3d.plot_surface(Ra * np.cos(Ta), Ra * np.sin(Ta), np.full_like(Ra, 1.9), color="#f4a261", alpha=0.22, shade=False)

    poly = Poly3DCollection(tris, facecolors=plt.cm.coolwarm(norm_rf(urf_vals)), edgecolors=(0.15, 0.15, 0.15, 0.28), linewidths=0.35)
    ax3d.add_collection3d(poly)
    ax3d.set_xlim(-1.15, 1.15)
    ax3d.set_ylim(-1.15, 1.15)
    ax3d.set_zlim(0.60, 3.35)
    ax3d.set_box_aspect((2.3, 2.3, 2.75))
    ax3d.view_init(elev=19, azim=-48)
    ax3d.set_xlabel(r"$x$", labelpad=2)
    ax3d.set_ylabel(r"$y$", labelpad=2)
    ax3d.set_zlabel(r"$z$", labelpad=4)
    ax3d.set_title(r"(a) Surface Reaction Potential $u_{\mathrm{rf}}$ ($z_c = 2.0$, 512 Panels)", fontweight="bold")
    sm = ScalarMappable(norm=norm_rf, cmap="coolwarm")
    sm.set_array([])
    fig.colorbar(sm, ax=ax3d, fraction=0.036, pad=0.16, shrink=0.76, label=r"Reaction potential $u_{\mathrm{rf}}$")

    # (b) Smooth cross-sectional reaction potential phi_rf(x, 0, z)
    ax_sl = fig.add_subplot(gs[0, 1])
    v_sl = vmax_rf
    im_sl = ax_sl.pcolormesh(Xsl, Zsl, np.clip(phi_rf_2d, -v_sl, v_sl), cmap="coolwarm", vmin=-v_sl, vmax=v_sl, shading="gouraud")
    ax_sl.contour(Xsl, Zsl, np.clip(phi_rf_2d, -v_sl, v_sl), levels=12, colors="k", linewidths=0.55, alpha=0.42)
    th_p = np.linspace(0.0, 2.0 * np.pi, 240)
    rp_outline = 0.76 * (1.0 + 0.26 * (1.5 * np.cos(th_p) ** 2 - 0.5))
    ax_sl.plot(0.88 * rp_outline * np.sin(th_p), zc_if + rp_outline * np.cos(th_p), "k-", lw=1.9, label=r"Cavity boundary $\Gamma$")
    ax_sl.axhline(1.9, color="#d95f02", ls="--", lw=1.35, label=r"Bilayer headgroup $z = +L_m$")
    ax_sl.axhspan(-0.35, 1.9, color="#f4a261", alpha=0.07)
    ax_sl.set_aspect("equal")
    ax_sl.set_xlim(-2.1, 2.1)
    ax_sl.set_ylim(-0.35, 4.55)
    ax_sl.set_xlabel(r"$x$")
    ax_sl.set_ylabel(r"Height $z$")
    ax_sl.set_title(r"(b) Cross-Sectional Reaction Potential $\phi_{\mathrm{rf}}(x, 0, z)$", fontweight="bold")
    ax_sl.legend(loc="upper right", fontsize=8.4, framealpha=0.92)
    fig.colorbar(im_sl, ax=ax_sl, fraction=0.044, pad=0.02, label=r"$\phi_{\mathrm{rf}}(x, 0, z)$")

    # (c) Membrane translocation solvation barrier relative to bulk Born limit
    ax_fe = fig.add_subplot(gs[1, 0])
    ddG_ion = (dG_ion - born_wat) * 1e3
    ddG_mol = (dG_mol - dG_mol[-1]) * 1e3 * 4.0
    h_span = ax_fe.axvspan(-1.9, 1.9, color="#f4a261", alpha=0.16, label=r"Core $|z| < L_m$ ($\varepsilon \approx 2$)")
    (h_ion,) = ax_fe.plot(z_sweep, ddG_ion, "-o", color="#0b559f", ms=4.5, lw=2.0, label=r"Spherical ion $\Delta G_{\mathrm{solv}} - \Delta G_{\mathrm{Born}}$")
    (h_mol,) = ax_fe.plot(z_sweep, ddG_mol, "-s", color="#d95f02", ms=4.2, lw=1.8, label=r"Diatomic dipole $\Delta\Delta G_{\mathrm{solv}} \times 4$")
    ax_fe.axhline(0.0, color="#0b559f", ls=":", lw=1.1, alpha=0.7)
    ax_fe.set_ylim(-2.0, 52.0)
    ax_fe.set_xlabel(r"Cavity center $z_c$")
    ax_fe.set_ylabel(r"Transfer free energy ($\times 10^{-3}$ a.u.)")
    ax_fe.set_title(r"(c) Bilayer Desolvation Barrier ($\varepsilon(z) \in [2, 80]$)", fontweight="bold")
    ax_fe.grid(True, alpha=0.25)

    ax_eps = ax_fe.twinx()
    z_dense = np.linspace(-4.8, 4.8, 300)
    eps_dense = bilayer_eps(np.column_stack([np.zeros_like(z_dense), np.zeros_like(z_dense), z_dense]))
    (h_eps,) = ax_eps.plot(z_dense, eps_dense, "--", color="#2a9d8f", lw=1.5, alpha=0.8, label=r"Permittivity $\varepsilon(z)$")
    ax_eps.set_ylabel(r"Permittivity $\varepsilon(z)$", color="#2a9d8f")
    ax_eps.tick_params(axis="y", labelcolor="#2a9d8f")
    ax_eps.set_ylim(-3.0, 114.0)

    ax_fe.legend(
        handles=[h_ion, h_mol, h_eps, h_span],
        loc="upper center",
        ncol=2,
        fontsize=7.6,
        framealpha=0.93,
        columnspacing=1.0,
    )

    # (d) 3D surface mesh refinement convergence
    ax_cv = fig.add_subplot(gs[1, 1])
    h_eff = 1.0 / np.sqrt(np.array(panel_counts, dtype=float))
    ax_cv.loglog(panel_counts, flux_errs, "-o", color="#0b559f", ms=5.8, lw=2.0, label=r"Conormal flux $\sigma = \varepsilon\partial_n u$ rel. $L^2$ err")
    ax_cv.loglog(panel_counts, pot_errs, "-s", color="#e7298a", ms=5.8, lw=2.0, label=r"Exterior potential $u(x_{\mathrm{ext}})$ rel. $L^2$ err")
    ax_cv.loglog(panel_counts, 1.25 * pot_errs[0] * (h_eff / h_eff[0]), "k--", lw=1.2, alpha=0.65, label=r"Reference $O(h_{\mathrm{panel}})$")
    ax_cv.set_xticks(panel_counts)
    ax_cv.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax_cv.get_xaxis().set_minor_locator(matplotlib.ticker.NullLocator())
    ax_cv.set_xlabel(r"Surface triangles $N_{\mathrm{panels}}$")
    ax_cv.set_ylabel(r"Relative $L^2$ error")
    ax_cv.set_title("(d) 3D Surface Mesh Refinement Convergence", fontweight="bold")
    ax_cv.grid(True, which="both", alpha=0.25)
    ax_cv.legend(loc="upper right", fontsize=8.4, framealpha=0.92)

    out_path = out_dir / "03_bempp_dielectric_membrane.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
