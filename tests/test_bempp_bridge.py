"""3D bempp-cl surface operator convergence and dielectric bilayer tests."""

from __future__ import annotations

import bempp_cl.api as bempp
import numpy as np

from numgreen import BemppOperators, PDE, SplitGreen


def _unit_sphere_grid(level: int, radius: float = 0.9):
    g0 = bempp.shapes.regular_sphere(level)
    verts = np.asfortranarray(radius * g0.vertices / np.linalg.norm(g0.vertices, axis=0, keepdims=True))
    elems = np.asfortranarray(g0.elements)
    return bempp.Grid(verts, elems)


def test_bempp_3d_variable_dielectric_mesh_convergence() -> None:
    """Check mesh-refinement convergence against an exact 3D variable-permittivity solution."""
    alpha = 0.55
    eps_fn = lambda X: np.exp(2.0 * alpha * X[:, 2])
    pde = PDE.dielectric(eps=eps_fn, dim=3)
    green = SplitGreen(pde, bounds=[(-4.8, 4.8)] * 3, shape=29, bc="absorbing")
    green.tabulate_sources(bounds=[(-1.05, 1.05)] * 3, shape=9)
    assert green.num_factorizations == 1

    R = 0.9
    y0 = np.array([0.12, -0.15, 0.22])

    def exact_u_and_flux(X: np.ndarray, normals: np.ndarray | None = None):
        d = X - y0[None, :]
        r = np.linalg.norm(d, axis=1)
        sx, s0 = np.exp(alpha * X[:, 2]), np.exp(alpha * y0[2])
        u = np.exp(-abs(alpha) * r) / (4.0 * np.pi * r * sx * s0)
        if normals is None:
            return u
        dg = -(abs(alpha) * r + 1.0) * np.exp(-abs(alpha) * r) / (4.0 * np.pi * r**3 * sx * s0)
        grad = dg[:, None] * d
        grad[:, 2] -= alpha * u
        return u, eps_fn(X) * np.sum(grad * normals, axis=1)

    ang = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    X_ext = np.column_stack([1.55 * np.cos(ang), 1.55 * np.sin(ang), np.linspace(-0.85, 0.85, 16)])
    u_ext_ex = exact_u_and_flux(X_ext)

    flux_errs, ext_errs = [], []
    for level in (1, 2, 3):
        grid = _unit_sphere_grid(level, radius=R)
        bop = BemppOperators(pde, green, grid)
        u_bnd, sigma_ex = exact_u_and_flux(bop.centroids, bop.normals)
        sigma_num = bop.solve_exterior_dirichlet(u_bnd)
        u_ext_num = bop.eval_exterior(X_ext, u_bnd, sigma_num)
        flux_errs.append(np.linalg.norm(sigma_num - sigma_ex) / np.linalg.norm(sigma_ex))
        ext_errs.append(np.linalg.norm(u_ext_num - u_ext_ex) / np.linalg.norm(u_ext_ex))

    assert flux_errs[0] > flux_errs[1] > flux_errs[2]
    assert ext_errs[0] > ext_errs[1] > ext_errs[2]
    assert flux_errs[2] < 2.2e-2
    assert ext_errs[2] < 2.0e-2


def test_bempp_born_solvation_and_membrane_translation_symmetry() -> None:
    """Check bulk-water Born solvation limit, z <-> -z symmetry, and midplane barrier."""
    eps_mem, eps_wat, L_m, w = 2.0, 80.0, 1.8, 0.55
    eps_fn = lambda X: eps_mem + 0.5 * (eps_wat - eps_mem) * (2.0 - np.tanh((X[:, 2] + L_m) / w) + np.tanh((X[:, 2] - L_m) / w))
    pde = PDE.dielectric(eps=eps_fn, dim=3)
    green = SplitGreen(pde, bounds=[(-4.4, 4.4), (-4.4, 4.4), (-7.5, 7.5)], shape=(23, 23, 39), bc="absorbing")
    green.tabulate_sources(bounds=[(-1.1, 1.1), (-1.1, 1.1), (-5.2, 5.2)], shape=(7, 7, 17))
    assert green.num_factorizations == 1

    R_ion = 0.85
    eps_in = 2.0
    q = [1.0]
    rq = np.array([[0.0, 0.0, 0.0]])
    grid = _unit_sphere_grid(3, radius=R_ion)
    bop = BemppOperators(pde, green, grid)

    pde_wat = PDE.dielectric(eps=eps_wat, dim=3)
    green_wat = SplitGreen(pde_wat, bounds=[(-3.0, 3.0)] * 3, shape=15, bc="absorbing")
    born_mesh_wat = BemppOperators(pde_wat, green_wat, grid).solve_dielectric_cavity(q, rq, eps_in=eps_in).energy
    born_exact = -(1.0**2) / (8.0 * np.pi * R_ion) * (1.0 / eps_in - 1.0 / eps_wat)

    sol_wat_pos = bop.solve_dielectric_cavity(q, rq, eps_in=eps_in, shift=(0.0, 0.0, 4.4))
    sol_wat_neg = bop.solve_dielectric_cavity(q, rq, eps_in=eps_in, shift=(0.0, 0.0, -4.4))
    sol_iface = bop.solve_dielectric_cavity(q, rq, eps_in=eps_in, shift=(0.0, 0.0, 1.8))
    sol_core = bop.solve_dielectric_cavity(q, rq, eps_in=eps_in, shift=(0.0, 0.0, 0.0))

    assert abs(sol_wat_pos.energy - born_mesh_wat) / abs(born_mesh_wat) < 2.0e-2
    assert abs(sol_wat_pos.energy - born_exact) / abs(born_exact) < 2.5e-2
    assert abs(sol_wat_pos.energy - sol_wat_neg.energy) / abs(born_exact) < 5e-3
    assert sol_wat_pos.energy < sol_iface.energy < sol_core.energy < 0.0
    assert sol_core.energy - sol_wat_pos.energy > 0.25 * abs(born_exact)
