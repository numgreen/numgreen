"""Singularity-split numerical Green's function on an enclosing Cartesian box."""

from __future__ import annotations

from itertools import product
from typing import Sequence

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .pde import PDE


def _cubic_weights(ax: np.ndarray, coord: np.ndarray, want_grad: bool):
    h = ax[1] - ax[0]
    u = np.clip((coord - ax[0]) / h, 1.0, len(ax) - 2.0 - 1e-9)
    i0 = np.floor(u).astype(int)
    t = u - i0
    w = np.column_stack([
        -0.5 * t * (1.0 - t) ** 2,
        1.0 - 2.5 * t**2 + 1.5 * t**3,
        0.5 * t * (1.0 + 4.0 * t - 3.0 * t**2),
        -0.5 * t**2 * (1.0 - t),
    ])
    if not want_grad:
        return i0, w, None
    dw = np.column_stack([
        -0.5 + 2.0 * t - 1.5 * t**2,
        -5.0 * t + 4.5 * t**2,
        0.5 + 4.0 * t - 4.5 * t**2,
        -t + 1.5 * t**2,
    ]) / h
    return i0, w, dw


def _interp_cubic(axes: list[np.ndarray], values: np.ndarray, X: np.ndarray, want_grad: bool = False):
    dim, M, K = len(axes), X.shape[0], values.shape[-1]
    st = [_cubic_weights(axes[d], X[:, d], want_grad) for d in range(dim)]
    val = np.zeros((M, K), dtype=values.dtype)
    grad = np.zeros((M, K, dim), dtype=values.dtype) if want_grad else None
    for offs in product(range(4), repeat=dim):
        v = values[tuple(st[d][0] + offs[d] - 1 for d in range(dim))]
        wp = st[0][1][:, offs[0]] * st[1][1][:, offs[1]] * (1.0 if dim == 2 else st[2][1][:, offs[2]])
        val += wp[:, None] * v
        if want_grad:
            for g_ax in range(dim):
                factors = [st[d][2 if d == g_ax else 1][:, offs[d]] for d in range(dim)]
                dwp = factors[0] * factors[1] * (1.0 if dim == 2 else factors[2])
                grad[:, :, g_ax] += dwp[:, None] * v
    return val, grad


def _d2_axis(ax: np.ndarray, pml_width: float, k_ref: complex | float):
    n, h = len(ax), ax[1] - ax[0]
    if pml_width <= 0.0 or k_ref == 0:
        return sp.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(n, n), dtype=float) / h**2
    xm = 0.5 * (ax[:-1] + ax[1:])
    scale = 12.0 / max(abs(k_ref) * pml_width, 1.0)

    def inv_s(c: np.ndarray) -> np.ndarray:
        depth = (np.maximum(0.0, (ax[0] + pml_width) - c) + np.maximum(0.0, c - (ax[-1] - pml_width))) / pml_width
        return 1.0 / (1.0 + 1j * scale * depth**3)

    in_n, in_m = inv_s(ax), inv_s(xm)
    up, lo = in_n[:-1] * in_m / h**2, in_n[1:] * in_m / h**2
    diag = np.zeros(n, dtype=complex)
    diag[:-1] -= up
    diag[1:] -= lo
    return sp.diags([lo, diag, up], [-1, 0, 1], shape=(n, n), dtype=complex)


def _kron_all(mats):
    out = mats[0]
    for m in mats[1:]:
        out = sp.kron(out, m)
    return out


class SplitGreen:
    """Singularity-split Green's function ``G(x, y) = G_0(x, y) + H(x, y)``.

    Discretizes the delta-free remainder equation ``L_x H = -(L_x - L_0) G_0`` by
    second-order finite differences on an enclosing Cartesian box ``bounds``,
    factors the sparse matrix once via ``scipy.sparse.linalg.splu``, and evaluates
    ``H`` and its gradients off-grid by ``C^1`` Catmull-Rom cubic interpolation.
    Target and source points must lie strictly inside the box interior (outside any
    PML layer).
    """

    def __init__(
        self,
        pde: PDE,
        bounds: Sequence[tuple[float, float]] | Sequence[float],
        shape: int | Sequence[int] = 96,
        bc: str = "absorbing",
        robin_alpha: complex | float | None = None,
        pml_width: float | None = None,
    ) -> None:
        self.pde = pde
        dim = pde.dim
        pairs = np.asarray(bounds, dtype=float).reshape(dim, 2)
        self.shape = (int(shape),) * dim if isinstance(shape, int) else tuple(int(s) for s in shape)
        self.axes = [np.linspace(pairs[d, 0], pairs[d, 1], self.shape[d]) for d in range(dim)]
        self.spacings = np.array([ax[1] - ax[0] for ax in self.axes])
        self.grid_points = np.column_stack([m.ravel() for m in np.meshgrid(*self.axes, indexing="ij")])

        use_pml = (bc == "absorbing") and (pde.k0 != 0)
        span_min = float(np.min(pairs[:, 1] - pairs[:, 0]))
        pw = (0.16 * span_min if pml_width is None else float(pml_width)) if use_pml else 0.0
        self._chi = np.ones(self.grid_points.shape[0], dtype=float)
        if pw > 0.0:
            for d in range(dim):
                coord = self.grid_points[:, d]
                xi = np.clip(np.maximum((pairs[d, 0] + 0.5 * pw) - coord, coord - (pairs[d, 1] - 0.5 * pw)) / (0.5 * pw), 0.0, 1.0)
                self._chi *= 0.5 * (1.0 + np.cos(np.pi * xi))

        dtype = complex if (pde.is_complex or use_pml or isinstance(robin_alpha, complex)) else float
        eyes = [sp.eye(n, dtype=dtype) for n in self.shape]
        self._d1_ops = [
            _kron_all([
                sp.diags([-0.5, 0.0, 0.5], [-1, 0, 1], shape=(self.shape[k], self.shape[k]), dtype=float) / self.spacings[k]
                if k == d else sp.eye(self.shape[k])
                for k in range(dim)
            ]).tocsc()
            for d in range(dim)
        ]

        if pde.is_anisotropic:
            self._s_grid = np.ones(self.grid_points.shape[0])
            self._A_grid = pde.eval_A(self.grid_points)
            self._b_grid = pde.eval_b(self.grid_points)
            self._c_grid = pde.eval_c(self.grid_points)
            op = sp.diags(self._c_grid.astype(dtype), 0, dtype=dtype)
            for p in range(dim):
                if pde.has_advection:
                    op = op + sp.diags(self._b_grid[:, p].astype(dtype)) @ self._d1_ops[p]
                for q in range(dim):
                    if p == q:
                        a_pq = self._A_grid[:, p, p].reshape(self.shape)
                        sl_lo, sl_hi = [slice(None)] * dim, [slice(None)] * dim
                        sl_lo[p], sl_hi[p] = slice(0, -1), slice(1, None)
                        af = 0.5 * (a_pq[tuple(sl_lo)] + a_pq[tuple(sl_hi)]) / self.spacings[p] ** 2
                        stride = int(np.prod(self.shape[p + 1 :]))
                        up_v, lo_v = np.zeros(self.shape), np.zeros(self.shape)
                        up_v[tuple(sl_lo)], lo_v[tuple(sl_hi)] = af, af
                        op = op - sp.diags([lo_v.ravel()[stride:], -(up_v + lo_v).ravel(), up_v.ravel()[:-stride]], [-stride, 0, stride], dtype=dtype)
                    else:
                        op = op - self._d1_ops[p] @ sp.diags(self._A_grid[:, p, q].astype(dtype)) @ self._d1_ops[q]
        else:
            a_g = pde.eval_a(self.grid_points)
            self._s_grid = np.sqrt(a_g)
            self._b_over_a = pde.eval_b(self.grid_points) / a_g[:, None]
            grad_log_s = pde.eval_grad_a(self.grid_points) / (2.0 * a_g[:, None])
            self._q_eff = (
                pde.eval_lap_sqrt_a(self.grid_points) / self._s_grid
                + pde.eval_c(self.grid_points) / a_g
                - np.sum(self._b_over_a * grad_log_s, axis=1)
            )
            d2s = [_d2_axis(self.axes[d], pw, pde.k0) for d in range(dim)]
            op = sp.diags(self._q_eff.astype(dtype), 0, dtype=dtype)
            for d in range(dim):
                op = op - _kron_all([d2s[k] if k == d else eyes[k] for k in range(dim)])
                if pde.has_advection:
                    op = op + sp.diags(self._b_over_a[:, d].astype(dtype)) @ self._d1_ops[d]

        op = op.tolil()
        idx_nd = np.arange(self.grid_points.shape[0]).reshape(self.shape)
        self._bnd_mask = np.zeros(self.grid_points.shape[0], dtype=bool)
        use_monopole = (bc == "absorbing") and (pde.k0 == 0) and (dim == 3) and (not pde.has_advection) and (not pde.is_anisotropic)
        for d in range(dim):
            for side in (0, self.shape[d] - 1):
                sl, sl_in = [slice(None)] * dim, [slice(None)] * dim
                sl[d], sl_in[d] = side, (1 if side == 0 else self.shape[d] - 2)
                b_idx, in_idx = idx_nd[tuple(sl)].ravel(), idx_nd[tuple(sl_in)].ravel()
                self._bnd_mask[b_idx] = True
                if use_monopole or bc == "robin":
                    h_d = self.spacings[d]
                    xb = self.grid_points[b_idx]
                    nsign = -1.0 if side == 0 else 1.0
                    imp = -(nsign * xb[:, d]) / np.maximum(np.sum(xb**2, axis=1), 1e-12) if use_monopole else complex(robin_alpha or 0.0)
                    for bi, ii, im in zip(b_idx, in_idx, np.broadcast_to(imp, b_idx.shape)):
                        op.rows[bi], op.data[bi] = [int(bi), int(ii)], [1.0 / h_d - im, -1.0 / h_d]
                else:
                    for bi in b_idx:
                        op.rows[bi], op.data[bi] = [int(bi)], [1.0]

        self._lu = spla.splu(op.tocsc(), permc_spec="MMD_AT_PLUS_A")
        self.num_factorizations = 1
        self._cache_Y: np.ndarray | None = None
        self._cache_Ht: np.ndarray | None = None
        self._src_axes: list[np.ndarray] | None = None
        self._src_table: np.ndarray | None = None

    def _solve_tilde(self, Y: np.ndarray) -> np.ndarray:
        Y = np.atleast_2d(np.asarray(Y, dtype=float))
        if self._cache_Y is not None and self._cache_Y.shape == Y.shape and np.array_equal(self._cache_Y, Y):
            return self._cache_Ht  # type: ignore[return-value]
        r_reg = 0.35 * float(np.mean(self.spacings))
        if self.pde.is_anisotropic:
            g, gx, _ = self.pde.std_kernel(self.grid_points, Y, grad_x=True, r_reg=r_reg)
            dA = self._A_grid[:, None, :, :] - self.pde.eval_A(Y)[None, :, :, :]
            flux = np.einsum("mnpq,mnq->mnp", dA, gx)
            rhs = sum(self._d1_ops[p] @ flux[:, :, p] for p in range(self.pde.dim)) - np.einsum("md,mnd->mn", self._b_grid, gx) - self._c_grid[:, None] * g
        elif not self.pde.has_advection:
            g, _, _ = self.pde.std_kernel(self.grid_points, Y, grad_x=False, r_reg=r_reg)
            rhs = -((self._q_eff + self.pde.k0**2) * self._chi)[:, None] * g
        else:
            g, gx, gamma_y = self.pde.std_kernel(self.grid_points, Y, grad_x=True, r_reg=r_reg)
            db = self._b_over_a[:, None, :] - (self.pde.eval_b(Y) / self.pde.eval_a(Y)[:, None])[None, :, :]
            rhs = -(np.sum(db * gx, axis=-1) + (self._q_eff[:, None] - gamma_y[None, :]) * g) * self._chi[:, None]
        rhs[self._bnd_mask, :] = 0.0
        if np.iscomplexobj(self._lu.L.data) and not np.iscomplexobj(rhs):
            rhs = rhs.astype(complex)
        Ht = self._lu.solve(rhs).reshape(*self.shape, Y.shape[0])
        self._cache_Y, self._cache_Ht = Y.copy(), Ht
        return Ht

    def tabulate_sources(self, bounds: Sequence[tuple[float, float]] | Sequence[float], shape: int | Sequence[int] = 12) -> "SplitGreen":
        """Tabulate ``tilde_H(x_grid, y_grid)`` on a Cartesian source grid ``bounds``."""
        dim = self.pde.dim
        pairs = np.asarray(bounds, dtype=float).reshape(dim, 2)
        s_shape = (int(shape),) * dim if isinstance(shape, int) else tuple(int(s) for s in shape)
        self._src_axes = [np.linspace(pairs[d, 0], pairs[d, 1], s_shape[d]) for d in range(dim)]
        src_pts = np.column_stack([m.ravel() for m in np.meshgrid(*self._src_axes, indexing="ij")])
        self._src_table = self._solve_tilde(src_pts).reshape(*self.shape, *s_shape)
        return self

    def eval_H(self, X: np.ndarray, Y: np.ndarray, grad_x: bool = False, grad_y: bool = False):
        """Evaluate remainder ``H(x, y)`` and optional target/source gradients."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        Y = np.atleast_2d(np.asarray(Y, dtype=float))
        M, N, dim = X.shape[0], Y.shape[0], self.pde.dim

        if self._src_axes is not None and self._src_table is not None:
            s_shape = tuple(len(a) for a in self._src_axes)
            tx, gtx = _interp_cubic(self.axes, self._src_table.reshape(*self.shape, -1), X, want_grad=grad_x)
            tx_src = np.moveaxis(tx.reshape(M, *s_shape), 0, -1)
            yt, gty_sw = _interp_cubic(self._src_axes, tx_src, Y, want_grad=grad_y)
            Ht = yt.T
            gy_Ht = np.swapaxes(gty_sw, 0, 1) if grad_y else None
            if grad_x:
                gtx_src = np.moveaxis(gtx.reshape(M, *s_shape, dim), (0, -1), (-2, -1)).reshape(*s_shape, M * dim)
                gx_Ht = np.moveaxis(_interp_cubic(self._src_axes, gtx_src, Y)[0].reshape(N, M, dim), 0, 1)
            else:
                gx_Ht = None
        else:
            Ht_grid = self._solve_tilde(Y)
            Ht, gx_Ht = _interp_cubic(self.axes, Ht_grid, X, want_grad=grad_x)
            gy_Ht = None
            if grad_y:
                sym = (not self.pde.has_advection) and (not self.pde.is_anisotropic)
                if sym and X.shape == Y.shape and np.allclose(X, Y):
                    gx_self = gx_Ht if grad_x else _interp_cubic(self.axes, Ht_grid, X, want_grad=True)[1]
                    gy_Ht = np.swapaxes(gx_self, 0, 1)
                elif sym and M <= 2 * dim * N:
                    gy_Ht = np.swapaxes(_interp_cubic(self.axes, self._solve_tilde(X), Y, want_grad=True)[1], 0, 1)
                else:
                    eps = 1e-4
                    shifts = np.vstack([Y + s * eps * np.eye(dim)[d] for d in range(dim) for s in (1.0, -1.0)])
                    vals, _ = _interp_cubic(self.axes, self._solve_tilde(shifts), X)
                    vals = vals.reshape(M, dim, 2, N)
                    gy_Ht = np.moveaxis((vals[:, :, 0, :] - vals[:, :, 1, :]) / (2.0 * eps), 1, -1)

        sx = np.ones(M) if self.pde.is_anisotropic else np.sqrt(self.pde.eval_a(X))
        sy = np.ones(N) if self.pde.is_anisotropic else np.sqrt(self.pde.eval_a(Y))
        inv_s = 1.0 / (sx[:, None] * sy[None, :])
        H = Ht * inv_s
        if not grad_x and not grad_y:
            return H
        out = [H]
        if grad_x:
            ga_x = np.zeros_like(X) if self.pde.is_anisotropic else self.pde.eval_grad_a(X) / (2.0 * sx[:, None] ** 2)
            out.append(gx_Ht * inv_s[:, :, None] - ga_x[:, None, :] * H[:, :, None])
        if grad_y:
            ga_y = np.zeros_like(Y) if self.pde.is_anisotropic else self.pde.eval_grad_a(Y) / (2.0 * sy[:, None] ** 2)
            out.append(gy_Ht * inv_s[:, :, None] - ga_y[None, :, :] * H[:, :, None])
        return tuple(out)

    def eval(self, X: np.ndarray, Y: np.ndarray, grad_x: bool = False, grad_y: bool = False):
        """Evaluate ``G(x, y) = G_0(x, y) + H(x, y)`` and optional gradients."""
        g0 = self.pde.eval_G0(X, Y, grad_x=grad_x, grad_y=grad_y)
        h = self.eval_H(X, Y, grad_x=grad_x, grad_y=grad_y)
        return (g0 + h) if (not grad_x and not grad_y) else tuple(a + b for a, b in zip(g0, h))
