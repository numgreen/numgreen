"""Linear second-order elliptic PDE specification in 2D and 3D."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.special import hankel1, kve

ArrayFn = Callable[[np.ndarray], np.ndarray]


@dataclass
class PDE:
    """Linear elliptic operator ``-div(A(x) grad u) + b(x)·grad u + c(x) u = 0``.

    Coordinate inputs have shape ``(M, dim)`` with ``dim`` in ``{2, 3}``.
    Provide either a positive ``C^2`` scalar coefficient ``a(x)`` (isotropic
    diffusion/permittivity) or a symmetric positive-definite tensor ``A(x)`` of
    shape ``(M, dim, dim)``. Outgoing wave problems use the ``exp(-i omega t)``
    convention with ``c(x) = -k(x)^2`` and reference wavenumber ``k0``.
    """

    dim: int = 2
    a: ArrayFn | float = 1.0
    A: ArrayFn | np.ndarray | None = None
    b: ArrayFn | Sequence[float] | None = None
    c: ArrayFn | complex | float = 0.0
    k0: complex | float = 0.0
    is_complex: bool = False
    name: str = "PDE"

    def __post_init__(self) -> None:
        if self.dim not in (2, 3):
            raise ValueError("PDE dimension `dim` must be 2 or 3.")
        self.has_advection = self.b is not None
        self.is_anisotropic = self.A is not None
        self.const_a = (not callable(self.a)) and (not self.is_anisotropic)
        if self.k0 != 0.0 or isinstance(self.k0, complex) or isinstance(self.c, complex):
            self.is_complex = True

    @classmethod
    def laplace(cls, dim: int = 2) -> "PDE":
        return cls(dim=dim, a=1.0, c=0.0, name=f"Laplace{dim}D")

    @classmethod
    def helmholtz(cls, k: ArrayFn | float, k0: float, dim: int = 2, name: str = "Helmholtz") -> "PDE":
        c_fn = (lambda X: -(np.asarray(k(X), dtype=complex) ** 2)) if callable(k) else -(complex(k) ** 2)
        return cls(dim=dim, a=1.0, c=c_fn, k0=k0, is_complex=True, name=name)

    @classmethod
    def advection_diffusion(
        cls,
        diffusivity: ArrayFn | float,
        velocity: ArrayFn | Sequence[float],
        reaction: ArrayFn | float = 0.0,
        dim: int = 2,
    ) -> "PDE":
        return cls(dim=dim, a=diffusivity, b=velocity, c=reaction, name="AdvectionDiffusion")

    @classmethod
    def dielectric(cls, eps: ArrayFn | float, screening: ArrayFn | float = 0.0, dim: int = 3) -> "PDE":
        return cls(dim=dim, a=eps, c=screening, name=f"Dielectric{dim}D")

    def eval_a(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        out = np.asarray(self.a(X) if callable(self.a) else self.a, dtype=float)
        return np.broadcast_to(out, (X.shape[0],)).copy()

    def eval_grad_a(self, X: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if self.const_a:
            return np.zeros_like(X)
        eye = eps * np.eye(self.dim)
        cols = [(self.eval_a(X + eye[d]) - self.eval_a(X - eye[d])) / (2.0 * eps) for d in range(self.dim)]
        return np.column_stack(cols)

    def eval_lap_sqrt_a(self, X: np.ndarray, eps: float = 2e-3) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if self.const_a:
            return np.zeros(X.shape[0], dtype=float)
        s0 = np.sqrt(self.eval_a(X))
        eye = eps * np.eye(self.dim)
        lap = np.zeros(X.shape[0], dtype=float)
        for d in range(self.dim):
            vs = [np.sqrt(self.eval_a(X + k * eye[d])) for k in (-2.0, -1.0, 1.0, 2.0)]
            lap += (-vs[0] + 16.0 * vs[1] - 30.0 * s0 + 16.0 * vs[2] - vs[3]) / (12.0 * eps**2)
        return lap

    def eval_A(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if not self.is_anisotropic:
            return self.eval_a(X)[:, None, None] * np.eye(self.dim)[None, :, :]
        raw = np.asarray(self.A(X) if callable(self.A) else self.A, dtype=float)
        return np.broadcast_to(raw, (X.shape[0], self.dim, self.dim)).copy()

    def eval_b(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if self.b is None:
            return np.zeros_like(X)
        raw = np.asarray(self.b(X) if callable(self.b) else self.b, dtype=float)
        return np.broadcast_to(raw, X.shape).copy()

    def eval_c(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        dtype = complex if self.is_complex else float
        raw = np.asarray(self.c(X) if callable(self.c) else self.c, dtype=dtype)
        return np.broadcast_to(raw, (X.shape[0],)).copy()

    def conormal(self, X: np.ndarray, normals: np.ndarray) -> np.ndarray:
        if not self.is_anisotropic:
            return self.eval_a(X)[:, None] * normals
        return np.einsum("mji,mj->mi", self.eval_A(X), normals)

    def std_kernel(self, X: np.ndarray, Y: np.ndarray, grad_x: bool = False, r_reg: float = 0.0):
        """Evaluate unscaled reference kernel ``g(x, y)`` for ``(-Delta_x + 2 beta_y·grad_x + gamma_y) g = delta``."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        Y = np.atleast_2d(np.asarray(Y, dtype=float))
        diff = X[:, None, :] - Y[None, :, :]
        r_raw = np.linalg.norm(diff, axis=-1)
        r = np.sqrt(r_raw**2 + r_reg**2) if r_reg > 0.0 else np.maximum(r_raw, 1e-14)
        mask = (r > 0.0) if r_reg > 0.0 else (r_raw > 1e-13)

        if self.is_anisotropic:
            Ay = 0.5 * (self.eval_A(Y) + np.swapaxes(self.eval_A(Y), -1, -2))
            inv_Ay = np.linalg.inv(Ay)
            sdet = np.sqrt(np.linalg.det(Ay))[None, :]
            Minv_d = np.einsum("nij,mnj->mni", inv_Ay, diff)
            rho = np.sqrt(np.maximum(np.sum(diff * Minv_d, axis=-1), 1e-28))
            g = -np.log(rho) / (2.0 * np.pi * sdet) if self.dim == 2 else 1.0 / (4.0 * np.pi * sdet * rho)
            fac = -1.0 / (2.0 * np.pi * sdet * rho**2) if self.dim == 2 else -1.0 / (4.0 * np.pi * sdet * rho**3)
            gx = np.where(mask[:, :, None], fac[:, :, None] * Minv_d, 0.0) if grad_x else None
            return g, gx, np.zeros(Y.shape[0])

        if self.has_advection:
            a_y = self.eval_a(Y)
            beta_y = self.eval_b(Y) / (2.0 * a_y[:, None])
            gamma_y = np.maximum(np.real(self.eval_c(Y)) / a_y, 0.0)
            mu_y = np.sqrt(np.sum(beta_y**2, axis=1) + gamma_y + 1e-12)
            mr = mu_y[None, :] * r
            E = np.exp(np.einsum("mnd,nd->mn", diff, beta_y) - mr)
            if self.dim == 2:
                g = (0.5 / np.pi) * E * kve(0, mr)
                gx = (
                    g[:, :, None] * beta_y[None, :, :]
                    - (mu_y[None, :, None] / (2.0 * np.pi))
                    * (E * kve(1, mr))[:, :, None]
                    * (diff / r[:, :, None])
                ) if grad_x else None
            else:
                g = E / (4.0 * np.pi * r)
                gx = g[:, :, None] * (beta_y[None, :, :] - ((1.0 + mr) / r**2)[:, :, None] * diff) if grad_x else None
        else:
            gamma_y = np.full(Y.shape[0], -(self.k0**2), dtype=complex if self.is_complex else float)
            if self.dim == 2:
                g = -np.log(r) / (2.0 * np.pi) if self.k0 == 0 else 0.25j * hankel1(0, self.k0 * r)
                dr = -1.0 / (2.0 * np.pi * r**2) if self.k0 == 0 else -0.25j * self.k0 * hankel1(1, self.k0 * r) / r
            else:
                ekr = 1.0 if self.k0 == 0 else np.exp(1j * self.k0 * r)
                g = ekr / (4.0 * np.pi * r)
                dr = -1.0 / (4.0 * np.pi * r**3) if self.k0 == 0 else ekr * (1j * self.k0 * r - 1.0) / (4.0 * np.pi * r**3)
            gx = (dr[:, :, None] * diff) if grad_x else None

        return g, (np.where(mask[:, :, None], gx, 0.0) if gx is not None else None), gamma_y

    def eval_G0(self, X: np.ndarray, Y: np.ndarray, grad_x: bool = False, grad_y: bool = False):
        """Evaluate singular reference parametrix ``G_0(x, y)`` for targets ``X`` and sources ``Y``."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        Y = np.atleast_2d(np.asarray(Y, dtype=float))
        need_gx = grad_x or (grad_y and not self.has_advection and not self.is_anisotropic)
        g, gx_std, _ = self.std_kernel(X, Y, grad_x=need_gx)
        sx = np.ones(X.shape[0]) if self.is_anisotropic else np.sqrt(self.eval_a(X))
        sy = np.ones(Y.shape[0]) if self.is_anisotropic else np.sqrt(self.eval_a(Y))
        inv_s = 1.0 / (sx[:, None] * sy[None, :])
        G0 = g * inv_s
        if not grad_x and not grad_y:
            return G0
        out = [G0]
        if grad_x:
            ga_x = np.zeros_like(X) if self.is_anisotropic else self.eval_grad_a(X) / (2.0 * sx[:, None] ** 2)
            out.append(gx_std * inv_s[:, :, None] - ga_x[:, None, :] * G0[:, :, None])
        if grad_y:
            if not self.has_advection and not self.is_anisotropic:
                ga_y = self.eval_grad_a(Y) / (2.0 * sy[:, None] ** 2)
                out.append(-gx_std * inv_s[:, :, None] - ga_y[None, :, :] * G0[:, :, None])
            else:
                eps, eye = 1e-6, np.eye(self.dim)
                cols = [(self.eval_G0(X, Y + eps * eye[d]) - self.eval_G0(X, Y - eps * eye[d])) / (2.0 * eps) for d in range(self.dim)]
                out.append(np.stack(cols, axis=-1))
        return tuple(out)
