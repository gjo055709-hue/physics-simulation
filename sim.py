#!/usr/bin/env python3
"""Stellar MAC fluid lab. Python 3.10–3.12, Taichi 1.7.4, NumPy.

    python -m pip install taichi numpy
    python sim.py --test karman_vortex --window-scale 0.5
    python sim.py --run-all --headless --output results
    python sim.py --width 128 --headless --test fusion_tokamak --cpu

Axes are (x,y), y up. p is gauge pressure in Pa. Boundary pressure inputs
are total Pa. Q is W/m³, external J×B is N/m³, and cp is J/(kg K).
A step uses bounded bilinear semi-Lagrangian transport, explicit diffusion
and a non-incremental Chorin projection. The projected fields stay available
while the UI time-slices the next pressure solve. Display and simulation Hz
are measured independently. Native 4K needs substantially more device memory.

The heated rods conduct heat into the coolant using one effective thermal
material. The tokamak solves a prescribed-force equilibrium.
Poiseuille starts with a developed profile computed by an independent 1D
finite-volume solve. Hydrostatic and tokamak start at zero gauge pressure.
"""

import argparse
import csv
import json
import logging
import math
import os
import sys
import time
import tempfile
from collections import deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from enum import IntEnum
from pathlib import Path

import numpy as np
import taichi as ti

LOG = logging.getLogger("stellar")


@contextmanager
def atomic_output(path, mode, **kwargs):
    path = Path(path)
    tmp = tempfile.NamedTemporaryFile(dir=path.parent, mode=mode, delete=False, **kwargs)
    try:
        with tmp as stream:
            yield stream
        os.replace(tmp.name, path)
    finally:
        Path(tmp.name).unlink(missing_ok=True)


class Wall(IntEnum):
    NO_SLIP = 0
    FREE_SLIP = 1
    INFLOW_VELOCITY = 2
    FIXED_PRESSURE = 3
    OUTFLOW_NEUMANN = 4


NOSLIP, SLIP, INFLOW, FIXED, OUTFLOW = (int(w) for w in Wall)


@dataclass(frozen=True)
class BoundaryCondition:
    kind: Wall = Wall.NO_SLIP
    velocity: tuple[float, float] = (0.0, 0.0)  # Cartesian components, m/s
    pressure: float = 0.0  # total Pa, for FIXED_PRESSURE
    temperature: float | None = None  # inlet K, defaults to T_ref
    rigid_datum: bool = False  # sealed no-slip wall with a mean pressure datum


@dataclass(frozen=True)
class SimConfig:
    name: str = "hydrostatic"
    Lx: float = 1.6
    Ly: float = 0.9
    rho: float = 1000.0
    mu: float = 1.0e-3
    beta: float = 0.0
    alpha_thermal: float = 1.4e-7
    cp: float = 4180.0
    T_ref: float = 293.15
    p_ref: float = 101325.0
    gx: float = 0.0
    gy: float = 0.0
    left: BoundaryCondition = field(default_factory=BoundaryCondition)
    right: BoundaryCondition = field(default_factory=BoundaryCondition)
    bottom: BoundaryCondition = field(default_factory=BoundaryCondition)
    top: BoundaryCondition = field(default_factory=BoundaryCondition)
    cfl: float = 0.8
    max_dt: float = 1 / 60
    solver_tol: float = 0.01  # volume-normalized L2 residual, Pa/m²
    max_iters: int = 300  # fine-grid red-black sweep pairs
    target_time: float = 0.1
    headless: bool = False

    @property
    def walls(self):
        return (self.left, self.right, self.bottom, self.top)

    def check(self):
        positive = (
            self.Lx,
            self.Ly,
            self.rho,
            self.cp,
            self.T_ref,
            self.cfl,
            self.max_dt,
            self.solver_tol,
            self.target_time,
        )
        numbers = (*positive, self.mu, self.alpha_thermal, self.p_ref, self.beta, self.gx, self.gy)
        if not all(map(math.isfinite, numbers)) or min(positive) <= 0:
            raise ValueError("Invalid physical constants, time limit or tolerance")
        if min(self.mu, self.alpha_thermal) < 0 or self.cfl > 1 or self.max_iters < 1:
            raise ValueError("Require nonnegative diffusivities, CFL <= 1 and positive iterations")
        if any(w.rigid_datum for w in self.walls):
            if sum(w.kind == FIXED for w in self.walls) != 1:
                raise ValueError("A rigid pressure datum requires one pressure reference")
        for w in self.walls:
            Wall(w.kind)
            if w.rigid_datum and w.kind != FIXED:
                raise ValueError("A rigid pressure datum requires FIXED_PRESSURE")
            if len(w.velocity) != 2 or not all(map(math.isfinite, (*w.velocity, w.pressure))):
                raise ValueError("Invalid wall data")
            if w.temperature is not None and not (
                math.isfinite(w.temperature) and w.temperature > 0
            ):
                raise ValueError("Inlet temperature must be positive in K")


@ti.kernel
def _probe64(x: ti.f64, out: ti.types.ndarray(dtype=ti.f64, ndim=1)):
    out[0], out[1], out[2] = 1.0, 2.0, 0.0
    for i in range(8):
        ti.atomic_add(out[0], x)
        ti.atomic_min(out[1], 1.0 + x)
        ti.atomic_max(out[2], 1.0 + x)


def init_taichi(cpu=False, single=False):
    """Select a precision before any persistent field trees are constructed."""
    arch = ti.cpu if cpu else ti.gpu
    if not cpu and sys.platform.startswith("linux"):
        # Compute on native CPU when Linux only exposes the Mesa software ICD.
        # Long Taichi 1.7 compute workloads can exhaust lavapipe descriptor pools.
        # Keep CUDA/AMD candidates, including remote CUDA drivers, in the selector.
        icds = os.environ.get("VK_DRIVER_FILES") or os.environ.get("VK_ICD_FILENAMES")
        if icds:
            software = all(
                "lvp" in Path(f).name or "lavapipe" in Path(f).name for f in icds.split(os.pathsep)
            )
        else:
            nodes = ("/dev/dxg", "/dev/nvidiactl", "/dev/mali0", "/dev/kgsl-3d0", "/dev/pvrsrvkm")
            software = not (
                any(Path(f).exists() for f in nodes)
                or any(Path("/dev/dri").glob("renderD*"))
                or any(Path("/dev/dri").glob("card*"))
            )
        if software:
            LOG.info("Software-only Linux graphics: prefer native CPU for compute")
            arch = [a for a in ti.gpu if a not in (ti.vulkan, ti.opengl, ti.gles)]
        elif not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            # GL context creation on a display-less host can terminate the process.
            arch = [a for a in ti.gpu if a not in (ti.opengl, ti.gles)]
    opts = dict(arch=arch, offline_cache=False)
    if single:
        ti.init(default_fp=ti.f32, **opts)
        fp = ti.f32
    else:
        try:
            ti.init(default_fp=ti.f64, **opts)
            active = ti.lang.impl.current_cfg().arch
            if active == ti.metal:
                raise RuntimeError("backend lacks double precision")
            delta = 2.0**-40
            probe = ti.ndarray(ti.f64, shape=3)
            _probe64(delta, probe)
            if not np.array_equal(probe.to_numpy(), [1.0 + 8 * delta, 1.0 + delta, 1.0 + delta]):
                raise RuntimeError("device failed the FP64 arithmetic probe")
            fp = ti.f64
        except Exception as exc:
            LOG.warning("FP64 unavailable (%s), selecting FP32", exc)
            ti.reset()
            ti.init(default_fp=ti.f32, **opts)
            fp = ti.f32
    active = ti.lang.impl.current_cfg().arch
    LOG.info("Backend %s, physics precision %s", active, fp)
    return fp


@dataclass
class Scene:
    """Host-side initial conditions, broadcastable over the fixed mesh."""

    sdf: object
    u: object = 0.0
    v: object = 0.0
    Q: object = 0.0
    em_x: object = 0.0
    em_y: object = 0.0
    temperature_rise: object = 0.0


class Mesh:
    def __init__(self, nx, ny, cfg):
        self.nx, self.ny = nx, ny
        self.dx, self.dy = cfg.Lx / nx, cfg.Ly / ny
        self.x = (np.arange(nx)[:, None] + 0.5) * self.dx
        self.y = (np.arange(ny)[None, :] + 0.5) * self.dy


def developed_channel(n, dy, forcing):
    """Solve -u''=forcing using reflected no-slip ghosts at half cells."""
    diagonal = np.full(n, 2.0)
    diagonal[[0, -1]] = 3.0
    rhs = np.full(n, forcing * dy**2)
    for j in range(1, n):
        diagonal[j] -= 1 / diagonal[j - 1]
        rhs[j] += rhs[j - 1] / diagonal[j - 1]
    rhs[-1] /= diagonal[-1]
    for j in reversed(range(n - 1)):
        rhs[j] = (rhs[j] + rhs[j + 1]) / diagonal[j]
    return rhs[None, :]


# Each registry entry builds a config and supplies a host-side geometry builder.
def hydrostatic():
    c = SimConfig(
        gy=-9.81, top=BoundaryCondition(Wall.FIXED_PRESSURE, pressure=101325.0, rigid_datum=True)
    )
    return c, lambda m, c: Scene(c.Lx + c.Ly)


def poiseuille():
    c = SimConfig(
        "poiseuille",
        rho=1000.0,
        mu=2.0,
        p_ref=0.0,
        left=BoundaryCondition(Wall.FIXED_PRESSURE, pressure=100.0),
        right=BoundaryCondition(Wall.FIXED_PRESSURE),
        target_time=0.005,
    )

    def geometry(m, c):
        forcing = (c.left.pressure - c.right.pressure) / (c.Lx * c.mu)
        return Scene(c.Lx + c.Ly, u=developed_channel(m.ny, m.dy, forcing))

    return c, geometry


def karman_vortex():
    speed, diameter = 1.0, 0.12
    slip = BoundaryCondition(Wall.FREE_SLIP)
    c = SimConfig(
        "karman_vortex",
        rho=1.0,
        mu=speed * diameter / 150,
        p_ref=0.0,
        left=BoundaryCondition(Wall.INFLOW_VELOCITY, (speed, 0.0)),
        right=BoundaryCondition(Wall.FIXED_PRESSURE),
        bottom=slip,
        top=slip,
        target_time=6.0,
        solver_tol=0.03,
    )

    def geometry(m, c):
        cx, cy = 0.38, c.Ly / 2
        sdf = np.hypot(m.x - cx, m.y - cy) - diameter / 2
        yf = np.arange(m.ny + 1)[None, :] * m.dy
        seed = 0.08 * speed * np.exp(-(((m.x - cx - 0.13) / 0.15) ** 2) - ((yf - cy) / 0.11) ** 2)
        return Scene(sdf, u=speed, v=seed)

    return c, geometry


def venturi_tube():
    c = SimConfig(
        "venturi_tube",
        rho=1000.0,
        mu=0.05,
        p_ref=101325.0,
        left=BoundaryCondition(Wall.INFLOW_VELOCITY, (0.5, 0.0)),
        right=BoundaryCondition(Wall.FIXED_PRESSURE, pressure=101325.0),
        target_time=1.0,
        solver_tol=3.0,
    )

    def geometry(m, c):
        r1 = np.hypot(m.x - c.Lx / 2, m.y + 0.2)
        r2 = np.hypot(m.x - c.Lx / 2, m.y - c.Ly - 0.2)
        sdf = np.minimum(r1, r2) - 0.48
        # A corner streamfunction makes the initial MAC field solenoidal, even
        # along stair-stepped walls. Both walls are streamlines of constant flux.
        fluid = sdf >= 0
        bottom = np.argmax(fluid, axis=1)
        top = m.ny - np.argmax(fluid[:, ::-1], axis=1)
        bottom_faces = np.maximum(np.r_[bottom[0], bottom], np.r_[bottom, bottom[-1]])
        top_faces = np.minimum(np.r_[top[0], top], np.r_[top, top[-1]])
        fraction = (np.arange(m.ny + 1)[None, :] - bottom_faces[:, None]) / (
            top_faces - bottom_faces
        )[:, None]
        psi = c.left.velocity[0] * c.Ly * np.clip(fraction, 0.0, 1.0)
        return Scene(sdf, u=np.diff(psi, axis=1) / m.dy, v=-np.diff(psi, axis=0) / m.dx)

    return c, geometry


def fission_core():
    c = SimConfig(
        "fission_core",
        rho=710.0,
        mu=0.001,
        p_ref=15.5e6,
        T_ref=560.0,
        beta=0.002,
        alpha_thermal=0.00015,
        cp=5000.0,
        gy=-9.81,
        bottom=BoundaryCondition(Wall.INFLOW_VELOCITY, (0.0, 0.20)),
        top=BoundaryCondition(Wall.FIXED_PRESSURE, pressure=15.5e6),
        target_time=1.0,
        solver_tol=2.0,
    )

    def geometry(m, c):
        sdf = np.full((m.nx, m.ny), c.Lx + c.Ly)
        for x, y in ((0.38, 0.28), (0.8, 0.28), (1.22, 0.28), (0.59, 0.55), (1.01, 0.55)):
            np.minimum(sdf, np.hypot(m.x - x, m.y - y) - 0.066, out=sdf)
        return Scene(sdf, v=0.20, Q=2.0e7 * (sdf < 0), temperature_rise=15.0 * (sdf < 0))

    return c, geometry


MINOR_RADIUS, CORE_PRESSURE = 0.38, 5.0e5


def fusion_tokamak():
    c = SimConfig("fusion_tokamak", p_ref=0.0, rho=1000.0, solver_tol=0.2, target_time=0.005)

    def geometry(m, c):
        x, y = m.x - 0.8, m.y - 0.45
        r2 = x * x + y * y
        a2 = MINOR_RADIUS**2
        factor = -4 * CORE_PRESSURE / a2 * np.maximum(0.0, 1.0 - r2 / a2)
        return Scene(MINOR_RADIUS - np.sqrt(r2), em_x=factor * x, em_y=factor * y)

    return c, geometry


EXPERIMENTS = {
    f.__name__: f
    for f in (hydrostatic, poiseuille, karman_vortex, venturi_tube, fission_core, fusion_tokamak)
}


@ti.func
def clamped_index(I, f: ti.template()):
    return ti.max(0, ti.min(I, ti.Vector([f.shape[0] - 1, f.shape[1] - 1])))


@ti.func
def bilerp(f: ti.template(), xy):
    pos = ti.max(0.0, ti.min(xy, ti.Vector([f.shape[0] - 1, f.shape[1] - 1])))
    a = ti.cast(ti.floor(pos), ti.i32)
    b = clamped_index(a + 1, f)
    w = pos - a
    return ti.math.mix(
        ti.math.mix(f[a], f[b.x, a.y], w.x), ti.math.mix(f[a.x, b.y], f[b], w.x), w.y
    )


@ti.data_oriented
class PressureLevel:
    """Positive -Laplacian. k stores east, north and Dirichlet conductance."""

    def __init__(self, shape, fp):
        self.shape = shape
        self.x = ti.field(fp, shape=shape)
        self.b = ti.field(fp, shape=shape)
        self.k = ti.Vector.field(3, fp, shape=shape)
        self.inv_diag = ti.field(fp, shape=shape)
        self.r = ti.field(fp, shape=shape)

    @ti.func
    def action(self, i, j):
        c = self.x[i, j]
        ke, kn, kd = self.k[i, j]
        ne = ti.min(i + 1, self.shape[0] - 1)
        nn = ti.min(j + 1, self.shape[1] - 1)
        val = kd * c + ke * (c - self.x[ne, j]) + kn * (c - self.x[i, nn])
        if i > 0:
            val += self.k[i - 1, j][0] * (c - self.x[i - 1, j])
        if j > 0:
            val += self.k[i, j - 1][1] * (c - self.x[i, j - 1])
        return val

    @ti.kernel
    def diagonal(self):
        for i, j in self.x:
            total = self.k[i, j].sum()
            if i > 0:
                total += self.k[i - 1, j][0]
            if j > 0:
                total += self.k[i, j - 1][1]
            self.inv_diag[i, j] = 0.0
            if total > 0.0:
                self.inv_diag[i, j] = 1.0 / total

    @ti.kernel
    def coarsen(self, fine: ti.template()):
        for I in ti.grouped(self.x):
            weights = ti.Vector([0.0, 0.0, 0.0])
            for a, b in ti.static(ti.ndrange(2, 2)):
                J = 2 * I + ti.Vector([a, b])
                if J.x < fine.shape[0] and J.y < fine.shape[1]:
                    k = fine.k[J]
                    if ti.static(a == 1):
                        weights.x += k.x
                    if ti.static(b == 1):
                        weights.y += k.y
                    weights.z += k.z
            self.k[I] = 0.5 * weights

    @ti.kernel
    def smooth(self, count: ti.template()):
        # Each offloaded loop is a device-wide color barrier. Batch the launches
        # without a CPU round trip between red and black.
        for sweep in ti.static(range(count)):
            for parity in ti.static(range(2)):
                for i, half in ti.ndrange(self.shape[0], (self.shape[1] + 1) // 2):
                    j = 2 * half + ((i + parity) & 1)
                    if j < self.shape[1]:
                        self.x[i, j] += self.inv_diag[i, j] * (self.b[i, j] - self.action(i, j))

    @ti.kernel
    def residual(self):
        for i, j in self.x:
            self.r[i, j] = self.b[i, j] - self.action(i, j)

    @ti.kernel
    def restrict(self, fine: ti.template()):
        for I in ti.grouped(self.x):
            total = 0.0
            for a, b in ti.static(ti.ndrange(2, 2)):
                J = 2 * I + ti.Vector([a, b])
                if J.x < fine.shape[0] and J.y < fine.shape[1]:
                    total += fine.r[J]
            self.b[I] = total
            self.x[I] = 0.0

    @ti.kernel
    def correct(self, coarse: ti.template()):
        for I in ti.grouped(self.x):
            if self.inv_diag[I] > 0.0:
                pos = (I + 0.5) * 0.5 - 0.5
                lo = ti.cast(ti.floor(pos), ti.i32)
                w = pos - lo
                norm, value = 0.0, 0.0
                for a, b in ti.static(ti.ndrange(2, 2)):
                    J = clamped_index(lo + ti.Vector([a, b]), coarse.x)
                    if coarse.inv_diag[J] > 0.0:
                        wt = (w.x if a else 1 - w.x) * (w.y if b else 1 - w.y)
                        value += wt * coarse.x[J]
                        norm += wt
                if norm > 0.0:
                    self.x[I] += value / norm


def device_field(dst, value, dtype):
    data = np.broadcast_to(np.asarray(value, dtype=dtype), dst.shape)
    dst.from_numpy(np.ascontiguousarray(data))


@ti.data_oriented
class MACFluidSolver:
    def __init__(self, nx=1024, ny=576, fp=ti.f64):
        if min(nx, ny) < 4:
            raise ValueError("At least four cells per axis are required")
        self.nx, self.ny, self.fp = nx, ny, fp
        self.np_fp = np.float64 if fp == ti.f64 else np.float32
        cells = (nx, ny)
        self.u, self.us = (ti.field(fp, shape=(nx + 1, ny)) for _ in range(2))
        self.v, self.vs = (ti.field(fp, shape=(nx, ny + 1)) for _ in range(2))
        self.T, self.Ts, self.Q, self.sdf, self.wall_rhs = (
            ti.field(fp, shape=cells) for _ in range(5)
        )
        self.f_EM = ti.Vector.field(2, fp, shape=cells)
        self.is_solid = ti.field(ti.i32, shape=cells)
        self.levels = [PressureLevel(cells, fp)]
        while max(cells) > 8:
            cells = tuple((k + 1) // 2 for k in cells)
            self.levels.append(PressureLevel(cells, fp))
        self.coarsest = self.levels[-1]
        n = math.prod(self.coarsest.shape)
        self.coarse_inverse = ti.field(fp, shape=(n, n))
        self.fine = self.levels[0]
        self.p = self.fine.x
        self.visible_p = ti.field(fp, shape=(nx, ny))
        self.curl = ti.field(fp, shape=(nx, ny))
        props = "dx dy rho nu beta alpha cp tref gx gy dt"
        self.par = ti.types.struct(**{k: fp for k in props.split()}).field(shape=())
        self.wall = ti.field(ti.i32, shape=4)
        self.wall_v = ti.Vector.field(2, fp, shape=4)
        self.wall_p = ti.field(fp, shape=4)
        self.wall_T = ti.field(fp, shape=4)
        self.stats = ti.field(fp, shape=10)
        self.sum_sq = ti.field(fp, shape=())
        self.net = ti.field(fp, shape=())
        self.scalar_offset = ti.field(fp, shape=())

    def load_config(self, config, geometry=None):
        config.check()
        if geometry is None:
            geometry = EXPERIMENTS[config.name]()[1]
        self.cfg = c = config
        m = Mesh(self.nx, self.ny, c)
        scene = geometry(m, c)
        self.dx, self.dy = m.dx, m.dy
        self.par[None] = dict(
            dx=m.dx,
            dy=m.dy,
            rho=c.rho,
            nu=c.mu / c.rho,
            beta=c.beta,
            alpha=c.alpha_thermal,
            cp=c.cp,
            tref=c.T_ref,
            gx=c.gx,
            gy=c.gy,
            dt=0.0,
        )
        walls = c.walls
        device_field(self.wall, [NOSLIP if w.rigid_datum else int(w.kind) for w in walls], np.int32)
        self.wall_v.from_numpy(np.array([w.velocity for w in walls], self.np_fp))
        device_field(self.wall_p, [w.pressure - c.p_ref for w in walls], self.np_fp)
        device_field(
            self.wall_T,
            [w.temperature if w.temperature is not None else c.T_ref for w in walls],
            self.np_fp,
        )
        sdf = np.broadcast_to(scene.sdf, (self.nx, self.ny))
        fluid = sdf >= 0
        self.nfluid = int(np.count_nonzero(fluid))
        if not self.nfluid:
            raise ValueError("Domain contains no fluid")
        edge_counts = [
            np.count_nonzero(a) for a in (fluid[0, :], fluid[-1, :], fluid[:, 0], fluid[:, -1])
        ]
        lengths = [n * h for n, h in zip(edge_counts, (m.dy, m.dy, m.dx, m.dx))]
        self.rigid_side = next((i for i, w in enumerate(walls) if w.rigid_datum), None)
        self.rigid_count = 0 if self.rigid_side is None else edge_counts[self.rigid_side]
        if self.rigid_side is not None and not self.rigid_count:
            raise ValueError("The pressure datum wall has no adjoining fluid")
        self.has_datum = any(
            w.kind == FIXED and not w.rigid_datum and length > 0
            for w, length in zip(walls, lengths)
        )
        self.outlet_length = sum(length for w, length in zip(walls, lengths) if w.kind == OUTFLOW)
        if not self.has_datum and not self.outlet_length:
            flux = sum(
                (1 if k % 2 else -1) * w.velocity[k // 2] * length
                for k, (w, length) in enumerate(zip(walls, lengths))
                if w.kind == INFLOW
            )
            if abs(flux) > 1e-12:
                raise ValueError("Net prescribed flux in a sealed incompressible domain")
        for dst, val in (
            (self.sdf, sdf),
            (self.u, scene.u),
            (self.v, scene.v),
            (self.Q, scene.Q),
            (self.T, c.T_ref + np.asarray(scene.temperature_rise)),
        ):
            device_field(dst, val, self.np_fp)
        device_field(self.is_solid, ~fluid, np.int32)
        self.f_EM.from_numpy(
            np.stack(np.broadcast_arrays(scene.em_x, scene.em_y, fluid)[:2], axis=-1).astype(
                self.np_fp
            )
        )
        self.p.fill(0)
        self._setup_operator()
        self.fine.diagonal()
        for fine, coarse in zip(self.levels, self.levels[1:]):
            coarse.coarsen(fine)
            coarse.diagonal()
        self._factor_coarsest()
        self._set_initial_faces(0, self.u)
        self._set_initial_faces(1, self.v)
        self.t = self.dt = self.residual = 0.0
        self.iterations = self.steps = self.capped = 0
        self.scalar_offset[None] = 0.0
        self._diagnose()
        self.probe_history = deque(maxlen=None if c.headless else 32768)
        self._record_probe()

    @ti.kernel
    def _setup_operator(self):
        for i, j in self.p:
            k = ti.Vector([0.0, 0.0, 0.0])
            source = 0.0
            if self.is_solid[i, j] == 0:
                wx, wy = 1 / self.par[None].dx ** 2, 1 / self.par[None].dy ** 2
                if i + 1 < self.nx and self.is_solid[ti.min(i + 1, self.nx - 1), j] == 0:
                    k.x = wx
                if j + 1 < self.ny and self.is_solid[i, ti.min(j + 1, self.ny - 1)] == 0:
                    k.y = wy
                for side in ti.static(range(4)):
                    edge = ti.Vector([i == 0, i == self.nx - 1, j == 0, j == self.ny - 1])
                    if edge[side] and self.wall[side] == FIXED:
                        w = 2 * (wx if ti.static(side < 2) else wy)
                        k.z += w
                        source += w * self.wall_p[side]
            self.fine.k[i, j] = k
            self.wall_rhs[i, j] = source

    @ti.func
    def _face_open(self, axis: ti.template(), I):
        D = ti.Vector([1 - axis, axis])
        lo = clamped_index(I - D, self.p)
        hi = clamped_index(I, self.p)
        return self.is_solid[lo] == 0 and self.is_solid[hi] == 0

    @ti.func
    def _side(self, axis: ti.template(), I, f: ti.template()):
        side = -1
        if I[axis] == 0:
            side = 2 * axis
        elif I[axis] == ti.static(f.shape[axis] - 1):
            side = 2 * axis + 1
        return side

    @ti.func
    def _prescribed(self, axis: ti.template(), I, side, value):
        result = value
        if not self._face_open(axis, I):
            result = 0.0
        elif side >= 0:
            kind = self.wall[side]
            if kind == INFLOW:
                result = self.wall_v[side][axis]
            elif kind == NOSLIP or kind == SLIP:
                result = 0.0
        return result

    @ti.kernel
    def _set_initial_faces(self, axis: ti.template(), f: ti.template()):
        for I in ti.grouped(f):
            f[I] = self._prescribed(axis, I, self._side(axis, I, f), f[I])

    @ti.func
    def velocity(self, pos):
        return ti.Vector(
            [
                bilerp(self.u, pos - ti.Vector([0.0, 0.5])),
                bilerp(self.v, pos - ti.Vector([0.5, 0.0])),
            ]
        )

    @ti.func
    def _face_neighbor(self, axis: ti.template(), d: ti.template(), I, sign, f: ti.template()):
        J = I + sign * ti.Vector([1 - d, d])
        K = clamped_index(J, f)
        q = f[K]
        if J[d] != K[d]:
            side = 2 * d + ti.cast(sign > 0, ti.i32)
            if ti.static(axis != d):
                if self.wall[side] == NOSLIP or self.wall[side] == INFLOW:
                    q = 2 * self.wall_v[side][axis] - f[I]
        elif ti.static(axis != d):
            if not self._face_open(axis, K):
                q = -f[I]  # tangential ghost reflected about the solid surface
        return q

    @ti.kernel
    def _predict(self, axis: ti.template(), f: ti.template(), dst: ti.template()):
        for I in ti.grouped(f):
            c = self.par[None]
            offset = 0.5 * ti.Vector([axis, 1 - axis])
            pos = I + offset
            back = pos - c.dt * self.velocity(pos) / ti.Vector([c.dx, c.dy])
            q = bilerp(f, back - offset)
            lap = 0.0
            for d in ti.static(range(2)):
                h = c.dx if ti.static(d == 0) else c.dy
                lap += (
                    self._face_neighbor(axis, d, I, -1, f)
                    - 2 * f[I]
                    + self._face_neighbor(axis, d, I, 1, f)
                ) / h**2
            D = ti.Vector([1 - axis, axis])
            a, b = clamped_index(I - D, self.p), clamped_index(I, self.p)
            acceleration = 0.5 * (self.f_EM[a][axis] + self.f_EM[b][axis]) / c.rho
            if ti.static(axis == 1):
                temperature = 0.5 * (self.Ts[a] + self.Ts[b])
                acceleration += c.gy * (1 - c.beta * (temperature - c.tref))
            else:
                acceleration += c.gx
            value = q + c.dt * (c.nu * lap + acceleration)
            dst[I] = self._prescribed(axis, I, self._side(axis, I, f), value)

    @ti.kernel
    def _temperature(self):
        for I in ti.grouped(self.T):
            c = self.par[None]
            temp = self.T[I]
            advected = temp
            if self.is_solid[I] == 0:
                pos = I + 0.5 - c.dt * self.velocity(I + 0.5) / ti.Vector([c.dx, c.dy])
                # Insulate the advective flux at rods; conduction uses both materials.
                lo = ti.cast(ti.floor(pos - 0.5), ti.i32)
                w = pos - 0.5 - lo
                norm, total = 0.0, 0.0
                for a, b in ti.static(ti.ndrange(2, 2)):
                    J = lo + ti.Vector([a, b])
                    K = clamped_index(J, self.T)
                    if self.is_solid[K] == 0:
                        v = self.T[K]
                        for d in ti.static(range(2)):
                            if J[d] != K[d]:
                                side = 2 * d + ti.cast(J[d] > K[d], ti.i32)
                                if self.wall[side] == INFLOW:
                                    v = 2 * self.wall_T[side] - self.T[K]
                        weight = (w.x if a else 1 - w.x) * (w.y if b else 1 - w.y)
                        total += weight * (v - temp)
                        norm += weight
                if norm > 0.0:
                    advected = temp + total / norm
            lap = 0.0
            for d in ti.static(range(2)):
                h = c.dx if ti.static(d == 0) else c.dy
                for sign in ti.static((-1, 1)):
                    J = I + sign * ti.Vector([1 - d, d])
                    K = clamped_index(J, self.T)
                    nb = self.T[K]
                    if K[d] != J[d] and self.wall[2 * d + int(sign > 0)] == INFLOW:
                        nb = 2 * self.wall_T[2 * d + int(sign > 0)] - temp
                    lap += (nb - temp) / h**2
            self.Ts[I] = advected + c.dt * (c.alpha * lap + self.Q[I] / (c.rho * c.cp))

    @ti.kernel
    def _extrapolate_outlets(self, axis: ti.template(), f: ti.template()):
        for k in range(ti.static(f.shape[1 - axis])):
            for end in ti.static(range(2)):
                if self.wall[2 * axis + end] == OUTFLOW:
                    n = ti.static((f.shape[axis] - 1) * end)
                    I = ti.Vector([n, k]) if ti.static(axis == 0) else ti.Vector([k, n])
                    J = I + (1 - 2 * end) * ti.Vector([1 - axis, axis])
                    if self._face_open(axis, I):
                        f[I] = f[J]

    @ti.kernel
    def _outflow_flux(self):
        self.net[None] = 0.0
        for j in range(self.ny):
            self.net[None] += self.par[None].dy * (self.us[self.nx, j] - self.us[0, j])
        for i in range(self.nx):
            self.net[None] += self.par[None].dx * (self.vs[i, self.ny] - self.vs[i, 0])

    @ti.kernel
    def _balance_outflow(self, axis: ti.template(), f: ti.template()):
        for k in range(ti.static(f.shape[1 - axis])):
            for end in ti.static(range(2)):
                if self.wall[2 * axis + end] == OUTFLOW:
                    n = ti.static((f.shape[axis] - 1) * end)
                    I = ti.Vector([n, k]) if ti.static(axis == 0) else ti.Vector([k, n])
                    if self._face_open(axis, I):
                        f[I] -= (2 * end - 1) * self.scalar_offset[None]

    @ti.kernel
    def _rhs(self):
        for i, j in self.p:
            c = self.par[None]
            div = (self.us[i + 1, j] - self.us[i, j]) / c.dx + (
                self.vs[i, j + 1] - self.vs[i, j]
            ) / c.dy
            self.fine.b[i, j] = self.wall_rhs[i, j] - c.rho / c.dt * div

    @ti.kernel
    def _reduce_residual(self):
        self.sum_sq[None] = 0.0
        for i, j in self.p:
            r = self.fine.b[i, j] - self.fine.action(i, j)
            self.sum_sq[None] += r * r

    @ti.kernel
    def _project(self, axis: ti.template(), f: ti.template(), star: ti.template()):
        for I in ti.grouped(f):
            v = star[I]
            if self._face_open(axis, I):
                c = self.par[None]
                D = ti.Vector([1 - axis, axis])
                a, b = clamped_index(I - D, self.p), clamped_index(I, self.p)
                dp = self.p[b] - self.p[a]
                side = self._side(axis, I, f)
                if side >= 0 and self.wall[ti.max(side, 0)] == FIXED:
                    dp = (1 - 2 * (side % 2)) * 2 * (self.p[b] - self.wall_p[side])
                h = c.dx if ti.static(axis == 0) else c.dy
                v -= c.dt / c.rho * dp / h
            f[I] = v

    @ti.func
    def _center(self, i, j):
        return 0.5 * ti.Vector([self.u[i, j] + self.u[i + 1, j], self.v[i, j] + self.v[i, j + 1]])

    @ti.kernel
    def _measure(self):
        for k in range(10):
            self.stats[k] = 0.0
        self.stats[3], self.stats[4] = 1.0e30, -1.0e30
        self.stats[7], self.stats[8] = 1.0e30, -1.0e30
        for i, j in self.p:
            self.curl[i, j] = 0.0
            if self.is_solid[i, j] == 0:
                c = self.par[None]
                uc, vc = self._center(i, j)
                im, ip = ti.max(i - 1, 0), ti.min(i + 1, self.nx - 1)
                jm, jp = ti.max(j - 1, 0), ti.min(j + 1, self.ny - 1)
                vort = (self._center(ip, j).y - self._center(im, j).y) / ((ip - im) * c.dx) - (
                    self._center(i, jp).x - self._center(i, jm).x
                ) / ((jp - jm) * c.dy)
                self.curl[i, j] = vort
                div = (self.u[i + 1, j] - self.u[i, j]) / c.dx + (
                    self.v[i, j + 1] - self.v[i, j]
                ) / c.dy
                p, t = self.p[i, j], self.T[i, j]
                ti.atomic_max(self.stats[0], ti.max(abs(self.u[i, j]), abs(self.u[i + 1, j])))
                ti.atomic_max(self.stats[1], ti.max(abs(self.v[i, j]), abs(self.v[i, j + 1])))
                ti.atomic_max(self.stats[2], ti.sqrt(uc * uc + vc * vc))
                ti.atomic_min(self.stats[3], p)
                ti.atomic_max(self.stats[4], p)
                ti.atomic_max(self.stats[5], abs(div))
                ti.atomic_max(self.stats[6], abs(vort))
                ti.atomic_min(self.stats[7], t)
                ti.atomic_max(self.stats[8], t)
                if ti.math.isnan(p + t + uc + vc) or ti.math.isinf(p + t + uc + vc):
                    ti.atomic_max(self.stats[9], 1.0)

    @ti.kernel
    def _pressure_trace(self, side: ti.template()):
        # Linear extrapolation from adjacent cell centers to a sealed wall.
        self.net[None] = 0.0
        for k in range(ti.static(self.ny if side < 2 else self.nx)):
            axis = ti.static(side // 2)
            n = ti.static(((self.nx, self.ny)[axis] - 1) * (side % 2))
            I = ti.Vector([n, k]) if ti.static(axis == 0) else ti.Vector([k, n])
            J = I + (1 - 2 * (side % 2)) * ti.Vector([1 - axis, axis])
            if self.is_solid[I] == 0:
                trace = self.p[I]
                if self.is_solid[J] == 0:
                    trace += 0.5 * (self.p[I] - self.p[J])
                self.net[None] += trace

    @ti.kernel
    def _publish(self):
        for I in ti.grouped(self.p):
            if self.is_solid[I] == 0:
                self.p[I] -= self.scalar_offset[None]
            self.visible_p[I] = self.p[I]

    def _diagnose(self):
        self._measure()
        s = self.stats.to_numpy().astype(float)
        if s[9] or not np.isfinite(s).all():
            raise FloatingPointError(f"{self.cfg.name} produced a non-finite state at t={self.t}")
        # A closed Neumann chamber fixes only pressure differences. Its minimum
        # fluid pressure sets the edge datum (second-order here since dp/dr vanishes).
        offset = 0.0 if self.has_datum else s[3]
        if self.rigid_side is not None:
            self._pressure_trace(self.rigid_side)
            wall = self.cfg.walls[self.rigid_side]
            offset = float(self.net[None]) / self.rigid_count - (wall.pressure - self.cfg.p_ref)
        self.scalar_offset[None] = offset
        self._publish()
        self.umax, self.vmax, self.speed = s[:3]
        self.gmin, self.gmax = s[3:5] - offset
        self.pmin, self.pmax = self.cfg.p_ref + self.gmin, self.cfg.p_ref + self.gmax
        self.divmax, self.vortmax, self.tmin, self.tmax = s[5:9]

    def _record_probe(self):
        if self.cfg.name == "karman_vortex":
            # Fixed wake probe, about 3 diameters downstream.
            i = min(self.nx - 1, int(0.74 / self.dx))
            j = min(self.ny, round(0.45 / self.dy))
            self.probe_history.append((self.t, float(self.v[i, j])))

    def _norm(self):
        self._reduce_residual()
        value = math.sqrt(float(self.sum_sq[None]) / self.nfluid)
        if not math.isfinite(value):
            raise FloatingPointError("Non-finite pressure residual")
        return value

    def _factor_coarsest(self):
        # Only this <=64-by-64 matrix crosses to the host, once per geometry.
        level = self.coarsest
        nx, ny = level.shape
        k = level.k.to_numpy().astype(np.float64)
        A = np.diag(k[:, :, 2].ravel())
        for i, j in np.ndindex(nx, ny):
            a = i * ny + j
            for b, weight in ((a + ny, k[i, j, 0]), (a + 1, k[i, j, 1])):
                if weight:
                    A[a, a] += weight
                    A[b, b] += weight
                    A[a, b] -= weight
                    A[b, a] -= weight
        # The pseudoinverse also handles Neumann gauges and empty coarse cells.
        self.coarse_inverse.from_numpy(np.linalg.pinv(A, hermitian=True).astype(self.np_fp))

    @ti.kernel
    def _solve_coarsest(self):
        for i, j in self.coarsest.x:
            ny = ti.static(self.coarsest.shape[1])
            value = 0.0
            for k in range(ti.static(self.coarse_inverse.shape[1])):
                value += self.coarse_inverse[i * ny + j, k] * self.coarsest.b[k // ny, k % ny]
            self.coarsest.x[i, j] = value

    def _coarse_correction(self, index):
        fine = self.levels[index]
        coarse = self.levels[index + 1]
        fine.residual()
        coarse.restrict(fine)
        if index + 2 == len(self.levels):
            self._solve_coarsest()
        else:
            coarse.smooth(2)
            yield from self._coarse_correction(index + 1)
            coarse.smooth(2)
        fine.correct(coarse)
        if index <= 1:
            yield

    def stable_dt(self):
        c = self.cfg
        adv = c.cfl * min(self.dx / (self.umax + 1.0e-30), self.dy / (self.vmax + 1.0e-30))
        # Reflected Dirichlet ghosts can increase a boundary row's diagonal.
        diff = max(c.mu / c.rho, c.alpha_thermal)
        diff_dt = 0.9 / (2 * diff * (1 / self.dx**2 + 1 / self.dy**2)) if diff else math.inf
        return min(c.max_dt, adv, diff_dt)

    def advance(self, remaining=math.inf):
        """Yield between GPU work groups; publish a consistent state at completion."""
        c = self.cfg
        dt = min(self.stable_dt(), remaining)
        if not (math.isfinite(dt) and dt > 0):
            raise ValueError("A positive finite time step is required")
        self.par.dt[None] = dt
        self._temperature()
        self._predict(0, self.u, self.us)
        self._predict(1, self.v, self.vs)
        if self.outlet_length:
            self._extrapolate_outlets(0, self.us)
            self._extrapolate_outlets(1, self.vs)
            if not self.has_datum:
                self._outflow_flux()
                self.scalar_offset[None] = float(self.net[None]) / self.outlet_length
                self._balance_outflow(0, self.us)
                self._balance_outflow(1, self.vs)
        self._rhs()
        it = 0
        residual = self._norm()
        while residual >= c.solver_tol and it < c.max_iters:
            stop = min(it + 25, c.max_iters)
            # Multigrid removes long-wave error; the fine operator and stopping
            # test are always the original, unmodified MAC equations.
            while it < stop:
                count = min(5, stop - it)
                pre = min(2, count)
                self.fine.smooth(pre)
                if len(self.levels) > 1:
                    yield from self._coarse_correction(0)
                if count > pre:
                    self.fine.smooth(count - pre)
                it += count
                yield
            residual = self._norm()
        self._project(0, self.u, self.us)
        self._project(1, self.v, self.vs)
        self.T.copy_from(self.Ts)
        self.t += dt
        self.dt, self.residual, self.iterations = dt, residual, it
        self.steps += 1
        if residual >= c.solver_tol:
            self.capped += 1
        self._diagnose()
        self._record_probe()

    def step(self, remaining=math.inf):
        for _ in self.advance(remaining):
            pass

    def snapshot(self, path):
        """Export physical coordinates with every staggered array; no pickle required."""
        p = self.visible_p.to_numpy()
        solid = self.is_solid.to_numpy().astype(bool)
        total = p.astype(np.float64) + self.cfg.p_ref
        total[solid] = np.nan
        # The fine p field may already be iterating in an interactive snapshot.
        payload = dict(
            p=p,
            P_total=total,
            u=self.u.to_numpy(),
            v=self.v.to_numpy(),
            T=self.T.to_numpy(),
            Q=self.Q.to_numpy(),
            f_EM=self.f_EM.to_numpy(),
            is_solid=solid,
            sdf=self.sdf.to_numpy(),
            vorticity=self.curl.to_numpy(),
            x_c=(np.arange(self.nx) + 0.5) * self.dx,
            y_c=(np.arange(self.ny) + 0.5) * self.dy,
            x_face=np.arange(self.nx + 1) * self.dx,
            y_face=np.arange(self.ny + 1) * self.dy,
            t=self.t,
            dt=self.dt,
            iterations=self.iterations,
            residual_Pa_m2=self.residual,
            max_div_s=self.divmax,
            config_json=json.dumps(asdict(self.cfg)),
            wake_probe_t_v=np.asarray(self.probe_history).reshape(-1, 2),
        )
        with atomic_output(path, "wb") as stream:
            np.savez_compressed(stream, **payload)


def relative_l2(measured, reference):
    return float(
        np.sqrt(
            np.sum((measured - reference) ** 2, dtype=np.float64)
            / np.sum(reference**2, dtype=np.float64)
        )
    )


class ExperimentManager:
    def __init__(self, args, fp):
        self.args = args
        self.names = list(EXPERIMENTS)
        self.solver = MACFluidSolver(args.width, args.width * 9 // 16, fp)
        self.out = Path(args.output)
        self.index = self.names.index(args.test)

    def _single_precision_tolerance(self, c):
        # An absolute Laplacian residual amplifies pressure ulps by O(1/h²).
        # Only built-in FP32 defaults are scaled. Explicit tolerances are exact.
        pressure_scale = {
            "hydrostatic": c.rho * abs(c.gy) * c.Ly,
            "poiseuille": 100.0,
            "fusion_tokamak": CORE_PRESSURE,
            "venturi_tube": 1200.0,
            "fission_core": c.rho * abs(c.gy) * c.Ly,
            "karman_vortex": 1.0,
        }[c.name]
        inv_h2 = (self.solver.nx / c.Lx) ** 2 + (self.solver.ny / c.Ly) ** 2
        return max(c.solver_tol, float(np.finfo(np.float32).eps) * pressure_scale * inv_h2)

    def load(self, index):
        self.index = index % len(self.names)
        c, geometry = EXPERIMENTS[self.names[self.index]]()
        updates = dict(headless=self.args.headless)
        if self.solver.fp == ti.f32 and self.args.solver_tol is None:
            updates["solver_tol"] = self._single_precision_tolerance(c)
            LOG.info("FP32 absolute Poisson tolerance %.5g Pa/m²", updates["solver_tol"])
        for key in ("target_time", "max_iters", "solver_tol", "max_dt"):
            if getattr(self.args, key) is not None:
                updates[key] = getattr(self.args, key)
        self.solver.load_config(replace(c, **updates), geometry)
        LOG.info(
            "%s, %d × %d, dx=%.5g m, dy=%.5g m",
            c.name,
            self.solver.nx,
            self.solver.ny,
            self.solver.dx,
            self.solver.dy,
        )

    def assess(self):
        s, c = self.solver, self.solver.cfg
        row = dict(
            test=c.name,
            grid=f"{s.nx}x{s.ny}",
            seconds=s.t,
            steps=s.steps,
            dt_s=s.dt,
            Pmin_Pa=s.pmin,
            Pmax_Pa=s.pmax,
            umax_m_s=s.speed,
            max_div_s=s.divmax,
            poisson_RMS_Pa_m2=s.residual,
            last_iterations=s.iterations,
            capped_steps=s.capped,
            metric="",
            relative_L2="",
            limit="",
            observations="",
        )
        m = Mesh(s.nx, s.ny, c)
        fluid = ~s.is_solid.to_numpy().astype(bool)
        p = s.visible_p.to_numpy().astype(np.float64)
        err = limit = None
        if c.name == "hydrostatic":
            hydro = np.broadcast_to(c.rho * abs(c.gy) * (c.Ly - m.y), fluid.shape)
            # Test the pressure head as well: a large atmospheric offset must
            # never hide an incorrect hydrostatic gradient.
            err = relative_l2(p[fluid], hydro[fluid])
            total_error = relative_l2(p[fluid] + c.p_ref, hydro[fluid] + c.p_ref)
            row["observations"] = json.dumps(dict(total_pressure_relative_L2=total_error))
            row["metric"], limit = "hydrostatic pressure head", 0.001
        elif c.name == "poiseuille":
            u = s.u.to_numpy().astype(np.float64)
            ref = ((c.left.pressure - c.right.pressure) / (2 * c.mu * c.Lx)) * m.y * (c.Ly - m.y)
            err = relative_l2(u, np.broadcast_to(ref, u.shape))
            row["metric"], limit = "developed axial velocity", 0.01
        elif c.name == "fusion_tokamak":
            r2 = (m.x - 0.8) ** 2 + (m.y - 0.45) ** 2
            ref = CORE_PRESSURE * (1 - r2[fluid] / MINOR_RADIUS**2) ** 2
            err = relative_l2(p[fluid], ref)
            row["metric"], limit = "radial confinement pressure", 0.01
        elif c.name == "venturi_tube":
            upstream, throat = s.nx // 8, s.nx // 2
            u = s.u.to_numpy()
            uc = 0.5 * (u[1:] + u[:-1])

            def mean_at(a, i):
                return float(a[i, fluid[i]].mean())

            row["observations"] = json.dumps(
                dict(
                    upstream_u_m_s=mean_at(uc, upstream),
                    throat_u_m_s=mean_at(uc, throat),
                    drop_Pa=mean_at(p, upstream) - mean_at(p, throat),
                )
            )
        elif c.name == "fission_core":
            t = s.T.to_numpy()
            row["observations"] = json.dumps(
                dict(
                    max_coolant_rise_K=float(t[fluid].max() - c.T_ref),
                    rod_heating_W_per_m=float(s.Q.to_numpy().sum(dtype=np.float64) * s.dx * s.dy),
                    axial_drop_Pa=float(p[:, 0].mean() - p[:, -1].mean()),
                )
            )
        elif c.name == "karman_vortex":
            history = np.array(s.probe_history)
            wake = history[history[:, 0] >= 0.5 * s.t]
            rising = np.flatnonzero((wake[:-1, 1] <= 0) & (wake[1:, 1] > 0))
            obs = dict(
                Re=c.rho * c.left.velocity[0] * 0.12 / c.mu,
                wake_rms_m_s=float(np.sqrt(np.mean(wake[:, 1] ** 2))),
                upcrossings=len(rising),
            )
            if len(rising) > 1:
                obs["shedding_Hz"] = 1.0 / float(np.diff(wake[rising, 0]).mean())
            row["observations"] = json.dumps(obs)
        if err is not None:
            row.update(relative_L2=err, limit=limit)
            row["status"] = "PASS" if err < limit else "FAIL"
        else:
            row["status"] = "MEASURED"
        if s.capped:
            row["status"] = "SOLVER_LIMIT"
        return row

    def run_batch(self):
        self.out.mkdir(parents=True, exist_ok=True)
        indices = range(len(self.names)) if self.args.run_all else [self.index]
        rows = []
        for i in indices:
            self.load(i)
            s = self.solver
            start = time.perf_counter()
            while s.cfg.target_time - s.t > 1e-13 * s.cfg.target_time:
                s.step(s.cfg.target_time - s.t)
            row = self.assess()
            row["elapsed_s"] = time.perf_counter() - start
            s.snapshot(self.out / f"{s.cfg.name}.npz")
            rows.append(row)
            # Flush a complete summary after every experiment, including failures.
            with atomic_output(self.out / "summary.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(rows)
            LOG.info(
                "%s %s: L2=%s, max |div|=%.3g /s, capped steps=%d",
                s.cfg.name,
                row["status"],
                row["relative_L2"],
                s.divmax,
                s.capped,
            )
        return 1 if any(row["status"] in ("FAIL", "SOLVER_LIMIT") for row in rows) else 0


def rgb(hex_color):
    return tuple(int(hex_color[k : k + 2], 16) / 255 for k in (0, 2, 4))


@ti.func
def blend(a, b, w):
    return a + (b - a) * w


@ti.func
def sample32(f: ti.template(), pos):
    """Interpolate in display precision, including when the simulation is FP64."""
    bounds = ti.Vector([f.shape[0] - 1, f.shape[1] - 1], dt=ti.f32)
    xy = ti.max(ti.f32(0.0), ti.min(bounds, pos))
    a = ti.cast(ti.floor(xy), ti.i32)
    b = clamped_index(a + 1, f)
    w = xy - ti.cast(a, ti.f32)
    return blend(
        blend(ti.f32(f[a]), ti.f32(f[b.x, a.y]), w.x),
        blend(ti.f32(f[a.x, b.y]), ti.f32(f[b]), w.x),
        w.y,
    )


@ti.func
def pixel_noise(i, j):
    h = ti.u32(i) * ti.u32(1597334677) ^ ti.u32(j) * ti.u32(3812015801)
    h = (h ^ (h >> 16)) * ti.u32(2246822519)
    return ti.f32((h ^ (h >> 13)) & ti.u32(65535)) / ti.f32(65536.0)


class FieldKeys:
    """Taichi 1.7 GGUI omits digit events. Read 1–4 for this window only.

    X11 subscribes to the simulation's KeyPress events. Win32 and Cocoa poll
    those four key states while this process owns the foreground window.
    The HUD field button also works when native key queries are unavailable.
    """

    def __init__(self, title):
        import ctypes as ct
        import ctypes.util

        self._poll = lambda: None
        self.close = lambda: None
        try:
            if sys.platform.startswith("linux") and os.environ.get("DISPLAY"):
                x = ct.CDLL(ctypes.util.find_library("X11") or "libX11.so.6")
                x.XOpenDisplay.restype = ct.c_void_p
                display = ct.c_void_p(x.XOpenDisplay(None))
                if not display:
                    return
                self.close = lambda: x.XCloseDisplay(display)
                window = ct.c_ulong()
                event = (ct.c_long * 24)()
                x.XLookupKeysym.restype = ct.c_ulong

                def x11():
                    if not window.value:
                        focus, revert = ct.c_ulong(), ct.c_int()
                        x.XGetInputFocus(display, ct.byref(focus), ct.byref(revert))
                        if focus.value <= 1:
                            return None
                        name = ct.c_char_p()
                        if not x.XFetchName(display, focus, ct.byref(name)) or not name:
                            return None
                        matches = name.value.startswith(title.encode())
                        x.XFree(name)
                        if not matches:
                            return None
                        window.value = focus.value
                        x.XSelectInput(display, window, ct.c_long(1))  # KeyPressMask
                        x.XFlush(display)
                    mode = None
                    while x.XCheckWindowEvent(display, window, ct.c_long(1), ct.byref(event)):
                        key = x.XLookupKeysym(ct.byref(event), 0)
                        if ord("1") <= key <= ord("4"):
                            mode = key - ord("0")
                    return mode

                self._poll = x11
            elif sys.platform == "win32":
                user = ct.windll.user32
                user.GetForegroundWindow.restype = ct.c_void_p

                def windows():
                    pid = ct.c_ulong()
                    win = ct.c_void_p(user.GetForegroundWindow())
                    user.GetWindowThreadProcessId(win, ct.byref(pid))
                    if pid.value == os.getpid():
                        return next(
                            (i for i in range(1, 5) if user.GetAsyncKeyState(48 + i) & 0x8000), None
                        )
                    return None

                self._poll = windows
            elif sys.platform == "darwin":
                objc = ct.CDLL(ctypes.util.find_library("objc"))
                quartz = ct.CDLL(ctypes.util.find_library("CoreGraphics"))
                objc.objc_getClass.restype = objc.sel_registerName.restype = ct.c_void_p
                send = ct.CFUNCTYPE(ct.c_void_p, ct.c_void_p, ct.c_void_p)(("objc_msgSend", objc))
                app = send(
                    objc.objc_getClass(b"NSApplication"),
                    objc.sel_registerName(b"sharedApplication"),
                )
                key_window = objc.sel_registerName(b"keyWindow")
                quartz.CGEventSourceKeyState.restype = ct.c_bool

                def cocoa():
                    if send(app, key_window):
                        return next(
                            (
                                i
                                for i, key in enumerate((18, 19, 20, 21), 1)
                                if quartz.CGEventSourceKeyState(0, key)
                            ),
                            None,
                        )
                    return None

                self._poll = cocoa
        except (OSError, AttributeError) as exc:
            LOG.info("Native digit-key support unavailable (%s)", exc)

    def read(self):
        return self._poll()


@ti.data_oriented
class Stellar4KVisualizer:
    names = ("Total pressure (Pa)", "Speed (m/s)", "Vorticity (1/s)", "Temperature (K)")

    def __init__(self, solver, win_w=3840, win_h=2160):
        self.s = solver
        self.win_w, self.win_h = win_w, win_h
        self.image = ti.Vector.field(3, ti.f32, shape=(win_w, win_h))
        self.palette = ti.Vector.field(3, ti.f32, shape=(4, 5))
        colors = (
            ("05070D", "1E1B4B", "7C3AED", "F97316", "FEF08A"),
            ("05070D", "0F766E", "06B6D4", "E0F2FE", "E0F2FE"),
            ("38BDF8", "195379", "05070D", "FB7185", "FBBF24"),
            ("05070D", "7F1D1D", "EA580C", "FACC15", "FFFFFF"),
        )
        self.palette.from_numpy(
            np.array([[rgb(color) for color in row] for row in colors], dtype=np.float32)
        )
        self.mode, self.grid, self.vectors = 1, True, False
        self.spark_pitch = float(max(12, round(win_w / 80)))
        self.bg, self.line = rgb("05070D"), rgb("0F172A")
        self.metal, self.rim = rgb("121722"), rgb("38BDF8")

    @ti.func
    def _color(self, mode, value):
        n = 3 if mode == 2 else 4
        z = ti.min(ti.f32(1.0), ti.max(ti.f32(0.0), value)) * ti.f32(n)
        k = ti.min(ti.i32(ti.floor(z)), n - 1)
        return blend(self.palette[mode - 1, k], self.palette[mode - 1, k + 1], z - ti.f32(k))

    @ti.kernel
    def render(
        self,
        mode: ti.i32,
        grid: ti.i32,
        vectors: ti.i32,
        low: ti.f32,
        inv_range: ti.f32,
        inv_speed: ti.f32,
        dx: ti.f32,
        dy: ti.f32,
    ):
        for i, j in self.image:
            pixel = ti.Vector([i, j], dt=ti.f32) + ti.f32(0.5)
            screen = ti.Vector([self.win_w, self.win_h], dt=ti.f32)
            domain = ti.Vector([self.s.nx, self.s.ny], dt=ti.f32)
            pos = pixel / screen * domain
            center = pos - ti.f32(0.5)
            vel = ti.Vector([ti.f32(0.0), ti.f32(0.0)])
            if vectors != 0 or mode == 2:
                vel.x = sample32(self.s.u, pos - ti.Vector([0.0, 0.5], dt=ti.f32))
                vel.y = sample32(self.s.v, pos - ti.Vector([0.5, 0.0], dt=ti.f32))
            value = ti.f32(0.0)
            if mode == 1:
                value = sample32(self.s.visible_p, center)
            elif mode == 2:
                value = ti.sqrt(vel.dot(vel))
            elif mode == 3:
                value = sample32(self.s.curl, center)
            else:
                value = sample32(self.s.T, center)
            a = ti.min(ti.f32(1.0), ti.max(ti.f32(0.0), (value - low) * inv_range))
            energy = a
            if mode == 3:
                energy = abs(ti.f32(2.0) * a - ti.f32(1.0))
            color = self._color(mode, a)
            quiet = (ti.f32(1.0) - energy) ** 2
            # Screen-space grid spans 16 x 9 divisions and is one pixel wide.
            spacing = screen / ti.Vector([16.0, 9.0], dt=ti.f32)
            if grid:
                dist = abs(ti.math.mod(pixel, spacing) - ti.f32(0.5))
                grid_alpha = ti.max(ti.f32(0.0), ti.f32(1.0) - ti.min(dist.x, dist.y))
                color += (
                    grid_alpha
                    * (ti.Vector(self.line, dt=ti.f32) - ti.Vector(self.bg, dt=ti.f32))
                    * (ti.f32(0.25) + ti.f32(0.75) * quiet)
                )
            noise = pixel_noise(i, j)
            if noise > ti.f32(0.9988):
                color += (
                    quiet
                    * (noise - ti.f32(0.9988))
                    * ti.f32(250.0)
                    * ti.Vector([0.65, 0.78, 1.0], dt=ti.f32)
                )
            if vectors:
                pitch = ti.f32(self.spark_pitch)
                d = ti.math.mod(pixel, pitch) - ti.f32(0.5) * pitch
                display_vel = vel * screen / (domain * ti.Vector([dx, dy]))
                direction = display_vel / ti.max(
                    ti.f32(1.0e-9), ti.sqrt(display_vel.dot(display_vel))
                )
                along = d.dot(direction)
                across = abs(d.x * direction.y - d.y * direction.x)
                amp = ti.min(ti.f32(1.0), ti.sqrt(vel.dot(vel)) * inv_speed)
                length = pitch * (ti.f32(0.10) + ti.f32(0.32) * ti.sqrt(amp))
                if abs(along) < length and across < ti.f32(1.2):
                    tail = (along / length + ti.f32(1.0)) * ti.f32(0.5)
                    alpha = ti.sqrt(amp) * tail * ti.min(ti.f32(1.0), ti.f32(1.2) - across)
                    color = blend(color, ti.Vector([0.80, 0.97, 1.0], dt=ti.f32), alpha)
            # SDF is positive in fluid. Estimate its change across a pixel so
            # the outline also antialiases under unequal physical axis scales.
            sd = sample32(self.s.sdf, center)
            pixel_size = ti.max(dx * domain.x / screen.x, dy * domain.y / screen.y)
            sd_px = sd / pixel_size
            metal = ti.Vector(self.metal, dt=ti.f32)
            rim = ti.Vector(self.rim, dt=ti.f32)
            cover = ti.min(ti.f32(1.0), ti.max(ti.f32(0.0), sd_px + ti.f32(0.5)))
            color = blend(metal, color, cover)
            if abs(sd_px) < ti.f32(7.0):
                halo = ti.exp(-abs(sd_px) * ti.f32(0.78)) * ti.f32(0.94)
                color = blend(color, rim, halo)
            self.image[i, j] = ti.min(ti.f32(1.0), ti.max(ti.f32(0.0), color))

    def draw(self):
        s = self.s
        # Subtract p_ref from both endpoints before mapping to preserve FP32 contrast.
        bounds = (
            (s.gmin, s.gmax),
            (0.0, max(1e-6, s.speed)),
            (-max(1e-8, s.vortmax), max(1e-8, s.vortmax)),
            (min(float(s.np_fp(s.cfg.T_ref)), s.tmin), max(float(s.np_fp(s.cfg.T_ref)), s.tmax)),
        )
        lo, hi = bounds[self.mode - 1]
        inv_range = 1.0 / (hi - lo) if hi > lo else 0.0
        self.render(
            self.mode,
            int(self.grid),
            int(self.vectors),
            lo,
            inv_range,
            1.0 / max(s.speed, 1.0e-12),
            s.dx,
            s.dy,
        )

    def hud(self, window, fps, sim_hz, paused):
        s = self.s
        gui = window.get_gui()
        width = min(0.97, 385 / self.win_w)
        height = min(0.95, 260 / self.win_h)
        with gui.sub_window("Stellar fluid lab", 0.015, 0.024, width, height) as panel:
            panel.text(s.cfg.name.replace("_", " ").title(), color=(0.35, 0.78, 0.98))
            if panel.button(self.names[self.mode - 1] + "  [1-4]"):
                self.mode = self.mode % 4 + 1
            panel.text(f"MAC {s.nx} x {s.ny}   |   Display {self.win_w} x {self.win_h}")
            panel.text(f"t {s.t:.6g} s   |   dt {s.dt:.4g} s")
            panel.text(f"Pmin {s.pmin:.8g} Pa   |   Pmax {s.pmax:.8g} Pa")
            panel.text(f"max speed {s.speed:.4g} m/s   |   max div {s.divmax:.3g} /s")
            panel.text(f"Poisson RMS {s.residual:.3g} Pa/m^2   |   {s.iterations} sweeps")
            label = "Paused" if paused else f"{sim_hz:.1f} steps/s"
            panel.text(f"{fps:.1f} FPS   |   {label}   |   capped steps {s.capped}")
            panel.text("1-4 field   V vectors   G grid   Space pause")
            panel.text("R reset   N next   S snapshot   Esc exit")

    def run(self, manager):
        title = "Stellar MAC | Fluid pressure"
        window = ti.ui.Window(title, (self.win_w, self.win_h), vsync=True, fps_limit=60)
        canvas = window.get_canvas()
        field_keys = FieldKeys(title)
        task = None
        paused = False
        image_key = None
        last = time.perf_counter()
        fps = physics_hz = 0.0
        frames = steps = 0
        try:
            while window.running:
                save = False
                digit = field_keys.read()
                if digit is not None:
                    self.mode = digit
                for event in window.get_events(ti.ui.PRESS):
                    key = event.key.lower()
                    if key == ti.ui.ESCAPE.lower():
                        window.running = False
                    elif key in ("1", "2", "3", "4"):
                        self.mode = int(key)
                    elif key == "g":
                        self.grid = not self.grid
                    elif key == "v":
                        self.vectors = not self.vectors
                    elif key == ti.ui.SPACE:
                        paused = not paused
                    elif key in ("r", "n"):
                        if task is not None:
                            task.close()
                        task = None
                        manager.load(manager.index + int(key == "n"))
                        steps = 0
                        image_key = None
                    elif key == "s":
                        save = True
                window.get_events(ti.ui.RELEASE)  # discard releases after handling press edges
                if not window.running:
                    break
                if not paused:
                    if task is None:
                        task = self.s.advance()
                    deadline = time.perf_counter() + 0.006
                    while True:
                        try:
                            next(task)
                        except StopIteration:
                            steps += 1
                            task = None
                            break
                        ti.sync()
                        if time.perf_counter() >= deadline:
                            break
                state_key = (self.s.steps, self.mode, self.grid, self.vectors)
                if state_key != image_key:
                    self.draw()
                    image_key = state_key
                canvas.set_image(self.image)
                self.hud(window, fps, physics_hz, paused)
                if save:
                    manager.out.mkdir(parents=True, exist_ok=True)
                    stem = f"{self.s.cfg.name}_{self.s.steps:08d}_{time.time_ns()}"
                    ti.tools.imwrite(self.image, str(manager.out / (stem + ".png")))
                    self.s.snapshot(manager.out / (stem + ".npz"))
                    LOG.info("Saved %s/%s (.png and .npz)", manager.out, stem)
                ti.sync()
                window.show()
                frames += 1
                now = time.perf_counter()
                if now - last >= 0.5:
                    fps, physics_hz = frames / (now - last), steps / (now - last)
                    frames = steps = 0
                    last = now
        finally:
            if task is not None:
                task.close()
            field_keys.close()
            window.destroy()

    def benchmark(self, frames):
        # Exclude JIT, include completed physics and full-size RGB generation;
        # presentation/swapchain latency is measured separately by the live HUD.
        s = self.s
        s.step()
        self.draw()
        ti.sync()
        durations = []
        for _ in range(frames):
            start = time.perf_counter()
            s.step()
            self.draw()
            ti.sync()
            durations.append(time.perf_counter() - start)
        times = np.array(durations)
        print(
            json.dumps(
                dict(
                    test=s.cfg.name,
                    physics_grid=[s.nx, s.ny],
                    display=[self.win_w, self.win_h],
                    frames=frames,
                    completed_step_and_render_Hz=1 / float(times.mean()),
                    median_ms=1000 * float(np.median(times)),
                    p95_ms=1000 * float(np.percentile(times, 95)),
                ),
                indent=2,
            )
        )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0], formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--test", choices=EXPERIMENTS, default="hydrostatic")
    parser.add_argument(
        "--run-all", action="store_true", help="run all cases to their target times"
    )
    parser.add_argument(
        "--headless", action="store_true", help="validate and export without a window"
    )
    res = parser.add_mutually_exclusive_group()
    res.add_argument("--native-4k", action="store_true", help="use a 3840 x 2160 physics grid")
    res.add_argument("--width", type=int, help="physics width divisible by 16, default 1024")
    parser.add_argument(
        "--window-scale", type=float, default=1.0, help="scale the 3840 x 2160 display"
    )
    parser.add_argument("--target-time", type=float, help="override each batch duration in seconds")
    parser.add_argument("--max-dt", type=float, help="upper bound on a physics step in seconds")
    parser.add_argument(
        "--max-iters", type=int, help="maximum fine-grid RBGS sweeps per projection"
    )
    parser.add_argument("--solver-tol", type=float, help="absolute Poisson RMS tolerance in Pa/m²")
    parser.add_argument("--output", default="stellar-results", help="snapshot and CSV directory")
    parser.add_argument("--cpu", action="store_true", help="select Taichi's CPU backend explicitly")
    parser.add_argument("--fp32", action="store_true", help="select single precision explicitly")
    parser.add_argument(
        "--benchmark-frames",
        type=int,
        metavar="N",
        help="time N completed steps and offscreen renders after warmup",
    )
    args = parser.parse_args(argv)
    args.width = 3840 if args.native_4k else (1024 if args.width is None else args.width)
    if args.width < 32 or args.width % 16:
        parser.error("--width must be a multiple of 16 and at least 32")
    if not math.isfinite(args.window_scale) or args.window_scale <= 0:
        parser.error("--window-scale must be positive and finite")
    for key in ("target_time", "max_dt", "max_iters", "solver_tol", "benchmark_frames"):
        value = getattr(args, key)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(f"--{key.replace('_', '-')} must be positive and finite")
    if args.benchmark_frames and args.run_all:
        parser.error("--benchmark-frames measures one selected case")
    args.headless = bool(args.headless or args.run_all or args.benchmark_frames)
    return args


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if (
        not args.headless
        and sys.platform.startswith("linux")
        and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    ):
        raise SystemExit("Interactive mode needs a display. Use --headless to run without one.")
    fp = init_taichi(args.cpu, args.fp32)
    manager = ExperimentManager(args, fp)
    if args.headless and not args.benchmark_frames:
        return manager.run_batch()
    ui = Stellar4KVisualizer(
        manager.solver,
        max(1, round(3840 * args.window_scale)),
        max(1, round(2160 * args.window_scale)),
    )
    manager.load(manager.index)
    if args.benchmark_frames:
        ui.benchmark(args.benchmark_frames)
    else:
        ui.run(manager)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
