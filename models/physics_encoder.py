"""
physics_encoder.py
──────────────────
Translates the declarative WorldSpec (entities + interactions + environment)
into concrete force vectors that the StateEngine integrator can consume at
every timestep.

Design philosophy
─────────────────
Each physical effect is an independent, composable `ForceModel`.  The
`PhysicsEncoder` aggregates them, iterates over all active interactions, and
returns a per-entity net-force dict ready for F = ma integration.

Supported interaction types (maps to WorldSpec Interaction.type values)
────────────────────────────────────────────────────────────────────────
  gravity      – constant body force, read from Environment.gravity
  friction     – kinetic friction opposing velocity (Coulomb model)
  drag         – quadratic aerodynamic drag  F = ½ρCdA|v|²
  contact      – normal force / ground reaction (simple y=0 plane)
  collision    – impulse-based (resolved as a discrete event, not a force)
  spring       – Hooke's law between two entities
  magnetic     – placeholder (raises NotImplementedError)

Force conventions
─────────────────
  All forces in Newtons, as (fx, fy, fz) tuples.
  x = East, y = Up, z = North (matches WorldSpec).
  Net force on an entity is the vector sum of all active models.

Usage (internal – called by StateEngine)
─────────────────────────────────────────
    encoder = PhysicsEncoder(spec)
    forces  = encoder.net_forces(states, t)
    # → {"e_car": (fx, fy, fz), "e_ball": (fx, fy, fz), ...}
"""

from __future__ import annotations
import math
from typing import NamedTuple

from models.world_spec import WorldSpec, Entity, Environment, Interaction


# ─────────────────────────────────────────────
# Canonical drag coefficient table
# ─────────────────────────────────────────────

_CD_TABLE: dict[str, float] = {
    "vehicle":    0.30,
    "projectile": 0.47,   # sphere
    "agent":      1.0,    # standing person
    "fluid":      0.0,
    "structure":  1.3,
    "terrain":    0.0,
    "object":     0.47,
}


# ─────────────────────────────────────────────
# State snapshot (what the integrator passes in)
# ─────────────────────────────────────────────

class EntityState(NamedTuple):
    """Minimal mutable view of one entity at one timestep."""
    entity_id: str
    x: float; y: float; z: float          # position m
    vx: float; vy: float; vz: float       # velocity m/s

    def speed(self) -> float:
        return math.sqrt(self.vx**2 + self.vy**2 + self.vz**2)

    def vel_unit(self) -> tuple[float, float, float]:
        s = self.speed()
        if s < 1e-9:
            return (0.0, 0.0, 0.0)
        return (self.vx / s, self.vy / s, self.vz / s)


# ─────────────────────────────────────────────
# Individual force models
# ─────────────────────────────────────────────

def _gravity_force(entity: Entity, env: Environment
                   ) -> tuple[float, float, float]:
    """
    F_gravity = m * g  (zero for static entities – ground absorbs it).
    """
    if entity.is_static:
        return (0.0, 0.0, 0.0)
    m = entity.mass
    return (
        m * env.gravity.x,
        m * env.gravity.y,
        m * env.gravity.z,
    )


def _aerodynamic_drag(entity: Entity, state: EntityState,
                       env: Environment) -> tuple[float, float, float]:
    """
    Quadratic drag: F_d = -½ρ Cd A |v|² v̂
    Frontal area A estimated from bounding box (height × depth face).
    """
    if entity.is_static:
        return (0.0, 0.0, 0.0)

    rho  = env.air_density                          # kg/m³
    cd   = _CD_TABLE.get(entity.entity_type, 0.47)
    bb   = entity.bounding_box
    area = bb.height * bb.depth                     # m²   (frontal, x-direction)

    v2   = state.speed() ** 2
    if v2 < 1e-12:
        return (0.0, 0.0, 0.0)

    mag  = 0.5 * rho * cd * area * v2
    ux, uy, uz = state.vel_unit()
    return (-mag * ux, -mag * uy, -mag * uz)


def _kinetic_friction(entity: Entity, state: EntityState,
                       env: Environment,
                       interaction: Interaction | None = None
                       ) -> tuple[float, float, float]:
    """
    Coulomb kinetic friction: F_f = -μ |F_n| v̂_horizontal
    Normal force = m * |g| when entity is on the ground (y ≈ 0).
    μ is taken from the Interaction.parameters if present,
    otherwise falls back to Entity.friction (blended with global).
    """
    if entity.is_static:
        return (0.0, 0.0, 0.0)

    # Only apply friction when entity is near ground
    if state.y > entity.bounding_box.height / 2 + 0.05:
        return (0.0, 0.0, 0.0)

    mu   = (interaction.parameters.get("friction_coefficient")
            if interaction and interaction.parameters
            else None)
    if mu is None:
        mu = (entity.friction + env.friction_global) / 2.0

    g_mag  = abs(env.gravity.y)
    f_norm = entity.mass * g_mag                    # N (normal force on flat ground)

    # Horizontal velocity only (friction acts in xz-plane)
    vx, vz = state.vx, state.vz
    v_horiz = math.sqrt(vx**2 + vz**2)
    if v_horiz < 1e-9:
        return (0.0, 0.0, 0.0)

    mag = mu * f_norm
    return (-mag * vx / v_horiz, 0.0, -mag * vz / v_horiz)


def _ground_normal(entity: Entity, state: EntityState,
                    env: Environment) -> tuple[float, float, float]:
    """
    Ground reaction force: prevents sinking below y=0.
    Returns an upward force equal to gravity when the entity is at ground level.
    This is a penalty-spring approximation (stiff spring, k = 1e5 N/m).
    """
    if entity.is_static:
        return (0.0, 0.0, 0.0)

    ground_y  = entity.bounding_box.height / 2.0
    penetration = ground_y - state.y
    if penetration <= 0:
        return (0.0, 0.0, 0.0)

    k      = 1e5   # N/m  stiff penalty spring
    damping = 2e3  # N·s/m
    fy = k * penetration - damping * min(state.vy, 0.0)
    return (0.0, max(fy, 0.0), 0.0)


def _spring_force(entity_a: Entity, entity_b: Entity,
                   state_a: EntityState, state_b: EntityState,
                   params: dict) -> tuple[tuple[float,float,float],
                                          tuple[float,float,float]]:
    """
    Hooke's law spring between two entities.
    Returns (force_on_a, force_on_b).
    params keys: k_Nm (stiffness), rest_length_m, damping_Nsm
    """
    k       = params.get("k_Nm",        100.0)
    l0      = params.get("rest_length_m", 1.0)
    damping = params.get("damping_Nsm",   0.0)

    dx = state_b.x - state_a.x
    dy = state_b.y - state_a.y
    dz = state_b.z - state_a.z
    dist = math.sqrt(dx**2 + dy**2 + dz**2)
    if dist < 1e-9:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)

    # Unit vector a → b
    ux, uy, uz = dx / dist, dy / dist, dz / dist

    # Extension
    extension = dist - l0

    # Relative velocity along spring axis
    dvx = state_b.vx - state_a.vx
    dvy = state_b.vy - state_a.vy
    dvz = state_b.vz - state_a.vz
    v_rel = dvx * ux + dvy * uy + dvz * uz

    mag = k * extension + damping * v_rel
    fa  = ( mag * ux,  mag * uy,  mag * uz)   # force on A (toward B)
    fb  = (-mag * ux, -mag * uy, -mag * uz)   # reaction on B
    return fa, fb


# ─────────────────────────────────────────────
# Wind helper
# ─────────────────────────────────────────────

def _wind_force(entity: Entity, state: EntityState,
                env: Environment) -> tuple[float, float, float]:
    """
    Aerodynamic drag relative to wind velocity.
    Wind blows in the xz-plane (no vertical component assumed).
    """
    if entity.is_static or env.wind.speed < 1e-3:
        return (0.0, 0.0, 0.0)

    wd  = env.wind.direction             # rad from North (z-axis)
    ws  = env.wind.speed                 # m/s
    # Wind velocity vector (in world frame)
    wx  =  ws * math.sin(wd)            # East component
    wz  =  ws * math.cos(wd)            # North component

    # Relative velocity = entity vel − wind vel
    rvx = state.vx - wx
    rvz = state.vz - wz

    rho  = env.air_density
    cd   = _CD_TABLE.get(entity.entity_type, 0.47)
    area = entity.bounding_box.height * entity.bounding_box.depth
    rv2  = rvx**2 + rvz**2
    if rv2 < 1e-12:
        return (0.0, 0.0, 0.0)

    rv_mag = math.sqrt(rv2)
    mag    = 0.5 * rho * cd * area * rv2
    return (-mag * rvx / rv_mag, 0.0, -mag * rvz / rv_mag)


# ─────────────────────────────────────────────
# PhysicsEncoder
# ─────────────────────────────────────────────

class PhysicsEncoder:
    """
    Encodes a WorldSpec into force functions callable at each integration step.

    Parameters
    ----------
    spec : WorldSpec
        The fully-assembled world specification.

    Public API
    ----------
    net_forces(states, t) → dict[str, tuple[float,float,float]]
        Given a dict mapping entity_id → EntityState and the current time,
        return the net force vector (fx, fy, fz) for each dynamic entity.
    """

    def __init__(self, spec: WorldSpec):
        self.spec    = spec
        self.env     = spec.environment
        self._entity_map: dict[str, Entity] = {e.id: e for e in spec.entities}

        # Classify interactions for fast lookup
        self._friction_ints:  list[Interaction] = []
        self._contact_ints:   list[Interaction] = []
        self._drag_ints:      list[Interaction] = []
        self._spring_ints:    list[Interaction] = []
        self._unsupported:    list[str]         = []

        for itr in spec.interactions:
            t = itr.type.lower()
            if t == "friction":
                self._friction_ints.append(itr)
            elif t in ("contact", "collision"):
                self._contact_ints.append(itr)
            elif t == "fluid_drag":
                self._drag_ints.append(itr)
            elif t == "spring":
                self._spring_ints.append(itr)
            elif t in ("magnetic",):
                self._unsupported.append(t)
            # gravity is always applied; no entry needed

        if self._unsupported:
            print(f"[PhysicsEncoder] NOTE: unsupported interaction types "
                  f"(will be skipped): {set(self._unsupported)}")

    # ── main entry point ─────────────────────

    def net_forces(self,
                   states: dict[str, EntityState],
                   t: float
                   ) -> dict[str, tuple[float, float, float]]:
        """
        Compute net force on every dynamic entity at time t.

        Parameters
        ----------
        states : dict[entity_id → EntityState]
        t      : current simulation time (s) – used for event-driven forces

        Returns
        -------
        dict[entity_id → (fx, fy, fz)]   units: Newtons
        """
        net: dict[str, list[float]] = {
            eid: [0.0, 0.0, 0.0]
            for eid, e in self._entity_map.items()
            if not e.is_static
        }

        for eid, acc in net.items():
            entity = self._entity_map[eid]
            state  = states.get(eid)
            if state is None:
                continue

            # 1. Gravity (always)
            gx, gy, gz = _gravity_force(entity, self.env)
            acc[0] += gx; acc[1] += gy; acc[2] += gz

            # 2. Ground normal force (always for non-flying entities)
            nx, ny, nz = _ground_normal(entity, state, self.env)
            acc[0] += nx; acc[1] += ny; acc[2] += nz

            # 3. Aerodynamic drag (always – zero if env.air_density ≈ 0)
            dx, dy, dz = _aerodynamic_drag(entity, state, self.env)
            acc[0] += dx; acc[1] += dy; acc[2] += dz

            # 4. Wind force
            wx, wy, wz = _wind_force(entity, state, self.env)
            acc[0] += wx; acc[1] += wy; acc[2] += wz

        # 5. Friction (from declared friction interactions)
        for itr in self._friction_ints:
            ea = self._entity_map.get(itr.entity_a)
            if ea is None or ea.is_static:
                continue
            sa = states.get(itr.entity_a)
            if sa is None:
                continue
            fx, fy, fz = _kinetic_friction(ea, sa, self.env, itr)
            net[itr.entity_a][0] += fx
            net[itr.entity_a][1] += fy
            net[itr.entity_a][2] += fz

        # 6. If no explicit friction interaction, apply default for ground contact
        friction_covered = {itr.entity_a for itr in self._friction_ints}
        for eid, acc in net.items():
            if eid in friction_covered:
                continue
            entity = self._entity_map[eid]
            state  = states.get(eid)
            if state is None:
                continue
            fx, fy, fz = _kinetic_friction(entity, state, self.env)
            acc[0] += fx; acc[1] += fy; acc[2] += fz

        # 7. Spring forces (bidirectional)
        for itr in self._spring_ints:
            ea = self._entity_map.get(itr.entity_a)
            eb = self._entity_map.get(itr.entity_b)
            if ea is None or eb is None:
                continue
            sa = states.get(itr.entity_a)
            sb = states.get(itr.entity_b)
            if sa is None or sb is None:
                continue
            fa, fb = _spring_force(ea, eb, sa, sb, itr.parameters)
            if not ea.is_static and itr.entity_a in net:
                net[itr.entity_a][0] += fa[0]
                net[itr.entity_a][1] += fa[1]
                net[itr.entity_a][2] += fa[2]
            if not eb.is_static and itr.entity_b in net:
                net[itr.entity_b][0] += fb[0]
                net[itr.entity_b][1] += fb[1]
                net[itr.entity_b][2] += fb[2]

        return {eid: (v[0], v[1], v[2]) for eid, v in net.items()}

    # ── event handling ───────────────────────

    def get_active_events(self, t: float, dt: float) -> list[dict]:
        """
        Return SimulationGraph events whose t_s falls in [t, t+dt).
        StateEngine calls this each step to apply discrete impulses.
        """
        lo, hi = t, t + dt
        return [
            ev for ev in self.spec.simulation_graph.events
            if lo <= ev.get("t_s", -1.0) < hi
        ]
