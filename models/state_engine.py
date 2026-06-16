"""
state_engine.py
────────────────
Deterministic time-integration engine for PhysWorldLM.

Pipeline:

    WorldSpec
        │
        ▼
    PhysicsEncoder  (gravity, friction, drag, wind, ground contact, springs)
        │
        ▼
    Net Forces
        │
        ▼
    Integration (Euler / RK4)
        │
        ▼
    Trajectory  (sampled at export_fps, integrated at dt)

This module owns the simulation loop. It does not invent physics models —
all force computation is delegated to ``PhysicsEncoder``. Its sole
responsibilities are:

  1. Maintaining mutable per-entity kinematic state.
  2. Driving the chosen numerical integrator (Euler or RK4) at a fixed
     timestep ``dt``.
  3. Sampling the resulting trajectory at ``export_fps`` and packaging it
     into ``Trajectory`` / ``Frame`` records, including kinetic and
     potential energy bookkeeping.
  4. Providing a small, stable public API (`simulate`, `step`, `reset`,
     `export_csv`) suitable for downstream tooling, notebooks, and tests.

Interface assumptions on collaborating modules
-----------------------------------------------
``physics_encoder.PhysicsEncoder``:

    class PhysicsEncoder:
        def __init__(self, spec: WorldSpec) -> None: ...

        @dataclass
        class EntityState:
            entity_id: str
            mass: float
            position: Vec3
            velocity: Vec3
            orientation: Vec3
            angular_velocity: Vec3

        def net_forces(
            self,
            states: dict[str, "PhysicsEncoder.EntityState"],
            t: float,
        ) -> dict[str, Vec3]:
            \"\"\"Return net force vector (N) acting on each dynamic entity,
            given the current state of *all* entities and the current
            simulation time.\"\"\"

``trajectory.Frame`` / ``trajectory.Trajectory``:

    @dataclass
    class Frame:
        t: float
        entity_id: str
        position: Vec3
        velocity: Vec3
        acceleration: Vec3
        orientation: Vec3
        angular_vel: Vec3
        kinetic_energy: float
        potential_energy: float

    class Trajectory:
        def __init__(self, scene_id: str) -> None: ...
        def add_frame(self, frame: Frame) -> None: ...
        # additional serialization helpers (to_json, save, ...) may exist
        # but are not required by StateEngine.

If the real modules differ slightly (e.g. ``EntityState`` field names),
only the small adapter methods ``MutableEntityState.to_entity_state`` and
``StateEngine._build_frame`` need to change — the integration core is
agnostic to these details.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field, replace
from typing import Optional

from models.world_spec import WorldSpec, Entity, Vec3
from models.physics_encoder import PhysicsEncoder, EntityState
from models.trajectory import Frame, Trajectory


# ─────────────────────────────────────────────
# MutableEntityState
# ─────────────────────────────────────────────

@dataclass
class MutableEntityState:
    """
    Mutable kinematic + dynamic state for a single dynamic entity.

    Unlike :class:`models.world_spec.PhysicsState` (an immutable snapshot
    used for serialization), ``MutableEntityState`` is the *working*
    representation that the integrator reads from and writes to on every
    timestep. Position, velocity, and acceleration are stored as flat
    scalar components (``x, y, z`` / ``vx, vy, vz`` / ``ax, ay, az``)
    rather than ``Vec3`` instances, which keeps the RK4 derivative
    arithmetic simple and avoids repeated object construction in the hot
    loop.

    Attributes
    ----------
    entity_id:
        Identifier matching :attr:`models.world_spec.Entity.id`.
    mass:
        Mass in kilograms. Constant throughout the simulation.
    x, y, z:
        Position components in metres (world frame: x=East, y=Up, z=North).
    vx, vy, vz:
        Velocity components in metres per second.
    ax, ay, az:
        Acceleration components in metres per second squared. Recomputed
        every timestep from the net force; retained between steps purely
        for reporting (e.g. in :class:`Frame`).
    orientation:
        Euler angles (radians) about x, y, z.
    angular_velocity:
        Angular velocity (radians per second) about x, y, z.
    """

    entity_id: str
    mass: float

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0

    ax: float = 0.0
    ay: float = 0.0
    az: float = 0.0

    orientation: Vec3 = field(default_factory=Vec3)
    angular_velocity: Vec3 = field(default_factory=Vec3)

    # ── conversions ──────────────────────────

    def to_entity_state(self) -> PhysicsEncoder.EntityState:
        """
        Convert this mutable state into the immutable
        ``PhysicsEncoder.EntityState`` snapshot expected by
        :meth:`PhysicsEncoder.net_forces`.

        Returns
        -------
        PhysicsEncoder.EntityState
            A read-only snapshot of position, velocity, orientation, and
            angular velocity, suitable for passing into the encoder.
        """
        return EntityState(
            entity_id=self.entity_id,
            x=self.x,
            y=self.y,
            z=self.z,
            vx=self.vx,
            vy=self.vy,
            vz=self.vz,
        )

    def position_vec(self) -> Vec3:
        """Return current position as a :class:`Vec3`."""
        return Vec3(self.x, self.y, self.z)

    def velocity_vec(self) -> Vec3:
        """Return current velocity as a :class:`Vec3`."""
        return Vec3(self.vx, self.vy, self.vz)

    def acceleration_vec(self) -> Vec3:
        """Return current (last-computed) acceleration as a :class:`Vec3`."""
        return Vec3(self.ax, self.ay, self.az)

    def copy(self) -> "MutableEntityState":
        """Return a deep-enough copy safe for RK4 stage perturbation."""
        return replace(
            self,
            orientation=Vec3(
                self.orientation.x, self.orientation.y, self.orientation.z
            ),
            angular_velocity=Vec3(
                self.angular_velocity.x,
                self.angular_velocity.y,
                self.angular_velocity.z,
            ),
        )


# ─────────────────────────────────────────────
# Internal derivative representation
# ─────────────────────────────────────────────

@dataclass(frozen=True)
class _Derivative:
    """
    Time-derivative of the integrable state vector for one entity.

    ``d(position)/dt = velocity`` and ``d(velocity)/dt = acceleration``.
    This lightweight container is used internally by the RK4 stages to
    avoid recomputing forces against a full :class:`MutableEntityState`
    when only the six-component derivative is needed.
    """

    vx: float
    vy: float
    vz: float
    ax: float
    ay: float
    az: float


# ─────────────────────────────────────────────
# StateEngine
# ─────────────────────────────────────────────

class StateEngine:
    """
    Deterministic physics state engine.

    ``StateEngine`` consumes a fully populated :class:`WorldSpec`,
    constructs mutable per-entity state, and integrates the equations of
    motion forward in time using forces supplied by
    :class:`models.physics_encoder.PhysicsEncoder`. The result is a
    :class:`models.trajectory.Trajectory` containing one
    :class:`models.trajectory.Frame` per exported sample, per entity.

    The engine is *deterministic*: given the same ``WorldSpec``, the same
    integrator, and the same ``dt``, repeated calls to :meth:`simulate`
    produce bit-identical trajectories (modulo floating point
    associativity, which is fixed by always applying operations in the
    same order).

    Parameters
    ----------
    spec:
        A fully populated :class:`models.world_spec.WorldSpec`. The engine
        reads ``dt``, ``duration``, ``export_fps``, and ``integrator`` from
        ``spec.simulation_graph``.

    Attributes
    ----------
    spec:
        The source :class:`WorldSpec` (not mutated).
    encoder:
        The :class:`PhysicsEncoder` instance used to compute net forces.
    dt:
        Integration timestep, in seconds.
    duration:
        Total simulated time, in seconds.
    export_fps:
        Frame export rate, in frames per second.
    integrator:
        Either ``"euler"`` or ``"rk4"``. Any other value in
        ``spec.simulation_graph.integrator`` (including ``"verlet"``,
        which is not implemented) falls back to ``"rk4"`` with a logged
        warning.
    states:
        Mapping of entity id → :class:`MutableEntityState` for every
        *dynamic* entity in the spec. Static entities are not integrated
        but their state is still made available (immutably) to the
        encoder via :attr:`static_states`.
    static_states:
        Mapping of entity id → :class:`MutableEntityState` for every
        *static* entity. These never change after initialization and are
        included in encoder calls so that static geometry (ground,
        walls, etc.) can participate in force computations (e.g. ground
        contact, spring anchors).
    t:
        Current simulation time, in seconds. Starts at ``0.0``.
    trajectory:
        The :class:`Trajectory` being built by the current/most recent
        call to :meth:`simulate`.
    """

    _VALID_INTEGRATORS = ("euler", "rk4")

    def __init__(self, spec: WorldSpec) -> None:
        self.spec: WorldSpec = spec
        self.encoder: PhysicsEncoder = PhysicsEncoder(spec)

        sg = spec.simulation_graph
        self.dt: float = float(sg.dt)
        self.duration: float = float(sg.duration)
        self.export_fps: int = int(sg.export_fps)

        integrator = (sg.integrator or "rk4").lower()
        if integrator not in self._VALID_INTEGRATORS:
            print(
                f"[StateEngine] WARNING integrator '{integrator}' not "
                f"supported, falling back to 'rk4'"
            )
            integrator = "rk4"
        self.integrator: str = integrator

        # ── build mutable states ─────────────
        self._initial_states: dict[str, MutableEntityState] = {}
        self._initial_static_states: dict[str, MutableEntityState] = {}
        for entity in spec.entities:
            mstate = self._mutable_state_from_entity(entity)
            if entity.is_static:
                self._initial_static_states[entity.id] = mstate
            else:
                self._initial_states[entity.id] = mstate

        # working copies (reset() restores these from the initial copies)
        self.states: dict[str, MutableEntityState] = {
            eid: s.copy() for eid, s in self._initial_states.items()
        }
        self.static_states: dict[str, MutableEntityState] = {
            eid: s.copy() for eid, s in self._initial_static_states.items()
        }

        self.t: float = 0.0
        self.trajectory: Trajectory = Trajectory(
            scene_id=spec.scene_id,
            description=spec.description,
            entity_ids=[
                e.id
                for e in spec.entities
                if not e.is_static
            ]
        )

        # number of integration steps and export cadence
        self._n_steps: int = max(1, round(self.duration / self.dt))
        export_period_s = 1.0 / self.export_fps if self.export_fps > 0 else self.dt
        self._export_every_n: int = max(1, round(export_period_s / self.dt))

    # ── construction helpers ──────────────────

    @staticmethod
    def _mutable_state_from_entity(entity: Entity) -> MutableEntityState:
        """
        Build a :class:`MutableEntityState` from an :class:`Entity`'s
        initial :class:`models.world_spec.PhysicsState`.

        Acceleration is initialized to zero; it is recomputed from forces
        on the first call to :meth:`StateEngine._compute_accelerations`
        (or the first RK4 stage) before any frame is exported.
        """
        ps = entity.state
        return MutableEntityState(
            entity_id=entity.id,
            mass=entity.mass,
            x=ps.position.x, y=ps.position.y, z=ps.position.z,
            vx=ps.velocity.x, vy=ps.velocity.y, vz=ps.velocity.z,
            ax=0.0, ay=0.0, az=0.0,
            orientation=Vec3(
                ps.orientation.x, ps.orientation.y, ps.orientation.z
            ),
            angular_velocity=Vec3(
                ps.angular_vel.x, ps.angular_vel.y, ps.angular_vel.z
            ),
        )

    # ── force / acceleration helpers ──────────

    def _all_entity_states(
        self, states: dict[str, MutableEntityState]
    ) -> dict[str, PhysicsEncoder.EntityState]:
        """
        Build the full ``{entity_id: EntityState}`` mapping (dynamic +
        static) required by :meth:`PhysicsEncoder.net_forces`.

        Parameters
        ----------
        states:
            The *dynamic* entity states to use (a working copy during RK4
            stage evaluation, or :attr:`self.states` for the canonical
            current state).
        """
        snapshot: dict[str, PhysicsEncoder.EntityState] = {
            eid: s.to_entity_state() for eid, s in states.items()
        }
        for eid, s in self.static_states.items():
            snapshot[eid] = s.to_entity_state()
        return snapshot

    def _compute_accelerations(
        self, states: dict[str, MutableEntityState], t: float
    ) -> dict[str, tuple[float, float, float]]:
        """
        Compute ``a = F / m`` for every dynamic entity.

        Parameters
        ----------
        states:
            Dynamic entity states at which to evaluate forces.
        t:
            Simulation time (seconds) at which to evaluate forces. Some
            force terms (e.g. scripted events, time-varying wind) may
            depend on ``t``.

        Returns
        -------
        dict[str, tuple[float, float, float]]
            Mapping of entity id → ``(ax, ay, az)`` in m/s².
        """
        snapshot = self._all_entity_states(states)
        forces = self.encoder.net_forces(snapshot, t)

        accelerations: dict[str, tuple[float, float, float]] = {}
        for eid, mstate in states.items():
            f = forces.get(eid, (0.0, 0.0, 0.0))
            m = mstate.mass
            if m <= 0:
                raise ValueError(
                    f"Entity '{eid}' has non-positive mass ({m} kg); "
                    f"cannot integrate F = m·a"
                )
            accelerations[eid] = (f[0] / m, f[1] / m, f[2] / m)
        return accelerations

    # ── derivative computation (for RK4) ──────

    def _derivative(
        self,
        states: dict[str, MutableEntityState],
        t: float,
    ) -> dict[str, _Derivative]:
        """
        Compute ``d(state)/dt`` for every dynamic entity at the given
        state and time.

        The integrable state vector for each entity is
        ``(x, y, z, vx, vy, vz)``; its derivative is
        ``(vx, vy, vz, ax, ay, az)``.
        """
        accelerations = self._compute_accelerations(states, t)
        derivatives: dict[str, _Derivative] = {}
        for eid, mstate in states.items():
            ax, ay, az = accelerations[eid]
            derivatives[eid] = _Derivative(
                vx=mstate.vx, vy=mstate.vy, vz=mstate.vz,
                ax=ax, ay=ay, az=az,
            )
        return derivatives

    @staticmethod
    def _apply_derivative(
        states: dict[str, MutableEntityState],
        derivatives: dict[str, _Derivative],
        scale: float,
    ) -> dict[str, MutableEntityState]:
        """
        Return a new state mapping advanced by ``scale * derivative``.

        Used to build the perturbed states ``y + h/2 * k1``, etc. required
        by intermediate RK4 stages. The originals in ``states`` are not
        mutated.
        """
        advanced: dict[str, MutableEntityState] = {}
        for eid, mstate in states.items():
            d = derivatives[eid]
            new = mstate.copy()
            new.x += scale * d.vx
            new.y += scale * d.vy
            new.z += scale * d.vz
            new.vx += scale * d.ax
            new.vy += scale * d.ay
            new.vz += scale * d.az
            advanced[eid] = new
        return advanced

    # ── integrators ────────────────────────────

    def _integrate_euler(self) -> None:
        """
        Advance :attr:`states` by one timestep using explicit (forward)
        Euler integration.

        ``v(t+dt) = v(t) + a(t)·dt``
        ``x(t+dt) = x(t) + v(t)·dt``

        Position is updated using the *pre-update* velocity, which is the
        standard (non-symplectic) forward-Euler formulation. Acceleration
        is stored on each :class:`MutableEntityState` for reporting.
        """
        dt = self.dt
        accelerations = self._compute_accelerations(self.states, self.t)

        for eid, mstate in self.states.items():
            ax, ay, az = accelerations[eid]
            mstate.ax, mstate.ay, mstate.az = ax, ay, az

            # position update uses velocity *before* it is advanced
            mstate.x += mstate.vx * dt
            mstate.y += mstate.vy * dt
            mstate.z += mstate.vz * dt

            mstate.vx += ax * dt
            mstate.vy += ay * dt
            mstate.vz += az * dt

    def _integrate_rk4(self) -> None:
        """
        Advance :attr:`states` by one timestep using classical fourth-order
        Runge-Kutta (RK4) integration on the 6-component state vector
        ``(x, y, z, vx, vy, vz)`` per entity.

        Standard RK4 update for state ``y`` and derivative function
        ``f(y, t)``::

            k1 = f(y,            t)
            k2 = f(y + dt/2·k1,  t + dt/2)
            k3 = f(y + dt/2·k2,  t + dt/2)
            k4 = f(y + dt·k3,    t + dt)

            y(t+dt) = y(t) + dt/6 · (k1 + 2 k2 + 2 k3 + k4)

        Acceleration stored on each entity for reporting purposes is taken
        from ``k1`` (i.e. the acceleration evaluated at the start of the
        step), consistent with the Euler integrator's convention.
        """
        dt = self.dt
        t = self.t

        k1 = self._derivative(self.states, t)

        states_k2 = self._apply_derivative(self.states, k1, dt / 2.0)
        k2 = self._derivative(states_k2, t + dt / 2.0)

        states_k3 = self._apply_derivative(self.states, k2, dt / 2.0)
        k3 = self._derivative(states_k3, t + dt / 2.0)

        states_k4 = self._apply_derivative(self.states, k3, dt)
        k4 = self._derivative(states_k4, t + dt)

        for eid, mstate in self.states.items():
            d1, d2, d3, d4 = k1[eid], k2[eid], k3[eid], k4[eid]

            mstate.x += (dt / 6.0) * (d1.vx + 2 * d2.vx + 2 * d3.vx + d4.vx)
            mstate.y += (dt / 6.0) * (d1.vy + 2 * d2.vy + 2 * d3.vy + d4.vy)
            mstate.z += (dt / 6.0) * (d1.vz + 2 * d2.vz + 2 * d3.vz + d4.vz)

            mstate.vx += (dt / 6.0) * (d1.ax + 2 * d2.ax + 2 * d3.ax + d4.ax)
            mstate.vy += (dt / 6.0) * (d1.ay + 2 * d2.ay + 2 * d3.ay + d4.ay)
            mstate.vz += (dt / 6.0) * (d1.az + 2 * d2.az + 2 * d3.az + d4.az)

            # store start-of-step acceleration for reporting / energy calc
            mstate.ax, mstate.ay, mstate.az = d1.ax, d1.ay, d1.az

    # ── energy ─────────────────────────────────

    def _kinetic_energy(self, mstate: MutableEntityState) -> float:
        """
        Kinetic energy ``KE = 1/2 · m · |v|²`` in joules.
        """
        v2 = mstate.vx ** 2 + mstate.vy ** 2 + mstate.vz ** 2
        return 0.5 * mstate.mass * v2

    def _potential_energy(self, mstate: MutableEntityState) -> float:
        """
        Gravitational potential energy ``PE = m · |g| · h`` in joules,
        where ``h`` is the entity's height (``y`` coordinate, metres)
        above the ``y = 0`` reference plane and ``|g|`` is the magnitude
        of :attr:`models.world_spec.Environment.gravity`.

        Negative heights (an entity below the reference plane) yield
        negative potential energy, which is physically consistent with the
        ``PE = m g h`` convention relative to a fixed datum.
        """
        g = self.spec.environment.gravity.magnitude()
        return mstate.mass * g * mstate.y

    # ── frame construction ─────────────────────

    def _build_frame(self, mstate: MutableEntityState) -> Frame:
        """
        Construct a :class:`models.trajectory.Frame` snapshot for one
        entity at the current simulation time :attr:`t`.
        """
        return Frame(
            t=self.t,
            entity_id=mstate.entity_id,
            position=mstate.position_vec(),
            velocity=mstate.velocity_vec(),
            acceleration=mstate.acceleration_vec(),
            orientation=Vec3(
                mstate.orientation.x, mstate.orientation.y, mstate.orientation.z
            ),
            angular_vel=Vec3(
                mstate.angular_velocity.x,
                mstate.angular_velocity.y,
                mstate.angular_velocity.z,
            ),
            kinetic_energy=self._kinetic_energy(mstate),
            potential_energy=self._potential_energy(mstate),
        )

    def _export_frames(self) -> None:
        """Append one :class:`Frame` per dynamic entity at the current time."""
        for mstate in self.states.values():
            self.trajectory.add_frame(self._build_frame(mstate))

    # ── public API ──────────────────────────────

    def step(self) -> None:
        """
        Advance the simulation by exactly one timestep ``dt``.

        Computes net forces via the :class:`PhysicsEncoder`, integrates
        all dynamic entities forward using the configured integrator
        (Euler or RK4), and advances :attr:`t` by :attr:`dt`. This method
        does not export frames; frame export cadence is managed by
        :meth:`simulate`.

        If there are no dynamic entities, this still advances time but
        performs no integration work.
        """
        if self.states:
            if self.integrator == "euler":
                self._integrate_euler()
            else:
                self._integrate_rk4()
        self.t += self.dt

    def reset(self) -> None:
        """
        Restore the engine to its initial conditions.

        Resets :attr:`states` and :attr:`static_states` to deep copies of
        the entity states as constructed from the source
        :class:`WorldSpec`, resets :attr:`t` to ``0.0``, and starts a fresh
        :class:`Trajectory`. After calling :meth:`reset`, :meth:`simulate`
        will reproduce the same trajectory as the first run (determinism
        guarantee).
        """
        self.states = {
            eid: s.copy() for eid, s in self._initial_states.items()
        }
        self.static_states = {
            eid: s.copy() for eid, s in self._initial_static_states.items()
        }
        self.t = 0.0
        self.trajectory = Trajectory(
            scene_id=self.spec.scene_id,
            description=self.spec.description,
            entity_ids=[
                e.id
                for e in spec.entities
                if not e.is_static
            ],
        )

    def simulate(self) -> Trajectory:
        """
        Run the complete simulation from ``t = 0`` to ``t = duration``.

        Integration proceeds at the fixed timestep :attr:`dt` using the
        configured integrator. Frames are exported into the returned
        :class:`Trajectory` at approximately :attr:`export_fps`, by
        exporting every ``round((1/export_fps) / dt)``-th integration
        step (and always exporting the initial state at ``t = 0``).

        Returns
        -------
        Trajectory
            The fully populated trajectory for this run. Also stored on
            :attr:`self.trajectory`.

        Notes
        -----
        Calling :meth:`simulate` multiple times without an intervening
        :meth:`reset` will continue integrating from wherever the previous
        call left off, accumulating additional frames into the *same*
        trajectory. Call :meth:`reset` first to start a fresh,
        independent run.
        """
        print(f"[StateEngine] START scene_id={self.spec.scene_id} "
              f"integrator={self.integrator} dt={self.dt} "
              f"duration={self.duration} export_fps={self.export_fps}")

        # Export the initial state (t = 0, or wherever we currently are).
        self._export_frames()

        exported = 1 if self.states else 0
        for step_index in range(1, self._n_steps + 1):
            self.step()
            if step_index % self._export_every_n == 0 or step_index == self._n_steps:
                self._export_frames()
                if self.states:
                    exported += 1

        print(f"[StateEngine] steps={self._n_steps}")
        print(f"[StateEngine] exported_frames={exported * max(1, len(self.states))}")
        print("[StateEngine] DONE")
        return self.trajectory

    def export_csv(self, path: str) -> None:
        """
        Export the most recent trajectory to a CSV file.

        The CSV contains one row per ``(time, entity)`` pair with columns::

            time, entity_id, x, y, z, vx, vy, vz, ax, ay, az

        Rows are written in the order frames were appended to
        :attr:`trajectory` (i.e. chronological, then by entity within each
        exported timestep).

        Parameters
        ----------
        path:
            Destination file path. Parent directories are not created
            automatically.

        Raises
        ------
        AttributeError
            If :attr:`trajectory` does not expose an iterable of frames
            under a ``frames`` attribute. ``simulate()`` must be called
            before exporting.
        """
        frames = getattr(self.trajectory, "frames", None)
        if frames is None:
            raise AttributeError(
                "Trajectory has no 'frames' attribute to export; "
                "call simulate() first."
            )

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["time", "entity_id", "x", "y", "z",
                 "vx", "vy", "vz", "ax", "ay", "az"]
            )
            for frame in frames:
                writer.writerow([
                    frame.t,
                    frame.entity_id,
                    frame.position.x, frame.position.y, frame.position.z,
                    frame.velocity.x, frame.velocity.y, frame.velocity.z,
                    frame.acceleration.x, frame.acceleration.y, frame.acceleration.z,
                ])

        print(f"[StateEngine] exported CSV → {path}")


# ─────────────────────────────────────────────
# Self-test / sanity scenario
# ─────────────────────────────────────────────

def _self_test() -> None:
    """
    Minimal self-test reproducing the canonical 'car at constant velocity'
    scenario described in the StateEngine specification.

    A 1200 kg car moving at 16.67 m/s with zero net horizontal force
    (i.e. a frictionless / drag-free PhysicsEncoder stub) should reach
    ``x ≈ 166.7 m`` after 10 seconds.

    This function is intentionally self-contained: it builds a minimal
    WorldSpec and monkey-patches a trivial PhysicsEncoder so that it can
    run without a full scene description, and is meant for quick manual
    verification (``python -m models.state_engine``) rather than as part
    of the automated test suite.
    """
    from models.world_spec import (
        WorldSpec, Entity, PhysicsState, Environment,
        SimulationGraph, BoundingBox, Vec3,
    )

    class _ZeroForceEncoder(PhysicsEncoder):
        """PhysicsEncoder stub returning zero net force on every entity."""

        def __init__(self, spec: WorldSpec) -> None:
            self.spec = spec

        def net_forces(self, states, t):
            return {eid: Vec3(0.0, 0.0, 0.0) for eid in states}

    car = Entity(
        id="e_car",
        label="car",
        entity_type="vehicle",
        is_static=False,
        mass=1200.0,
        bounding_box=BoundingBox(4.5, 1.5, 1.8),
        state=PhysicsState(
            position=Vec3(0.0, 0.0, 0.0),
            velocity=Vec3(16.67, 0.0, 0.0),
        ),
    )

    spec = WorldSpec(
        scene_id="selftest_car",
        description="A car moving at constant velocity on a flat road.",
        entities=[car],
        environment=Environment(gravity=Vec3(0, -9.81, 0)),
        simulation_graph=SimulationGraph(
            dt=0.01, duration=10.0, integrator="rk4", export_fps=30,
        ),
    )

    engine = StateEngine(spec)
    engine.encoder = _ZeroForceEncoder(spec)  # override with zero-force stub

    engine.simulate()
    final = engine.states["e_car"]
    print(f"[SelfTest] x(10s) = {final.x:.4f} m (expected ≈ 166.7 m)")
    assert abs(final.x - 166.7) < 1e-3, "RK4 constant-velocity integration mismatch"
    print("[SelfTest] PASSED")


if __name__ == "__main__":
    _self_test()
