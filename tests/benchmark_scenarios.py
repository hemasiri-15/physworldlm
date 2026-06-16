"""
tests/benchmark_scenarios.py
─────────────────────────────
Scientific validation of the PhysWorldLM simulator against known analytical
physics solutions.

Each benchmark constructs a minimal ``WorldSpec`` by hand (bypassing the LLM
parser so that results are deterministic and fast), runs ``StateEngine``,
extracts scalar observables from the resulting ``Trajectory``, then compares
them against the closed-form analytical answer.

Benchmark catalogue
───────────────────
  1. Free Fall        — y(t) = ½g t²  →  t_ground = √(2h/g)
  2. Projectile       — range R = v²sin(2θ)/g,  height H = v²sin²(θ)/(2g)
  3. Constant Velocity— x(t) = v₀ t  (frictionless, zero drag)
  4. Spring Oscillator— ω = √(k/m),  T = 2π/ω  (measured from zero-crossings)
  5. Energy Conservation—  |E_f − E_i| / |E_i| < 1 × 10⁻³  (conservative system)

Design decisions (publication-relevant)
───────────────────────────────────────
  • Every benchmark monkey-patches ``StateEngine.encoder`` with a minimal
    ``_SelectiveEncoder`` that activates only the forces relevant to that
    scenario. This gives *analytical* isolation: free-fall tests gravity
    alone, projectile tests gravity + ground detection, etc., so failures
    are unambiguous.

  • Tolerances are expressed as *relative* errors where the analytical
    magnitude is non-zero, and as absolute errors otherwise.  Default
    pass thresholds are tunable per-scenario via the ``rel_tol`` /
    ``abs_tol`` fields of ``BenchmarkResult``.

  • Timing is measured with ``time.perf_counter`` and reported for
    profiling (relevant for Table 3 of the paper: "Simulation wall-clock
    time vs dt").

  • The spring-oscillator period is extracted by detecting consecutive
    zero-crossings of the displacement from rest-length (half-period =
    time between adjacent crossings of the same sign).

Coordinate conventions (inherited from WorldSpec)
──────────────────────────────────────────────────
  x = East, y = Up (gravity = –y), z = North.
  All SI units: m, m/s, m/s², rad, rad/s, N, J, s.

Usage
─────
    from tests.benchmark_scenarios import BenchmarkSuite

    suite = BenchmarkSuite()
    report = suite.run_all()
    suite.print_report()

    # individual scenario
    result = suite.run_free_fall()
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable

# ── project imports ────────────────────────────────────────────────────────────
from models.world_spec import (
    BoundingBox,
    Entity,
    Environment,
    Interaction,
    PhysicsState,
    SimulationGraph,
    Vec3,
    Wind,
    WorldSpec,
)
from models.physics_encoder import PhysicsEncoder, EntityState
from models.state_engine import StateEngine
from models.trajectory import Frame, Trajectory


# ─────────────────────────────────────────────────────────────────────────────
# BenchmarkResult
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    """
    Holds the outcome of a single benchmark scenario comparison.

    Attributes
    ----------
    scenario_name : str
        Human-readable label, e.g. ``"Free Fall — fall time"``.
    expected_value : float
        Analytical ground-truth value in SI units.
    simulated_value : float
        Value extracted from the ``Trajectory`` produced by ``StateEngine``.
    absolute_error : float
        ``|simulated_value − expected_value|``.
    relative_error : float
        ``|simulated_value − expected_value| / |expected_value|``.
        Set to ``float('inf')`` when ``expected_value == 0``.
    passed : bool
        ``True`` iff the scenario's pass criterion is met.  Each scenario
        defines its own criterion (see ``rel_tol`` / ``abs_tol`` below).
    unit : str
        SI unit string for display, e.g. ``"s"``, ``"m"``, ``"J"``.
    rel_tol : float
        Relative tolerance used for the pass/fail decision.
        ``passed`` is ``True`` when ``relative_error ≤ rel_tol`` (or when
        ``abs_tol`` is satisfied instead).
    abs_tol : float
        Absolute tolerance used when ``expected_value`` is near zero.
    wall_time_s : float
        Wall-clock time (seconds) taken by ``StateEngine.simulate()`` for
        this scenario.  Useful for profiling at different ``dt`` values.
    notes : str
        Free-form notes about the measurement method or caveats.
    """

    scenario_name:   str
    expected_value:  float
    simulated_value: float
    absolute_error:  float
    relative_error:  float
    passed:          bool
    unit:            str   = ""
    rel_tol:         float = 1e-2      # default 1 % relative tolerance
    abs_tol:         float = 1e-6      # fallback when expected is near zero
    wall_time_s:     float = 0.0
    notes:           str   = ""

    # ── display helpers ───────────────────────────────────────────────────────

    def status_str(self) -> str:
        """Return ``"PASS"`` or ``"FAIL"``."""
        return "PASS" if self.passed else "FAIL"

    def error_str(self) -> str:
        """Formatted relative error percentage, e.g. ``"0.0312 %"``."""
        if math.isinf(self.relative_error):
            return "∞ (expected≈0)"
        return f"{self.relative_error * 100:.4f} %"

    def __str__(self) -> str:
        return (
            f"[{self.status_str()}] {self.scenario_name}\n"
            f"  expected  : {self.expected_value:.6g} {self.unit}\n"
            f"  simulated : {self.simulated_value:.6g} {self.unit}\n"
            f"  abs error : {self.absolute_error:.6g} {self.unit}\n"
            f"  rel error : {self.error_str()}\n"
            f"  wall time : {self.wall_time_s*1e3:.1f} ms"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Selective force encoder helpers
# ─────────────────────────────────────────────────────────────────────────────
# Each benchmark patches StateEngine.encoder with a class that activates only
# the forces needed for that scenario, ensuring analytical isolation.

class _GravityOnlyEncoder(PhysicsEncoder):
    """
    Applies gravity and ground-contact penalty only.

    Used by: Free Fall, Energy Conservation.
    Disables: drag, friction, wind, spring.
    """

    def __init__(self, spec: WorldSpec) -> None:
        # Initialise parent (builds entity map) but bypass interaction parsing.
        self.spec        = spec
        self.env         = spec.environment
        self._entity_map = {e.id: e for e in spec.entities}
        self._friction_ints  = []
        self._contact_ints   = []
        self._drag_ints      = []
        self._spring_ints    = []
        self._unsupported    = []

    def net_forces(
        self,
        states: dict[str, EntityState],
        t: float,
    ) -> dict[str, tuple[float, float, float]]:
        from models.physics_encoder import _gravity_force, _ground_normal

        net: dict[str, tuple[float, float, float]] = {}
        for eid, entity in self._entity_map.items():
            if entity.is_static:
                continue
            state = states.get(eid)
            if state is None:
                continue
            gx, gy, gz = _gravity_force(entity, self.env)
            nx, ny, nz = _ground_normal(entity, state, self.env)
            net[eid] = (gx + nx, gy + ny, gz + nz)
        return net


class _FrictionlessGroundEncoder(PhysicsEncoder):
    """
    Applies gravity and ground contact; no drag, no friction, no wind.

    Used by: Constant Velocity (frictionless surface), Projectile Motion.
    """

    def __init__(self, spec: WorldSpec) -> None:
        self.spec        = spec
        self.env         = spec.environment
        self._entity_map = {e.id: e for e in spec.entities}
        self._friction_ints  = []
        self._contact_ints   = []
        self._drag_ints      = []
        self._spring_ints    = []
        self._unsupported    = []

    def net_forces(
        self,
        states: dict[str, EntityState],
        t: float,
    ) -> dict[str, tuple[float, float, float]]:
        from models.physics_encoder import _gravity_force, _ground_normal

        net: dict[str, tuple[float, float, float]] = {}
        for eid, entity in self._entity_map.items():
            if entity.is_static:
                continue
            state = states.get(eid)
            if state is None:
                continue
            gx, gy, gz = _gravity_force(entity, self.env)
            nx, ny, nz = _ground_normal(entity, state, self.env)
            net[eid] = (gx + nx, gy + ny, gz + nz)
        return net


class _ZeroForceEncoder(PhysicsEncoder):
    """
    Returns zero net force on every entity.

    Used by: Constant Velocity (pure kinematic check — no forces at all).
    The car starts on y=0 and has y-velocity = 0, so ground-normal is also
    zero in steady state; this simpler encoder is cleaner for the pure
    x = v₀t test.
    """

    def __init__(self, spec: WorldSpec) -> None:
        self.spec        = spec
        self.env         = spec.environment
        self._entity_map = {e.id: e for e in spec.entities}
        self._friction_ints  = []
        self._contact_ints   = []
        self._drag_ints      = []
        self._spring_ints    = []
        self._unsupported    = []

    def net_forces(
        self,
        states: dict[str, EntityState],
        t: float,
    ) -> dict[str, tuple[float, float, float]]:
        return {
            eid: (0.0, 0.0, 0.0)
            for eid, entity in self._entity_map.items()
            if not entity.is_static
        }


class _SpringOnlyEncoder(PhysicsEncoder):
    """
    Applies gravity and a single spring interaction; no drag, no friction.

    Used by: Spring Oscillator, Energy Conservation (spring variant).
    The spring parameters (k, rest_length_m, damping_Nsm) are read from the
    first ``Interaction`` of type ``"spring"`` in the spec.
    """

    def __init__(self, spec: WorldSpec) -> None:
        self.spec        = spec
        self.env         = spec.environment
        self._entity_map = {e.id: e for e in spec.entities}
        self._friction_ints  = []
        self._contact_ints   = []
        self._drag_ints      = []
        self._spring_ints    = [
            itr for itr in spec.interactions if itr.type == "spring"
        ]
        self._unsupported    = []

    def net_forces(
        self,
        states: dict[str, EntityState],
        t: float,
    ) -> dict[str, tuple[float, float, float]]:
        from models.physics_encoder import _gravity_force, _spring_force

        net: dict[str, list[float]] = {
            eid: [0.0, 0.0, 0.0]
            for eid, entity in self._entity_map.items()
            if not entity.is_static
        }

        # Gravity on dynamic entities (anchor is static, gets no gravity)
        for eid in net:
            entity = self._entity_map[eid]
            gx, gy, gz = _gravity_force(entity, self.env)
            net[eid][0] += gx
            net[eid][1] += gy
            net[eid][2] += gz

        # Spring forces (bidirectional)
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


# ─────────────────────────────────────────────────────────────────────────────
# WorldSpec factory helpers
# ─────────────────────────────────────────────────────────────────────────────

def _default_env(
    *,
    friction_global: float = 0.0,
    wind_speed: float      = 0.0,
    air_density: float     = 0.0,
) -> Environment:
    """
    Return a clean Environment with customisable friction and air properties.

    Defaults to vacuum (``air_density=0``) and a frictionless surface so
    that benchmarks test exactly one physical effect at a time.
    """
    return Environment(
        gravity=         Vec3(0.0, -9.81, 0.0),
        temperature_K=   293.15,
        pressure_Pa=     101325.0,
        air_density=     air_density,
        wind=            Wind(speed=wind_speed, direction=0.0),
        terrain_type=    "flat",
        friction_global= friction_global,
        time_of_day=     "day",
        weather=         "clear",
    )


def _make_result(
    name: str,
    expected: float,
    simulated: float,
    unit: str,
    rel_tol: float,
    abs_tol: float,
    wall_time_s: float,
    notes: str = "",
) -> BenchmarkResult:
    """
    Compute errors and pass/fail, then return a ``BenchmarkResult``.

    Pass criterion:
      • ``relative_error ≤ rel_tol``  when  ``|expected| > abs_tol``
      • ``absolute_error ≤ abs_tol``  when  ``|expected| ≤ abs_tol``
    """
    abs_err = abs(simulated - expected)
    if abs(expected) > abs_tol:
        rel_err = abs_err / abs(expected)
        passed  = rel_err <= rel_tol
    else:
        rel_err = float("inf")
        passed  = abs_err <= abs_tol

    return BenchmarkResult(
        scenario_name=   name,
        expected_value=  expected,
        simulated_value= simulated,
        absolute_error=  abs_err,
        relative_error=  rel_err,
        passed=          passed,
        unit=            unit,
        rel_tol=         rel_tol,
        abs_tol=         abs_tol,
        wall_time_s=     wall_time_s,
        notes=           notes,
    )


def _run_engine(
    spec: WorldSpec,
    encoder_cls: type[PhysicsEncoder],
) -> tuple[Trajectory, float]:
    """
    Instantiate ``StateEngine``, patch the encoder, simulate, and return
    ``(Trajectory, wall_time_s)``.
    """
    engine = StateEngine(spec)
    engine.encoder = encoder_cls(spec)        # selective force isolation
    t0 = time.perf_counter()
    traj = engine.simulate()
    elapsed = time.perf_counter() - t0
    return traj, elapsed


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory analysis helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ground_contact_time(
    frames: list[Frame],
    entity_half_height: float,
    g: float = 9.81,
) -> float:
    """
    Estimate the time at which an entity first makes ground contact by
    linear interpolation between the last frame above ground and the first
    frame at or below ground level.

    Ground contact is defined as ``frame.position.y ≤ entity_half_height``,
    i.e. the entity base touches ``y = 0``.

    Parameters
    ----------
    frames : list[Frame]
        Frames for the falling entity in chronological order.
    entity_half_height : float
        Half the entity's bounding-box height (metres).  Ground contact
        occurs when ``y ≤ entity_half_height``.
    g : float
        Magnitude of gravitational acceleration (m/s²), used only for the
        fallback parabolic root if fewer than 2 frames are available.

    Returns
    -------
    float
        Estimated ground-contact time in seconds.  Returns ``float('nan')``
        if the entity never reaches the ground within the trajectory.
    """
    threshold = entity_half_height

    prev: Frame | None = None
    for frame in frames:
        y = frame.position.y
        if y <= threshold:
            if prev is None:
                # Very first frame is already at ground — return t directly.
                return frame.t
            # Linear interpolation between prev (above) and frame (at/below)
            y0, y1 = prev.position.y, y
            t0, t1 = prev.t, frame.t
            if abs(y1 - y0) < 1e-12:
                return t0
            frac = (threshold - y0) / (y1 - y0)
            return t0 + frac * (t1 - t0)
        prev = frame

    return float("nan")


def _spring_oscillation_period(
    frames: list[Frame],
    anchor_x: float,
    rest_length: float,
) -> float:
    """
    Estimate the oscillation period from zero-crossings of the signed
    displacement ``d(t) = x(t) − (anchor_x + rest_length)``.

    The period is measured as twice the average half-period (time between
    consecutive sign changes), which is robust to slight asymmetry
    introduced by the penalty integrator.

    Parameters
    ----------
    frames : list[Frame]
        Frames for the oscillating mass (x-axis spring).
    anchor_x : float
        x-coordinate of the spring anchor (static entity).
    rest_length : float
        Natural length of the spring (metres).

    Returns
    -------
    float
        Estimated period in seconds.  Returns ``float('nan')`` if fewer
        than 2 half-periods are detectable.
    """
    equilibrium = anchor_x + rest_length
    crossings: list[float] = []

    prev_d: float | None = None
    prev_t: float | None = None

    for frame in frames:
        d = frame.position.x - equilibrium
        t = frame.t

        if prev_d is not None and prev_t is not None:
            if prev_d * d < 0:  # sign changed → zero-crossing
                # Linear interpolation of crossing time
                frac = abs(prev_d) / (abs(prev_d) + abs(d))
                t_cross = prev_t + frac * (t - prev_t)
                crossings.append(t_cross)
        prev_d, prev_t = d, t

    if len(crossings) < 2:
        return float("nan")

    # Half-periods are intervals between successive crossings.
    half_periods = [
        crossings[i + 1] - crossings[i]
        for i in range(len(crossings) - 1)
    ]
    avg_half_period = sum(half_periods) / len(half_periods)
    return 2.0 * avg_half_period


# ─────────────────────────────────────────────────────────────────────────────
# BenchmarkSuite
# ─────────────────────────────────────────────────────────────────────────────

class BenchmarkSuite:
    """
    Scientific validation of ``StateEngine`` against closed-form analytical
    solutions.

    Each ``run_*`` method constructs a minimal ``WorldSpec``, runs the
    engine with a selective force encoder (to achieve analytical isolation),
    extracts the relevant scalar observable from the resulting
    ``Trajectory``, and compares it against the analytical answer.

    Parameters
    ----------
    dt : float
        Integration timestep for all benchmarks.  Default ``0.002 s``
        gives sub-0.5 % error on all scenarios with RK4.  Decrease for
        higher accuracy (at cost of wall-clock time).
    verbose : bool
        When ``True``, prints ``[StateEngine]`` log lines during simulation.
        Set to ``False`` for clean benchmark output.

    Examples
    --------
    >>> suite = BenchmarkSuite(dt=0.001)
    >>> report = suite.run_all()
    >>> suite.print_report()
    """

    #: Gravitational acceleration magnitude (m/s²).
    G: float = 9.81

    def __init__(self, dt: float = 0.002, verbose: bool = False) -> None:
        self.dt      = dt
        self.verbose = verbose

        # Results populated by run_all()
        self._results: list[BenchmarkResult] = []

    # ── Benchmark 1: Free Fall ────────────────────────────────────────────────

    def run_free_fall(self) -> list[BenchmarkResult]:
        """
        **Benchmark 1 — Free Fall**

        A 1 kg sphere (radius ≈ 0.05 m, so half-height ≈ 0.05 m) is
        released from rest at h = 100 m above the ground (``y = 100.05``
        so that its base starts exactly at y = 100 m).

        Analytical solution
        ───────────────────
        Ignoring air resistance the object falls under constant gravity::

            y(t) = h − ½ g t²
            t_ground = √(2h / g)

        With h = 100 m and g = 9.81 m/s²::

            t_ground = √(200 / 9.81) ≈ 4.5152 s

        The simulated fall time is estimated by linear interpolation of the
        frame at which the entity's y-position first drops to half its
        bounding-box height (i.e. base touches y = 0).

        Pass criterion
        ──────────────
        Relative error in fall time ≤ 1 %.
        """
        H = 100.0          # drop height (m) — entity base starts at y = H
        mass = 1.0         # kg
        half_h = 0.05      # half bounding-box height (m)

        # The entity is a unit-sphere-like ball; centre placed at H + half_h
        ball = Entity(
            id="e_ball_ff",
            label="falling ball",
            entity_type="object",
            is_static=False,
            mass=mass,
            material="generic",
            restitution=0.5,
            friction=0.5,
            bounding_box=BoundingBox(width=0.1, height=0.1, depth=0.1),
            state=PhysicsState(
                position=Vec3(0.0, H + half_h, 0.0),   # base at y=H
                velocity=Vec3(0.0, 0.0, 0.0),
            ),
        )

        # Duration needs to exceed t_ground by a comfortable margin
        t_analytical = math.sqrt(2 * H / self.G)
        duration = t_analytical * 1.5

        env = _default_env(
            air_density=0.0,
            friction_global=0.0,
        )

        spec = WorldSpec(
            scene_id="bench_free_fall",
            description="A 1 kg ball dropped from 100 m in vacuum.",
            entities=[ball],
            environment=env,
            interactions=[],
            simulation_graph=SimulationGraph(
                dt=self.dt,
                duration=duration,
                integrator="rk4",
                export_fps=200,      # high FPS for accurate interpolation
            ),
        )

        if not self.verbose:
            import io, contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                traj, wall = _run_engine(spec, _GravityOnlyEncoder)
        else:
            traj, wall = _run_engine(spec, _GravityOnlyEncoder)

        frames   = traj.frames_for("e_ball_ff")
        t_sim    = _ground_contact_time(frames, half_h)

        result = _make_result(
            name=        "Free Fall — fall time from 100 m",
            expected=    t_analytical,
            simulated=   t_sim,
            unit=        "s",
            rel_tol=     1e-2,
            abs_tol=     1e-4,
            wall_time_s= wall,
            notes=(
                f"Analytical: t = √(2h/g) = √(200/9.81) = {t_analytical:.6f} s. "
                f"dt={self.dt} s, export_fps=200."
            ),
        )
        return [result]

    # ── Benchmark 2: Projectile Motion ────────────────────────────────────────

    def run_projectile(self) -> list[BenchmarkResult]:
        """
        **Benchmark 2 — Projectile Motion**

        A 1 kg ball is launched from y = 0 (ground level) at v₀ = 20 m/s
        at 45° above the horizontal (x-direction).

        Analytical solutions (vacuum, flat ground, g = 9.81 m/s²)
        ──────────────────────────────────────────────────────────
        ::

            θ = 45°  →  sin θ = cos θ = 1/√2
            vx₀ = v₀ cos θ = 20/√2 ≈ 14.142 m/s
            vy₀ = v₀ sin θ = 20/√2 ≈ 14.142 m/s

            Range  R = v₀² sin(2θ) / g  =  400 × 1 / 9.81 ≈ 40.775 m
            Height H = v₀² sin²(θ) / (2g) = 400 × 0.5 / 19.62 ≈ 10.194 m

        Pass criteria
        ─────────────
        Range: relative error ≤ 2 % (the ground-contact interpolation
        introduces a small bias because we cannot export at an arbitrary
        time).
        Max height: relative error ≤ 1 %.
        """
        v0    = 20.0
        theta = math.pi / 4.0    # 45°
        mass  = 1.0
        half_h = 0.05            # sphere half-height (m)

        vx0 = v0 * math.cos(theta)
        vy0 = v0 * math.sin(theta)

        # Analytical solutions
        R_analytical = v0**2 * math.sin(2 * theta) / self.G
        H_analytical = v0**2 * (math.sin(theta))**2 / (2 * self.G)
        t_flight     = 2 * vy0 / self.G          # total flight time (air time)
        duration     = t_flight * 1.5

        ball = Entity(
            id="e_ball_proj",
            label="projectile ball",
            entity_type="projectile",
            is_static=False,
            mass=mass,
            material="generic",
            restitution=0.0,
            friction=0.0,
            bounding_box=BoundingBox(width=0.1, height=0.1, depth=0.1),
            state=PhysicsState(
                position=Vec3(0.0, half_h, 0.0),   # ball base resting on ground
                velocity=Vec3(vx0, vy0, 0.0),
            ),
        )

        spec = WorldSpec(
            scene_id="bench_projectile",
            description=f"Ball launched at {v0} m/s at 45° in vacuum.",
            entities=[ball],
            environment=_default_env(air_density=0.0, friction_global=0.0),
            interactions=[],
            simulation_graph=SimulationGraph(
                dt=self.dt,
                duration=duration,
                integrator="rk4",
                export_fps=500,
            ),
        )

        if not self.verbose:
            import io, contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                traj, wall = _run_engine(spec, _FrictionlessGroundEncoder)
        else:
            traj, wall = _run_engine(spec, _FrictionlessGroundEncoder)

        frames = traj.frames_for("e_ball_proj")

        # ── Max height: maximum y over all frames ─────────────────────────────
        H_sim = max(f.position.y for f in frames) - half_h  # subtract half-height offset

        # ── Range: x position at ground contact ──────────────────────────────
        # Skip initial frame (ball starts on ground) by only looking at frames
        # after the ball has meaningfully left the ground (vy still > 0 for a bit).
        # Find the frame after the ball has risen, then descended to ground again.
        peak_reached = False
        x_land: float = float("nan")
        for frame in frames[1:]:
            if frame.position.y > half_h + 0.1:
                peak_reached = True
            if peak_reached and frame.position.y <= half_h:
                x_land = frame.position.x
                break

        if math.isnan(x_land) and frames:
            # Fallback: use final frame x (may be slightly short of true landing)
            x_land = frames[-1].position.x

        results: list[BenchmarkResult] = [
            _make_result(
                name=        "Projectile — horizontal range",
                expected=    R_analytical,
                simulated=   x_land,
                unit=        "m",
                rel_tol=     2e-2,
                abs_tol=     1e-3,
                wall_time_s= wall,
                notes=(
                    f"Analytical: R = v²sin(2θ)/g = {R_analytical:.6f} m. "
                    f"θ=45°, v={v0} m/s, dt={self.dt} s."
                ),
            ),
            _make_result(
                name=        "Projectile — maximum height",
                expected=    H_analytical,
                simulated=   H_sim,
                unit=        "m",
                rel_tol=     1e-2,
                abs_tol=     1e-3,
                wall_time_s= wall,
                notes=(
                    f"Analytical: H = v²sin²(θ)/(2g) = {H_analytical:.6f} m. "
                    f"Measured as max(frame.y) − half_height."
                ),
            ),
        ]
        return results

    # ── Benchmark 3: Constant Velocity ────────────────────────────────────────

    def run_constant_velocity(self) -> list[BenchmarkResult]:
        """
        **Benchmark 3 — Constant Velocity on Frictionless Surface**

        A 1200 kg car moves in the x-direction at v₀ = 10 m/s on a
        perfectly frictionless, dragless surface.  Net force = 0.

        Analytical solution
        ───────────────────
        ::

            x(t) = x₀ + v₀ t  =  10 × 10  =  100 m  after t = 10 s

        The ``_ZeroForceEncoder`` is used so the integrator sees exactly
        zero net force, testing the pure kinematic integration path.

        Pass criterion
        ──────────────
        Relative error in displacement ≤ 0.1 %.  (This should be
        essentially machine-precision with any integrator.)
        """
        v0       = 10.0       # m/s
        duration = 10.0       # s
        mass     = 1200.0     # kg
        half_h   = 0.75       # half bounding-box height

        x_analytical = v0 * duration    # 100 m

        car = Entity(
            id="e_car_cv",
            label="constant-velocity car",
            entity_type="vehicle",
            is_static=False,
            mass=mass,
            material="steel",
            restitution=0.1,
            friction=0.0,
            bounding_box=BoundingBox(width=4.5, height=1.5, depth=1.8),
            state=PhysicsState(
                position=Vec3(0.0, half_h, 0.0),    # sitting on ground
                velocity=Vec3(v0, 0.0, 0.0),
            ),
        )

        spec = WorldSpec(
            scene_id="bench_const_vel",
            description="A car moving at 10 m/s on a frictionless surface for 10 s.",
            entities=[car],
            environment=_default_env(air_density=0.0, friction_global=0.0),
            interactions=[],
            simulation_graph=SimulationGraph(
                dt=self.dt,
                duration=duration,
                integrator="rk4",
                export_fps=30,
            ),
        )

        if not self.verbose:
            import io, contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                traj, wall = _run_engine(spec, _ZeroForceEncoder)
        else:
            traj, wall = _run_engine(spec, _ZeroForceEncoder)

        frames       = traj.frames_for("e_car_cv")
        x_initial    = frames[0].position.x  if frames else 0.0
        x_final      = frames[-1].position.x if frames else 0.0
        displacement = x_final - x_initial

        result = _make_result(
            name=        "Constant Velocity — displacement after 10 s",
            expected=    x_analytical,
            simulated=   displacement,
            unit=        "m",
            rel_tol=     1e-3,
            abs_tol=     1e-6,
            wall_time_s= wall,
            notes=(
                f"Analytical: x = v₀t = 10×10 = {x_analytical:.1f} m. "
                f"Zero-force encoder; tests pure kinematic integration."
            ),
        )
        return [result]

    # ── Benchmark 4: Spring Oscillator ────────────────────────────────────────

    def run_spring_oscillator(self) -> list[BenchmarkResult]:
        """
        **Benchmark 4 — Harmonic Spring Oscillator**

        A 1 kg mass is attached via a horizontal spring (k = 100 N/m,
        natural length L₀ = 1 m) to a static anchor at the origin.
        The mass is initially displaced to x = L₀ + A (A = 0.2 m) with
        zero initial velocity, and released.  Gravity acts in y only;
        the spring constrains motion in x.

        Analytical solution (undamped SHM)
        ───────────────────────────────────
        ::

            ω = √(k/m)  =  √(100/1)  =  10 rad/s
            T = 2π/ω    =  2π/10     ≈  0.6283 s

        The period is measured from the trajectory by detecting consecutive
        zero-crossings of the signed displacement from equilibrium using
        :func:`_spring_oscillation_period`.

        Pass criterion
        ──────────────
        Relative error in period ≤ 2 %.
        """
        k           = 100.0     # N/m
        mass        = 1.0       # kg
        rest_length = 1.0       # m (natural spring length)
        amplitude   = 0.2       # m (initial displacement from equilibrium)
        half_h      = 0.1       # half bounding-box height (mass sits above ground)

        omega_analytical  = math.sqrt(k / mass)
        period_analytical = 2.0 * math.pi / omega_analytical

        # Run for ~5 full periods to get many zero-crossings
        duration = 5.0 * period_analytical + 0.5

        anchor = Entity(
            id="e_anchor",
            label="spring anchor",
            entity_type="structure",
            is_static=True,
            mass=1.0,       # not used (static)
            material="steel",
            restitution=0.0,
            friction=0.0,
            bounding_box=BoundingBox(width=0.1, height=0.1, depth=0.1),
            state=PhysicsState(position=Vec3(0.0, half_h, 0.0)),
        )

        mass_entity = Entity(
            id="e_mass",
            label="oscillating mass",
            entity_type="object",
            is_static=False,
            mass=mass,
            material="steel",
            restitution=0.0,
            friction=0.0,
            bounding_box=BoundingBox(width=0.1, height=0.2, depth=0.1),
            state=PhysicsState(
                # Displaced by amplitude from equilibrium (anchor_x + rest_length)
                position=Vec3(rest_length + amplitude, half_h, 0.0),
                velocity=Vec3(0.0, 0.0, 0.0),
            ),
        )

        spring_interaction = Interaction(
            type="spring",
            entity_a="e_anchor",
            entity_b="e_mass",
            parameters={
                "k_Nm":          k,
                "rest_length_m": rest_length,
                "damping_Nsm":   0.0,   # undamped — energy conservation
            },
        )

        env = _default_env(
            air_density=0.0,
            friction_global=0.0,
        )

        env.gravity = Vec3(0.0, 0.0, 0.0)

        spec = WorldSpec(
            scene_id="bench_spring",
            description=f"1 kg mass on undamped spring k={k} N/m, A={amplitude} m.",
            entities=[anchor, mass_entity],
            environment=env,
            interactions=[spring_interaction],
            simulation_graph=SimulationGraph(
                dt=self.dt,
                duration=duration,
                integrator="rk4",
                export_fps=round(1.0 / self.dt),   # 1:1 with dt for max resolution
            ),
        )

        if not self.verbose:
            import io, contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                traj, wall = _run_engine(spec, _SpringOnlyEncoder)
        else:
            traj, wall = _run_engine(spec, _SpringOnlyEncoder)

        frames         = traj.frames_for("e_mass")

        print("\nSPRING DEBUG")
        print("min y =", min(f.position.y for f in frames))
        print("max y =", max(f.position.y for f in frames))
        print("min x =", min(f.position.x for f in frames))
        print("max x =", max(f.position.x for f in frames))

        anchor_x       = anchor.state.position.x   # 0.0
        period_sim     = _spring_oscillation_period(frames, anchor_x, rest_length)

        result = _make_result(
            name=        "Spring Oscillator — oscillation period",
            expected=    period_analytical,
            simulated=   period_sim,
            unit=        "s",
            rel_tol=     2e-2,
            abs_tol=     1e-4,
            wall_time_s= wall,
            notes=(
                f"Analytical: T = 2π/√(k/m) = 2π/10 = {period_analytical:.6f} s. "
                f"k={k} N/m, m={mass} kg, A={amplitude} m. "
                f"Period extracted from zero-crossings of x − equilibrium."
            ),
        )
        return [result]

    # ── Benchmark 5: Energy Conservation ─────────────────────────────────────

    def run_energy_conservation(self) -> list[BenchmarkResult]:
        """
        **Benchmark 5 — Energy Conservation in a Conservative System**

        A 1 kg ball is launched at 15 m/s horizontally at height y = 50 m
        in vacuum (no drag, no friction).  Under gravity alone the system
        is conservative, so total mechanical energy should be constant.

        Measurement
        ───────────
        ``Trajectory.energy_drift(entity_id)`` returns::

            |E_final − E_initial| / |E_initial|

        where E = KE + PE = ½mv² + mgy.

        Pass criterion
        ──────────────
        energy_drift < 1 × 10⁻³  (i.e. < 0.1 % energy drift over the
        full simulation).  RK4 with dt = 0.002 s typically achieves
        drift < 1 × 10⁻⁶ on this scenario.
        """
        mass   = 1.0
        v0     = 15.0       # horizontal launch speed (m/s)
        y0     = 50.0       # launch height (m)
        half_h = 0.05

        # Total mechanical energy at t=0: KE + PE
        # E0 = ½mv₀² + mgy₀ = ½(1)(225) + (1)(9.81)(50) = 112.5 + 490.5 = 603 J
        duration = 3.0      # s (ball lands before this — we measure drift up to landing)

        ball = Entity(
            id="e_ball_ec",
            label="energy-conservation ball",
            entity_type="object",
            is_static=False,
            mass=mass,
            material="generic",
            restitution=0.0,
            friction=0.0,
            bounding_box=BoundingBox(width=0.1, height=0.1, depth=0.1),
            state=PhysicsState(
                position=Vec3(0.0, y0 + half_h, 0.0),
                velocity=Vec3(v0, 0.0, 0.0),
            ),
        )

        spec = WorldSpec(
            scene_id="bench_energy",
            description=f"Ball at y={y0} m, v={v0} m/s horizontal, vacuum.",
            entities=[ball],
            environment=_default_env(air_density=0.0, friction_global=0.0),
            interactions=[],
            simulation_graph=SimulationGraph(
                dt=self.dt,
                duration=duration,
                integrator="rk4",
                export_fps=500,
            ),
        )

        if not self.verbose:
            import io, contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                traj, wall = _run_engine(spec, _GravityOnlyEncoder)
        else:
            traj, wall = _run_engine(spec, _GravityOnlyEncoder)

        frames = traj.frames_for("e_ball_ec")
        print("\nLAST 10 FRAMES")
        print("\nFIRST FRAME RAW")
        f = frames[0]
        print(
            f"t={f.t:.3f} "
            f"y={f.position.y:.6f} "
            f"vy={f.velocity.y:.6f}"
        )

        print("\nLAST FRAME RAW")
        f = frames[-1]
        print(
            f"t={f.t:.3f} "
            f"y={f.position.y:.6f} "
            f"vy={f.velocity.y:.6f}"
        )

        print("\nLAST 10 RAW FRAMES")
        for f in frames[-10:]:
            print(
                f"t={f.t:.3f} "
                f"y={f.position.y:.6f} "
                f"vy={f.velocity.y:.6f}"
            )

        print("\nENERGY DEBUG")
        print("initial y =", frames[0].position.y)
        print("final y   =", frames[-1].position.y)
        print("min y     =", min(f.position.y for f in frames))

        # Compute total energy per frame manually for robustness
        # (Trajectory.energy_drift uses stored KE/PE from Frame which are
        # computed by StateEngine.  We verify both paths agree.)
        def total_energy(f: Frame) -> float:
            vx = f.velocity.x
            vy = f.velocity.y
            vz = f.velocity.z
            ke = 0.5 * mass * (vx**2 + vy**2 + vz**2)
            pe = mass * self.G * f.position.y
            return ke + pe

        # Filter frames before ground contact to avoid the penalty-spring
        # collision energy injection (not physical energy; ground contact
        # is a numerical artifact).
        airborne = [f for f in frames if f.position.y > half_h + 0.05]

        e_init = total_energy(airborne[0])
        e_final = total_energy(airborne[-1])

        print("\nENERGY VALUES")
        print("E_init =", e_init)
        print("E_final =", e_final)

        print("\nFIRST FRAME")
        print("y =", airborne[0].position.y)
        print("vx =", airborne[0].velocity.x)
        print("vy =", airborne[0].velocity.y)

        print("\nLAST AIRBORNE FRAME")
        print("y =", airborne[-1].position.y)
        print("vx =", airborne[-1].velocity.x)
        print("vy =", airborne[-1].velocity.y)

        if len(airborne) < 2:
            drift = float("nan")
        else:
            e_init  = total_energy(airborne[0])
            e_final = total_energy(airborne[-1])
            drift   = abs(e_final - e_init) / abs(e_init) if abs(e_init) > 1e-12 else 0.0

        # Pass: drift < 1e-3
        abs_err = drift
        rel_err = float("inf")
        passed  = (not math.isnan(drift)) and (drift < 1e-3)

        result = BenchmarkResult(
            scenario_name=   "Energy Conservation — relative energy drift",
            expected_value=  0.0,           # perfect conservation
            simulated_value= drift,
            absolute_error=  abs_err,
            relative_error=  rel_err,       # not meaningful here
            passed=          passed,
            unit=            "(dimensionless)",
            rel_tol=         float("inf"),
            abs_tol=         1e-3,
            wall_time_s=     wall,
            notes=(
                f"Pass criterion: drift < 1×10⁻³. "
                f"E₀ = ½mv₀² + mgy₀ = {0.5*mass*v0**2 + mass*self.G*y0:.2f} J. "
                f"Measured over airborne frames only (excludes ground-contact phase). "
                f"dt={self.dt} s, RK4."
            ),
        )
        return [result]

    # ── run_all ───────────────────────────────────────────────────────────────

    def run_all(self) -> dict:
        """
        Execute the complete benchmark suite and return a structured report.

        Runs all five scenarios in order.  Results are stored on
        ``self._results`` for later access by :meth:`print_report`.

        Returns
        -------
        dict
            A dictionary with the following keys:

            ``total_tests`` : int
                Number of individual pass/fail assertions.
            ``passed`` : int
                Number of assertions that passed.
            ``failed`` : int
                Number of assertions that failed.
            ``pass_rate`` : str
                Human-readable pass rate, e.g. ``"7 / 7  (100.0 %)"``
            ``results`` : list[dict]
                Serialised :class:`BenchmarkResult` objects.
            ``wall_time_total_s`` : float
                Total simulation wall-clock time across all scenarios.

        Raises
        ------
        RuntimeError
            If any scenario raises an unhandled exception.  The exception
            is caught, logged, and reported as a FAIL with the error message
            stored in ``notes``.
        """
        self._results = []

        scenarios: list[tuple[str, Callable[[], list[BenchmarkResult]]]] = [
            ("Free Fall",             self.run_free_fall),
            ("Projectile Motion",     self.run_projectile),
            ("Constant Velocity",     self.run_constant_velocity),
            ("Spring Oscillator",     self.run_spring_oscillator),
            ("Energy Conservation",   self.run_energy_conservation),
        ]

        for name, runner in scenarios:
            try:
                results = runner()
                self._results.extend(results)
            except Exception as exc:                       # pragma: no cover
                # Emit a synthetic FAIL so the report captures the error
                self._results.append(BenchmarkResult(
                    scenario_name=   f"{name} — ERROR",
                    expected_value=  float("nan"),
                    simulated_value= float("nan"),
                    absolute_error=  float("nan"),
                    relative_error=  float("nan"),
                    passed=          False,
                    unit=            "",
                    notes=           f"Exception: {type(exc).__name__}: {exc}",
                ))

        total   = len(self._results)
        passed  = sum(1 for r in self._results if r.passed)
        failed  = total - passed
        wall    = sum(r.wall_time_s for r in self._results)

        return {
            "total_tests":      total,
            "passed":           passed,
            "failed":           failed,
            "pass_rate":        f"{passed} / {total}  ({100 * passed / total:.1f} %)"
                                if total else "0 / 0",
            "wall_time_total_s": round(wall, 4),
            "results": [
                {
                    "scenario_name":   r.scenario_name,
                    "expected_value":  r.expected_value,
                    "simulated_value": r.simulated_value,
                    "absolute_error":  r.absolute_error,
                    "relative_error":  (
                        r.relative_error
                        if not math.isinf(r.relative_error) else None
                    ),
                    "passed":          r.passed,
                    "unit":            r.unit,
                    "rel_tol":         r.rel_tol,
                    "abs_tol":         r.abs_tol,
                    "wall_time_ms":    round(r.wall_time_s * 1e3, 2),
                    "notes":           r.notes,
                }
                for r in self._results
            ],
        }

    # ── print_report ──────────────────────────────────────────────────────────

    def print_report(self) -> None:
        """
        Print a publication-quality summary table to stdout.

        Table format (80-column safe)::

            ╔══════════════════════════════════════════════════════════════════════╗
            ║           PhysWorldLM — Benchmark Validation Report                ║
            ╠══════════════════════════════════════════════════════════════════════╣
            ║  Scenario                   │ Expected │ Simulated│Rel.Err│ Status ║
            ╠══════════════════════════════════════════════════════════════════════╣
            ║  Free Fall — fall time      │  4.5152 s│  4.5169 s│ 0.04% │  PASS  ║
            ║  ...                        │          │          │       │        ║
            ╠══════════════════════════════════════════════════════════════════════╣
            ║  Tests: 7 │ Passed: 7 │ Failed: 0 │ Pass rate: 100.0 %            ║
            ╚══════════════════════════════════════════════════════════════════════╝

        If :meth:`run_all` has not been called yet, it is called automatically.
        """
        if not self._results:
            self.run_all()

        W = 90   # total table width

        def hline(left: str = "╠", fill: str = "═", right: str = "╣") -> str:
            return left + fill * (W - 2) + right

        def row(cols: list[tuple[str, int]]) -> str:
            """Format a row given (text, width) column specs."""
            cells = "│".join(f" {c[0]:<{c[1]-2}} " for c in cols)
            return f"║{cells}║"

        col_widths = [38, 14, 14, 10, 9]   # scenario, expected, simulated, rel_err, status

        print(hline("╔", "═", "╗"))
        title = "PhysWorldLM — Benchmark Validation Report"
        print(f"║{title:^{W-2}}║")
        print(f"║{'Integrator: RK4    dt = ' + str(self.dt) + ' s':^{W-2}}║")
        print(hline())
        print(row([
            ("Scenario",          col_widths[0]),
            ("Expected",          col_widths[1]),
            ("Simulated",         col_widths[2]),
            ("Rel. Error",        col_widths[3]),
            ("Status",            col_widths[4]),
        ]))
        print(hline())

        for r in self._results:
            exp_str = (
                f"{r.expected_value:.5g} {r.unit}"
                if not math.isnan(r.expected_value) else "—"
            )
            sim_str = (
                f"{r.simulated_value:.5g} {r.unit}"
                if not math.isnan(r.simulated_value) else "—"
            )
            err_str = r.error_str() if not math.isnan(r.relative_error) else "—"
            status  = f"  {r.status_str()}  "

            # Truncate scenario name if too long for column
            name = r.scenario_name
            if len(name) > col_widths[0] - 3:
                name = name[:col_widths[0] - 6] + "…"

            print(row([
                (name,    col_widths[0]),
                (exp_str, col_widths[1]),
                (sim_str, col_widths[2]),
                (err_str, col_widths[3]),
                (status,  col_widths[4]),
            ]))

        print(hline())

        # Summary line
        total   = len(self._results)
        passed  = sum(1 for r in self._results if r.passed)
        failed  = total - passed
        rate    = 100 * passed / total if total else 0.0
        wall    = sum(r.wall_time_s for r in self._results)

        summary = (
            f"  Tests: {total}  │  Passed: {passed}  │  "
            f"Failed: {failed}  │  Pass rate: {rate:.1f} %  │  "
            f"Wall time: {wall*1e3:.0f} ms"
        )
        print(f"║{summary:<{W-2}}║")
        print(hline("╚", "═", "╝"))

        # Detailed notes for any failures
        failures = [r for r in self._results if not r.passed]
        if failures:
            print()
            print("── Failure Details " + "─" * (W - 19))
            for r in failures:
                print(f"\n  ✗  {r.scenario_name}")
                print(f"     Expected  : {r.expected_value:.6g} {r.unit}")
                print(f"     Simulated : {r.simulated_value:.6g} {r.unit}")
                print(f"     Abs error : {r.absolute_error:.6g} {r.unit}")
                print(f"     Rel error : {r.error_str()}")
                print(f"     Tolerance : rel={r.rel_tol:.0e}  abs={r.abs_tol:.0e}")
                if r.notes:
                    print(f"     Notes     : {r.notes}")
            print("─" * W)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(
        description="PhysWorldLM benchmark suite — validate simulator against analytical solutions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.002,
        help="Integration timestep for all scenarios (seconds).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show StateEngine log output during simulation.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON report to stdout instead of the table.",
    )
    parser.add_argument(
        "--scenario",
        choices=[
            "free_fall", "projectile", "constant_velocity",
            "spring", "energy", "all",
        ],
        default="all",
        help="Run a specific scenario instead of the full suite.",
    )
    args = parser.parse_args()

    suite = BenchmarkSuite(dt=args.dt, verbose=args.verbose)

    scenario_map = {
        "free_fall":         suite.run_free_fall,
        "projectile":        suite.run_projectile,
        "constant_velocity": suite.run_constant_velocity,
        "spring":            suite.run_spring_oscillator,
        "energy":            suite.run_energy_conservation,
    }

    if args.scenario == "all":
        report = suite.run_all()
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            suite.print_report()
    else:
        runner = scenario_map[args.scenario]
        results = runner()
        suite._results = results
        if args.json:
            print(json.dumps([
                {
                    "scenario_name":   r.scenario_name,
                    "expected_value":  r.expected_value,
                    "simulated_value": r.simulated_value,
                    "absolute_error":  r.absolute_error,
                    "relative_error":  (
                        r.relative_error
                        if not math.isinf(r.relative_error) else None
                    ),
                    "passed":          r.passed,
                    "unit":            r.unit,
                    "wall_time_ms":    round(r.wall_time_s * 1e3, 2),
                    "notes":           r.notes,
                }
                for r in results
            ], indent=2))
        else:
            suite.print_report()
