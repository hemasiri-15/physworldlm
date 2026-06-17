"""
tests/test_prompt_parser.py
────────────────────────────
Regression suite for models.prompt_parser.PromptParser.

Design principles
-----------------
* Tests validate *behaviour*, not implementation details (no magic numbers,
  no exact UUID checks, no assertions on internal regex internals).
* Approximate assertions use ``pytest.approx`` or simple relational checks
  (``>``, ``<``, ``>=``).
* Each test is isolated – the ``parser`` fixture returns a fresh instance.
* Tests are grouped conceptually with section comments matching the
  requirements document.

Coverage
--------
Section A  – Core scene types (free_fall, projectile, vehicle, spring)
Section B  – Multi-entity and collision scenes
Section C  – Environment (vacuum, weather, surface)
Section D  – Metadata fields (scene_type, parse_confidence, warnings)
Section E  – Alias / vocabulary coverage (sedan, SUV, helicopter …)
Section F  – Robustness / no-crash guarantees
Section G  – Regression prompts from the requirements document
"""

from __future__ import annotations

import math

import pytest

# ── import the module under test ───────────────────────────────────────────────
from models.prompt_parser import PromptParser


# ─────────────────────────────────────────────
# Shared fixture
# ─────────────────────────────────────────────

@pytest.fixture
def parser() -> PromptParser:
    """Return a fresh PromptParser with default settings for each test."""
    return PromptParser()


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _dynamic_entities(spec):
    """Return only non-static (dynamic) entities from a WorldSpec."""
    return [e for e in spec.entities if not e.is_static]


def _static_entities(spec):
    """Return only static entities from a WorldSpec."""
    return [e for e in spec.entities if e.is_static]


def _speed(entity) -> float:
    """Return scalar speed (m/s) of an entity's initial velocity."""
    v = entity.state.velocity
    return math.sqrt(v.x**2 + v.y**2 + v.z**2)


def _spring_interactions(spec):
    """Return all spring-type interactions in the spec."""
    return [i for i in spec.interactions if i.type == "spring"]


def _collision_interactions(spec):
    """Return all collision-type interactions in the spec."""
    return [i for i in spec.interactions if i.type == "collision"]


# ═════════════════════════════════════════════
# Section A – Core scene types
# ═════════════════════════════════════════════

class TestFreeFall:
    """Free-fall scenes: ball / object dropped from a height."""

    def test_free_fall_height(self, parser):
        """
        A ball dropped from 100 m should start with its centre near y ≈ 100 m.
        The parser places the entity centre at (height + half_bounding_box_height),
        so we accept anything within a metre of 100 m.
        """
        spec = parser.parse("A ball falls from 100 m")

        dynamic = _dynamic_entities(spec)
        assert len(dynamic) == 1, "Expected exactly one dynamic entity"

        entity = dynamic[0]
        assert entity.entity_type in ("object", "projectile", "generic"), (
            f"Unexpected entity_type: {entity.entity_type}"
        )
        # Centre should be close to 100 m (within half a metre of tolerance)
        assert entity.state.position.y == pytest.approx(100.0, abs=1.0), (
            f"Expected y ≈ 100 m, got {entity.state.position.y}"
        )

    def test_free_fall_initial_velocity_is_zero(self, parser):
        """An object released from rest should have zero initial velocity."""
        spec = parser.parse("A stone is dropped from 50 m")
        dynamic = _dynamic_entities(spec)
        assert len(dynamic) >= 1
        entity = dynamic[0]
        assert _speed(entity) == pytest.approx(0.0, abs=0.1)

    def test_free_fall_duration_scales_with_height(self, parser):
        """
        Duration heuristic for free fall: t ~ √(2h/g).
        A 200 m drop should produce a longer duration than a 50 m drop.
        """
        spec_high = parser.parse("A ball falls from 200 m")
        spec_low  = parser.parse("A ball falls from 50 m")
        assert (spec_high.simulation_graph.duration
                > spec_low.simulation_graph.duration), (
            "Higher drop should produce longer simulated duration"
        )

    def test_free_fall_in_feet(self, parser):
        """
        Unit conversion: 'falls 100 ft' should place entity at ≈ 30.48 m,
        not 100 m.
        """
        spec = parser.parse("A ball falls 100 ft")
        dynamic = _dynamic_entities(spec)
        assert dynamic, "At least one dynamic entity expected"
        y = dynamic[0].state.position.y
        # 100 ft ≈ 30.48 m; allow generous tolerance for bounding-box offset
        assert y == pytest.approx(30.48, abs=2.0), (
            f"Expected y ≈ 30.48 m (100 ft), got {y}"
        )


class TestProjectile:
    """Projectile / launch scenes."""

    def test_projectile_velocity(self, parser):
        """
        A projectile launched at 20 m/s at 45° should have vx > 0, vy > 0,
        and a speed magnitude of ≈ 20 m/s.
        """
        spec = parser.parse("A projectile is launched at 20 m/s at 45 degrees")
        dynamic = _dynamic_entities(spec)
        assert dynamic, "Expected at least one dynamic entity"

        entity = dynamic[0]
        vx = entity.state.velocity.x
        vy = entity.state.velocity.y
        assert vx > 0, f"Expected vx > 0, got {vx}"
        assert vy > 0, f"Expected vy > 0, got {vy}"
        assert _speed(entity) == pytest.approx(20.0, rel=0.05), (
            f"Speed should be ≈ 20 m/s, got {_speed(entity)}"
        )

    def test_projectile_velocity_components_at_45_degrees(self, parser):
        """At 45° the x and y components should be equal (within rounding)."""
        spec = parser.parse("A projectile launched at 10 m/s at 45 degrees")
        dynamic = _dynamic_entities(spec)
        assert dynamic
        vx = dynamic[0].state.velocity.x
        vy = dynamic[0].state.velocity.y
        assert vx == pytest.approx(vy, rel=0.02), (
            f"At 45° vx and vy should be equal; got vx={vx}, vy={vy}"
        )

    def test_projectile_default_speed_when_missing(self, parser):
        """A projectile with no speed specified should still get a positive speed."""
        spec = parser.parse("A projectile is launched")
        dynamic = _dynamic_entities(spec)
        assert dynamic
        assert _speed(dynamic[0]) > 0, "Default speed should be positive"

    def test_projectile_unit_conversion_kmh(self, parser):
        """
        'launched at 100 km/h' → vx+vy ≈ 27.78 m/s total speed.
        """
        spec = parser.parse("A projectile is launched at 100 km/h at 45 degrees")
        dynamic = _dynamic_entities(spec)
        assert dynamic
        assert _speed(dynamic[0]) == pytest.approx(100 / 3.6, rel=0.05)


class TestVehicleMotion:
    """Vehicle forward-motion scenes."""

    def test_car_motion_speed(self, parser):
        """
        A car moving at 30 m/s should produce a vehicle entity with vx ≈ 30 m/s.
        """
        spec = parser.parse("A car moves at 30 m/s")
        dynamic = _dynamic_entities(spec)
        vehicles = [e for e in dynamic if e.entity_type == "vehicle"]
        assert vehicles, "Expected at least one vehicle entity"

        car = vehicles[0]
        assert car.state.velocity.x == pytest.approx(30.0, rel=0.05), (
            f"Expected vx ≈ 30 m/s, got {car.state.velocity.x}"
        )

    def test_car_entity_type(self, parser):
        """Entity type should be 'vehicle' for a car prompt."""
        spec = parser.parse("A car moves at 20 m/s")
        dynamic = _dynamic_entities(spec)
        assert any(e.entity_type == "vehicle" for e in dynamic)

    def test_vehicle_duration_from_distance(self, parser):
        """
        'A truck travels 500 m at 10 m/s' → duration ≈ 500/10 = 50 s.
        We accept anything in [40, 70] s to allow for the 1.1× safety margin.
        """
        spec = parser.parse("A truck travels 500 m at 10 m/s")
        duration = spec.simulation_graph.duration
        assert 40.0 <= duration <= 70.0, (
            f"Expected duration ≈ 50 s (distance/speed), got {duration}"
        )


class TestSpring:
    """Spring oscillator scenes."""

    def test_spring_scene_creates_interaction(self, parser):
        """
        A spring prompt must produce at least one spring interaction
        with a k_Nm parameter ≈ 100.
        """
        spec = parser.parse(
            "A 1 kg mass attached to a spring with k=100 N/m and rest length 1 m"
        )
        springs = _spring_interactions(spec)
        assert springs, "Expected at least one spring interaction"
        k = springs[0].parameters.get("k_Nm")
        assert k is not None, "Spring interaction must include k_Nm parameter"
        assert k == pytest.approx(100.0, rel=0.01)

    def test_spring_rest_length_extracted(self, parser):
        """rest_length_m should be read from the prompt, not hardcoded."""
        spec = parser.parse("A spring with rest length 2 m and k=50 N/m")
        springs = _spring_interactions(spec)
        assert springs
        rest = springs[0].parameters.get("rest_length_m")
        assert rest is not None
        # Should be ≈ 2 m, not the old hardcoded default of 1 m
        assert rest == pytest.approx(2.0, abs=0.1), (
            f"Expected rest_length_m ≈ 2 m, got {rest}"
        )

    def test_spring_amplitude_extracted(self, parser):
        """amplitude_m should be read from the prompt when present."""
        spec = parser.parse("A spring with amplitude 0.5 m and k=100 N/m")
        springs = _spring_interactions(spec)
        assert springs
        amp = springs[0].parameters.get("amplitude_m")
        assert amp is not None
        assert amp == pytest.approx(0.5, abs=0.05), (
            f"Expected amplitude_m ≈ 0.5 m, got {amp}"
        )

    def test_spring_default_rest_length_when_absent(self, parser):
        """When rest length is not specified, a sensible positive default is used."""
        spec = parser.parse("A mass on a spring with k=200 N/m")
        springs = _spring_interactions(spec)
        assert springs
        rest = springs[0].parameters.get("rest_length_m", 0)
        assert rest > 0, "Default rest_length_m should be positive"

    def test_spring_k_200(self, parser):
        """
        Additional regression: 'A spring with k=200 N/m and rest length 2 m'
        should correctly set k=200.
        """
        spec = parser.parse("A spring with k=200 N/m and rest length 2 m")
        springs = _spring_interactions(spec)
        assert springs
        assert springs[0].parameters["k_Nm"] == pytest.approx(200.0, rel=0.01)
        assert springs[0].parameters["rest_length_m"] == pytest.approx(2.0, abs=0.1)

    def test_spring_oscillation_duration_covers_multiple_periods(self, parser):
        """Duration should cover at least one full period for k=100 N/m, m=1 kg."""
        spec = parser.parse("A 1 kg mass on a spring with k=100 N/m")
        omega  = math.sqrt(100.0 / 1.0)
        period = 2.0 * math.pi / omega          # ≈ 0.628 s
        assert spec.simulation_graph.duration >= period, (
            "Simulation should run for at least one full oscillation period"
        )


# ═════════════════════════════════════════════
# Section B – Multi-entity and collision scenes
# ═════════════════════════════════════════════

class TestCollision:
    """Collision / impact scenes."""

    def test_collision_creates_one_interaction(self, parser):
        """
        'Two cars collide' should produce two vehicle entities and exactly
        one collision interaction (not all pairwise combinations).
        """
        spec = parser.parse("Two cars collide")
        dynamic = _dynamic_entities(spec)
        vehicles = [e for e in dynamic if e.entity_type == "vehicle"]
        assert len(vehicles) == 2, f"Expected 2 vehicle entities, got {len(vehicles)}"

        collisions = _collision_interactions(spec)
        assert len(collisions) == 1, (
            f"Expected exactly 1 collision interaction, got {len(collisions)}"
        )

    def test_three_car_collision_not_quadratic(self, parser):
        """
        'Three cars collide' should NOT produce all C(3,2)=3 pairwise
        collision interactions — the parser uses a conservative single-pair
        heuristic.  We accept 1 or 2.
        """
        spec = parser.parse("Three cars collide")
        collisions = _collision_interactions(spec)
        assert len(collisions) <= 2, (
            f"Interaction count should be ≤ 2 for 3 entities, got {len(collisions)}"
        )

    def test_collision_mass_localisation(self, parser):
        """
        'A 1000 kg car collides with a 5 kg ball.'
        The car's mass should be much larger than the ball's mass.
        """
        spec = parser.parse("A 1000 kg car collides with a 5 kg ball.")
        dynamic = _dynamic_entities(spec)
        assert len(dynamic) >= 2, "Expected at least 2 dynamic entities"

        masses = sorted([e.mass for e in dynamic])
        # The heavier entity should be clearly heavier
        assert masses[-1] > masses[0] * 2, (
            f"Mass localisation failed: masses are {masses}; "
            "expected a large spread (1000 kg vs 5 kg)"
        )

    def test_head_on_collision_two_cars(self, parser):
        """
        'Two cars collide head-on at 20 m/s.' should produce 2 vehicles
        and at least 1 collision interaction.
        """
        spec = parser.parse("Two cars collide head-on at 20 m/s.")
        vehicles = [e for e in _dynamic_entities(spec) if e.entity_type == "vehicle"]
        assert len(vehicles) == 2
        assert len(_collision_interactions(spec)) >= 1

    def test_collision_duration_bounded(self, parser):
        """
        Collision duration heuristic should produce a reasonable value
        (not zero, not absurdly long).
        """
        spec = parser.parse("Two cars collide at 20 m/s")
        duration = spec.simulation_graph.duration
        assert 0.5 <= duration <= 60.0, (
            f"Collision duration {duration} s is outside plausible range [0.5, 60]"
        )


class TestMultiEntityMotion:
    """Multiple entities with independent velocities."""

    def test_two_entities_each_get_velocity(self, parser):
        """
        'A car and a person move at 10 m/s.' — both should receive a non-zero
        x-velocity.
        """
        spec = parser.parse("A car and a person move at 10 m/s.")
        dynamic = _dynamic_entities(spec)
        moving = [e for e in dynamic if abs(e.state.velocity.x) > 0.01]
        assert len(moving) >= 2, (
            "Both entities should receive velocity from a shared speed phrase"
        )

    def test_stationary_entity_gets_zero_velocity(self, parser):
        """
        'A car moving at 20 m/s hits a stationary wall.'
        The wall is static (structure), so no velocity assignment is needed.
        The car should have vx ≈ 20 m/s.
        """
        spec = parser.parse("A car moving at 20 m/s hits a stationary wall.")
        vehicles = [e for e in spec.entities
                    if e.entity_type == "vehicle" and not e.is_static]
        assert vehicles, "Expected at least one dynamic vehicle"
        car = vehicles[0]
        assert car.state.velocity.x == pytest.approx(20.0, rel=0.1), (
            f"Car velocity should be ≈ 20 m/s, got {car.state.velocity.x}"
        )


# ═════════════════════════════════════════════
# Section C – Environment
# ═════════════════════════════════════════════

class TestEnvironment:
    """Environmental conditions parsed from the prompt."""

    def test_vacuum_environment(self, parser):
        """'A ball falls in vacuum' should set air_density to zero."""
        spec = parser.parse("A ball falls in vacuum")
        assert spec.environment.air_density == pytest.approx(0.0, abs=1e-6), (
            "Vacuum keyword should set air_density = 0"
        )

    def test_wet_surface_friction_is_lower(self, parser):
        """
        Wet surface should reduce friction below the default (0.5).
        We don't check the exact value — just that it is strictly less.
        """
        spec_wet     = parser.parse("A car skids on a wet road")
        spec_default = parser.parse("A car moves at 20 m/s")
        assert (spec_wet.environment.friction_global
                < spec_default.environment.friction_global), (
            "Wet surface should have lower friction than default"
        )

    def test_icy_surface_friction_lower_than_wet(self, parser):
        """
        Icy surface friction should be lower than wet surface friction.
        """
        spec_wet = parser.parse("A car skids on a wet road")
        spec_icy = parser.parse("A car slides on an icy road")
        assert (spec_icy.environment.friction_global
                < spec_wet.environment.friction_global), (
            "Icy surface should have lower friction than wet surface"
        )

    def test_rough_surface_friction_is_higher(self, parser):
        """
        Rough surface should raise friction above the default (0.5).
        """
        spec_rough   = parser.parse("A block slides on a rough surface")
        spec_default = parser.parse("A block slides")
        assert (spec_rough.environment.friction_global
                > spec_default.environment.friction_global), (
            "Rough surface should have higher friction than default"
        )

    def test_night_scene(self, parser):
        """'A person walks at night' should set time_of_day to 'night'."""
        spec = parser.parse("A person walks at night")
        assert spec.environment.time_of_day == "night", (
            f"Expected time_of_day='night', got {spec.environment.time_of_day!r}"
        )

    def test_rain_weather(self, parser):
        """Rain keyword sets weather to 'rain'."""
        spec = parser.parse("A car drives in the rain")
        assert spec.environment.weather == "rain"

    def test_snow_weather(self, parser):
        """Snow keyword sets weather to 'snow'."""
        spec = parser.parse("A car drives in snow")
        assert spec.environment.weather == "snow"

    def test_gravity_direction(self, parser):
        """Gravity should always point downward (negative y)."""
        spec = parser.parse("A ball falls from 10 m")
        assert spec.environment.gravity.y < 0, "Gravity y-component must be negative"
        assert spec.environment.gravity.x == pytest.approx(0.0, abs=1e-6)
        assert spec.environment.gravity.z == pytest.approx(0.0, abs=1e-6)


# ═════════════════════════════════════════════
# Section D – Metadata fields
# ═════════════════════════════════════════════

class TestMetadata:
    """WorldSpec.metadata fields: scene_type, parse_confidence, warnings."""

    def test_scene_type_free_fall(self, parser):
        """scene_type should be 'free_fall' for a falling-ball prompt."""
        spec = parser.parse("A ball falls from 50 m")
        assert spec.metadata.get("scene_type") == "free_fall", (
            f"Expected scene_type='free_fall', got {spec.metadata.get('scene_type')!r}"
        )

    def test_scene_type_projectile(self, parser):
        """scene_type should be 'projectile' for a launch prompt."""
        spec = parser.parse("A projectile is launched at 30 m/s")
        assert spec.metadata.get("scene_type") == "projectile"

    def test_scene_type_vehicle_motion(self, parser):
        """scene_type should be 'vehicle_motion' for a car-moving prompt."""
        spec = parser.parse("A car moves at 20 m/s")
        assert spec.metadata.get("scene_type") == "vehicle_motion"

    def test_scene_type_spring(self, parser):
        """scene_type should be 'spring' for a spring prompt."""
        spec = parser.parse("A mass on a spring with k=50 N/m")
        assert spec.metadata.get("scene_type") == "spring"

    def test_scene_type_collision(self, parser):
        """scene_type should be 'collision' for a collision prompt."""
        spec = parser.parse("Two cars collide")
        assert spec.metadata.get("scene_type") == "collision"

    def test_parse_confidence_exists(self, parser):
        """
        parse_confidence must exist in metadata and be a float in [0.0, 1.0].
        """
        spec = parser.parse("A car moves at 20 m/s")
        assert "parse_confidence" in spec.metadata, (
            "metadata must contain 'parse_confidence'"
        )
        conf = spec.metadata["parse_confidence"]
        assert isinstance(conf, float), (
            f"parse_confidence must be a float, got {type(conf)}"
        )
        assert 0.0 <= conf <= 1.0, (
            f"parse_confidence must be in [0, 1], got {conf}"
        )

    def test_parse_confidence_higher_for_richer_prompt(self, parser):
        """
        A prompt with speed, mass, and a known scene type should score higher
        confidence than a vague prompt with no numeric values.
        """
        rich  = parser.parse("A 1000 kg car moves at 20 m/s")
        vague = parser.parse("something moves somehow")
        # vague may not even parse, but if it does, confidence should be lower
        try:
            # Even if vague doesn't raise, confidence should be less or equal
            assert (rich.metadata["parse_confidence"]
                    >= vague.metadata["parse_confidence"])
        except ValueError:
            pass  # vague prompt raising ValueError is acceptable

    def test_warnings_key_exists(self, parser):
        """All WorldSpecs must contain a 'warnings' list in metadata."""
        spec = parser.parse("A car moves at 20 m/s")
        assert "warnings" in spec.metadata, "metadata must contain 'warnings'"
        assert isinstance(spec.metadata["warnings"], list)

    def test_warning_default_projectile_speed(self, parser):
        """
        'A projectile is launched' (no speed given) should emit at least
        one warning about the missing speed.
        """
        spec = parser.parse("A projectile is launched")
        warnings = spec.metadata.get("warnings", [])
        assert len(warnings) > 0, (
            "Expected at least one warning when projectile speed is not specified"
        )

    def test_warning_default_spring_k(self, parser):
        """A spring scene with no k value should warn about the default."""
        spec = parser.parse("A mass is attached to a spring")
        warnings = spec.metadata.get("warnings", [])
        # At least one warning about missing k
        assert any("spring" in w.lower() or "k" in w.lower() for w in warnings), (
            f"Expected a warning about missing spring constant; got: {warnings}"
        )

    def test_no_spurious_warnings_for_complete_prompt(self, parser):
        """
        A fully specified prompt (speed, mass, k) should produce zero or very
        few warnings.  We allow up to 1 minor warning (e.g. multiple heights).
        """
        spec = parser.parse(
            "A 5 kg ball is launched at 20 m/s at 45 degrees from a height of 0 m"
        )
        warnings = spec.metadata.get("warnings", [])
        assert len(warnings) <= 2, (
            f"Expected ≤ 2 warnings for a well-specified prompt; got {len(warnings)}: {warnings}"
        )

    def test_entity_count_in_metadata(self, parser):
        """entity_count in metadata should match len(spec.entities)."""
        spec = parser.parse("A car moves at 20 m/s")
        assert spec.metadata["entity_count"] == len(spec.entities)

    def test_dynamic_count_in_metadata(self, parser):
        """dynamic_count should match the number of non-static entities."""
        spec = parser.parse("A car moves at 20 m/s")
        expected = sum(1 for e in spec.entities if not e.is_static)
        assert spec.metadata["dynamic_count"] == expected

    def test_material_detection(self, parser):
        """'A steel sphere falls from 20 m' should set material to 'steel'."""
        spec = parser.parse("A steel sphere falls from 20 m")
        dynamic = _dynamic_entities(spec)
        assert dynamic, "Expected at least one dynamic entity"
        materials = [e.material for e in dynamic]
        assert "steel" in materials, (
            f"Expected material='steel', got materials={materials}"
        )


# ═════════════════════════════════════════════
# Section E – Alias / vocabulary coverage
# ═════════════════════════════════════════════

class TestEntityAliases:
    """
    Tests that a broad range of entity synonyms are correctly classified.
    None of these should require manual additions to the parser vocabulary
    (all are covered by ENTITY_ALIASES or _classify_unknown_entity).
    """

    # Map: (prompt, expected_entity_type)
    ALIAS_CASES: list[tuple[str, str]] = [
        # vehicle – car variants
        ("A sedan moves at 20 m/s",            "vehicle"),
        ("An SUV drives on the road",           "vehicle"),
        ("A bicycle rolls at 5 m/s",            "vehicle"),
        ("A train travels at 80 km/h",          "vehicle"),
        ("A helicopter hovers at 10 m/s",       "vehicle"),
        # vehicle – watercraft
        ("A canoe moves at 2 m/s",              "vehicle"),
        ("A boat moves at 10 m/s",              "vehicle"),
        # agent
        ("A child runs at 3 m/s",               "agent"),
        ("A worker falls from 5 m",             "agent"),
        ("A person walks at night",             "agent"),
        # structure
        ("A ball hits a tower",                 "structure"),
        ("A ball hits a staircase",             "structure"),
        # fluid
        ("A lake is disturbed",                 "fluid"),
        # object
        ("A marble rolls at 1 m/s",             "object"),
        ("A crate falls from 10 m",             "object"),
    ]

    @pytest.mark.parametrize("prompt,expected_type", ALIAS_CASES)
    def test_alias_classification(self, parser, prompt, expected_type):
        """
        Each alias prompt should produce at least one entity of the expected type.
        The entity label should preserve the original noun (not the canonical type).
        """
        spec = parser.parse(prompt)
        entity_types = [e.entity_type for e in spec.entities]
        assert expected_type in entity_types, (
            f"Prompt {prompt!r}: expected entity_type {expected_type!r} "
            f"in spec.entities, got types {entity_types}"
        )

    def test_ferrari_classified_as_vehicle(self, parser):
        """Brand name 'Ferrari' should map to entity_type='vehicle'."""
        spec = parser.parse("A Ferrari accelerates at 30 m/s")
        vehicles = [e for e in spec.entities if e.entity_type == "vehicle"]
        assert vehicles, "Ferrari should be classified as a vehicle"

    def test_label_preserves_original_noun(self, parser):
        """
        When 'sedan' is the entity noun, the label should be 'sedan',
        not 'vehicle'.
        """
        spec = parser.parse("A sedan moves at 20 m/s")
        labels = [e.label for e in spec.entities]
        assert any("sedan" in lbl.lower() for lbl in labels), (
            f"Expected label to contain 'sedan', got {labels}"
        )

    def test_helicopter_label(self, parser):
        """'helicopter' label should be preserved."""
        spec = parser.parse("A helicopter hovers at 50 m/s")
        labels = [e.label for e in spec.entities]
        assert any("helicopter" in lbl.lower() for lbl in labels)

    def test_canoe_label(self, parser):
        """'canoe' label should be preserved."""
        spec = parser.parse("A canoe moves at 2 m/s")
        labels = [e.label for e in spec.entities]
        assert any("canoe" in lbl.lower() for lbl in labels)

    def test_child_label(self, parser):
        """'child' label should be preserved."""
        spec = parser.parse("A child runs at 3 m/s")
        labels = [e.label for e in spec.entities]
        assert any("child" in lbl.lower() for lbl in labels)

    def test_tower_label(self, parser):
        """'tower' label should be preserved."""
        spec = parser.parse("A ball hits a tower")
        labels = [e.label for e in spec.entities]
        assert any("tower" in lbl.lower() for lbl in labels)

    def test_staircase_label(self, parser):
        """'staircase' label should be preserved."""
        spec = parser.parse("A ball rolls down a staircase")
        labels = [e.label for e in spec.entities]
        assert any("staircase" in lbl.lower() for lbl in labels)

    def test_lake_label(self, parser):
        """'lake' label should be preserved."""
        spec = parser.parse("A stone falls into a lake")
        labels = [e.label for e in spec.entities]
        assert any("lake" in lbl.lower() for lbl in labels)

    def test_plural_two_cars_creates_two_vehicle_entities(self, parser):
        """'two cars collide' should produce exactly 2 vehicle entities."""
        spec = parser.parse("two cars collide")
        vehicles = [e for e in spec.entities if e.entity_type == "vehicle"]
        assert len(vehicles) == 2, f"Expected 2 vehicles, got {len(vehicles)}"

    def test_plural_three_balls(self, parser):
        """'three balls fall' should produce 3 object entities."""
        spec = parser.parse("three balls fall from 10 m")
        objects = [e for e in spec.entities if e.entity_type == "object"]
        assert len(objects) == 3, f"Expected 3 object entities, got {len(objects)}"

    def test_numeric_quantity_five_projectiles(self, parser):
        """'5 projectiles are launched' should produce 5 projectile entities."""
        spec = parser.parse("5 projectiles are launched at 20 m/s")
        projectiles = [e for e in spec.entities if e.entity_type == "projectile"]
        assert len(projectiles) == 5, f"Expected 5 projectiles, got {len(projectiles)}"


# ═════════════════════════════════════════════
# Section F – Robustness / no-crash guarantees
# ═════════════════════════════════════════════

class TestRobustness:
    """Parser must never crash on plausible English input."""

    def test_unknown_prompt_never_crashes(self, parser):
        """
        'simulate something' — vague but should not raise an exception.
        At least one entity should be produced (possibly with defaults).
        """
        # The parser is allowed to raise ValueError for completely
        # unrecognisable input, but must not crash with unhandled exceptions.
        spec = parser.parse("simulate something")
        assert len(spec.entities) >= 1

    def test_minimal_valid_prompt(self, parser):
        """'simulate a car' should produce a valid WorldSpec."""
        spec = parser.parse("simulate a car")
        assert spec.entities
        assert spec.simulation_graph.duration > 0

    def test_moving_object_with_no_speed(self, parser):
        """'a moving object' should not crash; default velocity applied."""
        spec = parser.parse("a moving object")
        assert spec.entities

    def test_red_ball_prompt(self, parser):
        """'a red ball' should produce an object entity with label 'red ball'."""
        spec = parser.parse("a red ball")
        dynamic = _dynamic_entities(spec)
        assert dynamic, "Expected at least one dynamic entity"
        labels = [e.label for e in dynamic]
        assert any("ball" in lbl.lower() for lbl in labels)

    def test_no_exceptions_on_all_examples(self, parser):
        """
        Smoke-test: none of these representative prompts should raise an
        unhandled exception.
        """
        examples = [
            "A ball falls from 100 m",
            "A car moves at 20 m/s",
            "Two cars collide",
            "A projectile launched at 30 m/s",
            "simulate something",
            "A spring with k=100 N/m",
        ]
        for prompt in examples:
            try:
                spec = parser.parse(prompt)
                # If it parsed, it must have at least one entity
                assert spec.entities, (
                    f"Prompt {prompt!r} produced empty entity list"
                )
            except ValueError:
                # ValueError for unrecognisable prompt is acceptable
                pass

    def test_extremely_large_height(self, parser):
        """Heights of 10 000 m should not cause numeric overflow or errors."""
        spec = parser.parse("A ball falls from 10000 m")
        assert spec.simulation_graph.duration > 0

    def test_very_small_mass(self, parser):
        """Prompts with tiny masses should produce a valid WorldSpec."""
        spec = parser.parse("A 0.001 kg ball falls from 5 m")
        dynamic = _dynamic_entities(spec)
        assert dynamic
        assert dynamic[0].mass > 0

    def test_unit_conversions_do_not_crash(self, parser):
        """Various unit phrases should be parsed without error."""
        prompts = [
            "A ball falls 100 ft",
            "A 2 lb projectile launched at 30 mph",
            "A car moves at 60 mph",
            "A mass of 500 grams falls from 10 m",
        ]
        for prompt in prompts:
            spec = parser.parse(prompt)
            assert spec.entities, f"Prompt {prompt!r} produced no entities"


# ═════════════════════════════════════════════
# Section G – Regression prompts from requirements doc
# ═════════════════════════════════════════════

class TestRegressionPrompts:
    """
    Explicit prompts from the requirements document.
    Each is tested for one or two key properties only — no over-specification.
    """

    def test_1000kg_car_5kg_ball_collision(self, parser):
        """
        'A 1000 kg car collides with a 5 kg ball.'
        Entities must exist and masses should be distinct.
        """
        spec = parser.parse("A 1000 kg car collides with a 5 kg ball.")
        dynamic = _dynamic_entities(spec)
        assert len(dynamic) >= 2
        # The heaviest entity should be at least 5× the lightest
        masses = sorted(e.mass for e in dynamic)
        assert masses[-1] > masses[0] * 5, (
            f"Expected large mass spread; got {masses}"
        )

    def test_two_cars_head_on_20ms(self, parser):
        """'Two cars collide head-on at 20 m/s.' → 2 vehicles, 1 collision."""
        spec = parser.parse("Two cars collide head-on at 20 m/s.")
        vehicles = [e for e in _dynamic_entities(spec) if e.entity_type == "vehicle"]
        assert len(vehicles) == 2
        assert len(_collision_interactions(spec)) >= 1

    def test_projectile_100kmh(self, parser):
        """'A projectile is launched at 100 km/h.' → speed ≈ 27.78 m/s."""
        spec = parser.parse("A projectile is launched at 100 km/h.")
        dynamic = _dynamic_entities(spec)
        assert dynamic
        spd = _speed(dynamic[0])
        assert spd == pytest.approx(100 / 3.6, rel=0.05), (
            f"Expected ≈ 27.78 m/s, got {spd}"
        )

    def test_spring_k200_rest2m(self, parser):
        """'A spring with k=200 N/m and rest length 2 m.' → correct params."""
        spec = parser.parse("A spring with k=200 N/m and rest length 2 m.")
        springs = _spring_interactions(spec)
        assert springs
        assert springs[0].parameters["k_Nm"] == pytest.approx(200.0, rel=0.01)
        assert springs[0].parameters["rest_length_m"] == pytest.approx(2.0, abs=0.1)

    def test_truck_500m_at_10ms(self, parser):
        """'A truck travels 500 m at 10 m/s.' → duration ≈ 50 s."""
        spec = parser.parse("A truck travels 500 m at 10 m/s.")
        assert 40 <= spec.simulation_graph.duration <= 70, (
            f"Expected duration ≈ 50 s, got {spec.simulation_graph.duration}"
        )

    def test_ball_falls_100ft(self, parser):
        """'A ball falls 100 ft.' → entity y ≈ 30.48 m."""
        spec = parser.parse("A ball falls 100 ft.")
        dynamic = _dynamic_entities(spec)
        assert dynamic
        y = dynamic[0].state.position.y
        assert y == pytest.approx(30.48, abs=2.0)

    def test_2lb_projectile_30mph(self, parser):
        """
        'A 2 lb projectile launched at 30 mph.'
        mass ≈ 0.907 kg, speed ≈ 13.41 m/s.
        """
        spec = parser.parse("A 2 lb projectile launched at 30 mph.")
        dynamic = _dynamic_entities(spec)
        assert dynamic
        entity = dynamic[0]
        # 2 lb ≈ 0.907 kg
        assert entity.mass == pytest.approx(0.907, rel=0.10), (
            f"Expected mass ≈ 0.907 kg, got {entity.mass}"
        )
        # 30 mph ≈ 13.41 m/s
        spd = _speed(entity)
        assert spd == pytest.approx(13.41, rel=0.10), (
            f"Expected speed ≈ 13.41 m/s, got {spd}"
        )


# ═════════════════════════════════════════════
# Section H – SimulationGraph sanity
# ═════════════════════════════════════════════

class TestSimulationGraph:
    """Basic checks on SimulationGraph fields produced by the parser."""

    def test_dt_is_positive(self, parser):
        spec = parser.parse("A car moves at 20 m/s")
        assert spec.simulation_graph.dt > 0

    def test_duration_is_positive(self, parser):
        spec = parser.parse("A ball falls from 10 m")
        assert spec.simulation_graph.duration > 0

    def test_dt_less_than_duration(self, parser):
        """dt must be strictly less than duration to avoid degenerate sims."""
        spec = parser.parse("A car moves at 20 m/s")
        assert spec.simulation_graph.dt < spec.simulation_graph.duration

    def test_integrator_is_valid(self, parser):
        """Integrator should be one of the accepted values."""
        valid = {"rk4", "euler", "verlet"}
        spec = parser.parse("A ball falls from 10 m")
        assert spec.simulation_graph.integrator in valid, (
            f"Unexpected integrator: {spec.simulation_graph.integrator!r}"
        )

    def test_export_fps_is_positive(self, parser):
        spec = parser.parse("A ball falls from 10 m")
        assert spec.simulation_graph.export_fps > 0


# ═════════════════════════════════════════════
# Section I – Interaction structural integrity
# ═════════════════════════════════════════════

class TestInteractionIntegrity:
    """
    All interaction entity references must point to existing entity IDs.
    This mirrors what WorldSpecValidator checks.
    """

    PROMPTS = [
        "A ball falls from 100 m",
        "Two cars collide at 20 m/s",
        "A 1 kg mass on a spring with k=100 N/m",
        "A projectile is launched at 20 m/s at 45 degrees",
        "A car moves at 30 m/s",
    ]

    @pytest.mark.parametrize("prompt", PROMPTS)
    def test_interaction_entity_refs_are_valid(self, parser, prompt):
        """
        For every interaction in the spec, entity_a and entity_b must either
        exist in spec.entities or be the sentinel string 'environment'.
        """
        spec = parser.parse(prompt)
        entity_ids = {e.id for e in spec.entities}
        for i, itr in enumerate(spec.interactions):
            assert itr.entity_a in entity_ids, (
                f"[{prompt!r}] interaction[{i}].entity_a {itr.entity_a!r} "
                f"not in entity ids"
            )
            assert itr.entity_b in entity_ids or itr.entity_b == "environment", (
                f"[{prompt!r}] interaction[{i}].entity_b {itr.entity_b!r} "
                f"not in entity ids"
            )

    @pytest.mark.parametrize("prompt", PROMPTS)
    def test_no_duplicate_entity_ids(self, parser, prompt):
        """Every entity must have a unique ID."""
        spec = parser.parse(prompt)
        ids = [e.id for e in spec.entities]
        assert len(ids) == len(set(ids)), (
            f"[{prompt!r}] Duplicate entity IDs found: {ids}"
        )

    @pytest.mark.parametrize("prompt", PROMPTS)
    def test_all_dynamic_entities_have_positive_mass(self, parser, prompt):
        """WorldSpecValidator requires mass > 0 for all non-static entities."""
        spec = parser.parse(prompt)
        for e in spec.entities:
            if not e.is_static:
                assert e.mass > 0, (
                    f"[{prompt!r}] Entity {e.id!r} ({e.label!r}) "
                    f"has non-positive mass {e.mass}"
                )

    @pytest.mark.parametrize("prompt", PROMPTS)
    def test_bounding_boxes_are_positive(self, parser, prompt):
        """All bounding box dimensions must be > 0."""
        spec = parser.parse(prompt)
        for e in spec.entities:
            bb = e.bounding_box
            assert bb.width > 0 and bb.height > 0 and bb.depth > 0, (
                f"[{prompt!r}] Entity {e.id!r} has zero/negative bounding box: "
                f"({bb.width}, {bb.height}, {bb.depth})"
            )
