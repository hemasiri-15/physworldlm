"""
tests/test_world_parser.py
──────────────────────────
Pytest tests for spatial relationship parsing in PhysWorldLM.

Tests verify that WorldParser.parse() correctly resolves positional
relationships (above, below, left of, right of, near, behind, in front of,
and explicit distance) between entities in a returned WorldSpec.

Coordinate conventions (from world_spec.py):
    x = East / forward
    y = Up
    z = North
    gravity = -y
"""

import math
import pytest

from models.world_parser import WorldParser
from models.world_spec import WorldSpec

# ─────────────────────────────────────────────
# Fixture
# ─────────────────────────────────────────────

@pytest.fixture
def parser():
    return WorldParser()


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _find_entity(spec, *keywords):
    """
    Return the first entity whose label contains any of the given keywords
    (case-insensitive).  Raises AssertionError with a clear message when not found.
    """
    keywords_lower = [kw.lower() for kw in keywords]
    for entity in spec.entities:
        label = entity.label.lower()
        if any(kw in label for kw in keywords_lower):
            return entity
    labels = [e.label for e in spec.entities]
    raise AssertionError(
        f"No entity matching {keywords!r} found in spec. "
        f"Available labels: {labels}"
    )


def _euclidean_distance(pos_a, pos_b):
    """3-D Euclidean distance between two Vec3 positions."""
    return math.sqrt(
        (pos_a.x - pos_b.x) ** 2
        + (pos_a.y - pos_b.y) ** 2
        + (pos_a.z - pos_b.z) ** 2
    )


# ─────────────────────────────────────────────
# Test class
# ─────────────────────────────────────────────

class TestSpatialRelationships:

    # ── 1. Explicit metric distance ───────────────────────────────────────────

    def test_distance_relationship(self, parser):
        """
        'A car is 20 m from a wall.'
        The x-separation between the car and the wall should be ≈ 20 m.
        """
        spec = parser.parse("A car is 20 m from a wall.")

        car  = _find_entity(spec, "car")
        wall = _find_entity(spec, "wall")

        x_dist = abs(car.state.position.x - wall.state.position.x)

        assert x_dist == pytest.approx(20.0, rel=0.1), (
            f"Expected x-distance ≈ 20 m between car and wall, "
            f"got {x_dist:.3f} m  "
            f"(car.x={car.state.position.x:.3f}, wall.x={wall.state.position.x:.3f})"
        )

    # ── 2. Above ─────────────────────────────────────────────────────────────

    def test_above_relationship(self, parser):
        """
        'A ball above a platform.'
        The ball's y-coordinate must exceed the platform's y-coordinate.
        """
        spec = parser.parse("A ball above a platform.")

        ball     = _find_entity(spec, "ball")
        platform = _find_entity(spec, "platform")

        assert ball.state.position.y > platform.state.position.y, (
            f"Expected ball.y > platform.y, "
            f"got ball.y={ball.state.position.y:.3f}, "
            f"platform.y={platform.state.position.y:.3f}"
        )

    # ── 3. Below ─────────────────────────────────────────────────────────────

    def test_below_relationship(self, parser):
        """
        'A cube below a shelf.'
        The cube's y-coordinate must be less than the shelf's y-coordinate.
        """
        spec = parser.parse("A cube below a shelf.")

        cube  = _find_entity(spec, "cube")
        shelf = _find_entity(spec, "shelf")

        assert cube.state.position.y < shelf.state.position.y, (
            f"Expected cube.y < shelf.y, "
            f"got cube.y={cube.state.position.y:.3f}, "
            f"shelf.y={shelf.state.position.y:.3f}"
        )

    # ── 4. On (surface contact) ───────────────────────────────────────────────

    def test_on_relationship(self, parser):
        """
        'A box on a table.'
        The box's y-coordinate must exceed the table's y-coordinate,
        indicating the box rests on top of the table surface.
        """
        spec = parser.parse("A box on a table.")

        box   = _find_entity(spec, "box")
        table = _find_entity(spec, "table")

        assert box.state.position.y > table.state.position.y, (
            f"Expected box.y > table.y (box rests on table surface), "
            f"got box.y={box.state.position.y:.3f}, "
            f"table.y={table.state.position.y:.3f}"
        )

    # ── 5. Left of ───────────────────────────────────────────────────────────

    def test_left_of(self, parser):
        """
        'A cube left of a sphere.'
        The cube's x-coordinate must be less than the sphere's x-coordinate.
        (x = East / right direction)
        """
        spec = parser.parse("A cube left of a sphere.")

        cube   = _find_entity(spec, "cube")
        sphere = _find_entity(spec, "sphere")

        assert cube.state.position.x < sphere.state.position.x, (
            f"Expected cube.x < sphere.x (cube is to the left), "
            f"got cube.x={cube.state.position.x:.3f}, "
            f"sphere.x={sphere.state.position.x:.3f}"
        )

    # ── 6. Right of ──────────────────────────────────────────────────────────

    def test_right_of(self, parser):
        """
        'A cube right of a sphere.'
        The cube's x-coordinate must exceed the sphere's x-coordinate.
        """
        spec = parser.parse("A cube right of a sphere.")

        cube   = _find_entity(spec, "cube")
        sphere = _find_entity(spec, "sphere")

        assert cube.state.position.x > sphere.state.position.x, (
            f"Expected cube.x > sphere.x (cube is to the right), "
            f"got cube.x={cube.state.position.x:.3f}, "
            f"sphere.x={sphere.state.position.x:.3f}"
        )

    # ── 7. Behind ────────────────────────────────────────────────────────────

    def test_behind(self, parser):
        """
        'A car behind a truck.'
        'Behind' maps to a smaller x-coordinate (further west / rear).
        """
        spec = parser.parse("A car behind a truck.")

        car   = _find_entity(spec, "car")
        truck = _find_entity(spec, "truck")

        assert car.state.position.x < truck.state.position.x, (
            f"Expected car.x < truck.x (car is behind truck), "
            f"got car.x={car.state.position.x:.3f}, "
            f"truck.x={truck.state.position.x:.3f}"
        )

    # ── 8. In front of ───────────────────────────────────────────────────────

    def test_in_front_of(self, parser):
        """
        'A car in front of a truck.'
        'In front of' maps to a larger x-coordinate (further east / forward).
        """
        spec = parser.parse("A car in front of a truck.")

        car   = _find_entity(spec, "car")
        truck = _find_entity(spec, "truck")

        assert car.state.position.x > truck.state.position.x, (
            f"Expected car.x > truck.x (car is in front of truck), "
            f"got car.x={car.state.position.x:.3f}, "
            f"truck.x={truck.state.position.x:.3f}"
        )

    # ── 9. Near (proximity) ───────────────────────────────────────────────────

    def test_near_relationship(self, parser):
        """
        'A ball near a wall.'
        The 3-D Euclidean distance between the ball and the wall must be < 5 m.
        """
        spec = parser.parse("A ball near a wall.")

        ball = _find_entity(spec, "ball")
        wall = _find_entity(spec, "wall")

        dist = _euclidean_distance(ball.state.position, wall.state.position)

        assert dist < 5.0, (
            f"Expected distance < 5 m for 'near' relationship, "
            f"got {dist:.3f} m  "
            f"(ball={ball.state.position.x:.2f},{ball.state.position.y:.2f},{ball.state.position.z:.2f}  "
            f"wall={wall.state.position.x:.2f},{wall.state.position.y:.2f},{wall.state.position.z:.2f})"
        )

    # ── 10. Multiple simultaneous relationships ────────────────────────────────

    def test_multiple_relationships(self, parser):
        """
        'A ball above a platform near a wall.'
        Two constraints must both hold:
          • ball.y > platform.y   (above)
          • distance(ball, wall) < 5 m  (near)
        """
        spec = parser.parse("A ball above a platform near a wall.")

        ball     = _find_entity(spec, "ball")
        platform = _find_entity(spec, "platform")
        wall     = _find_entity(spec, "wall")

        assert ball.state.position.y > platform.state.position.y, (
            f"Expected ball.y > platform.y (ball above platform), "
            f"got ball.y={ball.state.position.y:.3f}, "
            f"platform.y={platform.state.position.y:.3f}"
        )

        dist_ball_wall = _euclidean_distance(ball.state.position, wall.state.position)
        assert dist_ball_wall < 5.0, (
            f"Expected ball–wall distance < 5 m (ball near wall), "
            f"got {dist_ball_wall:.3f} m"
        )

    # ── 11. Entity count (no duplicates) ──────────────────────────────────────

    def test_relationships_preserve_entity_count(self, parser):
        """
        'A car behind a truck.'
        Exactly two dynamic entities must be created — one car and one truck —
        with no spurious duplicates.
        """
        spec = parser.parse("A car behind a truck.")

        dynamic = [e for e in spec.entities if not e.is_static]

        assert len(dynamic) == 2, (
            f"Expected exactly 2 dynamic entities (car + truck), "
            f"got {len(dynamic)}: {[e.label for e in dynamic]}"
        )

        labels = [e.label.lower() for e in dynamic]
        has_car   = any("car"   in lbl for lbl in labels)
        has_truck = any("truck" in lbl for lbl in labels)

        assert has_car, (
            f"Expected a 'car' entity among dynamic entities, "
            f"found: {[e.label for e in dynamic]}"
        )
        assert has_truck, (
            f"Expected a 'truck' entity among dynamic entities, "
            f"found: {[e.label for e in dynamic]}"
        )

    # ── 12. Return type sanity ────────────────────────────────────────────────

    def test_parser_returns_worldspec(self, parser):
        """
        parser.parse() must return a non-None WorldSpec with at least
        two entities for a two-body scene.
        """
        spec = parser.parse("A ball above a platform.")

        assert spec is not None, "parser.parse() returned None"
        assert isinstance(spec, WorldSpec), (
            f"Expected WorldSpec instance, got {type(spec).__name__}"
        )
        assert len(spec.entities) >= 2, (
            f"Expected at least 2 entities (ball + platform), "
            f"got {len(spec.entities)}: {[e.label for e in spec.entities]}"
        )

    def test_entity_ids_unique(self, parser):
        """
        Entity IDs should all be unique.
        """

        spec = parser.parse(
            "A car behind a truck."
        )

        ids = [e.id for e in spec.entities]

        assert len(ids) == len(set(ids)), (
            f"Duplicate ids found: {ids}"
        )

    def test_unknown_relationship(self, parser):
        """
        Unknown relationship words should not crash parser.
        """

        spec = parser.parse(
            "A ball mysteriously adjacent to a platform."
        )

        assert spec is not None
        assert len(spec.entities) >= 2

    def test_unknown_entity_names(self, parser):
        """
        Parser should survive unknown nouns.
        """

        spec = parser.parse(
            "A thingamajig above a platform."
        )

        assert spec is not None
        assert len(spec.entities) >= 1

    def test_empty_prompt(self, parser):
        """
        Empty prompts should not crash.
        """

        spec = parser.parse("")

        assert spec is not None
