import os
import sys
import json
import copy

import pytest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _candidate in (
    _THIS_DIR,
    os.path.join(_THIS_DIR, ".."),
    os.path.join(_THIS_DIR, "..", "src"),
    os.path.join(_THIS_DIR, "..", "physworldlm"),
):
    _candidate = os.path.abspath(_candidate)
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

from world_spec import (
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
from compiler.worldspec_builder import (
    BuildStatus,
    Severity,
    SpecValidationError,
    ValidationPolicy,
    WorldSpecBuilder,
    WorldSpecBuilderConfig,
    WorldSpecBuilderError,
)
from compiler.scene_compiler import (
    CompilationStatus,
    CompilerConfig,
    NodeType,
    SceneCompiler,
    SceneGraph,
    ValidationMode,
)


# ══════════════════════════════════════════════════════════════════════
# Entity / WorldSpec construction helpers
# ══════════════════════════════════════════════════════════════════════

def make_entity(
    id,
    label=None,
    entity_type="vehicle",
    is_static=False,
    mass=1000.0,
    material="steel",
    restitution=0.5,
    friction=0.5,
    position=(0.0, 0.0, 0.0),
    velocity=(0.0, 0.0, 0.0),
    orientation=(0.0, 0.0, 0.0),
    bbox=(2.0, 2.0, 4.0),
    tags=None,
    constraints=None,
    forces=None,
):
    return Entity(
        id=id,
        label=label or id,
        entity_type=entity_type,
        is_static=is_static,
        mass=mass,
        material=material,
        restitution=restitution,
        friction=friction,
        bounding_box=BoundingBox(width=bbox[0], height=bbox[1], depth=bbox[2]),
        state=PhysicsState(
            position=Vec3(*position),
            velocity=Vec3(*velocity),
            acceleration=Vec3(0.0, 0.0, 0.0),
            orientation=Vec3(*orientation),
            angular_vel=Vec3(0.0, 0.0, 0.0),
        ),
        forces=forces or [],
        constraints=constraints or [],
        tags=tags or [],
    )


def make_environment(terrain_type="flat", weather="clear"):
    return Environment(
        gravity=Vec3(0.0, -9.81, 0.0),
        temperature_K=293.15,
        pressure_Pa=101325.0,
        air_density=1.225,
        wind=Wind(speed=2.0, direction=0.5),
        terrain_type=terrain_type,
        friction_global=0.5,
        time_of_day="day",
        weather=weather,
    )


def make_world_spec(
    scene_id,
    description,
    entities=None,
    environment=None,
    interactions=None,
    simulation_graph=None,
):
    return WorldSpec(
        scene_id=scene_id,
        description=description,
        entities=entities or [],
        environment=environment or make_environment(),
        interactions=interactions or [],
        simulation_graph=simulation_graph or SimulationGraph(dt=0.01, duration=5.0, integrator="rk4"),
    )


# ══════════════════════════════════════════════════════════════════════
# Scenario fixtures
# ══════════════════════════════════════════════════════════════════════

@pytest.fixture
def single_car_world():
    car = make_entity(
        id="car_1",
        entity_type="vehicle",
        is_static=False,
        mass=1500.0,
        material="steel",
        position=(0.0, 0.0, 0.0),
        bbox=(2.0, 1.5, 4.5),
        tags=["asset:vehicles/car.usd"],
    )
    return make_world_spec("scene_single_car", "a single car sits on flat ground", entities=[car])


@pytest.fixture
def two_vehicles_world():
    car_a = make_entity(id="car_a", entity_type="vehicle", mass=1500.0, material="steel", position=(-5.0, 0.0, 0.0))
    car_b = make_entity(id="car_b", entity_type="vehicle", mass=1800.0, material="steel", position=(5.0, 0.0, 0.0))
    return make_world_spec(
        "scene_two_vehicles",
        "two vehicles parked apart",
        entities=[car_a, car_b],
    )


@pytest.fixture
def spring_system_world():
    chassis = make_entity(
        id="chassis",
        entity_type="structure",
        is_static=True,
        mass=0.0,
        material="steel",
        position=(0.0, 1.0, 0.0),
        bbox=(2.0, 0.5, 4.0),
    )
    wheel = make_entity(
        id="wheel_1",
        entity_type="component",
        is_static=False,
        mass=20.0,
        material="rubber",
        position=(0.0, 0.0, 0.0),
        bbox=(0.6, 0.6, 0.3),
    )
    joint = Interaction(
        type="joint",
        entity_a="wheel_1",
        entity_b="chassis",
        parameters={"stiffness": 5000.0, "damping": 50.0},
    )
    return make_world_spec(
        "scene_spring_system",
        "a wheel mounted to a chassis via a spring joint",
        entities=[chassis, wheel],
        interactions=[joint],
    )


@pytest.fixture
def collision_scenario_world():
    ball_a = make_entity(id="ball_a", entity_type="projectile", mass=5.0, material="rubber",
                          position=(-10.0, 1.0, 0.0), velocity=(10.0, 0.0, 0.0), bbox=(0.5, 0.5, 0.5))
    ball_b = make_entity(id="ball_b", entity_type="projectile", mass=5.0, material="rubber",
                          position=(10.0, 1.0, 0.0), velocity=(-10.0, 0.0, 0.0), bbox=(0.5, 0.5, 0.5))
    collision = Interaction(
        type="collision",
        entity_a="ball_a",
        entity_b="ball_b",
        parameters={"restitution_override": 0.9},
    )
    return make_world_spec(
        "scene_collision",
        "two balls on a collision course",
        entities=[ball_a, ball_b],
        interactions=[collision],
    )


@pytest.fixture
def military_scene_world():
    tank = make_entity(id="tank_1", entity_type="vehicle", mass=54000.0, material="steel",
                        position=(0.0, 0.0, 0.0), bbox=(3.5, 2.5, 7.0),
                        tags=["asset:vehicles/tank.usd", "military"])
    soldier = make_entity(id="soldier_1", entity_type="agent", mass=90.0, material="flesh",
                           position=(5.0, 0.0, 0.0), bbox=(0.6, 1.8, 0.4), tags=["military"])
    bunker = make_entity(id="bunker_1", entity_type="structure", is_static=True, mass=0.0,
                          material="concrete", position=(20.0, 0.0, 0.0), bbox=(6.0, 3.0, 6.0),
                          tags=["military"])
    radar = make_entity(id="radar_1", entity_type="structure", is_static=True, mass=0.0,
                         material="steel", position=(-20.0, 0.0, 0.0), bbox=(2.0, 4.0, 2.0),
                         tags=["military", "sensor_platform"])
    mount = Interaction(type="mount", entity_a="radar_1", entity_b="bunker_1", parameters={})
    contact = Interaction(type="contact", entity_a="tank_1", entity_b="environment", parameters={})
    return make_world_spec(
        "scene_military",
        "a tank, a soldier, a bunker and a radar mounted to the bunker",
        entities=[tank, soldier, bunker, radar],
        interactions=[mount, contact],
        environment=make_environment(terrain_type="urban", weather="fog"),
    )


@pytest.fixture
def empty_scene_world():
    return make_world_spec("scene_empty", "an empty world with no entities")


@pytest.fixture(params=[
    "single_car_world",
    "two_vehicles_world",
    "spring_system_world",
    "collision_scenario_world",
    "military_scene_world",
    "empty_scene_world",
])
def any_scenario_world(request):
    return request.getfixturevalue(request.param)


# ══════════════════════════════════════════════════════════════════════
# Helpers to run the two pipeline stages
# ══════════════════════════════════════════════════════════════════════

import tempfile
from pathlib import Path

def run_worldspec_builder(
    world_spec,
    policy=ValidationPolicy.STRICT,
    asset_search_paths=None,
    deterministic=True,
):
    if asset_search_paths is None:
        temp_root = Path(tempfile.mkdtemp())

        # Automatically create placeholder assets referenced by this WorldSpec
        for entity in world_spec.entities:
            for tag in getattr(entity, "tags", []):
                if tag.startswith("asset:"):
                    rel = Path(tag[len("asset:"):])
                    asset_path = temp_root / rel
                    asset_path.parent.mkdir(parents=True, exist_ok=True)
                    asset_path.touch(exist_ok=True)

        asset_search_paths = [temp_root]

    config = WorldSpecBuilderConfig(
        validation_policy=policy,
        deterministic=deterministic,
        asset_search_paths=asset_search_paths,
    )

    builder = WorldSpecBuilder(config)
    with builder:
        return builder.build(world_spec)

def run_scene_compiler(world_spec, output_path, validation_mode=ValidationMode.STRICT):
    config = CompilerConfig(validation_mode=validation_mode, overwrite_existing=True)
    compiler = SceneCompiler(config=config)
    report = compiler.compile(world_spec, output_path=output_path)
    return report


def find_node(scene_graph, node_type=None, name=None):
    for node in scene_graph.root.walk():
        if node_type is not None and node.node_type is not node_type:
            continue
        if name is not None and node.name != name:
            continue
        return node
    return None


def find_all_nodes(scene_graph, node_type):
    return [n for n in scene_graph.root.walk() if n.node_type is node_type]


# ══════════════════════════════════════════════════════════════════════
# 1. Prompt/WorldSpec -> SceneGraph
# ══════════════════════════════════════════════════════════════════════

class TestWorldSpecToSceneGraph:

    def test_builder_succeeds_for_single_car(self, single_car_world):
        report = run_worldspec_builder(single_car_world)
        assert report.status is BuildStatus.SUCCESS
        assert report.success is True
        assert report.scene_graph is not None

    def test_scene_graph_exists_and_has_entities(self, single_car_world):
        report = run_worldspec_builder(single_car_world)
        entity_nodes = find_all_nodes(report.scene_graph, NodeType.ENTITY)
        assert len(entity_nodes) == 1
        assert entity_nodes[0].metadata["world_spec_id"] == "car_1"

    def test_validation_passes_for_well_formed_spec(self, two_vehicles_world):
        report = run_worldspec_builder(two_vehicles_world)
        assert report.errors() == []
        assert report.status in (BuildStatus.SUCCESS, BuildStatus.SUCCESS_WITH_WARNINGS)


# ══════════════════════════════════════════════════════════════════════
# 2. SceneGraph -> USD
# ══════════════════════════════════════════════════════════════════════

class TestSceneGraphToUSD:

    def test_scene_compiler_succeeds(self, single_car_world, tmp_path):
        output_path = tmp_path / "single_car.usda"
        report = run_scene_compiler(single_car_world, output_path)
        assert report.success is True
        assert report.status in (CompilationStatus.SUCCESS, CompilationStatus.SUCCESS_WITH_WARNINGS)

    def test_usd_stage_file_created(self, single_car_world, tmp_path):
        output_path = tmp_path / "single_car.usda"
        report = run_scene_compiler(single_car_world, output_path)
        assert output_path.exists()
        assert output_path.stat().st_size > 0
        assert report.statistics.exported_file_size_bytes > 0

    def test_no_exceptions_raised_during_compile(self, military_scene_world, tmp_path):
        output_path = tmp_path / "military.usda"
        try:
            report = run_scene_compiler(military_scene_world, output_path)
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"SceneCompiler.compile() raised an exception: {exc}")
        assert report.success is True


# ══════════════════════════════════════════════════════════════════════
# 3. Scene hierarchy
# ══════════════════════════════════════════════════════════════════════

class TestSceneHierarchy:

    def test_world_root_path(self, single_car_world):
        report = run_worldspec_builder(single_car_world)
        assert report.scene_graph.root.path == "/World"
        assert report.scene_graph.root.node_type is NodeType.WORLD

    def test_environment_path_exists(self, single_car_world):
        report = run_worldspec_builder(single_car_world)
        env_node = find_node(report.scene_graph, node_type=NodeType.ENVIRONMENT)
        assert env_node is not None
        assert env_node.path == "/World/Environment"

    def test_entities_path_exists(self, single_car_world):
        report = run_worldspec_builder(single_car_world)
        entities_node = find_node(report.scene_graph, node_type=NodeType.ENTITIES_GROUP)
        assert entities_node is not None
        assert entities_node.path == "/World/Entities"

    def test_hierarchy_also_holds_via_scene_compiler(self, single_car_world, tmp_path):
        report = run_scene_compiler(single_car_world, tmp_path / "scene.usda")
        assert report.scene_graph.root.path == "/World"
        assert find_node(report.scene_graph, node_type=NodeType.ENVIRONMENT).path == "/World/Environment"
        assert find_node(report.scene_graph, node_type=NodeType.ENTITIES_GROUP).path == "/World/Entities"


# ══════════════════════════════════════════════════════════════════════
# 4. Entity hierarchy (parent/child)
# ══════════════════════════════════════════════════════════════════════

class TestEntityHierarchy:

    def test_joint_reparents_child_under_parent(self, spring_system_world):
        report = run_worldspec_builder(spring_system_world)
        assert report.success is True

        wheel_node = None
        chassis_node = None
        for node in report.scene_graph.root.walk():
            if node.node_type is NodeType.ENTITY and node.metadata.get("world_spec_id") == "wheel_1":
                wheel_node = node
            if node.node_type is NodeType.ENTITY and node.metadata.get("world_spec_id") == "chassis":
                chassis_node = node

        assert wheel_node is not None
        assert chassis_node is not None
        assert wheel_node.parent is chassis_node
        assert wheel_node in chassis_node.children

    def test_mount_reparents_radar_under_bunker(self, military_scene_world):
        report = run_worldspec_builder(military_scene_world)
        assert report.success is True

        radar_node = None
        bunker_node = None
        for node in report.scene_graph.root.walk():
            if node.node_type is NodeType.ENTITY and node.metadata.get("world_spec_id") == "radar_1":
                radar_node = node
            if node.node_type is NodeType.ENTITY and node.metadata.get("world_spec_id") == "bunker_1":
                bunker_node = node

        assert radar_node is not None
        assert bunker_node is not None
        assert radar_node.parent is bunker_node

    def test_no_spurious_reparenting_without_joint_or_mount(self, two_vehicles_world):
        report = run_worldspec_builder(two_vehicles_world)
        entities_node = find_node(report.scene_graph, node_type=NodeType.ENTITIES_GROUP)
        entity_children_ids = {
            n.metadata.get("world_spec_id") for n in entities_node.children if n.node_type is NodeType.ENTITY
        }
        assert entity_children_ids == {"car_a", "car_b"}


# ══════════════════════════════════════════════════════════════════════
# 5. Physics
# ══════════════════════════════════════════════════════════════════════

class TestPhysics:

    def test_every_dynamic_entity_has_physics_body(self, military_scene_world):
        report = run_worldspec_builder(military_scene_world)
        assert report.success is True

        dynamic_ids = {e.id for e in military_scene_world.dynamic_entities()}
        for node in report.scene_graph.root.walk():
            if node.node_type is NodeType.ENTITY and node.metadata.get("world_spec_id") in dynamic_ids:
                assert "physics_ref" in node.components

        physics_bodies = find_all_nodes(report.scene_graph, NodeType.PHYSICS_BODY)
        assert len(physics_bodies) == len(military_scene_world.entities)

    def test_physics_body_mass_matches_entity(self, single_car_world):
        report = run_worldspec_builder(single_car_world)
        physics_bodies = find_all_nodes(report.scene_graph, NodeType.PHYSICS_BODY)
        assert len(physics_bodies) == 1
        body = physics_bodies[0].components["physics_body"]
        assert body["mass_kg"] == pytest.approx(1500.0)
        assert body["body_type"] == "dynamic"

    def test_static_entity_marked_static(self, spring_system_world):
        report = run_worldspec_builder(spring_system_world)
        chassis_physics = None
        for node in find_all_nodes(report.scene_graph, NodeType.PHYSICS_BODY):
            if node.name.startswith("chassis"):
                chassis_physics = node
        assert chassis_physics is not None
        assert chassis_physics.components["physics_body"]["body_type"] == "static"

    def test_zero_dynamic_entities_yields_zero_physics_error(self, empty_scene_world):
        report = run_worldspec_builder(empty_scene_world)
        assert report.success is True
        assert find_all_nodes(report.scene_graph, NodeType.PHYSICS_BODY) == []


# ══════════════════════════════════════════════════════════════════════
# 6. Materials
# ══════════════════════════════════════════════════════════════════════

class TestMaterials:

    def test_materials_attached_to_entities(self, two_vehicles_world):
        report = run_worldspec_builder(two_vehicles_world)
        for node in report.scene_graph.root.walk():
            if node.node_type is NodeType.ENTITY:
                assert "material_ref" in node.components

    def test_shared_material_deduplicated(self, two_vehicles_world):
        report = run_worldspec_builder(two_vehicles_world)
        material_nodes = find_all_nodes(report.scene_graph, NodeType.MATERIAL)
        steel_nodes = [m for m in material_nodes if m.name == "steel"]
        assert len(steel_nodes) == 1

        refs = set()
        for node in report.scene_graph.root.walk():
            if node.node_type is NodeType.ENTITY:
                refs.add(node.components["material_ref"])
        assert steel_nodes[0].node_uuid in refs

    def test_material_defaults_used_not_entity_overrides(self, single_car_world):
        # material node should carry canonical MATERIAL_DEFAULTS values,
        # not the per-entity restitution/friction override.
        from world_spec import MATERIAL_DEFAULTS
        report = run_worldspec_builder(single_car_world)
        steel_node = find_node(report.scene_graph, node_type=NodeType.MATERIAL, name="steel")
        assert steel_node is not None
        mat = steel_node.components["material"]
        assert mat["density"] == MATERIAL_DEFAULTS["steel"]["density"]
        assert mat["restitution"] == MATERIAL_DEFAULTS["steel"]["restitution"]
        assert mat["friction"] == MATERIAL_DEFAULTS["steel"]["friction"]

    def test_unknown_material_falls_back_to_generic_in_permissive_mode(self):
        entity = make_entity(id="mystery", material="unobtainium")
        spec = make_world_spec("scene_unknown_material", "entity with unknown material", entities=[entity])
        report = run_worldspec_builder(spec, policy=ValidationPolicy.PERMISSIVE)
        assert report.success is True
        generic_node = find_node(report.scene_graph, node_type=NodeType.MATERIAL, name="generic")
        assert generic_node is not None


# ══════════════════════════════════════════════════════════════════════
# 7. Terrain
# ══════════════════════════════════════════════════════════════════════

class TestTerrain:

    def test_terrain_node_exists_for_defined_environment(self, military_scene_world):
        report = run_worldspec_builder(military_scene_world)
        terrain_node = find_node(report.scene_graph, node_type=NodeType.TERRAIN)
        assert terrain_node is not None
        assert terrain_node.metadata["terrain_type"] == "urban"

    def test_terrain_validated_flag_set_for_known_type(self, single_car_world):
        report = run_worldspec_builder(single_car_world)
        terrain_node = find_node(report.scene_graph, node_type=NodeType.TERRAIN)
        assert terrain_node.metadata["validated"] is True

    def test_terrain_exists_even_for_empty_scene(self, empty_scene_world):
        report = run_worldspec_builder(empty_scene_world)
        terrain_node = find_node(report.scene_graph, node_type=NodeType.TERRAIN)
        assert terrain_node is not None
        assert terrain_node.metadata["terrain_type"] == "flat"


# ══════════════════════════════════════════════════════════════════════
# 8. Assets
# ══════════════════════════════════════════════════════════════════════

class TestAssets:

    def test_asset_reference_resolved_via_search_path(self, tmp_path):
        asset_file = tmp_path / "car.usd"
        asset_file.write_text("dummy usd content")

        car = make_entity(id="asset_car", tags=["asset:car.usd"])
        spec = make_world_spec("scene_asset_car", "a car referencing an external asset", entities=[car])

        report = run_worldspec_builder(spec, asset_search_paths=[tmp_path])
        assert report.success is True

        entity_node = find_node(report.scene_graph, node_type=NodeType.ENTITY, name="asset_car")
        assert entity_node is not None
        assert "assets" in entity_node.components
        assert any(str(asset_file) == p or p.endswith("car.usd") for p in entity_node.components["assets"])

    def test_scene_compiler_preserves_asset_tag_permissively(self, single_car_world, tmp_path):
        report = run_scene_compiler(single_car_world, tmp_path / "scene.usda")
        entity_node = find_node(report.scene_graph, node_type=NodeType.ENTITY, name="car_1")
        assert "assets" in entity_node.components
        assert len(entity_node.components["assets"]) == 1


# ══════════════════════════════════════════════════════════════════════
# 9. Serialization
# ══════════════════════════════════════════════════════════════════════

class TestSerialization:

    def test_scene_graph_survives_json_roundtrip(self, military_scene_world):
        report = run_worldspec_builder(military_scene_world)
        original_dict = report.scene_graph.to_dict()

        serialized = json.dumps(original_dict)
        deserialized = json.loads(serialized)

        assert deserialized == original_dict
        assert deserialized["name"] == "World"
        assert deserialized["type"] == "WORLD"

    def test_serialized_node_counts_are_stable(self, military_scene_world):
        report = run_worldspec_builder(military_scene_world)
        d = report.scene_graph.to_dict()

        def count_nodes(node_dict):
            return 1 + sum(count_nodes(c) for c in node_dict["children"])

        assert count_nodes(d) == report.scene_graph.node_count()

    def test_deep_copy_of_serialized_dict_is_independent(self, single_car_world):
        report = run_worldspec_builder(single_car_world)
        d1 = report.scene_graph.to_dict()
        d2 = copy.deepcopy(d1)
        d2["name"] = "Mutated"
        assert d1["name"] == "World"
        assert d2["name"] == "Mutated"


# ══════════════════════════════════════════════════════════════════════
# 10. Compiler report
# ══════════════════════════════════════════════════════════════════════

class TestCompilerReport:

    def test_report_success_true_for_valid_scene(self, two_vehicles_world, tmp_path):
        report = run_scene_compiler(two_vehicles_world, tmp_path / "scene.usda")
        assert report.success is True
        assert report.status is not CompilationStatus.FAILED

    def test_report_diagnostics_available(self, two_vehicles_world, tmp_path):
        report = run_scene_compiler(two_vehicles_world, tmp_path / "scene.usda")
        assert isinstance(report.diagnostics, list)
        assert len(report.diagnostics) > 0
        assert all(hasattr(d, "severity") for d in report.diagnostics)

    def test_report_statistics_populated(self, two_vehicles_world, tmp_path):
        report = run_scene_compiler(two_vehicles_world, tmp_path / "scene.usda")
        stats = report.statistics
        assert stats.entity_count == 2
        assert stats.material_count >= 1
        assert stats.compilation_time_s >= 0.0
        assert stats.success is True
        assert stats.exported_file_size_bytes > 0
        assert len(stats.stage_durations_s) > 0

    def test_builder_report_statistics_populated(self, military_scene_world):
        report = run_worldspec_builder(military_scene_world)
        stats = report.statistics
        assert stats.entity_count == 4
        assert stats.resolved_entity_count == 4
        assert stats.relationship_count >= 2
        assert stats.build_time_s >= 0.0
        assert stats.success is True
        assert len(stats.phase_durations_s) > 0


# ══════════════════════════════════════════════════════════════════════
# 11. Regression tests
# ══════════════════════════════════════════════════════════════════════

class TestRegressions:

    def test_duplicate_entity_ids_do_not_corrupt_graph(self):
        e1 = make_entity(id="dup", mass=100.0)
        e2 = make_entity(id="dup", mass=200.0)
        spec = make_world_spec("scene_dup_ids", "duplicate entity ids", entities=[e1, e2])

        report = run_worldspec_builder(spec, policy=ValidationPolicy.PERMISSIVE)

        entity_nodes = find_all_nodes(report.scene_graph, NodeType.ENTITY)
        assert len(entity_nodes) == 1
        error_messages = [d.message for d in report.diagnostics if d.severity in (Severity.ERROR, Severity.CRITICAL)]
        assert any("Duplicate entity id" in m for m in error_messages)

    def test_duplicate_entity_ids_fail_strict_build(self):
        e1 = make_entity(id="dup", mass=100.0)
        e2 = make_entity(id="dup", mass=200.0)
        spec = make_world_spec("scene_dup_ids_strict", "duplicate entity ids strict", entities=[e1, e2])

        report = run_worldspec_builder(spec, policy=ValidationPolicy.STRICT)
        assert report.status is BuildStatus.FAILED
        assert report.success is False

    def test_missing_parent_reference_fails_strict_build(self):
        child = make_entity(id="child_1", constraints=["nonexistent_parent"])
        spec = make_world_spec("scene_missing_parent", "entity constrained to a missing parent", entities=[child])

        report = run_worldspec_builder(spec, policy=ValidationPolicy.STRICT)
        assert report.status is BuildStatus.FAILED
        assert any("unknown entity" in d.message for d in report.errors())

    def test_missing_parent_reference_logged_in_permissive_mode(self):
        child = make_entity(id="child_2", constraints=["nonexistent_parent"])
        spec = make_world_spec("scene_missing_parent_permissive", "permissive missing parent", entities=[child])

        report = run_worldspec_builder(spec, policy=ValidationPolicy.PERMISSIVE)
        assert report.success is True
        assert any("unknown entity" in d.message for d in report.diagnostics)

    def test_invalid_transform_zero_bounding_box_fails(self):
        bad = make_entity(id="flat_entity", bbox=(0.0, 1.0, 1.0))
        spec = make_world_spec("scene_invalid_transform", "zero-width bounding box", entities=[bad])

        report = run_worldspec_builder(spec, policy=ValidationPolicy.STRICT)
        assert report.status is BuildStatus.FAILED
        assert any("bounding_box.width" in d.message for d in report.errors())

    def test_unresolved_asset_fails_strict_build(self):
        entity = make_entity(id="asset_missing", tags=["asset:does/not/exist.usd"])
        spec = make_world_spec("scene_unresolved_asset", "unresolved asset reference", entities=[entity])

        report = run_worldspec_builder(spec, policy=ValidationPolicy.STRICT, asset_search_paths=[])
        assert report.status is BuildStatus.FAILED
        assert any("Could not resolve asset reference" in d.message for d in report.errors())

    def test_unresolved_asset_logged_as_warning_in_permissive_mode(self):
        entity = make_entity(id="asset_missing_permissive", tags=["asset:does/not/exist.usd"])
        spec = make_world_spec("scene_unresolved_asset_permissive", "permissive unresolved asset", entities=[entity])

        report = run_worldspec_builder(spec, policy=ValidationPolicy.PERMISSIVE, asset_search_paths=[])
        assert report.success is True
        assert any("Could not resolve asset reference" in d.message for d in report.diagnostics)

    def test_invalid_physics_body_negative_mass_fails(self):
        bad = make_entity(id="ghost_car", is_static=False, mass=-5.0)
        spec = make_world_spec("scene_invalid_physics", "negative mass dynamic entity", entities=[bad])

        report = run_worldspec_builder(spec, policy=ValidationPolicy.STRICT)
        assert report.status is BuildStatus.FAILED

    def test_invalid_physics_body_zero_mass_fails_even_in_permissive_mode(self):
        # resolve_physics() treats non-positive mass on a dynamic entity
        # as always fatal, regardless of validation_policy.
        bad = make_entity(id="ghost_car_2", is_static=False, mass=0.0)
        spec = make_world_spec("scene_invalid_physics_permissive", "zero mass dynamic entity", entities=[bad])

        report = run_worldspec_builder(spec, policy=ValidationPolicy.PERMISSIVE)
        assert report.status is BuildStatus.FAILED

    def test_orphan_collider_constraint_does_not_crash_permissive_build(self):
        orphan = make_entity(id="orphan_collider", constraints=["ghost_reference"])
        spec = make_world_spec("scene_orphan_collider", "entity with an orphan collider constraint", entities=[orphan])

        report = run_worldspec_builder(spec, policy=ValidationPolicy.PERMISSIVE)
        assert report.success is True
        entity_node = find_node(report.scene_graph, node_type=NodeType.ENTITY, name="orphan_collider")
        assert entity_node is not None
        assert "physics_ref" in entity_node.components

    def test_dangling_interaction_edge_fails_strict_build(self):
        entity = make_entity(id="lonely_entity")
        dangling = Interaction(type="collision", entity_a="lonely_entity", entity_b="ghost_entity")
        spec = make_world_spec(
            "scene_dangling_edge",
            "interaction referencing a non-existent entity_b",
            entities=[entity],
            interactions=[dangling],
        )

        report = run_worldspec_builder(spec, policy=ValidationPolicy.STRICT)
        assert report.status is BuildStatus.FAILED
        assert any("unknown entity_b" in d.message for d in report.errors())

    def test_dangling_interaction_edge_skipped_in_permissive_mode(self):
        entity = make_entity(id="lonely_entity_2")
        dangling = Interaction(type="collision", entity_a="lonely_entity_2", entity_b="ghost_entity_2")
        spec = make_world_spec(
            "scene_dangling_edge_permissive",
            "permissive dangling interaction edge",
            entities=[entity],
            interactions=[dangling],
        )

        report = run_worldspec_builder(spec, policy=ValidationPolicy.PERMISSIVE)
        assert report.success is True
        entity_node = find_node(report.scene_graph, node_type=NodeType.ENTITY, name="lonely_entity_2")
        assert entity_node.components.get("relationships", []) == []

    def test_circular_dependency_detected(self):
        a = make_entity(id="node_a", constraints=["node_b"])
        b = make_entity(id="node_b", constraints=["node_a"])
        spec = make_world_spec("scene_circular", "two entities with circular constraints", entities=[a, b])

        report = run_worldspec_builder(spec, policy=ValidationPolicy.STRICT)
        assert report.status is BuildStatus.FAILED
        assert any("Circular dependency" in d.message for d in report.errors())

    def test_scene_compiler_raises_wrapped_dependency_error_when_out_of_order(self, single_car_world, tmp_path):
        # Directly invoking an internal stage out of order must not silently
        # succeed; SceneCompiler's public API always goes through compile(),
        # which enforces stage ordering via assert_dependencies_met().
        report = run_scene_compiler(single_car_world, tmp_path / "scene.usda")
        assert report.success is True
        assert set(report.statistics.stage_durations_s.keys()).issuperset(
            {"Validate World Spec", "Init Scene Graph", "Export Usd", "Produce Report"}
        )


# ══════════════════════════════════════════════════════════════════════
# 12. Multiple scenarios — full pipeline sanity across all fixtures
# ══════════════════════════════════════════════════════════════════════

class TestMultipleScenarios:

    def test_worldspec_builder_handles_every_scenario(self, any_scenario_world):
        report = run_worldspec_builder(any_scenario_world)
        assert report.status in (BuildStatus.SUCCESS, BuildStatus.SUCCESS_WITH_WARNINGS)
        assert report.scene_graph is not None
        assert report.scene_graph.root.path == "/World"

    def test_scene_compiler_handles_every_scenario(self, any_scenario_world, tmp_path):
        output_path = tmp_path / f"{any_scenario_world.scene_id}.usda"
        report = run_scene_compiler(any_scenario_world, output_path)
        assert report.success is True
        assert output_path.exists()
        assert report.scene_graph.node_count() > 0

    def test_entity_count_matches_worldspec_for_every_scenario(self, any_scenario_world):
        report = run_worldspec_builder(any_scenario_world)
        entity_nodes = find_all_nodes(report.scene_graph, NodeType.ENTITY)
        assert len(entity_nodes) == len(any_scenario_world.entities)

    def test_deterministic_node_ids_are_stable_across_runs(self, any_scenario_world):
        report_1 = run_worldspec_builder(any_scenario_world, deterministic=True)
        report_2 = run_worldspec_builder(any_scenario_world, deterministic=True)

        ids_1 = sorted(n.node_uuid for n in find_all_nodes(report_1.scene_graph, NodeType.ENTITY))
        ids_2 = sorted(n.node_uuid for n in find_all_nodes(report_2.scene_graph, NodeType.ENTITY))
        assert ids_1 == ids_2
