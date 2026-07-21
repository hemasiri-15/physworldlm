"""Structural and semantic validation for the scene graph IR.

Validation is deliberately separated from construction: a ``SceneGraph``
can exist in a transiently invalid state while ``WorldSpecBuilder`` is
still populating it, and is only required to pass validation immediately
before being handed to a downstream compiler (USD exporter, PhysX
exporter). This mirrors a traditional compiler's verifier pass running
after IR construction and before code generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from physworldlm.scene_graph.edges import EdgeKind, SceneEdge
from physworldlm.scene_graph.exceptions import SceneGraphValidationError
from physworldlm.scene_graph.graph import SceneGraph
from physworldlm.scene_graph.nodes import NodeKind, SceneNode

_BODY_BEARING_JOINT_KINDS: frozenset[EdgeKind] = frozenset(
    {
        EdgeKind.JOINT_SPRING,
        EdgeKind.JOINT_HINGE,
        EdgeKind.JOINT_FIXED,
        EdgeKind.JOINT_DISTANCE,
    }
)

_REQUIRED_PHYSICS_BODY_ATTRS: tuple[str, ...] = ("mass_kg", "is_static")
_REQUIRED_MATERIAL_ATTRS: tuple[str, ...] = ("friction", "restitution")
_REQUIRED_SPRING_ATTRS: tuple[str, ...] = ("k_Nm", "rest_length_m")


class ValidationSeverity(str, Enum):
    """The severity of a single :class:`ValidationIssue`."""

    ERROR = "error"
    """Renders the graph uncompilable; must be resolved before export."""

    WARNING = "warning"
    """Compilable, but likely to produce physically implausible results."""

    INFO = "info"
    """Purely informational; no action required."""


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """A single validation finding, scoped to an optional node/edge id."""

    severity: ValidationSeverity
    code: str
    message: str
    node_id: str | None = None
    edge_id: str | None = None


@dataclass(slots=True)
class ValidationReport:
    """The aggregate result of running every validation rule over a graph."""

    issues: list[ValidationIssue] = field(default_factory=list)

    def add(self, issue: ValidationIssue) -> None:
        """Append a single issue to the report."""
        self.issues.append(issue)

    def errors(self) -> list[ValidationIssue]:
        """Return only the :attr:`ValidationSeverity.ERROR` issues."""
        return [i for i in self.issues if i.severity is ValidationSeverity.ERROR]

    def warnings(self) -> list[ValidationIssue]:
        """Return only the :attr:`ValidationSeverity.WARNING` issues."""
        return [i for i in self.issues if i.severity is ValidationSeverity.WARNING]

    def is_valid(self) -> bool:
        """Return ``True`` if the report contains zero errors (warnings are allowed)."""
        return len(self.errors()) == 0

    def raise_if_invalid(self) -> None:
        """Raise :class:`SceneGraphValidationError` if any errors are present."""
        error_count = len(self.errors())
        if error_count:
            raise SceneGraphValidationError(error_count)


class SceneGraphValidator:
    """Runs a fixed suite of structural and semantic rules over a ``SceneGraph``.

    Each ``_check_*`` method is an independent rule that only appends to
    the shared :class:`ValidationReport`; this keeps rules independently
    testable and lets new rules be added without touching existing ones
    (open/closed with respect to the rule set).
    """

    def validate(self, graph: SceneGraph) -> ValidationReport:
        """Run every validation rule over ``graph`` and return the aggregate report."""
        report = ValidationReport()
        self._check_single_root(graph, report)
        self._check_dangling_hierarchy_pointers(graph, report)
        self._check_physics_body_completeness(graph, report)
        self._check_collider_attachment(graph, report)
        self._check_material_completeness(graph, report)
        self._check_joint_endpoints(graph, report)
        self._check_spring_joint_parameters(graph, report)
        self._check_unresolved_attributes(graph, report)
        self._check_environment_uniqueness(graph, report)
        return report

    # ------------------------------------------------------------------ #
    # Individual rules
    # ------------------------------------------------------------------ #

    @staticmethod
    def _check_single_root(graph: SceneGraph, report: ValidationReport) -> None:
        roots = [n for n in graph.nodes() if n.kind is NodeKind.ROOT]
        if len(roots) != 1:
            report.add(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="SG001",
                    message=f"expected exactly one ROOT node, found {len(roots)}",
                )
            )

    @staticmethod
    def _check_dangling_hierarchy_pointers(graph: SceneGraph, report: ValidationReport) -> None:
        for node in graph.nodes():
            if node.parent_id is not None and not graph.has_node(node.parent_id):
                report.add(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        code="SG002",
                        message=f"node references missing parent_id={node.parent_id!r}",
                        node_id=node.id,
                    )
                )
            for child_id in node.children_ids:
                if not graph.has_node(child_id):
                    report.add(
                        ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            code="SG003",
                            message=f"node references missing child_id={child_id!r}",
                            node_id=node.id,
                        )
                    )

    @staticmethod
    def _check_physics_body_completeness(graph: SceneGraph, report: ValidationReport) -> None:
        for node in graph.nodes_by_kind(NodeKind.PHYSICS_BODY):
            for attr_name in _REQUIRED_PHYSICS_BODY_ATTRS:
                if not node.attributes.has(attr_name):
                    report.add(
                        ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            code="SG010",
                            message=f"physics_body missing required attribute {attr_name!r}",
                            node_id=node.id,
                        )
                    )
            is_static = node.attribute_value("is_static", False)
            mass = node.attribute_value("mass_kg")
            if not is_static and (mass is None or mass <= 0.0):
                report.add(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        code="SG011",
                        message=(
                            f"dynamic physics_body has non-positive or missing "
                            f"mass_kg={mass!r}"
                        ),
                        node_id=node.id,
                    )
                )

    @staticmethod
    def _check_collider_attachment(graph: SceneGraph, report: ValidationReport) -> None:
        for node in graph.nodes_by_kind(NodeKind.COLLIDER):
            parent_id = node.parent_id
            if parent_id is None or not graph.has_node(parent_id):
                report.add(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        code="SG020",
                        message="collider has no valid parent node",
                        node_id=node.id,
                    )
                )
                continue
            parent = graph.get_node(parent_id)
            if parent.kind is not NodeKind.PHYSICS_BODY:
                report.add(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        code="SG021",
                        message=(
                            f"collider's parent must be a physics_body, "
                            f"got kind={parent.kind.value!r}"
                        ),
                        node_id=node.id,
                    )
                )

    @staticmethod
    def _check_material_completeness(graph: SceneGraph, report: ValidationReport) -> None:
        for node in graph.nodes_by_kind(NodeKind.MATERIAL):
            for attr_name in _REQUIRED_MATERIAL_ATTRS:
                if not node.attributes.has(attr_name):
                    report.add(
                        ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            code="SG030",
                            message=f"material missing required attribute {attr_name!r}",
                            node_id=node.id,
                        )
                    )

    @staticmethod
    def _check_joint_endpoints(graph: SceneGraph, report: ValidationReport) -> None:
        for edge in graph.edges():
            if edge.kind not in _BODY_BEARING_JOINT_KINDS:
                continue
            for endpoint_id in edge.endpoints():
                if not graph.has_node(endpoint_id):
                    report.add(
                        ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            code="SG040",
                            message=f"joint references missing node {endpoint_id!r}",
                            edge_id=edge.id,
                        )
                    )
                    continue
                endpoint_kind = graph.get_node(endpoint_id).kind
                if endpoint_kind not in (NodeKind.PHYSICS_BODY, NodeKind.ENTITY):
                    report.add(
                        ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            code="SG041",
                            message=(
                                f"joint endpoint {endpoint_id!r} has invalid kind "
                                f"{endpoint_kind.value!r} (expected physics_body or entity)"
                            ),
                            edge_id=edge.id,
                        )
                    )

    @staticmethod
    def _check_spring_joint_parameters(graph: SceneGraph, report: ValidationReport) -> None:
        for edge in graph.edges_by_kind(EdgeKind.JOINT_SPRING):
            for attr_name in _REQUIRED_SPRING_ATTRS:
                if not edge.attributes.has(attr_name):
                    report.add(
                        ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            code="SG050",
                            message=f"spring joint missing required attribute {attr_name!r}",
                            edge_id=edge.id,
                        )
                    )
            stiffness = edge.attribute_value("k_Nm")
            if stiffness is not None and stiffness <= 0.0:
                report.add(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        code="SG051",
                        message=f"spring stiffness k_Nm={stiffness!r} is non-positive",
                        edge_id=edge.id,
                    )
                )

    @staticmethod
    def _check_unresolved_attributes(graph: SceneGraph, report: ValidationReport) -> None:
        for node in graph.nodes():
            for attr in node.attributes.unresolved():
                report.add(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        code="SG060",
                        message=f"attribute {attr.name!r} is unresolved",
                        node_id=node.id,
                    )
                )
        for edge in graph.edges():
            for attr in edge.attributes.unresolved():
                report.add(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        code="SG061",
                        message=f"attribute {attr.name!r} is unresolved",
                        edge_id=edge.id,
                    )
                )

    @staticmethod
    def _check_environment_uniqueness(graph: SceneGraph, report: ValidationReport) -> None:
        environment_nodes = graph.nodes_by_kind(NodeKind.ENVIRONMENT)
        if len(environment_nodes) > 1:
            report.add(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="SG070",
                    message=(
                        f"expected at most one environment node, "
                        f"found {len(environment_nodes)}"
                    ),
                )
            )
