"""Backend-independent spatial primitives used throughout the scene graph IR.

The ``SceneGraph`` IR is deliberately decoupled from any single downstream
backend (OpenUSD, PhysX, Isaac Sim, ...). Upstream ``WorldSpec`` entities
represent orientation as an Euler-angle ``Vec3`` (radians); this module
provides the canonical conversions from that representation into the
quaternion + 4x4 matrix representations that USD/PhysX compilation
requires, so that no downstream compiler stage needs to reimplement
rotation math.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Vec3:
    """An immutable 3-component vector."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> "Vec3":
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    __rmul__ = __mul__

    def dot(self, other: "Vec3") -> float:
        """Return the dot product with ``other``."""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vec3") -> "Vec3":
        """Return the cross product with ``other``."""
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def magnitude(self) -> float:
        """Return the Euclidean length of the vector."""
        return math.sqrt(self.dot(self))

    def normalized(self) -> "Vec3":
        """Return a unit-length copy of this vector.

        Returns the zero vector unchanged rather than raising, since a
        zero-length normal/axis is a valid (if degenerate) input in scene
        authoring contexts.
        """
        mag = self.magnitude()
        if mag < 1e-12:
            return Vec3(0.0, 0.0, 0.0)
        return Vec3(self.x / mag, self.y / mag, self.z / mag)

    def to_tuple(self) -> tuple[float, float, float]:
        """Return the vector as a plain ``(x, y, z)`` tuple."""
        return (self.x, self.y, self.z)


@dataclass(frozen=True, slots=True)
class Quaternion:
    """An immutable unit quaternion in ``(w, x, y, z)`` scalar-first order."""

    w: float = 1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    @staticmethod
    def identity() -> "Quaternion":
        """Return the identity rotation."""
        return Quaternion(1.0, 0.0, 0.0, 0.0)

    @staticmethod
    def from_euler_xyz(euler_radians: Vec3) -> "Quaternion":
        """Build a quaternion from intrinsic XYZ Euler angles, in radians.

        This is the canonical conversion from the ``Vec3``-encoded Euler
        orientation used by ``PhysicsState.orientation`` upstream into the
        quaternion form required by USD ``UsdGeom.Xformable`` rotate ops
        and PhysX rigid body poses.
        """
        hx, hy, hz = euler_radians.x * 0.5, euler_radians.y * 0.5, euler_radians.z * 0.5
        cx, sx = math.cos(hx), math.sin(hx)
        cy, sy = math.cos(hy), math.sin(hy)
        cz, sz = math.cos(hz), math.sin(hz)

        # Intrinsic rotations applied in X, then Y, then Z order: q = qz * qy * qx
        qx = Quaternion(cx, sx, 0.0, 0.0)
        qy = Quaternion(cy, 0.0, sy, 0.0)
        qz = Quaternion(cz, 0.0, 0.0, sz)
        return qz.multiply(qy).multiply(qx).normalized()

    def multiply(self, other: "Quaternion") -> "Quaternion":
        """Return the Hamilton product ``self * other``."""
        w1, x1, y1, z1 = self.w, self.x, self.y, self.z
        w2, x2, y2, z2 = other.w, other.x, other.y, other.z
        return Quaternion(
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        )

    def magnitude(self) -> float:
        """Return the magnitude of the quaternion."""
        return math.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)

    def normalized(self) -> "Quaternion":
        """Return a unit-length copy of this quaternion, defaulting to identity."""
        mag = self.magnitude()
        if mag < 1e-12:
            return Quaternion.identity()
        return Quaternion(self.w / mag, self.x / mag, self.y / mag, self.z / mag)

    def to_tuple_wxyz(self) -> tuple[float, float, float, float]:
        """Return the quaternion as a ``(w, x, y, z)`` tuple."""
        return (self.w, self.x, self.y, self.z)

    def to_rotation_matrix(self) -> tuple[tuple[float, float, float], ...]:
        """Return the equivalent 3x3 row-major rotation matrix as nested tuples."""
        w, x, y, z = self.w, self.x, self.y, self.z
        return (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        )


@dataclass(frozen=True, slots=True)
class Transform:
    """An immutable local-space rigid + scale transform (TRS) for one node.

    ``Transform`` is composable: :meth:`compose` produces the transform of
    a child expressed in its parent's local space, applied in the
    conventional ``T * R * S`` order.
    """

    translation: Vec3 = Vec3(0.0, 0.0, 0.0)
    rotation: Quaternion = Quaternion.identity()
    scale: Vec3 = Vec3(1.0, 1.0, 1.0)

    @staticmethod
    def identity() -> "Transform":
        """Return the identity transform."""
        return Transform()

    @staticmethod
    def from_euler(
        translation: Vec3,
        euler_radians: Vec3,
        scale: Vec3 | None = None,
    ) -> "Transform":
        """Build a transform from a translation, Euler orientation, and scale."""
        return Transform(
            translation=translation,
            rotation=Quaternion.from_euler_xyz(euler_radians),
            scale=scale if scale is not None else Vec3(1.0, 1.0, 1.0),
        )

    def to_matrix4(self) -> tuple[tuple[float, ...], ...]:
        """Return the 4x4 row-major homogeneous transform matrix.

        The returned matrix follows the ``row-major, row-vector-on-left``
        convention expected by ``Gf.Matrix4d`` in OpenUSD.
        """
        r = self.rotation.to_rotation_matrix()
        sx, sy, sz = self.scale.x, self.scale.y, self.scale.z
        return (
            (r[0][0] * sx, r[0][1] * sy, r[0][2] * sz, self.translation.x),
            (r[1][0] * sx, r[1][1] * sy, r[1][2] * sz, self.translation.y),
            (r[2][0] * sx, r[2][1] * sy, r[2][2] * sz, self.translation.z),
            (0.0, 0.0, 0.0, 1.0),
        )

    def compose(self, child: "Transform") -> "Transform":
        """Return ``child``'s transform re-expressed in this transform's parent space.

        Rotation and translation are composed correctly for a rigid parent
        transform; non-uniform parent scale interacting with child rotation
        (shear) is intentionally not modelled, since neither USD's nor
        PhysX's authoring conventions rely on sheared transforms for
        physically simulated bodies.
        """
        rotated_child_translation = _rotate_vector(self.rotation, child.translation)
        scaled_child_translation = Vec3(
            rotated_child_translation.x * self.scale.x,
            rotated_child_translation.y * self.scale.y,
            rotated_child_translation.z * self.scale.z,
        )
        return Transform(
            translation=self.translation + scaled_child_translation,
            rotation=self.rotation.multiply(child.rotation).normalized(),
            scale=Vec3(
                self.scale.x * child.scale.x,
                self.scale.y * child.scale.y,
                self.scale.z * child.scale.z,
            ),
        )


def _rotate_vector(rotation: Quaternion, vector: Vec3) -> Vec3:
    """Rotate ``vector`` by unit ``rotation`` using the sandwich product formula."""
    m = rotation.to_rotation_matrix()
    return Vec3(
        m[0][0] * vector.x + m[0][1] * vector.y + m[0][2] * vector.z,
        m[1][0] * vector.x + m[1][1] * vector.y + m[1][2] * vector.z,
        m[2][0] * vector.x + m[2][1] * vector.y + m[2][2] * vector.z,
    )
