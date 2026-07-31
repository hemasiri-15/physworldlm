"""
omniverse
─────────
PhysWorldLM's Omniverse integration package.

``OmniverseConnector`` is the ONLY supported entry point into this
package. Everything else (``kit_locator``) is an internal implementation
detail exposed for testing/diagnostics, not part of the stable API.
"""

from .kit_locator import KitDiscoveryError, KitInstallation, KitLocator
from .omniverse_connector import (
    ConnectorState,
    ConnectorStatistics,
    KitAlreadyRunningError,
    KitLaunchError,
    KitNotFoundError,
    OmniverseConnector,
    OmniverseConnectorError,
    ConnectorStateError,
    StageLoadError,
)

__all__ = [
    "OmniverseConnector",
    "ConnectorState",
    "ConnectorStatistics",
    "OmniverseConnectorError",
    "ConnectorStateError",
    "KitNotFoundError",
    "KitLaunchError",
    "KitAlreadyRunningError",
    "StageLoadError",
    "KitLocator",
    "KitInstallation",
    "KitDiscoveryError",
]
