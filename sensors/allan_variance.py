"""
tools/allan_variance.py
══════════════════════════════════════════════════════════════════════════
Offline Allan variance analysis for logged IMU streams.

Deliberately NOT part of imu.py: Allan variance is a statistical
characterization computed over a long stationary capture (typically
hours), not a per-tick runtime behavior. It belongs with your other
offline validation/analysis tooling, consuming logged `IMUReading`
streams rather than being wired into the sensor itself.

Usage:
    from tools.allan_variance import allan_deviation, fit_noise_parameters

    taus, adev = allan_deviation(gyro_z_samples, dt=1.0 / 200.0)
    params = fit_noise_parameters(taus, adev)
    # params.angle_random_walk   -> ARW, deg/sqrt(hr) equivalent units
    # params.bias_instability    -> minimum of the Allan deviation curve
    # params.rate_random_walk    -> slope-based estimate past the bias floor
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["allan_deviation", "fit_noise_parameters", "NoiseParameters"]


def allan_deviation(
    samples: np.ndarray,
    dt: float,
    num_points: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the overlapping Allan deviation of a stationary time series.

    Parameters
    ----------
    samples : 1-D array of a single axis's raw sensor output (e.g. gyro_z
        in rad/s), captured at a fixed rate with the sensor stationary.
    dt : sample period in seconds (1 / update_rate_hz).
    num_points : number of log-spaced averaging-time (tau) values to
        evaluate between dt and N*dt/2.

    Returns
    -------
    (taus, adev) : averaging times in seconds, and the corresponding
        Allan deviation values (same units as `samples`).
    """
    samples = np.asarray(samples, dtype=np.float64)
    n = len(samples)
    if n < 4:
        raise ValueError("Need at least 4 samples to compute Allan deviation.")

    max_m = n // 2
    taus_candidate = np.unique(
        np.logspace(0, np.log10(max_m), num=min(num_points, max_m)).astype(int)
    )
    taus_candidate = taus_candidate[taus_candidate >= 1]

    # Cumulative sum trick for O(1) cluster averages per m.
    theta = np.concatenate(([0.0], np.cumsum(samples)))

    taus_out = []
    adev_out = []
    for m in taus_candidate:
        # Overlapping cluster averages of length m.
        cluster_avg = (theta[m:] - theta[:-m]) / m
        if len(cluster_avg) < m + 1:
            continue
        diff = cluster_avg[m:] - cluster_avg[:-m]
        variance = 0.5 * np.mean(diff**2)
        taus_out.append(m * dt)
        adev_out.append(np.sqrt(variance))

    return np.array(taus_out), np.array(adev_out)


@dataclass
class NoiseParameters:
    """Standard IMU noise parameters read off an Allan deviation curve.

    These correspond to the conventional slope regions of the log-log
    Allan deviation plot:
      - Angle/Velocity Random Walk: slope -1/2 region (short tau)
      - Bias Instability: local minimum (flicker-noise floor)
      - Rate Random Walk: slope +1/2 region (long tau)
    """

    angle_random_walk: float   # value of adev curve at tau=1, slope -1/2 fit
    bias_instability: float    # minimum of the adev curve
    bias_instability_tau: float
    rate_random_walk: float    # value of adev curve at tau=3, slope +1/2 fit


def fit_noise_parameters(taus: np.ndarray, adev: np.ndarray) -> NoiseParameters:
    """Extract ARW / bias instability / RRW from an Allan deviation curve.

    Uses local log-log slope to locate the -1/2 and +1/2 regions rather
    than assuming fixed tau values, since the exact location depends on
    sample rate and total capture length.
    """
    if len(taus) < 5:
        raise ValueError("Need at least 5 (tau, adev) points to fit noise parameters.")

    log_tau = np.log10(taus)
    log_adev = np.log10(adev)
    slopes = np.gradient(log_adev, log_tau)

    # Bias instability: minimum of the curve.
    min_idx = int(np.argmin(adev))
    bias_instability = float(adev[min_idx])
    bias_instability_tau = float(taus[min_idx])

    # ARW: point in the short-tau region closest to slope -0.5, extrapolated
    # to tau=1 via sigma(tau) = ARW / sqrt(tau).
    short_region = slice(0, min_idx) if min_idx > 0 else slice(0, len(taus))
    if len(slopes[short_region]) > 0:
        arw_idx = short_region.start + int(np.argmin(np.abs(slopes[short_region] + 0.5)))
        angle_random_walk = float(adev[arw_idx] * np.sqrt(taus[arw_idx]))
    else:
        angle_random_walk = float(adev[0] * np.sqrt(taus[0]))

    # RRW: point in the long-tau region closest to slope +0.5, extrapolated
    # to tau=3 via sigma(tau) = RRW * sqrt(tau / 3).
    long_region = slice(min_idx, len(taus))
    if len(slopes[long_region]) > 0:
        rrw_idx = min_idx + int(np.argmin(np.abs(slopes[long_region] - 0.5)))
        rate_random_walk = float(adev[rrw_idx] / np.sqrt(taus[rrw_idx] / 3.0))
    else:
        rate_random_walk = float(adev[-1] / np.sqrt(taus[-1] / 3.0))

    return NoiseParameters(
        angle_random_walk=angle_random_walk,
        bias_instability=bias_instability,
        bias_instability_tau=bias_instability_tau,
        rate_random_walk=rate_random_walk,
    )
