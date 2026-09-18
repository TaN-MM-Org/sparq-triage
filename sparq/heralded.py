"""Heralded single-photon sources: the exact purity limit set by the
coincidence-to-accidental ratio.

Solid-state emitters are one route to single photons; the other
workhorse is the heralded pair source (parametric down-conversion or
four-wave mixing), where detecting one photon of a pair heralds its
twin. Its purity metric is the heralded second-order correlation
g2_h(0), and its brightness-versus-purity trade is captured by one
routinely measured number: the coincidence-to-accidental ratio (CAR).
For a source pumped by a laser (Poissonian pair statistics) the two
are tied by an exact closed form,

    g2_h(0) = (2 CAR - 1) / CAR^2,

and for a thermal-like (single-mode spontaneous) pair distribution by

    g2_h(0) = (4 CAR + 2) / (CAR + 1)^2,

both given, with a source saturating the Poissonian limit, in H.
Wang et al., arXiv:2404.03236. This module ships the two closed
forms and their exact inversions, so a lab can answer both
directions: what purity does my measured CAR permit, and what CAR
must I reach for a target purity?

Honest limits, stated plainly: these are the LIMITS set by the pair
statistics alone -- multi-pair emission at the stated distribution,
perfect heralding logic. Real sources sit at or above the limit
(extra background, detector effects raise g2_h), so the forward
formula is a floor and the inverse `car_for_purity` is the MINIMUM
CAR a target purity requires. Neither formula applies to a
single-emitter (antibunched) source; that is what the rest of this
package is for.
"""
from __future__ import annotations

import numpy as np

__all__ = ["heralded_g2_limit", "car_for_purity"]


def heralded_g2_limit(car, statistics="poissonian"):
    """The lowest heralded g2(0) a given CAR permits.

    car : coincidence-to-accidental ratio (> 1; at CAR = 1 the
        "coincidences" are all accidental and the limit is 1).
    statistics : "poissonian" (laser-pumped pair generation, the
        usual case) or "thermal" (single-mode spontaneous source).

    Exact closed forms of Wang et al., arXiv:2404.03236; for large
    CAR both fall off as 2/CAR (Poissonian) and 4/CAR (thermal).
    """
    c = np.asarray(car, dtype=float)
    if np.any(~np.isfinite(c)) or np.any(c < 1.0):
        raise ValueError("CAR must be finite and >= 1 (below 1 the "
                         "'coincidences' are fewer than the "
                         "accidentals and no purity is certified)")
    if statistics == "poissonian":
        out = (2.0 * c - 1.0) / c ** 2
    elif statistics == "thermal":
        out = (4.0 * c + 2.0) / (c + 1.0) ** 2
    else:
        raise ValueError('statistics must be "poissonian" or '
                         '"thermal"')
    return float(out) if out.ndim == 0 else out


def car_for_purity(g2_target, statistics="poissonian"):
    """The minimum CAR a target heralded purity requires -- the exact
    inverse of `heralded_g2_limit`.

    Poissonian: g CAR^2 - 2 CAR + 1 = 0 gives
    CAR = (1 + sqrt(1 - g)) / g, the root on the physical branch
    (CAR >= 1). Thermal: with u = CAR + 1, g u^2 - 4 u + 2 = 0 gives
    u = (2 + sqrt(4 - 2 g)) / g on the physical branch; the tests
    verify both inversions as exact round trips.

    Refuses targets outside each formula's own range at CAR >= 1:
    (0, 1] for the Poissonian form and (0, 1.5] for the thermal one
    (at CAR = 1 the thermal formula gives exactly 3/2, the heralded
    remnant of thermal bunching); zero is unreachable at finite CAR.
    """
    g = float(g2_target)
    if statistics == "poissonian":
        if not (0.0 < g <= 1.0) or not np.isfinite(g):
            raise ValueError("g2_target must lie in (0, 1] for the "
                             "Poissonian form: zero is unreachable "
                             "at finite CAR, and its value at "
                             "CAR = 1 is exactly 1")
        return (1.0 + np.sqrt(1.0 - g)) / g
    if statistics == "thermal":
        if not (0.0 < g <= 1.5) or not np.isfinite(g):
            raise ValueError("g2_target must lie in (0, 1.5] for "
                             "the thermal form: zero is unreachable "
                             "at finite CAR, and its value at "
                             "CAR = 1 is exactly 3/2")
        # g (c+1)^2 = 4 c + 2, c = CAR: g u^2 - 4 u + 2 = 0 with
        # u = c + 1 gives u = (2 + sqrt(4 - 2 g)) / g on the
        # physical branch
        u = (2.0 + np.sqrt(4.0 - 2.0 * g)) / g
        return float(u - 1.0)
    raise ValueError('statistics must be "poissonian" or "thermal"')
