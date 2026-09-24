"""
sparq.physics
=============
Photophysics of solid-state single-photon emitters under CW excitation,
Hanbury Brown–Twiss (HBT) correlation statistics, and two simulators:

1. An *exact-statistics* histogram twin: the coincidence counts in each
   delay bin of an HBT histogram are Poisson-distributed with a mean set
   by the analytic second-order correlation function g2(tau).  This is
   the fast engine used to train estimators (thousands of synthetic
   acquisitions per second).

2. A *full Monte-Carlo photon-stream* simulator: a continuous-time Markov
   chain of the emitter level structure, photon-by-photon, with detector
   impairments (IRF jitter, dead time, afterpulsing) and blinking that
   the histogram twin does NOT model.  It serves (a) to validate the twin
   against the analytic law and (b) as the held-out "target domain" for
   the sim-to-real (GAN) experiments.

Analytic model (three-level system: ground g, excited e, shelving s):

    g2(tau) = 1 - (1 + a) exp(-|tau|/tau1) + a exp(-|tau|/tau2)

with antibunching time tau1, bunching amplitude a and bunching time tau2
(a = 0 recovers the two-level form).  For N identical independent
emitters:  g2_N = 1 + (g2_1 - 1)/N.   With uncorrelated (Poissonian)
background at signal fraction rho = S/(S+B):

    g2_meas(tau) = 1 + rho^2 (g2_N(tau) - 1).

Detector timing jitter convolves g2 with a Gaussian of width
sigma_pair = sqrt(2) * sigma_IRF.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field, asdict
from scipy.special import erf, erfcx  # noqa: F401 (erf kept for users)

# ----------------------------------------------------------------------
# Analytic correlation functions
# ----------------------------------------------------------------------

def g2_three_level(tau, tau1, tau2, a):
    """Ideal three-level CW g2(tau); tau in ns."""
    at = np.abs(tau)
    return 1.0 - (1.0 + a) * np.exp(-at / tau1) + a * np.exp(-at / max(tau2, 1e-9))


def g2_measured(tau, tau1, tau2, a, n_emitters=1, rho=1.0, sigma_irf=0.0):
    """Measured g2 for N emitters + background, IRF-convolved.

    The Gaussian convolution of exp(-|tau|/T) has the closed form
    used below (sigma_pair = sqrt(2) sigma_irf for two detectors).
    """
    g2_1 = lambda t, T: _exp_conv_gauss(t, T, np.sqrt(2.0) * sigma_irf)
    dip = (1.0 + a) * g2_1(tau, tau1) - a * g2_1(tau, tau2)
    g2n = 1.0 - dip / n_emitters
    return 1.0 + rho ** 2 * (g2n - 1.0)


def _exp_conv_gauss(tau, T, s):
    """Convolution of exp(-|tau|/T) with a normalized Gaussian of std s.

    Closed form, 1/2 e^{s^2/2T^2} [ e^{-t/T} erfc((s/T - t/s)/sqrt2)
    + e^{t/T} erfc((s/T + t/s)/sqrt2) ], evaluated through the scaled
    function erfcx(u) = e^{u^2} erfc(u): each term equals
    e^{-t^2/2s^2} erfcx(u) (and, for u < 0,
    2 e^{s^2/2T^2 -+ t/T} - e^{-t^2/2s^2} erfcx(-u)). Up to 0.9.1 the
    product e^{s^2/2T^2} (1 - erf(.)) was formed directly, which loses
    all precision once T is below about s/7 and overflows to NaN far
    from zero delay (new in 0.10.0; T and tau may be arrays that
    broadcast).
    """
    tau = np.asarray(tau, dtype=float)
    if s <= 1e-12:
        return np.exp(-np.abs(tau) / T)
    T = np.asarray(T, dtype=float)
    z = s / T
    g = np.exp(-0.5 * (tau / s) ** 2)
    out = 0.0
    for sg in (1.0, -1.0):
        u = (z - sg * tau / s) / np.sqrt(2.0)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            pos = g * erfcx(u)
            neg = (2.0 * np.exp(0.5 * z ** 2 - sg * tau / T)
                   - g * erfcx(-u))
        out = out + 0.5 * np.where(u >= 0.0, pos, neg)
    return np.clip(out, 0.0, 1.0)


def g2_zero(tau1, tau2, a, n_emitters=1, rho=1.0):
    """Physical (IRF-free) g2(0) used as the ground-truth label."""
    g2n = 1.0 - 1.0 / n_emitters
    return 1.0 + rho ** 2 * (g2n - 1.0)


# ----------------------------------------------------------------------
# Emitter platforms (literature-anchored photophysical parameter priors)
# ----------------------------------------------------------------------
# Each platform is described by the ranges of its three-level rates as
# reported in the literature (see manuscript references):
#   NV in nanodiamond, hBN monolayer/multilayer defects, GaN point
#   defects, SiV in diamond. Rates in ns / kcps at the detector.

@dataclass
class Platform:
    name: str
    tau1_rng: tuple          # antibunching time (ns) at operating power
    tau2_rng: tuple          # shelving/bunching time (ns)
    a_rng: tuple             # bunching amplitude
    rate_rng: tuple          # total detected count rate (kcps), both APDs
    rho_rng: tuple           # signal fraction S/(S+B)
    blink_p: float           # probability the emitter blinks
    blink_ton_rng: tuple     # mean on-time (ms)
    blink_toff_rng: tuple    # mean off-time (ms)
    # level-structure graph (nodes: g,e,s; edge rates 1/ns) for the GNN
    def graph(self, rng: np.random.Generator, params=None):
        """Return (node_feat [3,F], edge_index [2,E], edge_feat [E,G])."""
        p = params or self.sample(rng)
        k_exc = 1.0 / p["tau1"] * 0.4          # effective pump rate
        k_r = 1.0 / p["tau1"] * 0.6            # effective decay rate
        k_es = p["a"] / max(p["tau2"], 1.0)    # shelving in-rate (approx)
        k_se = 1.0 / max(p["tau2"], 1.0)       # deshelving rate
        node_feat = np.array([
            #  is_g, is_e, is_s, radiative?
            [1, 0, 0, 0.0],
            [0, 1, 0, 1.0],
            [0, 0, 1, 0.0],
        ], dtype=np.float32)
        edge_index = np.array([[0, 1, 1, 2], [1, 0, 2, 0]], dtype=np.int64)
        edge_feat = np.log10(np.array(
            [[k_exc], [k_r], [k_es], [k_se]], dtype=np.float32) + 1e-9)
        return node_feat, edge_index, edge_feat

    def sample(self, rng: np.random.Generator) -> dict:
        lo, hi = self.tau1_rng
        tau1 = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        lo, hi = self.tau2_rng
        tau2 = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        a = float(rng.uniform(*self.a_rng))
        rate = float(np.exp(rng.uniform(*np.log(self.rate_rng))))
        # Signal fraction is bimodal in surveyed fields: localized emitters
        # dominate their confocal spot (high rho), while a minority of
        # spots sit on strong background/clusters (low rho). See emitter
        # survey histograms in the characterization literature.
        r_lo, r_hi = self.rho_rng
        split = r_lo + 0.65 * (r_hi - r_lo)
        if rng.random() < 0.70:
            rho = float(rng.uniform(split, r_hi))
        else:
            rho = float(rng.uniform(r_lo, split))
        blinking = bool(rng.random() < self.blink_p)
        return dict(platform=self.name, tau1=tau1, tau2=tau2, a=a,
                    rate_kcps=rate, rho=rho, blinking=blinking,
                    t_on_ms=float(np.exp(rng.uniform(*np.log(self.blink_ton_rng)))),
                    t_off_ms=float(np.exp(rng.uniform(*np.log(self.blink_toff_rng)))))


# Parameter ranges anchored to published photophysics (citations in paper):
PLATFORMS = {
    # NV in nanodiamond: excited-state lifetime 12–25 ns, power-shortened
    # antibunching 8–25 ns, metastable singlet bunching 100–500 ns.
    "NV": Platform("NV", (8, 25), (100, 500), (0.1, 1.5),
                   (30, 350), (0.55, 0.98), 0.15, (5, 200), (0.5, 40)),
    # hBN defects: 2–4 ns lifetimes, strong bunching, bright, blinking common.
    "hBN": Platform("hBN", (1.5, 5), (20, 2500), (0.2, 3.0),
                    (60, 900), (0.6, 0.99), 0.35, (1, 100), (0.5, 60)),
    # GaN point defects: 0.7–1.6 ns lifetimes, bright and stable.
    "GaN": Platform("GaN", (0.6, 2.0), (10, 300), (0.1, 1.2),
                    (100, 1000), (0.6, 0.98), 0.10, (10, 300), (0.5, 20)),
    # SiV in diamond: ~1–1.8 ns, moderate bunching.
    "SiV": Platform("SiV", (0.8, 2.0), (15, 250), (0.1, 0.8),
                    (80, 800), (0.65, 0.99), 0.08, (10, 300), (0.5, 20)),
}


def register_platform(platform: Platform, overwrite: bool = False) -> Platform:
    """Register a user-defined emitter platform for use everywhere a platform
    name is accepted (sample_site, the dataset generators, the triage
    environment, the graph encoder's template).

    Provide literature-anchored ranges for your emitter: tau1_rng and
    tau2_rng in ns, a_rng dimensionless, rate_rng in kcps (both detectors),
    rho_rng the signal fraction, blink_p the blinking probability and
    blink_ton_rng / blink_toff_rng in ms.  Every range must be a (low, high)
    pair with 0 < low <= high; rho must lie in (0, 1].

    Returns the platform for chaining; raises ValueError on an invalid
    definition or a duplicate name unless overwrite=True.
    """
    if not isinstance(platform, Platform):
        raise ValueError("register_platform expects a Platform instance")
    if platform.name in PLATFORMS and not overwrite:
        raise ValueError(f"platform {platform.name!r} already exists; pass overwrite=True to replace it")
    for field_name in ("tau1_rng", "tau2_rng", "a_rng", "rate_rng", "rho_rng",
                       "blink_ton_rng", "blink_toff_rng"):
        rng = getattr(platform, field_name)
        try:
            lo, hi = float(rng[0]), float(rng[1])
        except (TypeError, IndexError, ValueError):
            raise ValueError(f"{field_name} must be a (low, high) pair") from None
        if not (lo <= hi):
            raise ValueError(f"{field_name}: low must not exceed high")
        if field_name != "a_rng" and not lo > 0:
            raise ValueError(f"{field_name}: bounds must be positive")
        if field_name == "a_rng" and lo < 0:
            raise ValueError("a_rng: bounds must be non-negative")
    if not (0.0 < platform.rho_rng[0] and platform.rho_rng[1] <= 1.0):
        raise ValueError("rho_rng must lie in (0, 1]")
    if not (0.0 <= platform.blink_p <= 1.0):
        raise ValueError("blink_p must lie in [0, 1]")
    PLATFORMS[platform.name] = platform
    return platform


@dataclass
class EmitterSite:
    """One candidate site in a confocal field."""
    params: dict
    n_emitters: int

    @property
    def g2_0(self):
        return g2_zero(self.params["tau1"], self.params["tau2"],
                       self.params["a"], self.n_emitters, self.params["rho"])

    @property
    def is_good(self):
        """'Good' = high-purity (g2(0) < 0.5), bright, non-blinking."""
        return (self.g2_0 < 0.5 and self.params["rate_kcps"] > 60
                and not self.params["blinking"])


def sample_site(rng, platform="NV", n_probs=(0.42, 0.30, 0.18, 0.10)):
    n = int(rng.choice([1, 2, 3, 4], p=n_probs))
    return EmitterSite(PLATFORMS[platform].sample(rng), n)


# ----------------------------------------------------------------------
# Histogram twin (exact Poisson statistics of the HBT histogram)
# ----------------------------------------------------------------------

@dataclass
class HBTConfig:
    """The histogram grid: bins of width 2 tau_max / n_bins from
    -tau_max to +tau_max (ns), and the per-detector timing jitter.

    bin_average (new in 0.10.0, default True): a correlator counts every
    pair whose delay falls anywhere in a bin, so the expected count of a
    bin is g2 AVERAGED over the bin. With True the simulator and the
    fits use that average (8-point Gauss-Legendre, 4 points on each half
    of the bin, so the kink of an unblurred dip at zero delay falls on a
    node boundary). With False they use g2 at the bin center, as every
    version up to 0.9.1 did; that biases fits of real data, whose bins
    are averages (for the example site of the README, g2(0) 0.086
    instead of 0.0975 on noise-free 1 ns bins).
    """
    tau_max: float = 60.5        # ns  (window +-tau_max)
    n_bins: int = 121            # odd -> a bin centered at tau = 0
    sigma_irf: float = 0.35      # ns, per-detector IRF sigma (~0.8 ns FWHM)
    bin_average: bool = True

    @property
    def bin_width(self):
        return 2 * self.tau_max / self.n_bins

    @property
    def bin_centers(self):
        return (np.arange(self.n_bins) + 0.5) * self.bin_width - self.tau_max

    def bin_nodes(self):
        """(nodes, weights): delays (n_bins, M) at which g2 is evaluated
        and weights (M,) that turn those values into the bin value
        (M = 8 with bin_average, else 1 at the center)."""
        c = self.bin_centers
        if not self.bin_average:
            return c[:, None], np.ones(1)
        x, wt = np.polynomial.legendre.leggauss(4)
        half = 0.5 * self.bin_width
        off = np.concatenate([(x - 1.0) * 0.5 * half,      # left half
                              (x + 1.0) * 0.5 * half])     # right half
        return c[:, None] + off[None, :], np.concatenate([wt, wt]) / 4.0


def mean_detected_rate_cps(site: EmitterSite) -> float:
    """Time-averaged detected rate (counts/s, both detectors, before
    dead time). `rate_kcps` is the rate while the emitter is on; a
    blinking emitter's light is off for a share t_off/(t_on + t_off) of
    the time, while the background (share 1 - rho) keeps going, as in
    the photon-by-photon simulator (new in 0.10.0; up to 0.9.1 the fast
    simulator switched the background off with the emitter)."""
    p = site.params
    r_on = p["rate_kcps"] * 1e3
    if not p["blinking"]:
        return r_on
    duty = p["t_on_ms"] / (p["t_on_ms"] + p["t_off_ms"])
    return r_on * (p["rho"] * duty + (1.0 - p["rho"]))


def expected_histogram(site: EmitterSite, T_s: float, cfg: HBTConfig,
                       dead_time_ns: float = 0.0):
    """Mean coincidence counts per bin for acquisition time T_s (seconds).

    Blinking (new in 0.10.0, exact): the emitter light is gated by an
    on/off (telegraph) process with exponential on and off times, the
    same process the photon-by-photon simulator uses, and independent
    of the emitter's own dynamics. The emitter part of g2 is then
    multiplied exactly by the gate's own correlation
        g2_gate(tau) = 1 + (t_off/t_on) exp(-|tau|/tau_c),
        1/tau_c = 1/t_on + 1/t_off,
    and the background, which is not gated, dilutes the result with the
    time-averaged signal share rho_b = rho d / (rho d + 1 - rho),
    d = t_on/(t_on + t_off):
        g2(tau) = 1 + rho_b^2 (g2_gate(tau) g2_N(tau) - 1).
    Blinking therefore raises g2 around the dip but cannot fill the dip
    itself. (Up to 0.9.1 blinking added a flat level everywhere,
    including at zero delay.) The instrument-response blur is applied
    to g2_N only, which is exact while tau_c is much longer than the
    jitter (microseconds and longer; the gate barely changes over a
    nanosecond).

    dead_time_ns (new in 0.10.0): non-paralyzable dead time of each
    detector. It lowers each detector's counted rate (see
    `deadtime_throughput`) and so the flat level. Any change it makes
    to the shape of g2 is not modelled; in the tests the result agrees
    with the photon-by-photon simulation (45 ns dead time, 300 kcps per
    detector) within Poisson noise. Afterpulsing is not modelled here;
    the photon-by-photon simulator has it.
    """
    r_a = r_b = 0.5 * mean_detected_rate_cps(site)
    td = float(dead_time_ns)
    if td < 0 or not np.isfinite(td):
        raise ValueError("dead_time_ns must be finite and >= 0")
    if td > 0:
        r_a = r_b = deadtime_throughput(site, r_a, td)
    nodes, wts = cfg.bin_nodes()
    g2 = site_g2(site, nodes, cfg.sigma_irf) @ wts
    flat = r_a * r_b * (cfg.bin_width * 1e-9) * T_s
    return flat * g2


def site_g2(site: EmitterSite, tau, sigma_irf: float = 0.0):
    """The normalized g2(tau) the fast simulator uses for a site: the
    N-emitter model with background, instrument response and, for a
    blinking site, the exact on/off gate (see `expected_histogram`)."""
    p = site.params
    tau = np.asarray(tau, dtype=float)
    if not p["blinking"]:
        return g2_measured(tau, p["tau1"], p["tau2"], p["a"],
                           site.n_emitters, p["rho"], sigma_irf)
    duty = p["t_on_ms"] / (p["t_on_ms"] + p["t_off_ms"])
    rho_b = p["rho"] * duty / (p["rho"] * duty + 1.0 - p["rho"])
    g2_n = g2_measured(tau, p["tau1"], p["tau2"], p["a"],
                       site.n_emitters, 1.0, sigma_irf)
    t_on, t_off = p["t_on_ms"] * 1e6, p["t_off_ms"] * 1e6          # ns
    tau_c = t_on * t_off / (t_on + t_off)
    gate = 1.0 + (t_off / t_on) * np.exp(-np.abs(tau) / tau_c)
    return 1.0 + rho_b ** 2 * (gate * g2_n - 1.0)


def deadtime_throughput(site: EmitterSite, rate_cps: float,
                        dead_time_ns: float) -> float:
    """Counted rate of one detector with non-paralyzable dead time that
    receives `rate_cps` of this site's light (new in 0.10.0).

    For uncorrelated (Poisson) light the exact result is
    r / (1 + r tau_d). Light that is antibunched or bunched on the
    scale of tau_d loses fewer or more counts, because the chance that
    the next photon arrives while the detector is blind is
    r * I with I = integral_0^tau_d g2(tau) dtau. This function returns
    r / (1 + r I), which is exact for Poisson light (g2 = 1, I = tau_d)
    and correct to first order in r tau_d otherwise; the tests compare
    it with the photon-by-photon detector simulation.
    """
    r = float(rate_cps)
    td = float(dead_time_ns)
    if td == 0.0:
        return r
    t = np.linspace(0.0, td, 4001)
    g = site_g2(site, t, 0.0)
    I_ns = float(np.sum(0.5 * (g[1:] + g[:-1]) * np.diff(t)))
    return r / (1.0 + r * I_ns * 1e-9)


def sample_histogram(site, T_s, cfg, rng):
    """Poisson-sampled HBT histogram (the fast twin)."""
    return rng.poisson(expected_histogram(site, T_s, cfg)).astype(np.float32)


def sample_event_stream(site, T_s, cfg, rng, n_slices):
    """Coincidence events resolved into n_slices time slices.

    Returns [n_slices, n_bins] Poisson counts whose sum over slices is a
    full histogram; this is the native event-driven input of the SNN.
    """
    mu = expected_histogram(site, T_s, cfg) / n_slices
    return rng.poisson(np.broadcast_to(mu, (n_slices, cfg.n_bins))).astype(np.float32)


# ----------------------------------------------------------------------
# Full Monte-Carlo photon-stream simulator (validation & target domain)
# ----------------------------------------------------------------------

@dataclass
class DetectorImpairments:
    dead_time_ns: float = 45.0        # APD dead time
    afterpulse_p: float = 0.02        # afterpulsing probability
    afterpulse_tau_ns: float = 80.0   # afterpulse delay scale
    sigma_irf_ns: float = 0.35        # Gaussian timing jitter (per detector)


def _site_rates(tau1, tau2, a):
    """Exact rates for the photon-by-photon simulator: pump fraction 0.4
    (the package's historical choice) when it works, otherwise the
    smallest pump fraction that does (stated in the message of
    `rates_for_params`)."""
    from .exact import rates_for_params
    try:
        return rates_for_params(tau1, tau2, a, 0.4)
    except ValueError as err:
        l1, l2 = 1.0 / tau1, 1.0 / tau2
        D = (1.0 + a) * l1 - a * l2
        k_se = l1 * l2 / D if D > 0 else np.nan
        Q = l1 + l2 - k_se
        R = k_se * (D - l1 - l2 + k_se)
        f_min = 4.0 * R / Q ** 2 if Q > 0 else np.inf
        if not (R > 0 and 0.4 < f_min < 1.0):
            raise ValueError(f"the photon-by-photon simulator cannot "
                             f"realize tau1={tau1}, tau2={tau2}, a={a}: "
                             f"{err}") from None
        return rates_for_params(tau1, tau2, a, min(f_min * (1 + 1e-9),
                                                   0.5 * (1 + f_min)))


def _emission_rate_per_ns(k_exc, k_r, k_es, k_se):
    """Steady-state photon emission rate k_r p_e of one emitter (1/ns)."""
    shelf = k_es / k_se if k_es > 0 else 0.0
    p_e = 1.0 / (1.0 + (k_r + k_es) / k_exc + shelf)
    return k_r * p_e


def _simulate_emission_times(p: dict, n_emitters: int, T_s: float,
                             rng: np.random.Generator, keep: float = 1.0):
    """Exact CTMC emission times for n independent three-level emitters,
    each photon kept with probability `keep` (the collection
    efficiency; applied block by block, so memory scales with the kept
    photons, not with all emitted ones).

    Per cycle from |g>: wait Exp(k_exc) to |e>; from |e>, with branching
    ratio phi emit a photon and return to |g>, else shelve to |s> and wait
    Exp(k_se).  The rates are the exact inverse of the site's
    (tau1, tau2, a) (`sparq.exact.rates_for_params`, new in 0.10.0), so
    the emitted stream has exactly the g2 of `g2_three_level`; up to
    0.9.1 an approximate mapping was used, whose g2 differed.
    """
    k_exc, k_r, k_es, k_se = _site_rates(p["tau1"], p["tau2"], p["a"])
    T_ns = T_s * 1e9
    all_times = []
    p_shelve = k_es / (k_r + k_es)
    for _ in range(n_emitters):
        times = []
        t = 0.0
        # vectorized block simulation
        block = max(1024, int(T_ns * k_exc * 0.6))
        block = min(block, 2_000_000)
        while t < T_ns:
            n = block
            dt_g = rng.exponential(1.0 / k_exc, n)
            dt_e = rng.exponential(1.0 / (k_r + k_es), n)
            shelved = rng.random(n) < p_shelve
            dt_s = np.where(shelved, rng.exponential(1.0 / k_se, n), 0.0) \
                if k_es > 0 else np.zeros(n)
            cyc = dt_g + dt_e + dt_s
            tt = t + np.cumsum(cyc)
            emit_t = tt - dt_s              # emission occurs at end of |e>
            emit = ~shelved & (emit_t < T_ns)
            if keep < 1.0:
                emit &= rng.random(n) < keep
            times.append(emit_t[emit])
            t = tt[-1]
        all_times.append(np.concatenate(times))
    em = np.sort(np.concatenate(all_times))
    return em


def simulate_photon_stream(site: EmitterSite, T_s: float,
                           rng: np.random.Generator,
                           imp: DetectorImpairments | None = None,
                           include_blinking=True):
    """Full MC HBT experiment. Returns (t_A, t_B) detector timestamp arrays (ns).

    `rate_kcps` is the detected rate while the emitter is on (both
    detectors, before dead time): a share `rho` of it comes from the
    emitter, the rest is background. Blinking switches the emitter
    light off and on; the background keeps going.
    """
    p = site.params
    rates = _site_rates(p["tau1"], p["tau2"], p["a"])
    # collection efficiency chosen to hit the site's detected signal rate
    # (from the exact steady-state emission rate, new in 0.10.0)
    r_signal = p["rate_kcps"] * 1e3 * p["rho"]
    emission_rate = site.n_emitters * _emission_rate_per_ns(*rates) * 1e9
    eta = min(1.0, r_signal / emission_rate)
    det = _simulate_emission_times(p, site.n_emitters, T_s, rng, keep=eta)
    # blinking telegraph gate
    if include_blinking and p["blinking"]:
        det = _telegraph_gate(det, p["t_on_ms"] * 1e6, p["t_off_ms"] * 1e6,
                              T_s * 1e9, rng)
    # Poissonian background
    r_bg = p["rate_kcps"] * 1e3 * (1.0 - p["rho"])
    n_bg = rng.poisson(r_bg * T_s)
    bg = rng.uniform(0, T_s * 1e9, n_bg)
    all_t = np.sort(np.concatenate([det, bg]))
    # 50/50 beamsplitter
    which = rng.random(len(all_t)) < 0.5
    t_a, t_b = all_t[which], all_t[~which]
    if imp is not None:
        t_a = _detector_chain(t_a, imp, rng)
        t_b = _detector_chain(t_b, imp, rng)
    return t_a, t_b


def _telegraph_gate(times, ton_ns, toff_ns, T_ns, rng):
    """Apply random-telegraph on/off blinking to a photon stream.

    The first state is drawn from the stationary probabilities and every
    on (off) period is exponential with mean ton_ns (toff_ns), so the
    gate is a stationary telegraph process. Periods are generated in
    vectorized blocks (new in 0.10.0; the earlier loop drew one period
    per Python step, which was too slow for microsecond blinking).
    """
    state0 = rng.random() < ton_ns / (ton_ns + toff_ns)
    n_est = int(2.2 * T_ns / (ton_ns + toff_ns)) + 16
    edges = [np.zeros(1)]
    t_end = 0.0
    start_on = state0
    while t_end < T_ns:
        n = n_est if n_est % 2 == 0 else n_est + 1
        first = ton_ns if start_on else toff_ns
        second = toff_ns if start_on else ton_ns
        dur = np.empty(n)
        dur[0::2] = rng.exponential(first, n // 2)
        dur[1::2] = rng.exponential(second, n // 2)
        e = t_end + np.cumsum(dur)
        edges.append(e)
        t_end = float(e[-1])        # an even number of periods keeps the
        #                             phase: the next block starts in the
        #                             same state as this one
    edges = np.concatenate(edges)
    idx = np.searchsorted(edges, times, side="right") - 1
    on = (idx % 2 == 0) == state0
    return times[on]


def _detector_chain(t, imp: DetectorImpairments, rng):
    """IRF jitter + dead time + afterpulsing."""
    t = np.sort(t + rng.normal(0, imp.sigma_irf_ns, len(t)))
    # dead time (sequential — vectorized via greedy pass)
    keep = np.ones(len(t), bool)
    last = -np.inf
    for i in range(len(t)):           # rates ~1e5/s -> arrays are small enough
        if t[i] - last >= imp.dead_time_ns:
            last = t[i]
        else:
            keep[i] = False
    t = t[keep]
    # afterpulsing
    ap = t[rng.random(len(t)) < imp.afterpulse_p]
    ap = ap + imp.dead_time_ns + rng.exponential(imp.afterpulse_tau_ns, len(ap))
    return np.sort(np.concatenate([t, ap]))


def correlate(t_a, t_b, cfg: HBTConfig):
    """HBT coincidence histogram from two timestamp arrays (ns).

    t_b must be sorted ascending (it is searched by bisection); an
    unsorted t_b is refused with ValueError. t_a may be in any order.
    """
    if np.any(np.diff(np.asarray(t_b, dtype=float)) < 0):
        raise ValueError("t_b must be sorted ascending")
    hist = np.zeros(cfg.n_bins)
    if len(t_a) == 0 or len(t_b) == 0:
        return hist
    lo = np.searchsorted(t_b, t_a - cfg.tau_max)
    hi = np.searchsorted(t_b, t_a + cfg.tau_max)
    diffs = []
    for i in range(len(t_a)):
        if hi[i] > lo[i]:
            diffs.append(t_b[lo[i]:hi[i]] - t_a[i])
    if not diffs:
        return hist
    d = np.concatenate(diffs)
    hist, _ = np.histogram(d, bins=cfg.n_bins, range=(-cfg.tau_max, cfg.tau_max))
    return hist.astype(np.float32)
