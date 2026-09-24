"""PicoQuant PTU files (new in 0.10.0). Files of every record type are
written from the bit layouts of PicoQuant's format description (T2 by
`save_ptu_t2`, T3 by an encoder in this file) and decoded by
`read_ptu`; when installed, two independent readers (`ptufile`,
`phconvert`) decode the same files and must agree."""
import struct

import numpy as np
import pytest

from sparq import (EmitterSite, HBTConfig, correlate,
                   simulate_photon_stream)
from sparq.ptu import (RECORD_TYPES, _tag, load_ptu_timetags, read_ptu,
                       save_ptu_t2)

T2 = [k for k, v in RECORD_TYPES.items() if v[1] == "T2"]
T3 = [k for k, v in RECORD_TYPES.items() if v[1] == "T3"]


def _t2_data(name, seed, big_gaps=True):
    rng = np.random.default_rng(seed)
    n = 3000
    gaps = rng.exponential(1e5, n)                    # ns
    if big_gaps:
        gaps[::500] += 5e6                   # > 128 overflows of 33.5 us
    t = np.cumsum(gaps)
    ch = rng.integers(0, 5, n)                        # 0 = sync (not PT)
    return ch, t


def _write_t3(path, name, ch, nsync, dtime, gres, res, rng):
    """T3 encoder written from the record layouts (independent of
    sparq.ptu's decoder), with overflow and marker records."""
    code, _, fam = RECORD_TYPES[name]
    wrap = 65536 if fam == "PT" else 1024
    recs, base = [], 0
    for c, ns, dt in zip(ch.tolist(), nsync.tolist(), dtime.tolist()):
        n_ofl = (ns - base) // wrap
        while n_ofl > 0:
            m = 1 if fam in ("PT", "HH1") else min(n_ofl, 1023)
            recs.append((15 << 28) if fam == "PT" else
                        (1 << 31) | (63 << 25) | (0 if fam == "HH1" else m))
            base += m * wrap
            n_ofl -= m
        if rng.random() < 0.02:                       # a marker record
            mk = int(rng.integers(1, 16))
            recs.append((15 << 28) | (mk << 16) | (ns - base) if fam == "PT"
                        else (1 << 31) | (mk << 25) | (ns - base))
        recs.append((c << 28) | (dt << 16) | (ns - base) if fam == "PT"
                    else (c << 25) | (dt << 10) | (ns - base))
    rec = np.array(recs, dtype="<u4")
    head = b"PQTTTR\0\0" + b"1.0.00\0\0"
    for t in (("Measurement_Mode", "Int8", 3),
              ("MeasDesc_GlobalResolution", "Float8", gres),
              ("MeasDesc_Resolution", "Float8", res),
              ("TTResult_SyncRate", "Int8", int(round(1 / gres))),
              ("TTResultFormat_TTTRRecType", "Int8", code),
              ("TTResultFormat_BitsPerRecord", "Int8", 32),
              ("TTResult_NumberOfRecords", "Int8", rec.size),
              ("Header_End", "Empty8", None)):
        head += _tag(*t)
    with open(path, "wb") as fh:
        fh.write(head + rec.tobytes())


def _t3_data(name, seed):
    rng = np.random.default_rng(seed)
    fam = RECORD_TYPES[name][2]
    n = 3000
    ns = np.sort(rng.integers(0, 3_000_000, n))
    ns[1500:] += 400_000                              # a long gap
    dt = rng.integers(0, 4095 if fam == "PT" else 12499, n)
    ch = rng.integers(1, 5, n) if fam == "PT" else rng.integers(0, 4, n)
    return ch, ns, dt, rng


@pytest.mark.parametrize("name", T2)
def test_t2_round_trip(tmp_path, name):
    ch, t = _t2_data(name, 1)
    fam = RECORD_TYPES[name][2]
    res = 4e-12 if fam == "PT" else 1e-12
    p = tmp_path / f"{name}.ptu"
    save_ptu_t2(p, ch, t, res, name)
    d = read_ptu(p)
    assert d["record_type"] == name and d["mode"] == "T2"
    assert np.array_equal(d["channel"], ch)
    ticks = np.rint(t * 1e-9 / res).astype(np.int64)
    assert np.array_equal(d["ticks"], ticks)
    assert np.abs(d["time_ns"] - t).max() <= 0.5 * res * 1e9 * (1 + 1e-9)


@pytest.mark.parametrize("name", T3)
def test_t3_decoding(tmp_path, name):
    ch, ns, dt, rng = _t3_data(name, 2)
    fam = RECORD_TYPES[name][2]
    res = 4e-12 if fam == "PT" else 1e-12
    p = tmp_path / f"{name}.ptu"
    _write_t3(p, name, ch, ns, dt, 12.5e-9, res, rng)
    d = read_ptu(p)
    assert d["mode"] == "T3"
    assert np.array_equal(d["channel"], ch)
    assert np.array_equal(d["ticks"], ns)
    assert np.array_equal(d["dtime"], dt)
    assert np.allclose(d["time_ns"], ns * 12.5 + dt * res * 1e9, rtol=0,
                       atol=1e-6)


def _phconvert(p):
    pq = pytest.importorskip("phconvert.pqreader")
    records, spec, _ = pq.ptu_reader(str(p))
    last = None
    for ovc in ("loop", "base"):      # one of them fails on some numpy
        try:
            return pq.process_pturecords(records.copy(), spec,
                                         ovcfunc=ovc), spec
        except Exception as err:      # noqa: BLE001 - try the other
            last = err
    raise last


@pytest.mark.parametrize("name", T2 + T3)
def test_independent_readers_agree(tmp_path, name):
    """ptufile and phconvert decode the same files to the same times,
    delays and (up to their channel numbering) channels. ptufile adds
    at most 127 overflows from one T2 overflow record of the types
    that store a count (it differs from PicoQuant's demo reader and
    phconvert once a record holds 128 or more), so its T2 check uses
    data without long gaps."""
    code, mode, fam = RECORD_TYPES[name]
    p = tmp_path / f"{name}.ptu"
    if mode == "T2":
        ch, t = _t2_data(name, 3)
        save_ptu_t2(p, ch, t, 1e-12, name)
    else:
        ch, ns, dt, rng = _t3_data(name, 4)
        _write_t3(p, name, ch, ns, dt, 12.5e-9, 1e-12, rng)
    d = read_ptu(p)
    try:
        (times, dets, dtime, _), spec = _phconvert(p)
    except pytest.skip.Exception:
        times = None
    if times is not None:
        dets = dets.astype(np.int64)
        if fam == "PT":
            keep = dets < 15
            ref_ch = dets[keep]
        else:
            sp = 1 << (spec["channel_bit"] - 1)
            if mode == "T2":
                keep = (dets < sp) | (dets == sp)
                ref_ch = np.where(dets[keep] == sp, 0, dets[keep] + 1)
            else:
                keep = dets < sp
                ref_ch = dets[keep]
        assert np.array_equal(times[keep].astype(np.int64), d["ticks"])
        assert np.array_equal(ref_ch, d["channel"])
        if mode == "T3":
            assert np.array_equal(dtime[keep].astype(np.int64), d["dtime"])
    ptufile = pytest.importorskip("ptufile")
    if mode == "T2":
        ch, t = _t2_data(name, 3, big_gaps=False)
        save_ptu_t2(p, ch, t, 1e-12, name)
        d = read_ptu(p)
    with ptufile.PtuFile(p) as f:
        r = f.decode_records()
    ph = r[r["channel"] >= 0]
    assert np.array_equal(ph["time"].astype(np.int64), d["ticks"])
    if mode == "T3":
        assert np.array_equal(ph["dtime"].astype(np.int64), d["dtime"])


def test_simulated_tags_through_a_ptu_file(tmp_path):
    site = EmitterSite(dict(tau1=15.0, tau2=250.0, a=0.3, rate_kcps=150.0,
                            rho=0.95, blinking=False), 1)
    t_a, t_b = simulate_photon_stream(site, 0.5, np.random.default_rng(5))
    ch = np.r_[np.ones(t_a.size, int), np.full(t_b.size, 2)]
    t = np.r_[t_a, t_b]
    o = np.argsort(t, kind="stable")
    p = tmp_path / "hbt.ptu"
    save_ptu_t2(p, ch[o], t[o], 1e-12, "HydraHarp2T2")
    a, b = load_ptu_timetags(p, 1, 2)
    assert a.size == t_a.size and b.size == t_b.size
    # rounded to 1 ps ticks (plus float round-off of times ~5e8 ns)
    assert np.abs(a - t_a).max() <= 5.01e-4
    assert np.abs(b - t_b).max() <= 5.01e-4
    h0 = correlate(t_a, t_b, HBTConfig())
    h1 = correlate(a, b, HBTConfig())
    assert abs(h0.sum() - h1.sum()) <= 2
    assert np.abs(h0 - h1).max() <= 2
    with pytest.raises(ValueError, match="no events"):
        load_ptu_timetags(p, 1, 7)


def test_refusals(tmp_path):
    bad = tmp_path / "bad.ptu"
    bad.write_bytes(b"NOTPTU\0\0" + b"\0" * 64)
    with pytest.raises(ValueError, match="PQTTTR"):
        read_ptu(bad)
    ch, t = _t2_data("GenericT2", 6)
    p = tmp_path / "trunc.ptu"
    save_ptu_t2(p, ch, t, 1e-12)
    raw = p.read_bytes()
    p.write_bytes(raw[:-40])
    with pytest.raises(ValueError, match="records"):
        read_ptu(p)
    with pytest.raises(ValueError):
        save_ptu_t2(p, [1, 2], [5.0, 1.0])            # not increasing
    with pytest.raises(ValueError):
        save_ptu_t2(p, [7], [1.0], record_type="PicoHarpT2")
    with pytest.raises(ValueError):
        save_ptu_t2(p, [1], [1.0], record_type="GenericT3")
    # an unknown record type is refused
    head = b"PQTTTR\0\0" + b"1.0.00\0\0" + _tag(
        "TTResultFormat_TTTRRecType", "Int8", 0x12345) + _tag(
        "TTResult_NumberOfRecords", "Int8", 0) + _tag(
        "MeasDesc_GlobalResolution", "Float8", 1e-12) + _tag(
        "Header_End", "Empty8", None)
    q = tmp_path / "unk.ptu"
    q.write_bytes(head)
    with pytest.raises(ValueError, match="record type"):
        read_ptu(q)
    assert struct.calcsize("<iI") == 8
