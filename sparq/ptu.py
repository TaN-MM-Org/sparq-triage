"""PicoQuant PTU time-tag files (new in 0.10.0).

A PTU file is a tagged header followed by 32-bit TTTR records, written
as the TCSPC hardware delivers them. This module reads the header and
decodes the records of all twelve record types that PicoQuant's file
documentation defines (PicoHarp 300, HydraHarp V1 and V2,
TimeHarp 260N and 260P, and the MultiHarp / PicoHarp 330 "generic"
formats, each in T2 and T3 mode). It follows the format description
and the demo reader that PicoQuant publishes
(https://github.com/PicoQuant/PicoQuant-Time-Tagged-File-Format-Demos,
files doc/apx_tttrrecords.html, doc/apx_tagtypes.html and
PTU/Python/Read_PTU.py), and numbers the channels as that demo prints
them:

* PicoHarp 300 T2: the record's 4-bit channel field (0 to 4).
* HydraHarp / TimeHarp 260 / generic T2: the sync input is channel 0,
  input k of the record's 6-bit field is channel k + 1.
* T3 (all): the record's channel field (PicoHarp 300: 1 to 4; the
  others: 0-based inputs).

Times. T2: overflow count times the overflow period plus the time
tag, times MeasDesc_GlobalResolution. T3: the same with nsync gives the
sync time, and dtime * MeasDesc_Resolution is added (the photon's
delay after its sync pulse). Times are returned in ns as float64, and
exactly as integers too: `ticks` (T2: the time in units of
MeasDesc_GlobalResolution; T3: the sync count) and, for T3, `dtime`
(in units of MeasDesc_Resolution).

The tests decode files of every record type, with overflows and
markers, and compare with two independent readers (`ptufile` by
C. Gohlke and `phconvert` by A. Ingargiola) when they are installed.
No PTU file recorded on real hardware was available for the tests;
the files are written to the format description by `save_ptu_t2`
and a T3 counterpart in the tests.

Other PicoQuant formats (PHU histograms, the older PT2/PT3/HT2/HT3
files) and other vendors' binary formats are not read.
"""
from __future__ import annotations

import struct

import numpy as np

__all__ = ["read_ptu", "load_ptu_timetags", "save_ptu_t2", "RECORD_TYPES"]

# tag types (doc/apx_tagtypes.html)
_TY = {
    0xFFFF0008: "Empty8", 0x00000008: "Bool8", 0x10000008: "Int8",
    0x11000008: "BitSet64", 0x12000008: "Color8", 0x20000008: "Float8",
    0x21000008: "TDateTime", 0x2001FFFF: "Float8Array",
    0x4001FFFF: "AnsiString", 0x4002FFFF: "WideString",
    0xFFFFFFFF: "BinaryBlob",
}

# record types: name -> (code, mode, family)
RECORD_TYPES = {
    "PicoHarpT2": (0x00010203, "T2", "PT"),
    "PicoHarpT3": (0x00010303, "T3", "PT"),
    "HydraHarpT2": (0x00010204, "T2", "HH1"),
    "HydraHarpT3": (0x00010304, "T3", "HH1"),
    "HydraHarp2T2": (0x01010204, "T2", "HH2"),
    "HydraHarp2T3": (0x01010304, "T3", "HH2"),
    "TimeHarp260NT2": (0x00010205, "T2", "HH2"),
    "TimeHarp260NT3": (0x00010305, "T3", "HH2"),
    "TimeHarp260PT2": (0x00010206, "T2", "HH2"),
    "TimeHarp260PT3": (0x00010306, "T3", "HH2"),
    "GenericT2": (0x00010207, "T2", "HH2"),
    "GenericT3": (0x00010307, "T3", "HH2"),
}
_BY_CODE = {v[0]: k for k, v in RECORD_TYPES.items()}


def _read_header(fh):
    magic = fh.read(8)
    if magic.rstrip(b"\0") != b"PQTTTR":
        raise ValueError("not a PTU file (magic is not PQTTTR)")
    version = fh.read(8).rstrip(b"\0").decode("ascii", "replace")
    tags = {}
    while True:
        raw = fh.read(48)
        if len(raw) < 48:
            raise ValueError("PTU header ends before Header_End")
        ident = raw[:32].rstrip(b"\0").decode("ascii", "replace")
        idx, typ = struct.unpack("<iI", raw[32:40])
        val = raw[40:48]
        kind = _TY.get(typ)
        if kind is None:
            raise ValueError(f"unknown tag type 0x{typ:08X} in {ident!r}")
        if kind in ("Empty8",):
            v = None
        elif kind == "Bool8":
            v = struct.unpack("<q", val)[0] != 0
        elif kind in ("Int8", "BitSet64", "Color8"):
            v = struct.unpack("<q", val)[0]
        elif kind in ("Float8", "TDateTime"):
            v = struct.unpack("<d", val)[0]
        else:
            n = struct.unpack("<q", val)[0]
            data = fh.read(n)
            if len(data) != n:
                raise ValueError(f"PTU header truncated in {ident!r}")
            if kind == "AnsiString":
                v = data.rstrip(b"\0").decode("utf-8", "replace")
            elif kind == "WideString":
                v = data.decode("utf-16le", "replace").rstrip("\0")
            elif kind == "Float8Array":
                v = np.frombuffer(data, "<f8").copy()
            else:
                v = data
        name = ident if idx < 0 else f"{ident}({idx})"
        tags[name] = v
        if ident == "Header_End":
            return version, tags


def _decode(rec, code, mode, family):
    """Photon events of the records: (channel, coarse, dtime), coarse in
    global-resolution units (T2 time, T3 sync count)."""
    rec = rec.astype(np.uint64)
    if family == "PT":
        ch = (rec >> np.uint64(28)).astype(np.int64)
        if mode == "T2":
            tt = rec & np.uint64(0x0FFFFFFF)
            special = ch == 15
            ofl = special & ((tt & np.uint64(0xF)) == 0)
            inc = np.where(ofl, 210698240, 0).astype(np.uint64)
            dt = np.zeros(rec.size, np.int64)
        else:
            dt = ((rec >> np.uint64(16)) & np.uint64(0xFFF)).astype(np.int64)
            tt = rec & np.uint64(0xFFFF)
            special = ch == 15
            ofl = special & (dt == 0)
            inc = np.where(ofl, 65536, 0).astype(np.uint64)
        photon = ~special
    else:
        sp = (rec >> np.uint64(31)).astype(bool)
        ch = ((rec >> np.uint64(25)) & np.uint64(0x3F)).astype(np.int64)
        if mode == "T2":
            tt = rec & np.uint64(0x1FFFFFF)
            dt = np.zeros(rec.size, np.int64)
            wrap = 33552000 if family == "HH1" else 33554432
        else:
            dt = ((rec >> np.uint64(10)) & np.uint64(0x7FFF)).astype(np.int64)
            tt = rec & np.uint64(0x3FF)
            wrap = 1024
        ofl = sp & (ch == 63)
        if family == "HH1":
            n_ofl = np.ones(rec.size, np.uint64)
        else:                       # count in the time field; 0 = one
            n_ofl = np.where(tt == 0, np.uint64(1), tt)
        inc = np.where(ofl, n_ofl * np.uint64(wrap), np.uint64(0))
        if mode == "T2":
            sync = sp & (ch == 0)   # the sync input, reported as channel 0
            photon = ~sp | sync
            ch = np.where(sync, 0, ch + 1)
        else:
            photon = ~sp
    coarse = np.cumsum(inc, dtype=np.uint64) + tt
    return ch[photon], coarse[photon].astype(np.int64), dt[photon]


def read_ptu(path):
    """Read a PTU file.

    Returns a dict with record_type (name, see RECORD_TYPES), mode ("T2"
    or "T3"), tags (the header tags, name -> value; indexed tags as
    "Name(i)"), channel (int64 per photon event, numbered as described
    in the module docstring), time_ns (float64, arrival times in ns),
    ticks (int64: T2 time in units of the global resolution; T3 sync
    count), dtime (int64, T3 only, in units of MeasDesc_Resolution;
    zeros for T2), global_resolution_s and resolution_s. Markers and
    overflow records are consumed and not returned.

    Refuses files whose magic is not PQTTTR, unknown tag or record
    types, and files with fewer records than TTResult_NumberOfRecords
    says.
    """
    with open(path, "rb") as fh:
        version, tags = _read_header(fh)
        code = int(tags.get("TTResultFormat_TTTRRecType", -1))
        if code not in _BY_CODE:
            raise ValueError(f"unknown or missing TTTR record type "
                             f"0x{code & 0xFFFFFFFF:08X}")
        n = int(tags.get("TTResult_NumberOfRecords", -1))
        if n < 0:
            raise ValueError("TTResult_NumberOfRecords is missing")
        rec = np.fromfile(fh, dtype="<u4", count=n)
    if rec.size != n:
        raise ValueError(f"file ends after {rec.size} of {n} records")
    name = _BY_CODE[code]
    _, mode, family = RECORD_TYPES[name]
    gres = float(tags["MeasDesc_GlobalResolution"])
    res = float(tags.get("MeasDesc_Resolution", 0.0))
    ch, coarse, dt = _decode(rec, code, mode, family)
    time_ns = coarse * (gres * 1e9)
    if mode == "T3":
        time_ns = time_ns + dt * (res * 1e9)
    return dict(record_type=name, mode=mode, version=version, tags=tags,
                channel=ch, ticks=coarse, dtime=dt, time_ns=time_ns,
                global_resolution_s=gres, resolution_s=res)


def load_ptu_timetags(path, channel_a=1, channel_b=2):
    """Two sorted arrays of arrival times (ns) from a PTU file, ready
    for `correlate`: (t_a, t_b) for channels `channel_a` and `channel_b`
    (numbered as in the module docstring; for a HydraHarp in T2 mode the
    first two inputs are 1 and 2). Refuses a channel with no events,
    naming the channels that have some."""
    d = read_ptu(path)
    out = []
    for c in (channel_a, channel_b):
        m = d["channel"] == c
        if not m.any():
            have = sorted(set(d["channel"].tolist()))
            raise ValueError(f"channel {c} has no events; channels with "
                             f"events: {have}")
        out.append(np.sort(d["time_ns"][m]))
    return out[0], out[1]


def _tag(ident, typ, value, idx=-1):
    code = {v: k for k, v in _TY.items()}[typ]
    head = ident.encode("ascii").ljust(32, b"\0") + struct.pack("<iI", idx,
                                                              code)
    if typ in ("Int8", "BitSet64", "Color8"):
        return head + struct.pack("<q", int(value))
    if typ == "Bool8":
        return head + struct.pack("<q", 1 if value else 0)
    if typ in ("Float8", "TDateTime"):
        return head + struct.pack("<d", float(value))
    if typ == "Empty8":
        return head + b"\0" * 8
    if typ == "AnsiString":
        data = value.encode("utf-8") + b"\0"
        data = data.ljust((len(data) + 7) // 8 * 8, b"\0")
        return head + struct.pack("<q", len(data)) + data
    raise ValueError(typ)


def save_ptu_t2(path, channels, times_ns, resolution_s=1e-12,
                record_type="GenericT2"):
    """Write photon events as a T2-mode PTU file (new in 0.10.0).

    channels : channel of each event, numbered as `read_ptu` returns
        them (PicoHarpT2: 0 to 4; the other T2 types: 0 = sync input,
        k + 1 = input k, up to 64).
    times_ns : arrival times (ns), non-negative; they are rounded to
        whole ticks of `resolution_s` (the global resolution) and must
        be in increasing order after rounding.
    record_type : any T2 name of RECORD_TYPES.

    Overflow records are inserted where the time tag wraps (for the
    types that allow it, several overflows in one record). The header
    holds the tags `read_ptu` needs plus a file comment; files are
    meant for tests and for exchanging data with programs that read
    PTU, and the tests check them with two independent readers.
    """
    if record_type not in RECORD_TYPES or \
            RECORD_TYPES[record_type][1] != "T2":
        raise ValueError("record_type must be a T2 type of RECORD_TYPES")
    code, _, family = RECORD_TYPES[record_type]
    ch = np.asarray(channels, dtype=np.int64)
    t = np.asarray(times_ns, dtype=float)
    if ch.shape != t.shape or ch.ndim != 1:
        raise ValueError("channels and times_ns must be equal-length 1-D")
    if not (resolution_s > 0):
        raise ValueError("resolution_s must be positive")
    ticks = np.rint(t * 1e-9 / resolution_s).astype(np.int64)
    if np.any(ticks < 0) or np.any(np.diff(ticks) < 0):
        raise ValueError("times must be non-negative and increasing")
    if family == "PT":
        if np.any((ch < 0) | (ch > 4)):
            raise ValueError("PicoHarpT2 channels are 0 to 4")
        wrap, many = 210698240, False
    else:
        if np.any((ch < 0) | (ch > 64)):
            raise ValueError("channels are 0 (sync) to 64")
        wrap, many = (33552000, False) if family == "HH1" else \
            (33554432, True)
    recs = []
    base = 0
    for c, k in zip(ch.tolist(), ticks.tolist()):
        n_ofl = (k - base) // wrap
        while n_ofl > 0:
            m = min(n_ofl, 0x1FFFFFF) if many else 1
            if family == "PT":
                recs.append((15 << 28) | 0)           # overflow
            else:
                recs.append((1 << 31) | (63 << 25) | (m if many else 0))
            base += m * wrap
            n_ofl -= m
        tt = k - base
        if family == "PT":
            recs.append((c << 28) | tt)
        elif c == 0:
            recs.append((1 << 31) | (0 << 25) | tt)   # sync event
        else:
            recs.append(((c - 1) << 25) | tt)
    rec = np.array(recs, dtype="<u4")
    head = b"PQTTTR\0\0" + b"1.0.00\0\0"
    head += _tag("File_Comment", "AnsiString",
                 "written by sparq.ptu.save_ptu_t2")
    head += _tag("Measurement_Mode", "Int8", 2)
    head += _tag("MeasDesc_GlobalResolution", "Float8", resolution_s)
    head += _tag("MeasDesc_Resolution", "Float8", resolution_s)
    head += _tag("TTResultFormat_TTTRRecType", "Int8", code)
    head += _tag("TTResultFormat_BitsPerRecord", "Int8", 32)
    head += _tag("TTResult_NumberOfRecords", "Int8", rec.size)
    head += _tag("Header_End", "Empty8", None)
    with open(path, "wb") as fh:
        fh.write(head)
        fh.write(rec.tobytes())
