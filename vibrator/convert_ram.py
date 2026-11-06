#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: The LineageOS Project
# SPDX-License-Identifier: Apache-2.0
#

"""convert_ram: convert Awinic haptic .bin files to editable JSON and back.

    convert_ram.py haptic_ram.bin  -o haptic_ram.json        # bin  -> json
    convert_ram.py --compile haptic_ram.json -o haptic_ram.bin   # json -> bin
    convert_ram.py haptic_rtp.bin --split -o haptic_rtp.json    # split large RTP

Two bin kinds are understood (detected automatically when decoding):

  ram  haptic_ram.bin, the SRAM waveform library written by the driver's ram_update.
       Big endian: [0:2] checksum = sum(file[2:]) & 0xFFFF, [2:4] base address,
       then the SRAM image: version byte, N x (start, end) u16 pairs (inclusive
       addresses), then N waves of signed 8-bit samples placed back to back.
       Wave addresses and the checksum are derived, so you can change a wave's
       length freely; everything is re-laid-out when compiling.

  rtp  haptic_rtp*.bin, headerless signed 8-bit PCM streamed through the RTP FIFO.

JSON keys that start with "_" are informational and ignored by --compile.
In "samples", an int is one int8 sample and [value, count] is a run of `count`
identical samples (used to keep long streams small).

--split (RTP only): large RTP streams such as haptic_rtp*.bin contain repeated
~1.6 s active sections separated by long near-silence gaps (~41 k samples /
~1.7 s).  --split cuts the stream at the mid-points of those gaps and writes
one JSON file per section (output_part00.json, output_part01.json, ...).
Concatenating the compiled .bin parts recovers the original bytes.
"""
import argparse
import json
import os
import struct
import sys

SRAM_END = 0x1FFF
RATE = 24000
RAM_MAX_WAVES = 127  # sequencer slot values >= 0x80 mean "wait"
WRAP = 96
# Silence longer than this (samples) is treated as a natural RTP split boundary.
SPLIT_THRESHOLD = 10000
HELP = ("samples: int = one int8 sample, [value, count] = run of count identical samples. "
        "Keys starting with '_' are informational and ignored when compiling.")


class Error(Exception):
    pass


class Rows(list):
    """Sample list; rendered wrapped over several lines."""


# ---------------------------------------------------------------- samples ---

def to_i8(data):
    return [x - 256 if x > 127 else x for x in data]


def compress(samples, min_run):
    """int8 list -> tokens (ints and [value, count] runs of >= min_run)."""
    if min_run < 2:
        return Rows(samples)
    out = Rows()
    i, n = 0, len(samples)
    while i < n:
        v = samples[i]
        j = i + 1
        while j < n and samples[j] == v:
            j += 1
        if j - i >= min_run:
            out.append([v, j - i])
        else:
            out.extend(samples[i:j])
        i = j
    return out


def expand(tokens, where):
    if not isinstance(tokens, list):
        raise Error("%s: must be a list" % where)
    out = bytearray()
    for i, t in enumerate(tokens):
        if isinstance(t, list):
            if len(t) != 2 or any(type(x) is not int for x in t):
                raise Error("%s[%d]: a run must be [value, count]" % (where, i))
            v, n = t
            if n < 1:
                raise Error("%s[%d]: run count must be >= 1" % (where, i))
            _check(v, "%s[%d]" % (where, i))
            out += bytes((v & 0xFF,)) * n
        else:
            _check(t, "%s[%d]" % (where, i))
            out.append(t & 0xFF)
    return bytes(out)


def _check(v, where):
    if type(v) is not int:
        raise Error("%s: sample must be an integer, got %r" % (where, v))
    if not -128 <= v <= 127:
        raise Error("%s: sample %d outside int8 range -128..127" % (where, v))


def number(v, name):
    """int, or string like '0x0800'."""
    try:
        if type(v) is int:
            return v
        if isinstance(v, str):
            return int(v, 0)
    except ValueError:
        pass
    raise Error("%s: expected an integer or a string like '0x0800', got %r" % (name, v))


# -------------------------------------------------------------------- ram ---

def parse_ram(blob):
    """-> (base, version, [wave bytes]); strict about the canonical layout."""
    if len(blob) < 9:
        raise Error("too short for a RAM image (%d bytes)" % len(blob))
    stored = struct.unpack(">H", blob[:2])[0]
    if stored != sum(blob[2:]) & 0xFFFF:
        raise Error("RAM checksum mismatch")
    base = struct.unpack(">H", blob[2:4])[0]
    image = blob[4:]
    if base + len(image) - 1 > SRAM_END:
        raise Error("image ends at 0x%04X, beyond SRAM 0x%04X" % (base + len(image) - 1, SRAM_END))
    first = struct.unpack(">H", image[1:3])[0]
    n = (first - base - 1) // 4
    if n < 1 or first != base + 1 + 4 * n or 1 + 4 * n > len(image):
        raise Error("bad wave table (first wave at 0x%04X)" % first)
    waves, pos = [], first
    for i in range(n):
        start, end = struct.unpack(">HH", image[1 + 4 * i: 5 + 4 * i])
        if start != pos or end < start or end - base >= len(image):
            raise Error("wave %d: range 0x%04X-0x%04X is not contiguous" % (i + 1, start, end))
        waves.append(image[start - base: end - base + 1])
        pos = end + 1
    if pos - base != len(image):
        raise Error("%d trailing bytes after the last wave" % (len(image) - (pos - base)))
    return base, image[0], waves


def build_ram(base, version, ram_end, waves):
    n = len(waves)
    if not 1 <= n <= RAM_MAX_WAVES:
        raise Error("need 1..%d waves, got %d" % (RAM_MAX_WAVES, n))
    pos = base + 1 + 4 * n
    table = bytearray()
    for i, w in enumerate(waves):
        if not w:
            raise Error("waves[%d]: empty wave" % i)
        table += struct.pack(">HH", pos & 0xFFFF, (pos + len(w) - 1) & 0xFFFF)
        pos += len(w)
    if pos - 1 > ram_end:
        raise Error("RAM overflow: ends at 0x%04X, limit 0x%04X (%d bytes too many)"
                    % (pos - 1, ram_end, pos - 1 - ram_end))
    body = struct.pack(">H", base) + bytes((version,)) + bytes(table) + b"".join(waves)
    return struct.pack(">H", sum(body) & 0xFFFF) + body


def ram_to_doc(blob, rate):
    base, version, waves = parse_ram(blob)
    pos = base + 1 + 4 * len(waves)
    items = []
    for i, w in enumerate(waves):
        items.append({
            "_index": i + 1,
            "_start": "0x%04X" % pos,
            "_end": "0x%04X" % (pos + len(w) - 1),
            "_length": len(w),
            "samples": Rows(to_i8(w)),
            "_duration_ms": round(len(w) * 1000.0 / rate, 3),
        })
        pos += len(w)
    used = pos - base
    return {
        "type": "ram",
        "_help": HELP,
        "sample_rate_hz": rate,
        "base_addr": "0x%04X" % base,
        "version": "0x%02X" % version,
        "ram_end": "0x%04X" % SRAM_END,
        "_info": {"file_bytes": len(blob), "ram_used": used,
                  "ram_free": SRAM_END - base + 1 - used},
        "waves": items,
    }


def compile_ram(doc):
    base = number(doc.get("base_addr", 0x0800), "base_addr")
    version = number(doc.get("version", 0x85), "version")
    ram_end = number(doc.get("ram_end", SRAM_END), "ram_end")
    if not 0 <= base <= 0xFFFF or not 0 <= version <= 0xFF or not 0 <= ram_end <= 0xFFFF:
        raise Error("base_addr/ram_end must fit u16 and version u8")
    items = doc.get("waves")
    if not isinstance(items, list) or not items:
        raise Error("'waves' must be a non-empty list")
    waves = []
    for i, w in enumerate(items):
        if not isinstance(w, dict) or "samples" not in w:
            raise Error("waves[%d]: needs a 'samples' list" % i)
        waves.append(expand(w["samples"], "waves[%d].samples" % i))
    return build_ram(base, version, ram_end, waves)


# -------------------------------------------------------------------- rtp ---

def rtp_to_doc(blob, rate, min_run, part_index=None, part_count=None):
    info = {"length_samples": len(blob),
            "duration_ms": round(len(blob) * 1000.0 / rate, 3)}
    if part_index is not None and part_count is not None:
        info["part"] = part_index
        info["parts"] = part_count
    return {
        "type": "rtp",
        "_help": HELP,
        "sample_rate_hz": rate,
        "_info": info,
        "samples": compress(to_i8(blob), min_run),
    }


def find_rtp_split_points(blob, threshold=SPLIT_THRESHOLD):
    """Return ordered sample indices at which to cut an RTP stream.

    Cuts are placed at the mid-point of every near-silence run whose length
    is >= threshold.  Leading/trailing silence is kept with the first/last
    part so that concatenating the compiled parts recovers the original bytes.
    Returns an empty list when no qualifying gaps exist (caller keeps one part).
    """
    samples = to_i8(blob)
    n = len(samples)
    mids = []
    i = 0
    while i < n:
        if abs(samples[i]) <= 1:
            j = i
            while j < n and abs(samples[j]) <= 1:
                j += 1
            if j - i >= threshold:
                # Skip pure leading silence (no preceding activity) and pure
                # trailing silence (nothing after); only inter-section gaps
                # become split points.
                if i > 0 and j < n:
                    mids.append((i + j) // 2)
            i = j
        else:
            i += 1
    return mids


def split_rtp_blob(blob, threshold=SPLIT_THRESHOLD):
    """-> list of (start, end) half-open ranges covering the whole blob."""
    cuts = find_rtp_split_points(blob, threshold)
    if not cuts:
        return [(0, len(blob))]
    bounds = [0] + cuts + [len(blob)]
    return list(zip(bounds[:-1], bounds[1:]))


def compile_rtp(doc):
    data = expand(doc.get("samples"), "samples")
    if not data:
        raise Error("'samples' is empty")
    return data


# ------------------------------------------------------------------- json ---

def _tok(t):
    return "[%d, %d]" % (t[0], t[1]) if isinstance(t, list) else str(t)


def emit(o, lvl=0):
    pad, pad1 = "  " * lvl, "  " * (lvl + 1)
    if isinstance(o, Rows):
        if not o:
            return "[]"
        lines, cur = [], ""
        for t in o:
            piece = _tok(t) + ","
            if cur and len(pad1) + len(cur) + 1 + len(piece) > WRAP:
                lines.append(cur)
                cur = piece
            else:
                cur = cur + " " + piece if cur else piece
        lines.append(cur)
        lines[-1] = lines[-1][:-1]
        return "[\n" + "\n".join(pad1 + ln for ln in lines) + "\n" + pad + "]"
    if isinstance(o, dict):
        body = ",\n".join("%s%s: %s" % (pad1, json.dumps(k), emit(v, lvl + 1)) for k, v in o.items())
        return "{\n" + body + "\n" + pad + "}"
    if isinstance(o, list):
        if not o:
            return "[]"
        return "[\n" + ",\n".join(pad1 + emit(v, lvl + 1) for v in o) + "\n" + pad + "]"
    return json.dumps(o)


def compile_doc(doc):
    if not isinstance(doc, dict):
        raise Error("top level must be a JSON object")
    kind = doc.get("type")
    if kind == "ram":
        return compile_ram(doc)
    if kind == "rtp":
        return compile_rtp(doc)
    raise Error("'type' must be \"ram\" or \"rtp\", got %r" % (kind,))


# -------------------------------------------------------------------- cli ---

def decode(blob, kind, rate, min_run, part_index=None, part_count=None):
    if kind in ("auto", "ram"):
        try:
            doc = ram_to_doc(blob, rate)
        except Error as e:
            if kind == "ram":
                raise
            print("convert_ram: not a RAM image (%s); treating as raw RTP" % e, file=sys.stderr)
            doc = rtp_to_doc(blob, rate, min_run, part_index, part_count)
    else:
        doc = rtp_to_doc(blob, rate, min_run, part_index, part_count)
    text = emit(doc) + "\n"
    if compile_doc(json.loads(text)) != blob:
        raise Error("internal: JSON does not round-trip to the original bytes")
    return text


def decode_split(blob, kind, rate, min_run, threshold, out_base):
    """Write one JSON per natural RTP section; return list of paths written."""
    if kind == "ram":
        raise Error("--split is only meaningful for RTP streams")
    # Force RTP path (auto would still fall through after a failed RAM parse).
    ranges = split_rtp_blob(blob, threshold)
    if len(ranges) == 1:
        print("convert_ram: no silence gaps >= %d samples; writing a single part"
              % threshold, file=sys.stderr)
    paths = []
    n = len(ranges)
    for i, (start, end) in enumerate(ranges):
        part = blob[start:end]
        text = decode(part, "rtp", rate, min_run, part_index=i, part_count=n)
        path = "%s_part%02d.json" % (out_base, i)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        paths.append(path)
        print("convert_ram: wrote %s (%d samples, %.1f ms)"
              % (path, end - start, (end - start) * 1000.0 / rate), file=sys.stderr)
    return paths


def main(argv=None):
    ap = argparse.ArgumentParser(description="Awinic haptic .bin <-> editable JSON (bin -> json by default).")
    ap.add_argument("input")
    ap.add_argument("-o", "--output", help="default: input with .json/.bin suffix")
    ap.add_argument("-c", "--compile", action="store_true", help="JSON -> bin")
    ap.add_argument("-t", "--type", choices=("auto", "ram", "rtp"), default="auto",
                    help="bin kind when decoding (default: auto)")
    ap.add_argument("--rate", type=int, default=RATE, help="sample rate for the informational durations")
    ap.add_argument("--min-run", type=int, default=8,
                    help="rtp: fold runs of >= N equal samples into [value, count]; 0 disables")
    ap.add_argument("--split", action="store_true",
                    help="RTP: write one JSON per section at natural silence gaps "
                         "(output becomes <base>_partNN.json)")
    ap.add_argument("--split-threshold", type=int, default=SPLIT_THRESHOLD,
                    help="silence length (samples) that triggers a split (default: %d)"
                         % SPLIT_THRESHOLD)
    args = ap.parse_args(argv)
    out = args.output or os.path.splitext(args.input)[0] + (".bin" if args.compile else ".json")
    try:
        if args.compile:
            if args.split:
                raise Error("--split is only used when decoding to JSON")
            with open(args.input, "r", encoding="utf-8") as f:
                try:
                    doc = json.load(f)
                except ValueError as e:
                    raise Error("%s: invalid JSON: %s" % (args.input, e))
            result = compile_doc(doc)
            mode = "wb"
            with open(out, mode) as f:
                f.write(result)
        else:
            with open(args.input, "rb") as f:
                blob = f.read()
            if args.split:
                base = os.path.splitext(out)[0]
                decode_split(blob, args.type, args.rate, args.min_run,
                             args.split_threshold, base)
            else:
                result = decode(blob, args.type, args.rate, args.min_run)
                with open(out, "w", encoding="utf-8", newline="\n") as f:
                    f.write(result)
    except (Error, OSError) as e:
        print("convert_ram: error: %s" % e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
