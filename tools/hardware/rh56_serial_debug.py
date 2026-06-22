"""
Verbose serial-level diagnostics for RH56 hand register communication.

This module bypasses higher-level helpers and shows raw bytes, frame sync,
checksum validation, and parsed payload for read/write probing.

Example:
  UV_PROJECT_ENVIRONMENT=.venv310 uv run python -m rh56_controller.rh56_serial_debug \
      --port /dev/ttyUSB3 --ids 1 2 --retries 3 --verbose
"""

from __future__ import annotations

import argparse
import struct
import time
from dataclasses import dataclass
from typing import Iterable

import serial


_CMD_READ = 0x11
_CMD_WRITE = 0x12


@dataclass(frozen=True)
class RegisterProbe:
    name: str
    address: int
    read_len: int


DEFAULT_PROBES = (
    RegisterProbe("HAND_ID", 0x05E8, 1),
    RegisterProbe("STATUS", 0x064C, 6),
    RegisterProbe("ANGLE_ACT", 0x060A, 12),
    RegisterProbe("FORCE_ACT", 0x062E, 12),
    RegisterProbe("CURRENT", 0x063A, 12),
    RegisterProbe("TEMP", 0x0652, 6),
)


def _hex(data: bytes) -> str:
    return " ".join(f"{b:02X}" for b in data)


def _checksum(payload_from_id: bytes) -> int:
    return sum(payload_from_id) & 0xFF


def build_frame(hand_id: int, command: int, address: int, data: bytes) -> bytes:
    frame = bytearray([0xEB, 0x90, hand_id])
    data_len = len(data) + 3
    frame.extend([data_len, command])
    frame.extend(struct.pack("<H", address))
    frame.extend(data)
    frame.append(_checksum(bytes(frame[2:])))
    return bytes(frame)


def extract_frame(buffer: bytearray) -> bytes | None:
    """Extract one 90 EB response frame from buffer if complete."""
    while True:
        idx = buffer.find(b"\x90\xEB")
        if idx < 0:
            # Keep a tiny suffix in case next chunk starts a header continuation.
            if len(buffer) > 1:
                del buffer[:-1]
            return None
        if idx > 0:
            del buffer[:idx]
        if len(buffer) < 4:
            return None
        data_len = int(buffer[3])
        total = data_len + 5
        if total < 8 or total > 255:
            # Corrupt length; shift by one and continue searching.
            del buffer[0]
            continue
        if len(buffer) < total:
            return None
        out = bytes(buffer[:total])
        del buffer[:total]
        return out


def read_response_frame(
    ser: serial.Serial,
    expected_id: int,
    timeout_s: float,
    verbose: bool,
) -> tuple[bytes | None, dict]:
    """
    Read serial bytes and return the first valid frame for expected_id.

    Also returns diagnostics for discarded frames and residual bytes.
    """
    diag = {
        "rx_chunks": [],
        "discarded_frames": [],
        "residual": b"",
    }
    deadline = time.monotonic() + timeout_s
    buf = bytearray()

    while time.monotonic() < deadline:
        want = ser.in_waiting or 1
        chunk = ser.read(want)
        if chunk:
            if verbose:
                print(f"    RX chunk ({len(chunk):02d} B): {_hex(chunk)}")
            diag["rx_chunks"].append(chunk)
            buf.extend(chunk)

        while True:
            frame = extract_frame(buf)
            if frame is None:
                break

            valid_checksum = _checksum(frame[2:-1]) == frame[-1]
            frame_id = frame[2] if len(frame) > 2 else None
            if not valid_checksum:
                diag["discarded_frames"].append(("checksum", frame))
                if verbose:
                    print(f"    Discard frame (bad checksum): {_hex(frame)}")
                continue
            if frame_id != expected_id:
                diag["discarded_frames"].append((f"id={frame_id}", frame))
                if verbose:
                    print(
                        f"    Discard frame (ID mismatch expected={expected_id} got={frame_id}): {_hex(frame)}"
                    )
                continue
            return frame, diag

    diag["residual"] = bytes(buf)
    return None, diag


def parse_response_frame(frame: bytes) -> dict:
    data_len = frame[3]
    cmd = frame[4]
    addr = struct.unpack("<H", frame[5:7])[0]
    payload = frame[7:-1]
    return {
        "id": frame[2],
        "data_len": data_len,
        "cmd": cmd,
        "address": addr,
        "payload": payload,
        "checksum_ok": (_checksum(frame[2:-1]) == frame[-1]),
    }


def probe_read(
    ser: serial.Serial,
    hand_id: int,
    reg: RegisterProbe,
    response_timeout: float,
    verbose: bool,
) -> tuple[bool, dict]:
    req = build_frame(hand_id, _CMD_READ, reg.address, bytes([reg.read_len]))

    ser.reset_input_buffer()
    ser.write(req)
    ser.flush()

    if verbose:
        print(
            f"  TX READ {reg.name:10s} id={hand_id} addr=0x{reg.address:04X} len={reg.read_len:2d} : {_hex(req)}"
        )

    frame, diag = read_response_frame(
        ser=ser,
        expected_id=hand_id,
        timeout_s=response_timeout,
        verbose=verbose,
    )
    if frame is None:
        return False, {
            "request": req,
            "error": "timeout waiting for valid response frame",
            "diag": diag,
        }

    parsed = parse_response_frame(frame)
    payload_len = len(parsed["payload"])
    ok_len = payload_len >= reg.read_len
    ok_cmd = parsed["cmd"] == _CMD_READ
    ok_addr = parsed["address"] == reg.address
    ok = parsed["checksum_ok"] and ok_len and ok_cmd and ok_addr

    return ok, {
        "request": req,
        "response": frame,
        "parsed": parsed,
        "diag": diag,
        "ok_len": ok_len,
        "ok_cmd": ok_cmd,
        "ok_addr": ok_addr,
    }


def probe_write_noop(
    ser: serial.Serial,
    hand_id: int,
    response_timeout: float,
    verbose: bool,
) -> tuple[bool, dict]:
    """
    Do a benign write to CLEAR_ERROR (0x03EC) with value 1.
    This is useful to verify write path + response path at serial level.
    """
    req = build_frame(hand_id, _CMD_WRITE, 0x03EC, bytes([1]))
    ser.reset_input_buffer()
    ser.write(req)
    ser.flush()

    if verbose:
        print(f"  TX WRITE CLEAR_ERROR id={hand_id}: {_hex(req)}")

    frame, diag = read_response_frame(
        ser=ser,
        expected_id=hand_id,
        timeout_s=response_timeout,
        verbose=verbose,
    )
    if frame is None:
        return False, {
            "request": req,
            "error": "timeout waiting for write ACK frame",
            "diag": diag,
        }

    parsed = parse_response_frame(frame)
    ok = (
        parsed["checksum_ok"]
        and parsed["cmd"] == _CMD_WRITE
        and parsed["address"] == 0x03EC
    )
    return ok, {
        "request": req,
        "response": frame,
        "parsed": parsed,
        "diag": diag,
    }


def run_probe(
    port: str,
    baudrate: int,
    parity: str,
    stopbits: float,
    ids: Iterable[int],
    retries: int,
    io_timeout: float,
    response_timeout: float,
    verbose: bool,
    include_write_check: bool,
) -> int:
    print("=== RH56 Serial Register Probe ===")
    print(
        f"port={port} baudrate={baudrate} parity={parity} stopbits={stopbits} "
        f"ids={list(ids)} retries={retries}"
    )
    print(f"serial_timeout={io_timeout:.3f}s response_timeout={response_timeout:.3f}s")

    failures = 0
    with serial.Serial(
        port=port,
        baudrate=baudrate,
        bytesize=8,
        parity=parity,
        stopbits=serial.STOPBITS_ONE if stopbits == 1.0 else serial.STOPBITS_TWO,
        timeout=io_timeout,
        exclusive=True,
    ) as ser:
        print(f"Opened {port} (exclusive lock acquired).")
        ser.reset_input_buffer()
        ser.reset_output_buffer()

        for hand_id in ids:
            print(f"\n--- Hand ID {hand_id} ---")

            if include_write_check:
                ok, details = probe_write_noop(
                    ser=ser,
                    hand_id=hand_id,
                    response_timeout=response_timeout,
                    verbose=verbose,
                )
                if ok:
                    print("[WRITE-ACK] OK")
                else:
                    failures += 1
                    print("[WRITE-ACK] FAIL")
                    if "response" in details:
                        parsed = details["parsed"]
                        print(
                            "  response="
                            f"cmd=0x{parsed['cmd']:02X} addr=0x{parsed['address']:04X} "
                            f"payload={_hex(parsed['payload'])}"
                        )
                    else:
                        print(f"  error={details['error']}")
                    residual = details["diag"].get("residual", b"")
                    if residual:
                        print(f"  residual={_hex(residual)}")

            for reg in DEFAULT_PROBES:
                reg_ok = False
                last_details = None

                for attempt in range(1, retries + 1):
                    ok, details = probe_read(
                        ser=ser,
                        hand_id=hand_id,
                        reg=reg,
                        response_timeout=response_timeout,
                        verbose=verbose,
                    )
                    last_details = details
                    if ok:
                        reg_ok = True
                        parsed = details["parsed"]
                        payload = parsed["payload"]
                        print(
                            f"[READ] {reg.name:10s} OK "
                            f"addr=0x{parsed['address']:04X} len={len(payload):2d} payload={_hex(payload)}"
                        )
                        break

                    if verbose:
                        print(f"  attempt {attempt}/{retries} failed: {details.get('error', 'validation failed')}")
                    time.sleep(0.01)

                if not reg_ok:
                    failures += 1
                    print(f"[READ] {reg.name:10s} FAIL")
                    if last_details is not None:
                        if "response" in last_details:
                            parsed = last_details["parsed"]
                            print(
                                "  response="
                                f"cmd=0x{parsed['cmd']:02X} addr=0x{parsed['address']:04X} "
                                f"checksum_ok={parsed['checksum_ok']} payload={_hex(parsed['payload'])}"
                            )
                            print(
                                f"  matches: cmd={last_details['ok_cmd']} addr={last_details['ok_addr']} len={last_details['ok_len']}"
                            )
                        else:
                            print(f"  error={last_details['error']}")
                        residual = last_details["diag"].get("residual", b"")
                        if residual:
                            print(f"  residual={_hex(residual)}")

    print("\n=== Summary ===")
    if failures == 0:
        print("All probes passed.")
        return 0
    print(f"Probe failures: {failures}")
    return 2


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Low-level RH56 serial probe with verbose register read diagnostics."
    )
    ap.add_argument("--port", required=True, help="Serial device, e.g. /dev/ttyUSB3")
    ap.add_argument("--baudrate", type=int, default=115200)
    ap.add_argument("--parity", choices=["N", "E", "O"], default="N")
    ap.add_argument("--stopbits", type=float, choices=[1.0, 2.0], default=1.0)
    ap.add_argument("--ids", nargs="+", type=int, default=[1, 2], help="Hand IDs to probe")
    ap.add_argument("--retries", type=int, default=3, help="Read retries per register")
    ap.add_argument("--io-timeout", type=float, default=0.006, help="Serial read timeout in seconds")
    ap.add_argument(
        "--response-timeout",
        type=float,
        default=0.060,
        help="Overall timeout waiting for a complete response frame in seconds",
    )
    ap.add_argument("--verbose", action="store_true", help="Print raw RX chunks and frame discard reasons")
    ap.add_argument(
        "--include-write-check",
        action="store_true",
        help="Also send a benign CLEAR_ERROR write and verify ACK",
    )
    ap.add_argument(
        "--sweep-line-settings",
        action="store_true",
        help="Try common UART framing combinations: 8N1, 8E1, 8O1",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    retries = max(1, args.retries)
    io_timeout = max(0.001, args.io_timeout)
    response_timeout = max(0.005, args.response_timeout)

    if args.sweep_line_settings:
        combos = [("N", 1.0), ("E", 1.0), ("O", 1.0)]
        exit_code = 2
        for parity, stopbits in combos:
            print("\n========================================")
            print(f"Line setting probe: parity={parity} stopbits={stopbits}")
            result = run_probe(
                port=args.port,
                baudrate=args.baudrate,
                parity=parity,
                stopbits=stopbits,
                ids=args.ids,
                retries=retries,
                io_timeout=io_timeout,
                response_timeout=response_timeout,
                verbose=args.verbose,
                include_write_check=bool(args.include_write_check),
            )
            if result == 0:
                exit_code = 0
                break
        return exit_code

    return run_probe(
        port=args.port,
        baudrate=args.baudrate,
        parity=args.parity,
        stopbits=args.stopbits,
        ids=args.ids,
        retries=retries,
        io_timeout=io_timeout,
        response_timeout=response_timeout,
        verbose=args.verbose,
        include_write_check=bool(args.include_write_check),
    )


if __name__ == "__main__":
    raise SystemExit(main())
