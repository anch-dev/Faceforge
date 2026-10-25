"""Faceforge offload protocol v1 (see docs/offload-protocol.md)."""
from __future__ import annotations

import asyncio
import json
import struct
from dataclasses import dataclass
from typing import Any, Dict, Tuple

MAGIC = b"FFG1"
HEADER = struct.Struct(">4sBBHI")      # magic, type, flags, reserved, length
FRAME_HDR = struct.Struct(">IQ")       # frame_id, client_time_us
RESULT_HDR = struct.Struct(">IQfB")    # frame_id, client_time_us, server_ms, face_found
MAX_PAYLOAD = 16 * 1024 * 1024
PROTOCOL_VERSION = 1
DEFAULT_PORT = 47800

HELLO, WELCOME, SOURCE, SOURCE_ACK, FRAME, RESULT, SETTINGS, ERROR = range(1, 9)
NAMES = {HELLO: "HELLO", WELCOME: "WELCOME", SOURCE: "SOURCE", SOURCE_ACK: "SOURCE_ACK",
         FRAME: "FRAME", RESULT: "RESULT", SETTINGS: "SETTINGS", ERROR: "ERROR"}


class ProtocolError(Exception):
    pass


def encode(msg_type: int, payload: bytes = b"") -> bytes:
    if msg_type not in NAMES:
        raise ProtocolError(f"unknown message type {msg_type}")
    if len(payload) > MAX_PAYLOAD:
        raise ProtocolError(f"payload {len(payload)} bytes exceeds {MAX_PAYLOAD}")
    return HEADER.pack(MAGIC, msg_type, 0, 0, len(payload)) + payload


def encode_json(msg_type: int, obj: Dict[str, Any]) -> bytes:
    return encode(msg_type, json.dumps(obj, separators=(",", ":"), sort_keys=True).encode())


def decode_header(header: bytes) -> Tuple[int, int]:
    if len(header) != HEADER.size:
        raise ProtocolError("short header")
    magic, msg_type, _flags, _reserved, length = HEADER.unpack(header)
    if magic != MAGIC:
        raise ProtocolError(f"bad magic {magic!r}")
    if msg_type not in NAMES:
        raise ProtocolError(f"unknown message type {msg_type}")
    if length > MAX_PAYLOAD:
        raise ProtocolError(f"payload length {length} exceeds limit")
    return msg_type, length


async def read_message(reader: asyncio.StreamReader) -> Tuple[int, bytes]:
    msg_type, length = decode_header(await reader.readexactly(HEADER.size))
    payload = await reader.readexactly(length) if length else b""
    return msg_type, payload


@dataclass
class Frame:
    frame_id: int
    client_time_us: int
    jpeg: bytes


@dataclass
class Result:
    frame_id: int
    client_time_us: int
    server_ms: float
    face_found: bool
    jpeg: bytes


def pack_frame(f: Frame) -> bytes:
    return encode(FRAME, FRAME_HDR.pack(f.frame_id, f.client_time_us) + f.jpeg)


def unpack_frame(payload: bytes) -> Frame:
    if len(payload) < FRAME_HDR.size:
        raise ProtocolError("short FRAME")
    fid, ts = FRAME_HDR.unpack_from(payload)
    return Frame(fid, ts, payload[FRAME_HDR.size:])


def pack_result(r: Result) -> bytes:
    return encode(RESULT, RESULT_HDR.pack(r.frame_id, r.client_time_us, r.server_ms, int(r.face_found)) + r.jpeg)


def unpack_result(payload: bytes) -> Result:
    if len(payload) < RESULT_HDR.size:
        raise ProtocolError("short RESULT")
    fid, ts, ms, found = RESULT_HDR.unpack_from(payload)
    return Result(fid, ts, ms, bool(found), payload[RESULT_HDR.size:])


def parse_json(payload: bytes) -> Dict[str, Any]:
    try:
        obj = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ProtocolError(f"bad JSON payload: {e}") from e
    if not isinstance(obj, dict):
        raise ProtocolError("JSON payload must be an object")
    return obj
