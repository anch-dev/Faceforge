"""Faceforge offload server: receives camera frames from the iPhone, swaps
faces on the PC's GPU, returns the result. Protocol: docs/offload-protocol.md."""
from __future__ import annotations

import asyncio
import hmac
import json
import os
import socket
import time
from typing import Callable, List, Optional

import cv2
import numpy as np

from faceforge_eval.compositor import Compositor
from faceforge_eval.pipeline import Pipeline

from . import protocol as P


class SwapEngine:
    """Owns the models and the pipeline. Thread-confined: called from one
    worker thread at a time."""

    def __init__(self, detector, recognizer, swapper, provider: str, device: str = ""):
        self.detector = detector
        self.recognizer = recognizer
        self.swapper = swapper
        self.provider = provider
        self.device = device
        self.detect_every = 2
        self.jpeg_quality = 80
        self.source_embedding: Optional[np.ndarray] = None
        self._new_pipeline()

    def _new_pipeline(self):
        self.pipeline = Pipeline(self.detector, self.swapper, Compositor(), detect_every=self.detect_every)

    def apply_settings(self, s: dict) -> None:
        if "detect_every" in s:
            self.detect_every = max(1, min(30, int(s["detect_every"])))
            self._new_pipeline()
        if "jpeg_quality" in s:
            self.jpeg_quality = max(30, min(95, int(s["jpeg_quality"])))

    def set_source(self, jpeg: bytes) -> dict:
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return {"ok": False, "faces": 0, "message": "could not decode image"}
        faces = self.detector.detect(img)
        if not faces:
            return {"ok": False, "faces": 0, "message": "no face found in source image"}
        best = max(faces, key=lambda f: f.area)
        self.source_embedding = self.recognizer.embed(img, best)
        self._new_pipeline()
        return {"ok": True, "faces": len(faces), "message": ""}

    def process(self, frame: P.Frame) -> P.Result:
        t0 = time.perf_counter()
        img = cv2.imdecode(np.frombuffer(frame.jpeg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise P.ProtocolError("could not decode frame JPEG")
        found = False
        if self.source_embedding is not None:
            out, timing = self.pipeline.process(img, self.source_embedding)
            found = "swap" in timing.stages_ms
        else:
            out = img
        ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            raise P.ProtocolError("could not encode result JPEG")
        ms = (time.perf_counter() - t0) * 1000
        return P.Result(frame.frame_id, frame.client_time_us, ms, found, enc.tobytes())


class OffloadServer:
    def __init__(self, engine: SwapEngine, token: str, printer: Callable[[str], None] = print):
        self.engine = engine
        self.token = token
        self.printer = printer
        self._busy = False
        self.frames_served = 0

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        peer = writer.get_extra_info("peername")
        loop = asyncio.get_running_loop()

        async def send(data: bytes):
            writer.write(data)
            await writer.drain()

        async def fail(msg: str):
            await send(P.encode_json(P.ERROR, {"message": msg}))
            writer.close()

        if self._busy:
            await fail("busy: another device is connected")
            return
        try:
            msg_type, payload = await asyncio.wait_for(P.read_message(reader), timeout=10)
            if msg_type != P.HELLO:
                await fail("expected HELLO")
                return
            hello = P.parse_json(payload)
            if not hmac.compare_digest(str(hello.get("token", "")), self.token):
                self.printer(f"[server] rejected {peer}: wrong pairing code")
                await fail("wrong pairing code")
                return
            self._busy = True
            self.printer(f"[server] {peer} connected ({hello.get('client', '?')})")
            await send(P.encode_json(P.WELCOME, {
                "protocol": P.PROTOCOL_VERSION, "server": "faceforge",
                "provider": self.engine.provider, "device": self.engine.device,
                "swapper": "inswapper_128"}))
            while True:
                msg_type, payload = await P.read_message(reader)
                if msg_type == P.FRAME:
                    frame = P.unpack_frame(payload)
                    result = await loop.run_in_executor(None, self.engine.process, frame)
                    self.frames_served += 1
                    await send(P.pack_result(result))
                elif msg_type == P.SOURCE:
                    ack = await loop.run_in_executor(None, self.engine.set_source, payload)
                    self.printer(f"[server] source image: {ack}")
                    await send(P.encode_json(P.SOURCE_ACK, ack))
                elif msg_type == P.SETTINGS:
                    self.engine.apply_settings(P.parse_json(payload))
                else:
                    await fail(f"unexpected {P.NAMES.get(msg_type, msg_type)}")
                    return
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        except (P.ProtocolError, asyncio.TimeoutError, ValueError) as e:
            try:
                await fail(str(e) or type(e).__name__)
            except Exception:
                pass
        finally:
            if self._busy:
                self.printer(f"[server] {peer} disconnected after {self.frames_served} frames")
            self._busy = False
            self.frames_served = 0
            try:
                writer.close()
            except Exception:
                pass


def lan_addresses() -> List[str]:
    addrs = set()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))  # no packet is sent; picks the LAN interface
            addrs.add(s.getsockname()[0])
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            addrs.add(info[4][0])
    except OSError:
        pass
    return sorted(a for a in addrs if not a.startswith("127."))


def advertise_bonjour(port: int, printer: Callable[[str], None] = print):
    """Advertise _faceforge._tcp if the optional 'zeroconf' package is installed."""
    try:
        from zeroconf import ServiceInfo, Zeroconf
    except ImportError:
        printer("[server] (install 'zeroconf' to let the iPhone find this PC automatically)")
        return None
    addrs = lan_addresses()
    if not addrs:
        return None
    zc = Zeroconf()
    info = ServiceInfo("_faceforge._tcp.local.", f"Faceforge on {socket.gethostname()}._faceforge._tcp.local.",
                       addresses=[socket.inet_aton(a) for a in addrs], port=port,
                       properties={"protocol": str(P.PROTOCOL_VERSION)})
    zc.register_service(info)
    printer("[server] advertising via Bonjour as _faceforge._tcp")
    return zc, info
