"""Consistency checks between the Swift sources and the verified Python side.
(The Swift code itself is compiled and unit-tested on the macOS CI runner.)"""
import re

import yaml

from conftest import REPO
from faceforge_eval import models as M
from faceforge_server import protocol as P

IOS = REPO / "ios"


def test_model_catalog_matches_python():
    swift = (IOS / "FaceforgeCore/Sources/FaceforgeCore/ModelCatalog.swift").read_text()
    for a in (M.INSWAPPER, M.BUFFALO_L):
        assert a.url in swift and str(a.size) in swift and a.sha256 in swift, a.name
    for name, (size, sha) in M.BUFFALO_MEMBERS.items():
        assert f'("{name}", {size}, "{sha}")' in swift


def test_protocol_constants_match_python():
    swift = (IOS / "FaceforgeCore/Sources/FaceforgeCore/OffloadProtocol.swift").read_text()
    assert f'Array("{P.MAGIC.decode()}".utf8)' in swift
    assert f"defaultPort: UInt16 = {P.DEFAULT_PORT}" in swift
    assert f"maxPayload = {P.MAX_PAYLOAD // (1024 * 1024)} * 1024 * 1024" in swift
    types = re.search(r"case hello = 1, welcome, source, sourceAck, frame, result, settings, error", swift)
    assert types and [P.HELLO, P.WELCOME, P.SOURCE, P.SOURCE_ACK, P.FRAME, P.RESULT, P.SETTINGS, P.ERROR] == list(range(1, 9))


def test_project_declares_required_privacy_keys():
    proj = yaml.safe_load((IOS / "project.yml").read_text())
    props = proj["targets"]["Faceforge"]["info"]["properties"]
    for key in ("NSCameraUsageDescription", "NSLocalNetworkUsageDescription",
                "NSPhotoLibraryAddUsageDescription"):
        assert props.get(key), key
    assert "_faceforge._tcp" in props["NSBonjourServices"]
    # Logs/ in Documents must be reachable from the Files app.
    assert props.get("UIFileSharingEnabled") is True and props.get("LSSupportsOpeningDocumentsInPlace") is True
    # HTTP to the PC's model hand-off is allowed on the LAN only, never arbitrary loads.
    ats = props["NSAppTransportSecurity"]
    assert ats == {"NSAllowsLocalNetworking": True}


def test_fast_model_handoff_matches_python():
    from faceforge_server import ios_model as IM
    swift = (IOS / "Faceforge/Engine/ModelStore.swift").read_text()
    assert f"port: Int = {IM.DEFAULT_PORT}" in swift
    assert 'appendingPathComponent("index.json")' in swift
    assert "let name: String; let size: Int64; let sha256: String; let kind: String" in swift
    assert IM.FP16_ONNX in swift and IM.MLPACKAGE.replace(".mlpackage", ".mlmodelc") in swift
    bat = (REPO / "server/Send-Model-To-iPhone.bat").read_text()
    assert "-m faceforge_server.ios_model" in bat


def test_app_never_bundles_weights():
    for p in IOS.rglob("*"):
        assert p.suffix not in {".onnx", ".mlpackage", ".mlmodel", ".mlmodelc"} or "fixtures" in p.parts, p
