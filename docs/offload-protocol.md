# Faceforge offload protocol (v1)

The iPhone sends camera frames to the PC, and the PC returns swapped frames. Both
sides implement this spec, and `tests/fixtures/protocol_v1.json` holds golden
byte strings that the Python and Swift tests both check.

**Transport:** plain TCP on the local network, default port **47800**. One
client at a time; a second client gets `ERROR` ("busy") and is disconnected.

**Pairing:** the server prints a 6-digit code at startup. The client must send
it in `HELLO`; a wrong code gets `ERROR` and the connection closes. This keeps
other devices on the network from using your PC or receiving your frames.
Traffic is not encrypted, so use it only on a network you trust.

## Framing

Every message is a 12-byte header followed by `length` payload bytes. All
integers are big-endian.

| Offset | Size | Field |
|---|---|---|
| 0 | 4 | magic `FFG1` (ASCII) |
| 4 | 1 | type |
| 5 | 1 | flags (0) |
| 6 | 2 | reserved (0) |
| 8 | 4 | payload length, at most 16 MiB |

## Messages

| Type | Name | Direction | Payload |
|---|---|---|---|
| 1 | `HELLO` | client → server | JSON `{"protocol":1,"token":"123456","client":"ios"}` |
| 2 | `WELCOME` | server → client | JSON `{"protocol":1,"server":"faceforge","provider":"CUDAExecutionProvider","device":"...","swapper":"inswapper_128"}` |
| 3 | `SOURCE` | client → server | JPEG of the source face photo |
| 4 | `SOURCE_ACK` | server → client | JSON `{"ok":true,"faces":1,"message":""}` |
| 5 | `FRAME` | client → server | `frame_id` u32, `client_time_us` u64, then JPEG |
| 6 | `RESULT` | server → client | `frame_id` u32, `client_time_us` u64, `server_ms` f32, `face_found` u8, then JPEG |
| 7 | `SETTINGS` | client → server | JSON `{"detect_every":2,"jpeg_quality":80}` (any subset) |
| 8 | `ERROR` | server → client | JSON `{"message":"..."}` |

## Rules

- The client sends `HELLO` first. Anything else first is answered with `ERROR` and the connection closes.
- `FRAME` before any `SOURCE` is returned unchanged, with `face_found=0`.
- The server answers every `FRAME` with exactly one `RESULT`, in order.
- The client keeps at most **2** frames in flight and skips camera frames while the window is full. That bounds latency instead of letting a queue build up.
- Round-trip latency = now − `client_time_us`, measured on the client's own clock, so no clock sync is needed.
- Orientation: the client sends upright images. The server does not rotate.
