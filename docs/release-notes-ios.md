### Install

1. Download `Faceforge.ipa` on your iPhone.
2. Open it with **SideStore** (or AltStore), which signs it with your Apple ID. A free account needs re-signing every 7 days.

### Two ways to run

- **On iPhone.** Open the **Models** tab, read and accept InsightFace's licence (non-commercial research use), and download the models (about 840 MB, straight from InsightFace's GitHub releases). The first Live start compiles them for the Neural Engine, which takes a while once.
- **PC offload.** On your Windows PC run `server\Start-Faceforge-Server.bat` from the Faceforge repo (needs Python 3.11; the NVIDIA build needs only the GPU driver). Enter the address and pairing code it prints in **Settings**, or tap your PC under "PCs found on this network". Both devices must be on the same Wi-Fi.

Then choose a **Source** face, open **Live**, pick the mode, and press **Start**. The stats overlay shows the backend actually in use, FPS, 1%-low FPS, latency, and the iPhone's thermal state. Live mode slows down or pauses on its own when the phone gets hot.

### Limits

- iOS has no supported way for an app to act as a camera or microphone for other apps (FaceTime, Zoom...). Faceforge works inside its own app and saves photos to your library.
- PC offload traffic is not encrypted; use it only on your own network.

> Only use faces you have permission to use, and label anything you share as a deepfake.
