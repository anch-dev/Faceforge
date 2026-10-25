### Install

1. Download `Faceforge.ipa` on your iPhone.
2. Open it with **SideStore** (or AltStore), which signs it with your Apple ID. A free account needs re-signing every 7 days.

### Two ways to run

- **On iPhone.** Open the **Models** tab, read and accept InsightFace's licence (non-commercial research use), and download the models (about 840 MB, straight from InsightFace's GitHub releases). The first Live start compiles them for the Neural Engine, which takes a while once.
- **PC offload.** On your Windows PC run `server\Start-Faceforge-Server.bat` from the Faceforge repo (needs Python 3.11; the NVIDIA build needs only the GPU driver). Enter the address and pairing code it prints in **Settings**, or tap your PC under "PCs found on this network". Both devices must be on the same Wi-Fi.

### Faster on iPhone (0.1.3)

The swap model the app downloads is 32-bit, and the Neural Engine only runs 16-bit models, so on its own it runs on the GPU (about 3 FPS in 0.1.2). To use the Neural Engine:

1. On your PC run `server\Send-Model-To-iPhone.bat` (after `Start-Faceforge-Server.bat` has run once). It makes a 16-bit copy from your own copy of the model, prints a **download code**, and waits. Nothing is uploaded.
2. On the iPhone: **Settings** › PC offload › PC address, then **Models** › Fast swap model › enter the code › **Get from PC**.

The Live stats line shows which swap backend is used and its measured time. **Smooth preview** (on by default) shows every camera frame and keeps the latest swapped face on your head between swaps. **Swap runs on** picks Neural Engine, GPU (Metal), or all.

Then choose a **Source** face, open **Live**, pick the mode, and press **Start**. The stats overlay shows the backend actually in use, FPS, 1%-low FPS, latency, and the iPhone's thermal state. Live mode slows down or pauses on its own when the phone gets hot.

### Limits

- iOS has no supported way for an app to act as a camera or microphone for other apps (FaceTime, Zoom...). Faceforge works inside its own app and saves photos to your library.
- PC offload traffic is not encrypted; use it only on your own network.

> Only use faces you have permission to use, and label anything you share as a deepfake.
