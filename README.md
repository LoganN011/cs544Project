# Face Recognition Password Vault (Client-Server)

A secure, facial-recognition-based password manager using deep learning (FaceNet) and liveness detection.

**Authors**: Logan Nunno and Quinn Sena

---

## Installation

Run the following single command to install all necessary dependencies:

```bash
pip install flask cryptography requests facenet-pytorch torch torchvision mediapipe opencv-python
```

---

## How to Run

The system requires two separate terminal windows to be running simultaneously on the same machine (or on the same network).

### 1. Start the Server
In the first terminal, run:
```bash
python server.py
```
```bash
.venv\Scripts\python.exe server.py
```
*The server handles facial embedding generation, encrypted storage, and matching.*

### 2. Start the Client
In a second terminal, run:
```bash
python client.py
```
```bash
.venv\Scripts\python.exe client.py
```
*The client handles camera capture, liveness detection (blinking), and the user interface.*

---

## Key Features

- **Multi-Password Vault**: Store multiple passwords for different services (Google, Bank, etc.) under a single face identity.
- **Liveness Verification**: Built-in blink detection ensures that a photo or video cannot be used to spoof the system.
- **Secure Encryption**: Passwords are encrypted using AES (Fernet) on the server. The encryption key is generated locally on your machine.
- **Intelligent Enrollment**: The capture loop only records frames when a face is clearly visible, ensuring high-quality training data.

# Things to talk about in the report
- Explain the "Password Fatigue" problem and why MFA (Multi-Factor Authentication) using biometrics is a solution.
- Why facial recognition? Discuss the balance between usability and security.
- Deep dive into MTCNN for alignment and InceptionResnetV1 for feature extraction. Explain the math behind Eye Aspect Ratio (EAR).
- Detail the Flask server-client architecture and the SQLite database implementation.
- How do we prevent spoofing? Discuss the liveness detection and moire detection.
- What are the limitations of our system? How could it be improved?
- Test the system against high-resolution photos or videos of yourself on a tablet. Document if the current EAR (Eye Aspect Ratio) threshold is enough to stop a video replay.
- Research and document how small digital perturbations (noise) could potentially fool the InceptionResnetV1 model into recognizing you as someone else.
- The client communicates with the server via HTTP. Discuss or implement HTTPS (using a self-signed certificate) to show you understand how to protect biometric data in transit.
- Discuss the "secret.key" file. In a real-world scenario, how would this be protected? (e.g., Hardware Security Modules or Environment Variables).
- Explain the risk if the encodings.pickle file is lost or stolen.
- Research "cancelable biometrics" or "fuzzy extractors"
- Future things to add: Mention 3D depth sensing (like FaceID) or multi-modal biometrics (voice + face).
- discuss the ethical implications of storing biometric "embeddings."
- 