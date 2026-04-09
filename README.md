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
*The server handles facial embedding generation, encrypted storage, and matching.*

### 2. Start the Client
In a second terminal, run:
```bash
python client.py
```
*The client handles camera capture, liveness detection (blinking), and the user interface.*

---

## Key Features

- **Multi-Password Vault**: Store multiple passwords for different services (Google, Bank, etc.) under a single face identity.
- **Liveness Verification**: Built-in blink detection ensures that a photo or video cannot be used to spoof the system.
- **Secure Encryption**: Passwords are encrypted using AES (Fernet) on the server. The encryption key is generated locally on your machine.
- **Intelligent Enrollment**: The capture loop only records frames when a face is clearly visible, ensuring high-quality training data.