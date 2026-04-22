# Face Recognition Password Vault (Client-Server)

A secure, facial-recognition-based password manager using deep learning (FaceNet) and liveness detection.

**Authors**: Logan Nunno and Quinn Sena

---

## Getting Started

The system is divided into two parts: the Server (Dockerized for cross-platform compatibility) and the Client (run natively to access your webcam).

### 1. Start the Server (Docker)
Since installing deep learning dependencies can be tricky across different computers, the server runs in Docker.

**Prerequisites:** Ensure you have [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running on your system.

In your terminal, run:
```bash
docker compose up --build
```
*The server will handle facial embedding generation, encrypted storage, and matching on `http://localhost:5000`. Thanks to Docker volumes, your passwords and encodings will persist in the project folder.*

### 2. Start the Client (Local)
In a **second terminal window**, set up and run the client natively to seamlessly access your webcam.

**Install Client Dependencies:**
```bash
pip install -r requirements-client.txt
```

**Run the Client:**
```bash
python client.py
```
*(If you are using a virtual environment, activate it first or use `.venv\Scripts\python.exe client.py`)*
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