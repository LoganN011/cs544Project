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
    People struggle to remember many complex passwords (fatigue), leading to poor security habits like reusing simple ones. Multi-Factor Authentication (MFA) using biometrics solves this by using "who you are" as a key, making access both easier and more secure.

- Why facial recognition? Discuss the balance between usability and security.
    It offers high usability because it is touchless and fast, balancing convenience for the user with the high security of unique biological traits.

- Deep dive into MTCNN for alignment and InceptionResnetV1 for feature extraction. Explain the math behind Eye Aspect Ratio (EAR).
    MTCNN is a multi-task cascaded convolutional neural network used for detecting faces and finding facial landmarks. It consists of three sequential networks: a Proposal Network (P-Net) to generate candidate face regions, a Refine Network (R-Net) to filter these candidates, and a Output Network (O-Net) to fine-tune the bounding boxes and facial landmarks. The P-Net acts as a fast first-pass filter, using a single convolutional layer to generate proposals, while the R-Net and O-Net use deeper architectures to increase accuracy. These networks work together to identify faces and extract key facial landmarks (eyes, nose, mouth) from an image.

    InceptionResnetV1, commonly known as FaceNet, is a deep convolutional neural network architecture developed by Google that is specifically designed for face recognition. It is based on the Inception architecture, which uses "inception modules" that contain multiple parallel convolutional layers with different filter sizes to capture features at various scales. The ResNet aspect refers to the integration of residual connections, which allow for the training of very deep networks by providing skip connections that help mitigate the vanishing gradient problem. FaceNet is trained using a "triplet loss" function, which learns to map faces to a lower-dimensional Euclidean space (embeddings) where faces of the same person are close together and faces of different people are far apart. This makes it highly effective for tasks like face verification and clustering.

- Detail the Flask server-client architecture and the SQLite database implementation.
    The system uses a Flask server to handle the heavy computational tasks of facial recognition and secure storage. The server communicates with the client via HTTP RESTful APIs. The client captures the image and sends it to the server for processing. The server uses the MTCNN model to detect and align the face, then extracts a 128-dimensional face embedding using the InceptionResnetV1 model. This embedding is then compared to the embeddings stored in the SQLite database. The SQLite database is used to store the face embeddings and the corresponding passwords for each user. The database is populated during the enrollment process, where the user enrolls their face for the first time. During authentication, the server retrieves the user's embedding from the database and compares it to the query embedding. If the distance is below a certain threshold, the server returns the corresponding password to the client.

- How do we prevent spoofing? Discuss the liveness detection and moire detection.
    To prevent spoofing, we use two methods: liveness detection and moire detection. Liveness detection is used to detect if the face is real or not. We do this by detecting if the user is blinking. If the user is not blinking, we assume that the face is not real. Moire detection is used to detect if the face is a photo or not. We do this by detecting if there is a moire pattern in the image. If there is a moire pattern in the image, we assume that the face is a photo.

- What are the limitations of our system? How could it be improved?
    One limitation of our system is that it is not very robust to changes in lighting conditions. If the lighting conditions are too dark or too bright, the system may not be able to detect the face. Another limitation is that the system is not very robust to changes in head position or orientation. If the user is not looking directly at the camera, the system may not be able to detect the face. A third limitation is that the system is not very robust to changes in facial expression. If the user is not smiling or frowning, the system may not be able to detect the face. 

- Test the system against high-resolution photos or videos of yourself on a tablet. Document if the current EAR (Eye Aspect Ratio) threshold is enough to stop a video replay.

- Research and document how small digital perturbations (noise) could potentially fool the InceptionResnetV1 model into recognizing you as someone else.
    

- The client communicates with the server via HTTP. Discuss or implement HTTPS (using a self-signed certificate) to show you understand how to protect biometric data in transit.
    Since the client transmits sensitive facial data to the server for processing, using HTTP means that biometric "fingerprints" (embeddings) are sent in plain text. An attacker on the same network could use a packet sniffer to intercept these embeddings and potentially reuse them to impersonate you.

- Discuss the "secret.key" file. In a real-world scenario, how would this be protected? (e.g., Hardware Security Modules or Environment Variables).
    Instead of a file, the key is stored in the operating system's environment. This prevents the key from being accidentally committed to version control. A more robust solution would be to use a Hardware Security Module (HSM), which is a physical computing device that safeguards and manages digital keys.

- Explain the risk if the encodings.pickle file is lost or stolen.
    Since the embeddings are directly derived from your biometric data, if this file were stolen, an attacker could potentially reconstruct your facial features or use them to bypass security systems that rely on these embeddings.

- Research "cancelable biometrics" or "fuzzy extractors"
    Cancelable biometrics is a technique that transforms biometric data into a non-invertible, revocable format. This means that the transformed data cannot be used to reconstruct the original biometric data, and it can be "revoked" or replaced if it is compromised.

- Future things to add: Mention 3D depth sensing (like FaceID) or multi-modal biometrics (voice + face).


- discuss the ethical implications of storing biometric "embeddings."
