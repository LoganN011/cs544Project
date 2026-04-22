import cv2
import requests
import argparse
import os
import sys
import time
import numpy as np
from liveness import LivenessDetector

# Default server address
DEFAULT_SERVER = "http://localhost:5000"


class RecognitionClient:
    def __init__(self, server_url):
        self.server_url = server_url.rstrip('/')
        self.liveness = LivenessDetector(ear_threshold=0.18, frames_to_blink=2)

    def enroll(self, username, label, password, num_samples=25):
        print(f"Enrolling user: {username} for service: {label}")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return

        captured_frames = []
        print(f"Please look at the camera. Capturing {num_samples} samples...")
        
        # Load a simple face detector to ensure we only capture faces
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        while len(captured_frames) < num_samples:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to grayscale for faster face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            # Show feedback
            color = (0, 255, 0) if len(faces) > 0 else (0, 0, 255)
            status = f"Capturing: {len(captured_frames)}/{num_samples}" if len(faces) > 0 else "FACE NOT DETECTED"
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            cv2.putText(frame, status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.imshow('Enrollment - Capture', frame)

            if len(faces) > 0:
                # Delay slightly to get diverse frames, then capture
                _, buffer = cv2.imencode('.jpg', frame)
                captured_frames.append(('images', ('image.jpg', buffer.tobytes(), 'image/jpeg')))
                time.sleep(0.05) # Small delay once face is found

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        if len(captured_frames) < num_samples:
            print("Capture cancelled.")
            return

        print("Uploading to server...")
        try:
            response = requests.post(
                f"{self.server_url}/enroll",
                data={'username': username, 'label': label, 'password': password},
                files=captured_frames
            )
            print(response.json().get('message', 'Done'))
        except Exception as e:
            print(f"Error connecting to server: {e}")

    def retrieve(self):
        print("Requesting challenge from server...")
        challenge = "blink"
        try:
            resp = requests.get(f"{self.server_url}/challenge")
            if resp.status_code == 200:
                challenge = resp.json().get('challenge', 'blink')
        except Exception as e:
            print(f"Could not get challenge, retreating to default. {e}")
            
        print(f"Retrieving password. Liveness Challenge: {challenge.upper()}.")
        self.liveness.reset()
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return

        verified_frame = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            is_live, message = self.liveness.process_frame(frame, challenge=challenge)
            
            color = (0, 0, 255) if not is_live else (0, 255, 0)
            status_text = f"CHALLENGE: {challenge.replace('_', ' ').upper()}" if not is_live else "VERIFIED - SENDING..."
            if message and "SPOOF" in str(message):
                status_text = "SPOOF DETECTED"
            
            cv2.putText(frame, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            if message:
                cv2.putText(frame, str(message), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
            cv2.imshow('Retrieval - Liveness Check', frame)

            if is_live:
                verified_frame = frame.copy()
                cv2.waitKey(500) # Show verified state for a moment
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        if verified_frame is not None:
            print("Liveness verified. Requesting your vault from server...")
            _, buffer = cv2.imencode('.jpg', verified_frame)
            try:
                response = requests.post(
                    f"{self.server_url}/recognize",
                    files={'image': ('image.jpg', buffer.tobytes(), 'image/jpeg')}
                )
                res_data = response.json()
                if res_data.get('status') == 'success':
                    username = res_data.get('username')
                    labels = res_data.get('labels', [])
                    
                    if not labels:
                        print(f"Welcome {username}, but you have no passwords saved yet.")
                        return

                    print(f"\nWelcome back, {username}!")
                    print("Select a service to retrieve your password:")
                    for i, label in enumerate(labels):
                        print(f"{i+1}. {label}")
                    
                    choice = input("Enter number (or 'q' to cancel): ").strip()
                    if choice.lower() == 'q':
                        return
                        
                    if choice.isdigit() and 1 <= int(choice) <= len(labels):
                        selected_label = labels[int(choice)-1]
                        # Request specific password
                        pass_resp = requests.post(
                            f"{self.server_url}/get_password",
                            json={'username': username, 'label': selected_label}
                        )
                        pass_data = pass_resp.json()
                        if pass_data.get('status') == 'success':
                            print(f"\n======================================")
                            print(f"SERVICE: {selected_label}")
                            print(f"PASSWORD: {pass_data.get('password')}")
                            print(f"======================================\n")
                        else:
                            print(f"Error: {pass_data.get('message')}")
                    else:
                        print("Invalid selection.")
                else:
                    print(f"Recognition failed: {res_data.get('message')}")
            except Exception as e:
                print(f"Error connecting to server: {e}")
        else:
            print("Verification failed or cancelled.")

def main():
    parser = argparse.ArgumentParser(description="Face Recognition Client")
    parser.add_argument("--server", type=str, default=DEFAULT_SERVER, help="Server URL (default: http://localhost:5000)")
    args = parser.parse_args()

    client = RecognitionClient(args.server)

    while True:
        print("\n=== Secure Password Vault (Face Recognition) ===")
        print("1. Enroll New User & Save Password")
        print("2. Retrieve My Password (Requires Face)")
        print("3. Exit")
        choice = input("Select an option (1-3): ")

        if choice == '1':
            name = input("Enter your name (Identity): ").strip()
            if not name: continue
            label = input("Enter service label (e.g., Google, Bank): ").strip()
            if not label: continue
            password = input("Enter password to save: ").strip()
            if not password: continue
            client.enroll(name, label, password)
        elif choice == '2':
            client.retrieve()
        elif choice == '3':
            break
        else:
            print("Invalid choice.")

if __name__ == '__main__':
    main()
