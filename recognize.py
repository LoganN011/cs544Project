import cv2
import pickle
import numpy as np
import torch
import os
import sys
from facenet_pytorch import MTCNN, InceptionResnetV1
from liveness import LivenessDetector

def get_resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def recognize_users():
    if not os.path.exists('encodings.pickle'):
        print("Model not trained yet! Run train.py first.")
        return

    with open('encodings.pickle', 'rb') as f:
        user_encodings = pickle.load(f)
    print(f"Loaded profiles for: {list(user_encodings.keys())}")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Running on device: {device}')

    # Model for live parsing. keep_all=True detects all faces in frame
    mtcnn = MTCNN(image_size=160, margin=0, keep_all=True, device=device)
    
    resnet = InceptionResnetV1(pretrained=None).eval()
    weights_path = get_resource_path(os.path.join('models', 'vggface2.pt'))
    
    if not os.path.exists(weights_path):
        import urllib.request
        print(f"Downloading vggface2.pt to {weights_path}...")
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        url = 'https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt'
        urllib.request.urlretrieve(url, weights_path)
    
    # Strip logits if they exist so it matches the expected model architecture
    state_dict = torch.load(weights_path, map_location=device)
    state_dict = {k: v for k, v in state_dict.items() if 'logits' not in k}
    
    resnet.load_state_dict(state_dict)
    resnet = resnet.to(device)
    
    liveness = LivenessDetector(ear_threshold=0.20, frames_to_blink=2)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    print("Look at the camera. Press 'q' to quit.")
    
    distance_threshold = 0.80

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect face boxes and aligned face cropped tensors
        boxes, probs = mtcnn.detect(img_rgb)
        
        best_name = "Unknown"
        is_live = False
        ear = None

        if boxes is not None:
            # Get aligned face tensors
            x_aligned = mtcnn(img_rgb)

            if x_aligned is not None:
                # Compute embeddings using Resnet
                embeddings = resnet(x_aligned.to(device)).detach().cpu().numpy()

                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = [int(b) for b in box]
                    embed = embeddings[i]
                    prob = probs[i]

                    if prob < 0.90:
                        continue
                        
                    # Compare embedding with known profiles using Euclidean Distance
                    best_match = "Unknown"
                    min_dist = float('inf')

                    for name, known_embed in user_encodings.items():
                        dist = np.linalg.norm(embed - known_embed)
                        if dist < min_dist:
                            min_dist = dist
                            best_match = name

                    if min_dist < distance_threshold:
                        best_name = best_match

                    # Run liveness detection. MediaPipe runs on the full frame
                    # For simplicity, we assign the liveness state to the largest/best match face
                    is_live, ear = liveness.process_frame(frame, best_name)

                    # Colors
                    if best_name == "Unknown":
                        color = (0, 0, 255) # Red for unknown
                        text = "Unknown Face"
                    elif not is_live:
                        color = (0, 165, 255) # Orange for waiting liveness
                        text = f"{best_name} - PLEASE BLINK"
                    else:
                        color = (0, 255, 0) # Green for verified live person
                        text = f"{best_name} - VERIFIED"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, text, (x1, max(y1 - 10, 10)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    if ear is not None:
                         cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            # No face detected
            pass
            
        cv2.imshow('Live Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    recognize_users()
