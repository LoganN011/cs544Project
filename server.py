import torch
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
import pickle
import numpy as np
import cv2
from flask import Flask, request, jsonify
from facenet_pytorch import MTCNN, InceptionResnetV1
from database import Database

app = Flask(__name__)
db = Database()

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Server running on device: {device}")

# Initialize models
mtcnn = MTCNN(image_size=160, margin=0, keep_all=False, device=device)
resnet = InceptionResnetV1(pretrained=None).eval()
weights_path = os.path.join('models', 'vggface2.pt')

if not os.path.exists(weights_path):
    import urllib.request
    print(f"Downloading weights to {weights_path}...")
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    url = 'https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt'
    urllib.request.urlretrieve(url, weights_path)

state_dict = torch.load(weights_path, map_location=device)
state_dict = {k: v for k, v in state_dict.items() if 'logits' not in k}
resnet.load_state_dict(state_dict)
resnet = resnet.to(device)

# Load existing encodings
ENCODINGS_FILE = 'encodings.pickle'
if os.path.exists(ENCODINGS_FILE):
    with open(ENCODINGS_FILE, 'rb') as f:
        user_encodings = pickle.load(f)
else:
    user_encodings = {}

@app.route('/enroll', methods=['POST'])
def enroll():
    data = request.form
    username = data.get('username')
    password = data.get('password')
    label = data.get('label', 'General') # Default to General if not provided
    
    if not username or not password:
        return jsonify({"status": "error", "message": "Username and password required"}), 400

    files = request.files.getlist('images')
    if not files:
        return jsonify({"status": "error", "message": "No images provided"}), 400

    embeddings = []
    for file in files:
        # Convert file to opencv image
        nparr = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            continue
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x_aligned, prob = mtcnn(img_rgb, return_prob=True)
        
        if x_aligned is not None and prob > 0.90:
            aligned = x_aligned.unsqueeze(0).to(device)
            with torch.no_grad():
                embed = resnet(aligned).detach().cpu().numpy()[0]
            embeddings.append(embed)

    if len(embeddings) < 5:
        return jsonify({"status": "error", "message": f"Only found {len(embeddings)} valid faces. Need at least 5."}), 400

    # Calculate mean embedding
    mean_embedding = np.mean(embeddings, axis=0)
    mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)
    
    # Save to memory and file
    user_encodings[username] = mean_embedding
    with open(ENCODINGS_FILE, 'wb') as f:
        pickle.dump(user_encodings, f)
        
    # Save password to DB with label
    db.save_password(username, label, password)
    
    print(f"User {username} enrolled session for {label} successfully.")
    return jsonify({"status": "success", "message": f"Saved {label} password for {username}."})

@app.route('/recognize', methods=['POST'])
def recognize():
    if 'image' not in request.files:
        return jsonify({"status": "error", "message": "No image provided"}), 400
        
    file = request.files['image']
    nparr = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return jsonify({"status": "error", "message": "Invalid image"}), 400
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x_aligned, prob = mtcnn(img_rgb, return_prob=True)
    
    if x_aligned is None or prob < 0.90:
        return jsonify({"status": "unknown", "message": "No face detected or low confidence"})

    aligned = x_aligned.unsqueeze(0).to(device)
    with torch.no_grad():
        embed = resnet(aligned).detach().cpu().numpy()[0]
        
    # Match
    best_match = "Unknown"
    min_dist = float('inf')
    distance_threshold = 0.80

    for name, known_embed in user_encodings.items():
        dist = np.linalg.norm(embed - known_embed)
        if dist < min_dist:
            min_dist = dist
            best_match = name

    if min_dist < distance_threshold:
        # Get all labels for this user
        labels = db.get_labels(best_match)
        return jsonify({
            "status": "success",
            "username": best_match,
            "labels": labels,
            "distance": float(min_dist)
        })
    else:
        return jsonify({"status": "unknown", "message": "Face not recognized"})

@app.route('/get_password', methods=['POST'])
def get_password():
    data = request.json
    username = data.get('username')
    label = data.get('label')
    
    if not username or not label:
        return jsonify({"status": "error", "message": "Missing username or label"}), 400
        
    password = db.get_password(username, label)
    if password:
        return jsonify({"status": "success", "password": password})
    else:
        return jsonify({"status": "error", "message": "Password not found"}), 404

if __name__ == '__main__':
    # Run the server
    app.run(host='0.0.0.0', port=5000, threaded=False)
