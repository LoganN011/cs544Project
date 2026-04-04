import os
import cv2
import numpy as np
import pickle
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

def train_model(data_dir='data', output_file='encodings.pickle'):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Running on device: {device}')

    # MTCNN for face cropping
    mtcnn = MTCNN(image_size=160, margin=0, keep_all=False, device=device)
    # InceptionResnetV1 for 512-d embeddings
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    user_encodings = {}

    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} not found.")
        return

    users = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    for user in users:
        print(f"Processing images for {user}...")
        user_dir = os.path.join(data_dir, user)
        embeddings = []

        for filename in os.listdir(user_dir):
            if not (filename.endswith('.jpg') or filename.endswith('.png')):
                continue

            img_path = os.path.join(user_dir, filename)
            img = cv2.imread(img_path)
            if img is None:
                continue

            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # mtcnn returns a normalized tensor ready for resnet
            x_aligned, prob = mtcnn(img_rgb, return_prob=True)
            
            if x_aligned is not None and prob > 0.90:
                # Get embedding
                aligned = x_aligned.unsqueeze(0).to(device)
                with torch.no_grad():
                    embed = resnet(aligned).detach().cpu().numpy()[0]
                embeddings.append(embed)

        if len(embeddings) > 0:
            # We average the embeddings to get a single prototype vector for the user
            mean_embedding = np.mean(embeddings, axis=0)
            # Normalize to length 1
            mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)
            user_encodings[user] = mean_embedding
            print(f"Successfully processed {len(embeddings)} images for {user}")
        else:
            print(f"Warning: No valid faces found for {user}")

    with open(output_file, 'wb') as f:
        pickle.dump(user_encodings, f)
    
    print(f"Training complete. Saved {len(user_encodings)} user profiles to {output_file}.")

if __name__ == '__main__':
    train_model()
