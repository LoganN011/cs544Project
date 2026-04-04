import cv2
import os
import argparse
from facenet_pytorch import MTCNN
import torch

def capture_new_user(user_name, num_samples=100, save_dir='data'):
    path = os.path.join(save_dir, user_name)
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory created for: {user_name}")
    else:
        print(f"User {user_name} already exists. Adding more photos.")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))
    mtcnn = MTCNN(keep_all=False, device=device) # keep_all=False means return only one face

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    count = 0
    print("Look at the camera. Capturing images...")

    while count < num_samples:
        ret, frame = cap.read()
        if not ret: 
            break
        
        # Convert BGR to RGB for MTCNN
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect face boxes
        boxes, _ = mtcnn.detect(img_rgb)
        
        if boxes is not None and len(boxes) > 0:
            box = boxes[0]
            x, y, x2, y2 = [int(b) for b in box]
            
            # Bound the coordinates
            x = max(0, x)
            y = max(0, y)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            
            # Make sure face is valid
            if (x2 - x) > 0 and (y2 - y) > 0:
                count += 1
                file_name = f"{path}/{user_name}_{count}.jpg"
                cv2.imwrite(file_name, frame) # Save full frame

                cv2.rectangle(frame, (x, y), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"Captured: {count}/{num_samples}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Capturing Face Data', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Successfully captured {count} images for {user_name}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Capture face images for a user.')
    parser.add_argument('--name', type=str, required=True, help='Name of the user to capture.')
    parser.add_argument('--samples', type=int, default=100, help='Number of sample images to record.')
    args = parser.parse_args()
    
    capture_new_user(args.name, args.samples)
