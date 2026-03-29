import cv2
import os
import numpy as np

IMG_SIZE = (200,200)

def preprocess_face(face_img):
    face_img = cv2.resize(face_img, IMG_SIZE)
    face_img = cv2.equalizeHist(face_img)
    return face_img


def capture_new_user(user_name):
    path = f'data/{user_name}'
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory created for: {user_name}")
    else:
        print(f"User {user_name} already exists. Adding more photos.")

    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('resources/haarcascade_frontalface_default.xml')

    count = 0
    print("Look at the camera. Capturing images...")

    while count < 100:
        ret, frame = cap.read()
        if not ret: break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            face_img = gray[y:y + h, x:x + w]
            file_name = f"{path}/{user_name}_{count}.jpg"
            cv2.imwrite(file_name, face_img)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, f"Captured: {count}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Capturing Face Data', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Successfully captured {count} images for {user_name}")


def train_model():
    #Make a better way to train the model because this is bad and does not work well
    # It thinks everyone is the same person
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    face_samples = []
    ids = []

    name_map = {}
    current_id = 0
    path = 'data'

    user_folders = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

    for user_name in user_folders:
        name_map[current_id] = user_name
        subject_path = os.path.join(path, user_name)
        for filename in os.listdir(subject_path):
            img_path = os.path.join(subject_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            face_samples.append(img)
            ids.append(current_id)

        current_id += 1

    recognizer.train(face_samples, np.array(ids))
    recognizer.save('trainer.yml')
    print(f"Trained on: {name_map}")
    return name_map


def get_registered_names(data_path='data'):
    if not os.path.exists(data_path):
        return []

    names = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]

    return names

# capture_new_user("Logan")
# train_model()

cap = cv2.VideoCapture(0)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')
names = get_registered_names()

face_cascade = cv2.CascadeClassifier('resources/haarcascade_frontalface_default.xml')
# This works bad with glasses and does not detected closed eyes but we need to use another data set
# to do this but works ok for now
eye_cascade = cv2.CascadeClassifier('resources/haarcascade_eye.xml')

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id_label, confidence = recognizer.predict(gray[y:y + h, x:x + w])

        if confidence < 40: #Figure out a way to make this higher but it does not work
            name = names[id_label]
            color = (0, 255, 0)
        else:
            name = "Unknown"
            color = (0, 0, 255)

        cv2.putText(frame, name, (x + 5, y - 5), cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 1)

        face_roi_color = frame[y:y + h, x:x + w]
        face_roi_gray = gray[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(face_roi_gray, 1.1, 10)
        for (ex, ey, ew, eh) in eyes:
            if ey + (eh / 2) < h * 0.6:
                cv2.circle(face_roi_color, (ex + ew // 2, ey + eh // 2),
                           int(ew / 2.5), (255, 0, 0), 2)

    cv2.imshow('Live Feed', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
