import cv2

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('resources/haarcascade_frontalface_default.xml')
##The eyes part does not work very well and has alot of extra things that show
eye_cascade = cv2.CascadeClassifier('resources/haarcascade_eye.xml')

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    #Mess around with the value of this or preprocess the img more
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)


    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    for (x, y, w, h) in eyes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow('Live Feed', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()