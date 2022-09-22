import cv2
import pickle


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyes_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

labels = {"person_name": 1}
with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0)
cap.set(3, 640) # set video widht
cap.set(4, 480)
# Define min window size to be recognized as a face
minW = 0.1*cap.get(3)
minH = 0.1*cap.get(4)


while True :
    _, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,
                                          1.2,
                                          minNeighbors=5,
                                          minSize = (int(minW), int(minH)))

    for(x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]

        #recognizer
        id_, confidence = recognizer.predict(roi_gray)
        # if conf>=45 and conf <= 80:
        #     print(id_)
        #     print(labels[id_])
        #roi_color = img[y:y+h, x:x+w]
        #img_item = "my-image.png"
        #cv2.imwrite(img_item, roi_color)
        #cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        #eyes = eyes_cascade.detectMultiScale(roi_gray)
        if (confidence < 100):
            id = labels[id_]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
        # for (ex, ey, ew, eh) in eyes:
        #     cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(img, str(id_), (x + 5, y - 5), font , 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence), (x + 5, y + h - 5),font , 1, (255, 255, 0), 1)
        print(labels[id_])
    # Display the resulting frame
    cv2.imshow('img', img)
    if cv2.waitKey(30) & 0xff == ord('q'):
        break

cap.release()