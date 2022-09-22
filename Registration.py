import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import cv2
from PyQt5 import QtCore, uic
import os.path, os
from PIL import Image
import numpy as np
import pickle
import re
from PyQt5.uic import loadUi

class Registration(QMainWindow):
    def __init__(self):
        super(Registration, self).__init__()
        #uic.loadUi('Registration.ui', self)
        loadUi('Registration.ui', self)
        self.image=None
        self.screenshot = False
        self._image_counter = 1
        self.start_webcam()
        self.saveButton.clicked.connect(self.save)
        self.pushCapture.clicked.connect(self.nameFirstLast)
        self.trainButton.clicked.connect(self.train)
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


    @QtCore.pyqtSlot()
    def start_webcam(self):
        self.capture=cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH,640)
        self.timer=QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(5)

    @QtCore.pyqtSlot()
    def update_frame(self):
        ret, self.image=self.capture.read()
        if self.screenshot == True:
            if self._image_counter != 200:
                firstName1 = self.firstNameInput.toPlainText().replace(" ", "").lower()
                lastName1 = self.lastNameInput.toPlainText().replace(" ", "").lower()
                fullName1 = firstName1 + '-' + lastName1
                path = os.path.join('C:\\Users\\johnc\\Desktop\\MINE\\images\\',fullName1)
                name = "{}.png".format(self._image_counter+1)
                cv2.imwrite(os.path.join(path,name), self.image)
                self._image_counter += 1
                self.warningText.setText(str(self._image_counter))

            else:
                self.screenshot = False
        else:
            self.screenshot = False
            self._image_counter = 0
        self.image=cv2.flip(self.image,1)
        self.displayImage(self.image, 1)
        detected_image=self.faceDetection(self.image)
        self.displayImage(detected_image,1)



    @QtCore.pyqtSlot()
    def faceDetection(self,img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray,1.2,minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        return img

    def displayImage(self,img,window=1):
        qformat=QImage.Format_Indexed8
        if len(img.shape)==3:
            if img.shape[2]==4:
                qformat=QImage.Format_RGBA8888
            else:
                qformat=QImage.Format_RGB888
        outImage=QImage(img,img.shape[1],img.shape[0],img.strides[0],qformat)
        outImage=outImage.rgbSwapped()

        if window==1:
            self.displayVideo.setPixmap(QPixmap.fromImage(outImage))
            self.displayVideo.setScaledContents(True)

    @QtCore.pyqtSlot()
    def train(self):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        image_dir = os.path.join(BASE_DIR, "images")
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        recognizer = cv2.face.LBPHFaceRecognizer_create()

        current_id = 0
        label_ids = {}
        y_labels = []
        x_train = []

        for root, dirs, files in os.walk(image_dir):
           # self.LoadingScreen().startAnimation
            for file in files:
                if file.endswith("png") or file.endswith("jpg"):
                    path = os.path.join(root, file)
                    label = os.path.basename(root).replace(" ", "-").lower()  # change the name format
                    # print(label, path)
                    if not label in label_ids:
                        label_ids[label] = current_id
                        current_id += 1
                    id_ = label_ids[label]
                    # print(label_ids)
                    # y_labels.append(label) # some number
                    # x_train.append(path) # Verify  this image, turn into a numpy array, GRAY
                    pil_image = Image.open(path).convert("L")  # ("L") grayscale
                    # final_image = pil_image.resize((550, 550), Image.Resampling.LANCZOS)
                    # image_array = np.array(final_image, "uint8")
                    image_array = np.array(pil_image, "uint8")
                    # print(image_array)
                    faces = face_cascade.detectMultiScale(image_array)  # , 1.2, minNeighbors=5)

                    for (x, y, w, h) in faces:
                        roi = image_array[y:y + h, x:x + w]
                        x_train.append(roi)
                        y_labels.append(id_)
        # print(y_labels)
        # print(x_train)

        with open("labels.pickle", 'wb') as f:
            pickle.dump(label_ids, f)

        recognizer.train(x_train, np.array(y_labels))
        recognizer.save("trainner.yml")

        self.warningText.setText("Done Training")

        #self.LoadingScreen().startAnimation
    @QtCore.pyqtSlot()
    def nameFirstLast(self):
        firstName = self.firstNameInput.toPlainText().replace(" ", "").lower()
        lastName = self.lastNameInput.toPlainText().replace(" ", "").lower()
        fullName = firstName + '-' + lastName

        special_char = re.compile('[@_!#$%^&*()<>?/\|}{~:]')
        if ((any(map(str.isdigit, firstName)) == True) or (special_char.search(firstName) != None)) or ((any(map(str.isdigit, lastName)) == True) or (special_char.search(lastName) != None)):
            self.warningText.setText("please don't use number/special characters")
        else:
            if os.path.exists(fullName) == False:
                os.chdir("images")
                os.makedirs(fullName)
                self.screenshot=True
                self.warningText.setText('')
                self.firstNameInput.setReadOnly(True)
                self.lastNameInput.setReadOnly(True)
            else:
                self.warningText.setText("Name Already Taken")




    @QtCore.pyqtSlot()
    def save(self):
        self.firstNameInput.setReadOnly(False)
        self.lastNameInput.setReadOnly(False)
        self.firstNameInput.clear()
        self.lastNameInput.clear()
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

class LoadingScreen(QWidget):
    def __init__(self):
        super.__init__()
        self.setFixedSize(200,200)
        self.setWindowFlag(Qt.WindowStaysOnTopHint | Qt.CustomizeWindowHint)
        self.label_animation = Qlabel(self)
        self.movie = QMovie('loadingTrain.gif')
        self.label_animation.setMovie(self.movie)

    def startAnimation(self):
        self.movie.start()

    def stopAnimation(self):
        self.movie.stop()
        self.close()


app=QApplication(sys.argv)
window = Registration()
window.show()
try:
    sys.exit(app.exec_())
except:
    print('exiting')

