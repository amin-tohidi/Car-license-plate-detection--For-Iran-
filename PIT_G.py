import os
import cv2
import sys 
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from segmention import SEGMENT
import matplotlib.pyplot as plt
import shutil
from keras.models import load_model    
import keras
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense,Flatten 
from keras.layers import Convolution2D, MaxPooling2D
from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(719, 468)
        MainWindow.setStyleSheet("background-color: rgb(152, 199, 240);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser.setGeometry(QtCore.QRect(10, 30, 341, 251))
        self.textBrowser.setStyleSheet("background-color: rgb(240, 240, 240);")
        self.textBrowser.setObjectName("textBrowser")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(570, 20, 141, 261))
        self.groupBox.setStyleSheet("background-color: rgb(0, 85, 255);")
        self.groupBox.setObjectName("groupBox")
        self.pelak = QtWidgets.QPushButton(self.groupBox)
        self.pelak.setGeometry(QtCore.QRect(20, 20, 101, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pelak.setFont(font)
        self.pelak.setStyleSheet("background-color: rgb(83, 255, 112);")
        self.pelak.setObjectName("pelak")
        self.train = QtWidgets.QPushButton(self.groupBox)
        self.train.setGeometry(QtCore.QRect(20, 80, 101, 51))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.train.setFont(font)
        self.train.setStyleSheet("background-color: rgb(141, 255, 80);")
        self.train.setObjectName("train")
        self.clear = QtWidgets.QPushButton(self.groupBox)
        self.clear.setGeometry(QtCore.QRect(20, 200, 101, 51))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.clear.setFont(font)
        self.clear.setStyleSheet("background-color: rgb(253, 255, 94);")
        self.clear.setObjectName("clear")
        self.save = QtWidgets.QPushButton(self.groupBox)
        self.save.setGeometry(QtCore.QRect(20, 140, 101, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.save.setFont(font)
        self.save.setStyleSheet("background-color: rgb(186, 255, 82);")
        self.save.setObjectName("save")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(370, 20, 191, 261))
        self.groupBox_2.setStyleSheet("background-color: rgb(0, 170, 255);")
        self.groupBox_2.setObjectName("groupBox_2")
        self.label_2 = QtWidgets.QLabel(self.groupBox_2)
        self.label_2.setGeometry(QtCore.QRect(70, 30, 111, 20))
        self.label_2.setStyleSheet("background-color: rgb(152, 199, 240);")
        self.label_2.setObjectName("label_2")
        self.filter1 = QtWidgets.QSpinBox(self.groupBox_2)
        self.filter1.setGeometry(QtCore.QRect(20, 30, 42, 22))
        self.filter1.setStyleSheet("background-color: rgb(225, 225, 225);")
        self.filter1.setObjectName("filter1")
        self.label_3 = QtWidgets.QLabel(self.groupBox_2)
        self.label_3.setGeometry(QtCore.QRect(70, 60, 111, 20))
        self.label_3.setStyleSheet("background-color: rgb(152, 199, 240);")
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.groupBox_2)
        self.label_4.setGeometry(QtCore.QRect(70, 90, 111, 20))
        self.label_4.setStyleSheet("background-color: rgb(152, 199, 240);")
        self.label_4.setObjectName("label_4")
        self.filter11 = QtWidgets.QSpinBox(self.groupBox_2)
        self.filter11.setGeometry(QtCore.QRect(20, 60, 42, 22))
        self.filter11.setStyleSheet("background-color: rgb(225, 225, 225);")
        self.filter11.setObjectName("filter11")
        self.filter2 = QtWidgets.QSpinBox(self.groupBox_2)
        self.filter2.setGeometry(QtCore.QRect(20, 90, 42, 22))
        self.filter2.setStyleSheet("background-color: rgb(225, 225, 225);")
        self.filter2.setObjectName("filter2")
        self.label_8 = QtWidgets.QLabel(self.groupBox_2)
        self.label_8.setGeometry(QtCore.QRect(70, 120, 111, 20))
        self.label_8.setStyleSheet("background-color: rgb(152, 199, 240);")
        self.label_8.setObjectName("label_8")
        self.filter22 = QtWidgets.QSpinBox(self.groupBox_2)
        self.filter22.setGeometry(QtCore.QRect(20, 120, 42, 22))
        self.filter22.setStyleSheet("background-color: rgb(225, 225, 225);")
        self.filter22.setObjectName("filter22")
        self.label_9 = QtWidgets.QLabel(self.groupBox_2)
        self.label_9.setGeometry(QtCore.QRect(70, 210, 111, 20))
        self.label_9.setStyleSheet("background-color: rgb(152, 199, 240);")
        self.label_9.setObjectName("label_9")
        self.epock = QtWidgets.QSpinBox(self.groupBox_2)
        self.epock.setGeometry(QtCore.QRect(20, 210, 42, 22))
        self.epock.setStyleSheet("background-color: rgb(225, 225, 225);")
        self.epock.setObjectName("epock")
        self.label_10 = QtWidgets.QLabel(self.groupBox_2)
        self.label_10.setGeometry(QtCore.QRect(70, 180, 111, 20))
        self.label_10.setStyleSheet("background-color: rgb(152, 199, 240);")
        self.label_10.setObjectName("label_10")
        
        self.learning = QtWidgets.QLineEdit(self.groupBox_2)
        self.learning.setGeometry(QtCore.QRect(20, 180, 42, 22))
        self.learning.setStyleSheet("background-color: rgb(225, 225, 225);")
        self.learning.setObjectName("learning")
        
        # self.learning_rate = QtWidgets.QLineEdit(self.groupBox)
        # self.learning_rate.setGeometry(QtCore.QRect(10, 220, 101, 21))
        # self.learning_rate.setStyleSheet("background-color: rgb(236, 236, 236);")
        # self.learning_rate.setObjectName("learning_rate")
        
        
        
        
        
        
        
        self.label_11 = QtWidgets.QLabel(self.groupBox_2)
        self.label_11.setGeometry(QtCore.QRect(70, 150, 111, 20))
        self.label_11.setStyleSheet("background-color: rgb(152, 199, 240);")
        self.label_11.setObjectName("label_11")
        self.noron = QtWidgets.QSpinBox(self.groupBox_2)
        self.noron.setGeometry(QtCore.QRect(20, 150, 42, 22))
        self.noron.setStyleSheet("background-color: rgb(225, 225, 225);")
        self.noron.setObjectName("noron")
        self.textBrowser_2 = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser_2.setGeometry(QtCore.QRect(10, 310, 701, 51))
        self.textBrowser_2.setStyleSheet("background-color: rgb(199, 199, 199);")
        self.textBrowser_2.setObjectName("textBrowser_2")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(150, -10, 41, 31))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.exit = QtWidgets.QPushButton(self.centralwidget)
        self.exit.setGeometry(QtCore.QRect(610, 390, 91, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.exit.setFont(font)
        self.exit.setStyleSheet("background-color: rgb(255, 189, 189);")
        self.exit.setObjectName("exit")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(-10, 370, 781, 20))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.label_12 = QtWidgets.QLabel(self.centralwidget)
        self.label_12.setGeometry(QtCore.QRect(0, 390, 441, 20))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.label_12.setFont(font)
        self.label_12.setObjectName("label_12")
        self.label_13 = QtWidgets.QLabel(self.centralwidget)
        self.label_13.setGeometry(QtCore.QRect(0, 410, 301, 20))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.label_13.setFont(font)
        self.label_13.setObjectName("label_13")
        self.label_14 = QtWidgets.QLabel(self.centralwidget)
        self.label_14.setGeometry(QtCore.QRect(0, 430, 301, 20))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.label_14.setFont(font)
        self.label_14.setObjectName("label_14")
        self.line_2 = QtWidgets.QFrame(self.centralwidget)
        self.line_2.setGeometry(QtCore.QRect(-10, 290, 781, 20))
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 719, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.exit.clicked.connect(MainWindow.close)

        self.pelak.clicked.connect(self.DETECT)
        self.train.clicked.connect(self.train2)
        self.clear.clicked.connect(self.cl)

    def cl(self):
        self.textBrowser_2.clear()
        self.textBrowser.clear()
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox.setTitle(_translate("MainWindow", "منو"))
        self.pelak.setText(_translate("MainWindow", "انتخاب پلاک"))
        self.train.setText(_translate("MainWindow", "آموزش مدل جدید"))
        self.clear.setText(_translate("MainWindow", "پاک کردن"))
        self.save.setText(_translate("MainWindow", "ذخیره اطاعات"))
        self.groupBox_2.setTitle(_translate("MainWindow", "مشخصات مدل کانولوشن"))
        self.label_2.setText(_translate("MainWindow", "تعداد فیلتر لایه اول"))
        self.label_3.setText(_translate("MainWindow", "اندازه فیلتر لایه اول"))
        self.label_4.setText(_translate("MainWindow", "تعداد فیلتر لایه دوم"))
        self.label_8.setText(_translate("MainWindow", "اندازه فیلتر لایه دوم"))
        self.label_9.setText(_translate("MainWindow", "              Epock تعداد "))
        self.label_10.setText(_translate("MainWindow", "نرخ یادگیری"))
        self.label_11.setText(_translate("MainWindow", "تعداد نورون های لایه FC"))
        self.label.setText(_translate("MainWindow", "نتایج"))
        self.exit.setText(_translate("MainWindow", "خروج"))
        self.label_12.setText(_translate("MainWindow", "دانشگاه مالک اشتر - دانشکده برق و کامپیوتر - گروه آموزشی هوش مصنوعی و کامپیوتر"))
        self.label_13.setText(_translate("MainWindow", "امین توحیدی فر    :             amin.tohidifar@yahoo.com"))
        self.label_14.setText(_translate("MainWindow", "علیرضا خرم آبادی  :   alirezakhoramabadi96@gmail.com"))



    def DETECT(self):
        
        self.textBrowser_2.clear()
        # self.textBrowser.clear()

        self.textBrowser_2.append('برنامه در حال خواندن پلاک و انجام سگمنتیشن و  OCRمی باشد\n')
        self.textBrowser_2.append('لطفا صبر کنید ....')
        Tk().withdraw() 
        filename2 = askopenfilename() 
        img1=cv2.imread(filename2, 1)
        
        gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        
        filtered = cv2.GaussianBlur(gray, (5,5), 0)
        
        edged = cv2.Canny(filtered, 10, 100) 
        cv2.imshow('2211', edged)
        
        contours, hir = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        plate = False
        for c in contours:
            perimeter = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
            if len(approx) == 4 and cv2.contourArea(c) > 1000:
                x, y, w, h = cv2.boundingRect(c)
                if 2.5 < w / h < 4.1:
                    plate = True
                    cv2.drawContours(img1, c, -1, (0, 255, 0), 3)
                    cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    break


                    
        if not plate:       
            for c in contours:
                perimeter = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
                if len(approx) >= 4:
                     x, y, w, h = cv2.boundingRect(c)
                     if 2.5 < w / h < 4.5 and 10000 <= (w * h):
                         plate = True
                         cv2.drawContours(img1, c, -1, (0, 0, 255), 1)
                         cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 0,255), 3)
                         break
        

        cv2.imshow('TASVIR ASLI', img1)
            
        if plate:
            cropped = img1[y-10:y+h+25, x-10:x+w+30]
            cv2.imshow('PELAK', cropped)
            cv2.imwrite('plate.jpg', cropped)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()    
        

        for root, dirs, files in os.walk('C:/Users/AMIN/Desktop/PLT/SEG'):
            for f in files:
                os.unlink(os.path.join(root, f))
            for d in dirs:
                shutil.rmtree(os.path.join(root, d))
        
      
        img=cv2.imread("C:/Users/AMIN/Desktop/PLT/plate.jpg", 0)
        
        # plt.imshow(img)
        
        num,plate=SEGMENT(img)
        plt.figure(figsize = (5,10))
        plt.imshow(plate)
        
        k=20
        j=0  
                        
        for i in num:
        
            ax=plt.subplot(10, 20, k+1)
            ax.axis('off')
            plt.imshow(i)
            k+=2
            # plt.figure(figsize = (10,10))
            # plt.imshow(i)
            cv2.imwrite('SEG/'  + 'AA'+str(j)+'.jpg', i)
            j += 1
            
                  
        
        model = keras.models.load_model('C:/Users/AMIN/Desktop/PLT/20.h5')
        
            
            
        directory = 'C:/Users/AMIN/Desktop/PLT/SEG'
        number_of_files = len([item for item in os.listdir(directory) 
                                   if os.path.isfile(os.path.join(directory, item))])

        
        self.textBrowser.append('\n از عدد های موجود در پلاک ' + str(number_of_files) +'عدد شناسایی شد ' )
        
        
        p=[]
       
        for i in range(0,number_of_files):
                img = cv2.imread('C:/Users/AMIN/Desktop/PLT/SEG/AA'+ str(i) +'.jpg',0)
                plt.imshow(img, cmap="Greys")
                
                img = cv2.resize(img, (60, 120))
                
                img = np.reshape(img, [1, 60, 120, 1])
                index =np.argmax(model.predict(img))
                p.append(index)
                i=i+1
                


        img2 = cv2.imread("C:/Users/AMIN/Desktop/PLT/plate.jpg")
        plt.imshow(img2, cmap="Greys")
             
        self.textBrowser.append('\n نتیجه تشخیص اعداد شناسایی شده در پلاک به صورت زیر است       :\n\n      '   +  (str(p)))

        if len(p)>8:
            self.textBrowser.append("\n\n متاسفانه بعضی ازاشکالی که شبیه کاراکتر بودند، به اشتباه کاراکتر تشخیص داده شده اند")
        else :
                if len(p)<8 :
                        self.textBrowser.append("\n\n متاسفانه بعضی از کاراکتر های پلاک شناسایی نشدند")

            

        self.textBrowser.append('\n ***************************************************')
        self.textBrowser_2.clear()
        self.textBrowser_2.append('\n تشخیص پلاک با موفقیت انجام شد.')


    def train2(self):
        
        self.textBrowser.clear()
        self.textBrowser_2.clear()
        self.textBrowser_2.append('در حال آموزش مدل جدید -  لطفا صبر کنید ممکن است چند دقیقه طول بکشد .....' )

        if self.learning.text() !=""  :
        
            n1=self.filter1.text()
            n1=int(n1)
            n2=self.filter11.text()
            n2=int(n2)
            n3=self.filter2.text()
            n3=int(n3)
            n4=self.filter22.text()
            n4=int(n4)
            n5=self.epock.text()
            n5=int(n5)
            n6=self.noron.text()
            n6=int(n6)
            n7=self.learning.text()
            n7=float(n7)


            img_gen=ImageDataGenerator(rescale=1./255, validation_split=0.2)
     
            train_set=img_gen.flow_from_directory('C:/Users/AMIN/Desktop/PLT/DATA',
                                                                      target_size=(60,120),
                                                                      batch_size=100,
                                                                      class_mode='categorical',
                                                                      color_mode='grayscale',
                                                                      subset='training',
                                                                      seed=1)
                        
            valid_set=img_gen.flow_from_directory('C:/Users/AMIN/Desktop/PLT/DATA',
                                                                      target_size=(60,120),
                                                                      batch_size=100,
                                                                      class_mode='categorical',
                                                                      color_mode='grayscale',
                                                                      subset='validation',
                                                                      seed=1)
            
    
            model = Sequential()
            model.add(Convolution2D(n1,n2, activation='relu',input_shape=(60,120,1)))
            model.add(Convolution2D(n3, n4, activation='relu'))
            model.add(MaxPooling2D(pool_size=(2,2)))
            model.add(Flatten())
            model.add(Dense(n6, activation='relu'))
            model.add(Dense( 9, activation='softmax'))
                                    
                     
            model.compile(loss='categorical_crossentropy',
                          optimizer=Adam(learning_rate=n7, name='Adam'),
                          metrics=['accuracy'])
                        
                      
            history = model.fit(x=train_set, validation_data=valid_set,
                                            epochs=n5, verbose=1)

            
            model.save("model_jadid.h5")
        
            self.textBrowser.append('مدل ذخیره شد' )
     
            x=max(history.history['accuracy'])
            y=min(history.history['loss'])
            xx=max(history.history['val_accuracy'])
            yy=min(history.history['val_loss']) 
            
            self.textBrowser.append('\n دقت روی دااده های آموزش   :       '   +  (str(x)))
            self.textBrowser.append('\n خطا روی داده های آموزش   :        '   +str(y))
            self.textBrowser.append('\n دقت روی دادهای ارزیابی    :          '   +str(xx))
            self.textBrowser.append('\n خطا روی داده های ارزیابی    :         '   +str(yy))
            
            self.textBrowser_2.clear()
            self.textBrowser_2.append('مدل با موفقیت آموزش داده شد و برای سهولت در استفاده ذخیره شد .' )

        else:
            self.textBrowser.append('            \n\n\n\n\n\tلطفا مقادیر مشخص شده را وارد نمایید ' )
        


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
