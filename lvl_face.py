# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 13:27:36 2018

@author: user
"""

import os

import cv2
import numpy as np
from PIL import Image
from shutil import copy2

from PyQt5.QtCore import QDir
from PyQt5.QtGui import QImage, QPalette, QPixmap
from PyQt5.QtWidgets import (QAction, QApplication, QFileDialog, QLabel,
        QMainWindow, QMenu, QMessageBox, QScrollArea, QSizePolicy, QProgressBar, QDialog)
from PyQt5.QtPrintSupport import QPrinter

#import tensorflow as tf
#from keras.backend.tensorflow_backend import set_session
from keras.models import load_model

#config = tf.ConfigProto(allow_soft_placement = True)
#config.gpu_options.allow_growth = True
##config.gpu_options.per_process_gpu_memory_fraction = 0.3
#set_session(tf.Session(config=config))


class progress(QDialog):
    def __init__(self):
        QDialog.__init__(self)
        
#        progress = QProgressBar(self)
#        progress.setGeometry(30,40,200,25)
#        progress.setProperty("value", 0)  
        
        self.progressBar = QProgressBar(self)
        self.progressBar.setGeometry(30,40,200,25)
        self.progressBar.setProperty("value",0)
        
        self.show()
        self.setWindowTitle('Progress')
        
    def update(self,n):
        self.progressBar.setValue(n)
        
    def total(self,total):
        self.progressBar.setMaximum(total)

class ImageViewer(QMainWindow):
    def __init__(self):
        super(ImageViewer, self).__init__()

        self.printer = QPrinter()
        self.scaleFactor = 0.0

        self.imageLabel = QLabel()
        self.imageLabel.setBackgroundRole(QPalette.Base)
        self.imageLabel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.imageLabel.setScaledContents(True)

        self.scrollArea = QScrollArea()
        self.scrollArea.setBackgroundRole(QPalette.Dark)
        self.scrollArea.setWidget(self.imageLabel)
        self.setCentralWidget(self.scrollArea)

        self.createActions()
        self.createMenus()

        self.setWindowTitle("Lovelyz classifier")
        self.resize(500, 400)
        
              
#        with tf.device('/cpu:0'):
        self.model = load_model('model_facenet.h5')
        self.faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    def prewhiten(self, x):
        if x.ndim == 4:
            axis = (1, 2, 3)
            size = x[0].size
        elif x.ndim == 3:
            axis = (0, 1, 2)
            size = x.size
        else:
            raise ValueError('Dimension should be 3 or 4')
    
        mean = np.mean(x, axis=axis, keepdims=True)
        std = np.std(x, axis=axis, keepdims=True)
        std_adj = np.maximum(std, 1.0/np.sqrt(size))
        y = (x - mean) / std_adj
        return y
    
    def extractThreeFrame(self, inGif):
        out = []
        frame = Image.open(inGif)
        
        frame.seek(0)
        mypalette = frame.getpalette()
        frame.putpalette(mypalette)
        new_im = Image.new("RGB", frame.size)
        new_im.paste(frame)
        new_im = cv2.cvtColor(np.array(new_im), cv2.COLOR_RGB2BGR)
        out.append(new_im)
        
        frame.seek(int(frame.n_frames/2))
        mypalette = frame.getpalette()
        frame.putpalette(mypalette)
        new_im = Image.new("RGB", frame.size)
        new_im.paste(frame)
        new_im = cv2.cvtColor(np.array(new_im), cv2.COLOR_RGB2BGR)
        out.append(new_im)
        
        frame.seek(frame.n_frames-1)
        mypalette = frame.getpalette()
        frame.putpalette(mypalette)
        new_im = Image.new("RGB", frame.size)
        new_im.paste(frame)
        new_im = cv2.cvtColor(np.array(new_im), cv2.COLOR_RGB2BGR)
        out.append(new_im)
        
        return out
    
    def readImage(self, fileName):
        stream = open(fileName, "rb")
        bt = bytearray(stream.read())
        numpyarray = np.asarray(bt, dtype=np.uint8)
        bgrImage = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
        
        if bgrImage.dtype == 'uint16':
            bgrImage = cv2.convertScaleAbs(bgrImage,alpha=(255.0/65535.0))
        if bgrImage.shape[2] == 4:
            bgrImage = cv2.cvtColor(bgrImage, cv2.COLOR_BGRA2BGR)
            
        if np.max(bgrImage.shape) > 1000:
            if bgrImage.shape[0] > bgrImage.shape[1]:
                sc = 1000/bgrImage.shape[0]
            else:
                sc = 1000/bgrImage.shape[1]
            bgrImage = cv2.resize(bgrImage, (int(bgrImage.shape[1]*sc), int(bgrImage.shape[0]*sc)))
        
        return bgrImage

    def findFaceAndClassify(self, fileName):
        bgrImage = self.readImage(fileName)        
        
        faces = self.faceCascade.detectMultiScale(bgrImage, scaleFactor=1.2, minNeighbors=5, minSize=(48,48))
        subjects = ["soul", "jiae", "jisoo", "mijoo", "jiyeon", "myungeun", "soojung", "yein"]

        outimg = bgrImage.copy()
        for k,(x, y, w, h) in enumerate(faces):
        	# extract the confidence (i.e., probability) associated with the
        	# prediction
            im = cv2.resize(outimg[y:y+h, x:x+w],(160,160)).astype('float32')
            im = np.expand_dims(im, axis=0)
            testim = self.prewhiten(im)
            y_prod = self.model.predict(testim)
        #    y_prod = np.random.rand((8)).reshape((1,8))
         
        #    idx = np.argmax(y_prod)
            pt1 = (x, y)
            pt2 = (x+w, y+h)
            cv2.rectangle(outimg,pt1,pt2,(255,0,0),2) 
            
            dx = x-156 if x+w+156 > outimg.shape[1] else x+w+3
            y0, dy = y, 15
            for i, m in enumerate(subjects):
                yo = y0 + (i+1)*dy
                cv2.putText(outimg, m + ' : {:.4f}'.format(y_prod[0][i]), (dx, yo), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))
#        
        return outimg
    
    def findFaceAndClassifyForAuto(self, fileName):
        bgrImage = self.readImage(fileName)        
        
        faces = self.faceCascade.detectMultiScale(bgrImage, scaleFactor=1.2, minNeighbors=5, minSize=(48,48))

        batches = np.zeros((len(faces),160,160,3), dtype='float32')
        for k,(x, y, w, h) in enumerate(faces):
        	# extract the confidence (i.e., probability) associated with the
        	# prediction
            im = cv2.resize(bgrImage[y:y+h, x:x+w],(160,160)).astype('float32')
            im = np.expand_dims(im, axis=0)
            testim = self.prewhiten(im)
            batches[k] = testim
        
        y_prod = self.model.predict(batches)
         
        return y_prod
    
    def findFaceAndClassifyForAuto_gif(self, fileName):
        bgrImage = self.extractThreeFrame(fileName)        
        
        batches_house = []
        for im in bgrImage:
            faces = self.faceCascade.detectMultiScale(im, scaleFactor=1.2, minNeighbors=5, minSize=(48,48))
    
            batchest = np.zeros((len(faces),160,160,3), dtype='float32')
            for k,(x, y, w, h) in enumerate(faces):
            	# extract the confidence (i.e., probability) associated with the
            	# prediction
                im_ = cv2.resize(im[y:y+h, x:x+w],(160,160)).astype('float32')
                im_ = np.expand_dims(im_, axis=0)
                testim = self.prewhiten(im_)
                batchest[k] = testim
            
            batches_house.append(batchest)
            
        batches = np.concatenate(batches_house)
        y_prod = self.model.predict(batches)
         
        return y_prod
                    
    def open(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open File",
                QDir.currentPath())
        if fileName:
#            fileName = 'data/soojung/1513023157032.jpg'     
            #    cv2.getTextSize('myungeun : 0.3513', cv2.FONT_HERSHEY_SIMPLEX, 0.5,1)
            #    cv2.rectangle(bgrImage, (dx+w, y), (dx+w+94, yo+3), (255,255,255), -1)
                
            outimg = self.findFaceAndClassify(fileName)
            outimg = cv2.cvtColor(outimg, cv2.COLOR_BGR2RGB)
            h, w, c = outimg.shape
            bytesPerLine = 3 * w
            image = QImage(outimg, w, h, bytesPerLine, QImage.Format_RGB888)
            if image.isNull():
                QMessageBox.information(self, "Image Viewer",
                        "Cannot load %s." % fileName)
                return
            
            self.imageLabel.setPixmap(QPixmap.fromImage(image))
            self.scaleFactor = 1.0

            self.fitToWindowAct.setEnabled(True)
            self.updateActions()
            self.resize(w, h)
            if not self.fitToWindowAct.isChecked():
                self.imageLabel.adjustSize()

    def zoomIn(self):
        self.scaleImage(1.25)

    def zoomOut(self):
        self.scaleImage(0.8)

    def normalSize(self):
        self.imageLabel.adjustSize()
        self.scaleFactor = 1.0

    def fitToWindow(self):
        fitToWindow = self.fitToWindowAct.isChecked()
        self.scrollArea.setWidgetResizable(fitToWindow)
        if not fitToWindow:
            self.normalSize()

        self.updateActions()

    def autoClassify(self):
        dirName = QFileDialog.getExistingDirectory()
        
        subjects_hangle = ["소울", "지애", "지수", "미주", "지연", "명은", "수정", "예인"]
        for m in subjects_hangle:
            pp = os.path.join(dirName, m)
            if not os.path.isdir(pp):
                os.mkdir(pp)
        
        filenames = os.listdir(dirName)
        
        ims = []
        for filename in filenames:            
            fullname = os.path.join(dirName, filename)
            if not os.path.isdir(fullname):
                ims.append(fullname)
        
        cantOpen = []
        failed = []
        prog = progress()
        prog.total(len(ims))
        for k, im in enumerate(ims):            
#            progress.setValue(k * 100/len(ims))
            prog.update(k)
            QApplication.processEvents()
            prediction = None
            try:
                if im.split('.')[-1] == 'gif':
                    prediction = self.findFaceAndClassifyForAuto_gif(im)
                else:
                    prediction = self.findFaceAndClassifyForAuto(im)
            
            
                isex = False
                for cand, prob in enumerate(prediction):
                    if np.max(prob) < 0.8:
                        isex = True
                    else:
                        copy2(im, os.path.join(dirName, subjects_hangle[np.argmax(prob)]))
                    
                if isex:
                    failed.append(im)
            except:
                print('Can\'t open ',im)
                cantOpen.append(im)
                pass
            
        np.savetxt(os.path.join(dirName,'잘뭐르게써여.txt'), failed, fmt='%s')
        np.savetxt(os.path.join(dirName,'열지 못한 파일.txt'), cantOpen, fmt='%s')
        del prog
    
    def createActions(self):
        self.openAct = QAction("&Open...", self, shortcut="Ctrl+O",
                triggered=self.open)


        self.exitAct = QAction("E&xit", self, shortcut="Ctrl+Q",
                triggered=self.close)

        self.zoomInAct = QAction("Zoom &In (25%)", self, shortcut="Ctrl++",
                enabled=False, triggered=self.zoomIn)

        self.zoomOutAct = QAction("Zoom &Out (25%)", self, shortcut="Ctrl+-",
                enabled=False, triggered=self.zoomOut)

        self.normalSizeAct = QAction("&Normal Size", self, shortcut="Ctrl+S",
                enabled=False, triggered=self.normalSize)

        self.fitToWindowAct = QAction("&Fit to Window", self, enabled=False,
                checkable=True, shortcut="Ctrl+F", triggered=self.fitToWindow)

        self.autoClassifyAct = QAction("&자동분류", self, triggered=self.autoClassify)

    def createMenus(self):
        self.fileMenu = QMenu("&File", self)
        self.fileMenu.addAction(self.openAct)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.exitAct)

        self.viewMenu = QMenu("&View", self)
        self.viewMenu.addAction(self.zoomInAct)
        self.viewMenu.addAction(self.zoomOutAct)
        self.viewMenu.addAction(self.normalSizeAct)
        self.viewMenu.addSeparator()
        self.viewMenu.addAction(self.fitToWindowAct)

        self.dirMenu = QMenu("&Auto", self)
        self.dirMenu.addAction(self.autoClassifyAct)

        self.menuBar().addMenu(self.fileMenu)
        self.menuBar().addMenu(self.viewMenu)
        self.menuBar().addMenu(self.dirMenu)

    def updateActions(self):
        self.zoomInAct.setEnabled(not self.fitToWindowAct.isChecked())
        self.zoomOutAct.setEnabled(not self.fitToWindowAct.isChecked())
        self.normalSizeAct.setEnabled(not self.fitToWindowAct.isChecked())

    def scaleImage(self, factor):
        self.scaleFactor *= factor
        self.imageLabel.resize(self.scaleFactor * self.imageLabel.pixmap().size())

        self.adjustScrollBar(self.scrollArea.horizontalScrollBar(), factor)
        self.adjustScrollBar(self.scrollArea.verticalScrollBar(), factor)

        self.zoomInAct.setEnabled(self.scaleFactor < 3.0)
        self.zoomOutAct.setEnabled(self.scaleFactor > 0.333)

    def adjustScrollBar(self, scrollBar, factor):
        scrollBar.setValue(int(factor * scrollBar.value()
                                + ((factor - 1) * scrollBar.pageStep()/2)))


if __name__ == '__main__':

    import sys

    app = QApplication(sys.argv)
    imageViewer = ImageViewer()
    imageViewer.show()
    sys.exit(app.exec_())