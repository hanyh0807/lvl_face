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
        QMainWindow, QMenu, QMessageBox, QScrollArea, QSizePolicy, QProgressBar, QDialog, QActionGroup)
from PyQt5.QtPrintSupport import QPrinter

import face_recognition

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
        
        self.isclosed = False
    
    def closeEvent(self, event):
      self.isclosed = True
#      print('closed!!!')
      
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
        self.face_detector = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt','res10_300x300_ssd_iter_140000.caffemodel')
        self.model = cv2.dnn.readNetFromTensorflow('model.pb','model.pbtxt')
        
        self.useHOG = True

    def prewhiten(self, x):
        """
        for facenet
        """
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
    
    def preprocess_input(self, x, data_format='channels_last', version=1):
        """
        for vggface
        """
        x_temp = np.copy(x)
    
        if version == 1: # VGG
            if data_format == 'channels_first':
                x_temp = x_temp[:, ::-1, ...]
                x_temp[:, 0, :, :] -= 93.5940
                x_temp[:, 1, :, :] -= 104.7624
                x_temp[:, 2, :, :] -= 129.1863
            else:
                x_temp = x_temp[..., ::-1]
                x_temp[..., 0] -= 93.5940
                x_temp[..., 1] -= 104.7624
                x_temp[..., 2] -= 129.1863
    
        elif version == 2: # RESNET50, SENET50
            if data_format == 'channels_first':
                x_temp = x_temp[:, ::-1, ...]
                x_temp[:, 0, :, :] -= 91.4953
                x_temp[:, 1, :, :] -= 103.8827
                x_temp[:, 2, :, :] -= 131.0912
            else:
                x_temp = x_temp[..., ::-1]
                x_temp[..., 0] -= 91.4953
                x_temp[..., 1] -= 103.8827
                x_temp[..., 2] -= 131.0912
        else:
            raise NotImplementedError
    
        return x_temp
    
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
            
        maxpix = 1024
        if np.max(bgrImage.shape) > maxpix:
            if bgrImage.shape[0] > bgrImage.shape[1]:
                sc = maxpix/bgrImage.shape[0]
            else:
                sc = maxpix/bgrImage.shape[1]
            bgrImage = cv2.resize(bgrImage, (int(bgrImage.shape[1]*sc), int(bgrImage.shape[0]*sc)))
        
        bgrImage= cv2.copyMakeBorder(bgrImage,30,30,30,30,cv2.BORDER_CONSTANT,value=0)
        
        return bgrImage
    
    def detectFace(self, im):
        (h, w) = im.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(im, (300, 300)), 1.0,	(300, 300), (104.0, 177.0, 123.0))

        self.face_detector.setInput(blob)
        detections = self.face_detector.forward()

        faces = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                faces.append(box.astype("int")) # (startX, startY, endX, endY)
                
        return faces

        
    def imaugmentAndBatch(self, im):
        rows, cols = im.shape[:2]
        M = cv2.getRotationMatrix2D((cols/2,rows/2),10,1)
        Mm = cv2.getRotationMatrix2D((cols/2,rows/2),-10,1)
        im2 = cv2.warpAffine(im, M, (cols, rows))
        im3 = cv2.warpAffine(im, Mm, (cols, rows))
        im4 = cv2.flip(im,1)
        im = np.expand_dims(im, axis=0)
        im2= np.expand_dims(im2, axis=0)
        im3 = np.expand_dims(im3, axis=0)
        im4 = np.expand_dims(im4, axis=0)
        
        batches = np.zeros((4,224,224,3), dtype='float32')
#        batches[0] = self.preprocess_input(im)
#        batches[1] = self.preprocess_input(im2)
#        batches[2] = self.preprocess_input(im3)
#        batches[3] = self.preprocess_input(im4)
        batches[0] = im
        batches[1] = im2
        batches[2] = im3
        batches[3] = im4
        batches = cv2.dnn.blobFromImages(batches, 1.0,	(224, 224), (93.594, 104.7624, 129.1863))
        
        return batches

    def findFaceAndClassify(self, fileName):
        bgrImage = self.readImage(fileName)               
        
        if self.useHOG:
            faces = face_recognition.face_locations(cv2.cvtColor(bgrImage, cv2.COLOR_BGR2GRAY))
        else:
            faces = self.detectFace(bgrImage)

        subjects = ["soul", "jiae", "jisoo", "mijoo", "jiyeon", "myungeun", "soojung", "yein"]

        outimg = bgrImage.copy()
        for k,(startX, startY, endX, endY) in enumerate(faces):            
        	# extract the confidence (i.e., probability) associated with the
        	# prediction
            if self.useHOG:
                y1 = startX; y2 = endX; x1 = endY; x2 = startY
                im = cv2.resize(outimg[y1:y2, x1:x2],(224,224)).astype('float32')
            else:
                y1 = startY; y2 = endY; x1 = startX; x2 = endX
                im = cv2.resize(outimg[y1:y2, x1:x2],(224,224)).astype('float32')
            batches = self.imaugmentAndBatch(im)     
            
            self.model.setInput(batches)
            y_prod = self.model.forward()
            
            cors = np.corrcoef(y_prod)
            np.fill_diagonal(cors,0)
            e = cors.sum(axis=1)
            wei = np.exp(e)/np.exp(e).sum()
            y_prod = (y_prod*wei.reshape([-1,1])).sum(axis=0)
#            print(y_prod)
            
            if np.median(y_prod) > 0.08:
                continue
         
            pt1 = (x1, y1)
            pt2 = (x2, y2)
            cv2.rectangle(outimg,pt1,pt2,(255,0,0),2) 
            
            dx = x1-156 if x2+156 > outimg.shape[1] else x2+3
            y0, dy = y1, 15
            for i, m in enumerate(subjects):
                yo = y0 + (i+1)*dy
#                cv2.putText(outimg, m + ' : {:.4f}'.format(y_prod[0][i]), (dx, yo), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))
                cv2.putText(outimg, m + ' : {:.4f}'.format(y_prod[i]), (dx, yo), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))
#        
        return outimg
    
    def findFaceAndClassifyForAuto(self, fileName):
        bgrImage = self.readImage(fileName)        

        if self.useHOG:
            faces = face_recognition.face_locations(cv2.cvtColor(bgrImage, cv2.COLOR_BGR2GRAY))
        else:
            faces = self.detectFace(bgrImage)
        
        batches_house = []
        for k,(startX, startY, endX, endY) in enumerate(faces):
        	# extract the confidence (i.e., probability) associated with the
        	# prediction
            if self.useHOG:
                y1 = startX; y2 = endX; x1 = endY; x2 = startY
                im = cv2.resize(bgrImage[y1:y2, x1:x2],(224,224)).astype('float32')
            else:
                y1 = startY; y2 = endY; x1 = startX; x2 = endX
                im = cv2.resize(bgrImage[y1:y2, x1:x2],(224,224)).astype('float32')
            testim = self.imaugmentAndBatch(im)
            batches_house.append(testim)
        
        batches = np.concatenate(batches_house)
        self.model.setInput(batches)
        y_prod = self.model.forward()
         
        return y_prod
    
    def findFaceAndClassifyForAuto_gif(self, fileName):
        bgrImage = self.extractThreeFrame(fileName)        
        
        batches_house = []
        for im in bgrImage:
            if self.useHOG:
                faces = face_recognition.face_locations(cv2.cvtColor(bgrImage, cv2.COLOR_BGR2GRAY))
            else:
                faces = self.detectFace(bgrImage)
            
            batchest = []
            for k,(startX, startY, endX, endY) in enumerate(faces):
            	# extract the confidence (i.e., probability) associated with the
            	# prediction
                if self.useHOG:
                    y1 = startX; y2 = endX; x1 = endY; x2 = startY
                    im_ = cv2.resize(im[y1:y2, x1:x2],(224,224)).astype('float32')
                else:
                    y1 = startY; y2 = endY; x1 = startX; x2 = endX
                    im_ = cv2.resize(im[y1:y2, x1:x2],(224,224)).astype('float32')
                
                testim = self.imaugmentAndBatch(im_)
                batchest.append(testim)
            
            batchest = np.concatenate(batchest)
            batches_house.append(batchest)
            
        batches = np.concatenate(batches_house)
        self.model.setInput(batches)
        y_prod = self.model.forward()
         
        return y_prod
    
    def findFaceAndClassifyForAuto_wrapper(self, fileName):
        pass
    
    
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
        
        if dirName:
            subjects_hangle = ["소울", "지애", "지수", "미주", "지연", "명은", "수정", "예인"]
            for m in subjects_hangle:
                pp = os.path.join(dirName, m)
                if not os.path.isdir(pp):
                    os.mkdir(pp)
            alayalay = os.path.join(dirName, '미묘미묘해')
            if not os.path.isdir(alayalay):
                os.mkdir(alayalay)
            
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
                if prog.isclosed:
                    break
                prog.update(k)
                QApplication.processEvents()
                prediction = None
                try:
                    if im.split('.')[-1] == 'gif':
                        prediction = self.findFaceAndClassifyForAuto_gif(im)
                    else:
                        prediction = self.findFaceAndClassifyForAuto(im)                    
                
                    isex = False                    
                    for cand in range(int(prediction.shape[0]/4)):
                        prob = prediction[(cand*4):((cand+1)*4)]
                        cors = np.corrcoef(prob)
                        np.fill_diagonal(cors,0)
                        e = cors.sum(axis=1)
                        wei = np.exp(e)/np.exp(e).sum()
                        y_prob = (prob*wei.reshape([-1,1])).sum(axis=0)
                        if np.median(y_prob) > 0.08:
                            continue
                        if np.max(y_prob) < 0.65:
                            isex = True
                            copy2(im, alayalay)
                        else:
                            copy2(im, os.path.join(dirName, subjects_hangle[np.argmax(y_prob)]))
                        
                    if isex:
                        failed.append(im)
                except:
                    print('Can\'t open ',im)
                    cantOpen.append(im)
                    pass
                
            np.savetxt(os.path.join(dirName,'미묘미묘해.txt'), failed, fmt='%s')
            np.savetxt(os.path.join(dirName,'열지 못한 파일.txt'), cantOpen, fmt='%s')
            del prog
    
    def hogTrue(self):
        self.useHOG = True
    
    def dnnTrue(self):
        self.useHOG = False
    
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
        
        self.setHOG = QAction("&HOG", self, triggered=self.hogTrue, checkable=True)
        self.setDNN = QAction("&DNN", self, triggered=self.dnnTrue, checkable=True)

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
        
        self.methodMenu = QMenu("&Method", self)
        self.ag = QActionGroup(self)        
        self.methodMenu.addAction(self.ag.addAction(self.setHOG))        
        self.methodMenu.addAction(self.ag.addAction(self.setDNN))


        self.menuBar().addMenu(self.fileMenu)
        self.menuBar().addMenu(self.viewMenu)
        self.menuBar().addMenu(self.dirMenu)
        self.menuBar().addMenu(self.methodMenu)

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