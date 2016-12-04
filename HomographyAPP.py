from PySide.QtGui import *
from PySide.QtCore import *
from HomographyGUI import *
import numpy as np
import scipy.misc as msc
from Homography import *
import os
import math
import sys


class HomographyAPP(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(HomographyAPP, self).__init__(parent)
        self.setupUi(self)

        #connect all buttons

        self.srcIMG = None
        self.targetIMG = None
        self.srcSubregionIMG = None

        self.effect = None
        self.pretgtImgFP = None

        self.status = None
        self.srcType = None
        self.tgtType = None

        self.srcUpLeftX = -1
        self.srcUpLeftY = -1
        
        self.srcLowerRghtX = -1
        self.srcLowerRghtY = -1
        
        
        
        
        self.tgtUpLeftX = -1
        self.tgtUpLeftY = -1
        
        self.tgtUpRghtX = -1
        self.tgtUpRghtY = -1
        
        self.tgtLowerLeftX = -1
        self.tgtLowerLeftY = -1
        
        self.tgtLowerRghtX = -1
        self.tgtLowerRghtY = -1
        
        



        self.loadSrcPB.clicked.connect(self.loadsrcPrs)
        self.loadTgtPB.clicked.connect(self.loadtgtPrs)
        self.acqPtsPB.clicked.connect(self.acqPtsPrs)
        self.acqPtsPB_extra.clicked.connect(self.acqPts_extraPrs)
        self.resetPB.clicked.connect(self.resetPrs)
        self.savePB.clicked.connect(self.savePrs)
        self.transfPB.clicked.connect(self.transfPrs)








    def loadsrcPrs(self):
        filepath = self.loadData()
        self.srcGraphicsView.setEnabled(True)
        self.srcIMG = msc.imread(filepath)
        if len(self.srcIMG.shape) == 2 :
            self.srcType = "Grey"
        else:
            self.srcType = "Color"

        scene = QtGui.QGraphicsScene(self)
        tmp = QtGui.QPixmap(filepath)
        pix = QGraphicsPixmapItem(tmp)

        scene.addItem(pix)
        self.srcGraphicsView.setScene(scene)
        self.srcGraphicsView.fitInView(scene.itemsBoundingRect(), Qt.KeepAspectRatio)
        pix.mousePressEvent = self.srcPixelSelect


        self.acqPtsPB_extra.setChecked(False)
        self.srcSubregionIMG = None
        self.acqPtsPB.setEnabled(True)
        self.acqPtsPB_extra.setEnabled(True)
        self.srcPt1LE.setEnabled(True)
        self.srcPt2LE.setEnabled(True)
        self.tgtPt1LE.setEnabled(True)
        self.tgtPt2LE.setEnabled(True)
        self.tgtPt3LE.setEnabled(True)
        self.tgtPt4LE.setEnabled(True)
        self.srcPt1LE.setText("")
        self.srcPt2LE.setText("")
        self.srcUpLeftX = -1
        self.srcUpLeftY = -1
        self.srcLowerRghtX = -1
        self.srcLowerRghtY = -1


        if self.status == "Transformed State":
            self.tgtGraphicsView.setEnabled(True)
            self.targetIMG = msc.imread(self.pretgtImgFP)

            scene = QtGui.QGraphicsScene(self)
            pix = QGraphicsPixmapItem(QtGui.QPixmap(self.pretgtImgFP))
            scene.addItem(pix)

            self.tgtGraphicsView.setScene(scene)
            self.tgtGraphicsView.fitInView(scene.itemsBoundingRect(), Qt.KeepAspectRatio)
            pix.mousePressEvent = self.tgtPixelSelect
            self.tgtPt1LE.setText("")
            self.tgtPt2LE.setText("")
            self.tgtPt3LE.setText("")
            self.tgtPt4LE.setText("")
            self.tgtUpLeftX = -1
            self.tgtUpLeftY = -1

            self.tgtUpRghtX = -1
            self.tgtUpRghtY = -1

            self.tgtLowerLeftX = -1
            self.tgtLowerLeftY = -1

            self.tgtLowerRghtX = -1
            self.tgtLowerRghtY = -1
        self.status = "Loaded State"


    def loadtgtPrs(self):
        filepath = self.loadData()

        self.tgtGraphicsView.setEnabled(True)
        self.targetIMG = msc.imread(filepath)
        self.pretgtImgFP = filepath
        if len(self.targetIMG.shape) == 2 :
            self.tgtType = "Grey"
        else:
            self.tgtType = "Color"
        scene = QtGui.QGraphicsScene(self)
        pix = QGraphicsPixmapItem(QtGui.QPixmap(filepath))
        scene.addItem(pix)

        self.tgtGraphicsView.setScene(scene)
        self.tgtGraphicsView.fitInView(scene.itemsBoundingRect(), Qt.KeepAspectRatio)
        pix.mousePressEvent = self.tgtPixelSelect

        self.acqPtsPB.setChecked(False)
        self.acqPtsPB.setEnabled(True)
        self.acqPtsPB_extra.setEnabled(True)
        self.srcPt1LE.setEnabled(True)
        self.srcPt2LE.setEnabled(True)
        self.tgtPt1LE.setEnabled(True)
        self.tgtPt2LE.setEnabled(True)
        self.tgtPt3LE.setEnabled(True)
        self.tgtPt4LE.setEnabled(True)
        self.tgtPt1LE.setText("")
        self.tgtPt2LE.setText("")
        self.tgtPt3LE.setText("")
        self.tgtPt4LE.setText("")
        self.tgtUpLeftX = -1
        self.tgtUpLeftY = -1

        self.tgtUpRghtX = -1
        self.tgtUpRghtY = -1

        self.tgtLowerLeftX = -1
        self.tgtLowerLeftY = -1

        self.tgtLowerRghtX = -1
        self.tgtLowerRghtY = -1
    def srcPixelSelect(self, event):
        if self.acqPtsPB_extra.isChecked():
             if self.srcUpLeftX == -1:
                 self.srcUpLeftX = round(event.pos().x(),1)
                 self.srcUpLeftY = round(event.pos().y(),1)
                 self.srcPt1LE.setText(str(self.srcUpLeftX)+", "+str(self.srcUpLeftY))
             elif self.srcLowerRghtX == -1:
                 self.srcLowerRghtX = round(event.pos().x(),1)
                 self.srcLowerRghtY = round(event.pos().y(),1)
                 self.srcPt2LE.setText(str(self.srcLowerRghtX)+", "+str(self.srcLowerRghtY))
    def tgtPixelSelect(self, event):
        if self.acqPtsPB.isChecked():
            if self.tgtUpLeftX == -1:
                     self.tgtUpLeftX = round(event.pos().x(),1)
                     self.tgtUpLeftY = round(event.pos().y(),1)
                     self.tgtPt1LE.setText(str(self.tgtUpLeftX)+", "+str(self.tgtUpLeftY))
            elif self.tgtUpRghtX == -1:
                 self.tgtUpRghtX = round(event.pos().x(),1)
                 self.tgtUpRghtY = round(event.pos().y(),1)
                 self.tgtPt2LE.setText(str(self.tgtUpRghtX)+", "+str(self.tgtUpRghtY))
            elif self.tgtLowerLeftX == -1:
                 self.tgtLowerLeftX = round(event.pos().x(),1)
                 self.tgtLowerLeftY = round(event.pos().y(),1)
                 self.tgtPt3LE.setText(str(self.tgtLowerLeftX)+", "+str(self.tgtLowerLeftY))
            elif self.tgtLowerRghtX == -1:
                 self.tgtLowerRghtX = round(event.pos().x(),1)
                 self.tgtLowerRghtY = round(event.pos().y(),1)
                 self.tgtPt4LE.setText(str(self.tgtLowerRghtX)+", "+str(self.tgtLowerRghtY))


    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Backspace and self.acqPtsPB.isChecked():
            if self.tgtLowerRghtX != -1:
                self.tgtLowerRghtX = -1
                self.tgtLowerRghtY = -1
                self.tgtPt4LE.setText("")
            elif self.tgtLowerLeftX != -1:
                self.tgtLowerLeftX = -1
                self.tgtLowerLeftY = -1
                self.tgtPt3LE.setText("")
            elif self.tgtUpRghtX != -1:
                self.tgtUpRghtX = -1
                self.tgtUpRghtY = -1
                self.tgtPt2LE.setText("")
            elif self.tgtUpLeftX != -1:
                self.tgtUpLeftX = -1
                self.tgtUpLeftY = -1
                self.tgtPt1LE.setText("")
            


    def loadData(self):

        #filePath, _ = QFileDialog.getOpenFileName(self, caption='Open XML file ...', filter="XML files (*.xml)")
        filePath, _ = QFileDialog.getOpenFileName(self, caption='Open PNG file ...', filter="PNG files (*.png)")


        # if not filePath:
        #     return

        #self.loadDataFromFile(filePath)
        return filePath
    def saveData(self):

        #filePath, _ = QFileDialog.getOpenFileName(self, caption='Open XML file ...', filter="XML files (*.xml)")
        filePath, _ = QFileDialog.getSaveFileName(self, caption='save as...', filter="PNG files (*.png)")


        # if not filePath:
        #     return

        #self.loadDataFromFile(filePath)
        return filePath

    def acqPtsPrs(self):
        if self.loadTgtPB.isEnabled() and self.targetIMG is not None:
            #self.acqPtsPB.toggle()
            self.loadTgtPB.setDisabled(True)
        elif (not self.loadTgtPB.isEnabled()) and self.tgtLowerRghtX != -1:
            self.acqPtsPB.setChecked(False)
            self.loadTgtPB.setEnabled(True)


    def acqPts_extraPrs(self):
        if self.loadSrcPB.isEnabled() and self.srcIMG is not None:
            #self.acqPtsPB.toggle()
            self.loadSrcPB.setDisabled(True)
        elif (not self.loadSrcPB.isEnabled()) and self.srcLowerRghtX != -1:

            self.acqPtsPB_extra.setChecked(False)
            self.loadSrcPB.setEnabled(True)
            generalTrans = GeneralTransformation(self.srcIMG,
                                                 [[self.srcUpLeftX,self.srcUpLeftY],
                                                  [self.srcLowerRghtX,self.srcLowerRghtY]])
            self.srcSubregionIMG = generalTrans.getSubregionIMG()
            #self.srcSubregionIMG = self.srcIMG[
            #                       math.ceil(self.srcUpLeftY):math.ceil(self.srcLowerRghtY),
            #                       math.ceil(self.srcUpLeftX):math.ceil(self.srcLowerRghtX)]



    def getEffect(self):
        if self.comboBox.currentText() == "Nothing":
            self.effect = None
        elif self.comboBox.currentText() == "Rotate 90":
            self.effect = Effect.rotate90
        elif self.comboBox.currentText() == "Rotate 180":
            self.effect = Effect.rotate180
        elif self.comboBox.currentText() == "Rotate 270":
            self.effect = Effect.rotate270
        elif self.comboBox.currentText() == "Flip Horizontally":
            self.effect = Effect.flipHorizontally
        elif self.comboBox.currentText() == "Flip Vertically":
            self.effect = Effect.flipVertically
        elif self.comboBox.currentText() == "Transpose":
            self.effect = Effect.transpose
    def srcGreytgtGrey(self):
        if self.srcSubregionIMG is None:
            transf = Transformation(self.srcIMG)
        else:

            transf = Transformation(self.srcSubregionIMG)

        transf.setupTransformation([[self.tgtUpLeftX, self.tgtUpLeftY],
                                     [self.tgtUpRghtX, self.tgtUpRghtY],
                                     [self.tgtLowerLeftX, self.tgtLowerLeftY],
                                     [self.tgtLowerRghtX, self.tgtLowerRghtY]], self.effect)
        transf.transformImageOnto(self.targetIMG)
    def srcGreytgtColor(self):
        if self.srcSubregionIMG is None:
            srccopy = np.dstack((self.srcIMG, self.srcIMG, self.srcIMG))
        else:
            srccopy = np.dstack((self.srcSubregionIMG, self.srcSubregionIMG, self.srcSubregionIMG))
        transf = ColorTransformation(srccopy)
        transf.setupTransformation([[self.tgtUpLeftX, self.tgtUpLeftY],
                                     [self.tgtUpRghtX, self.tgtUpRghtY],
                                     [self.tgtLowerLeftX, self.tgtLowerLeftY],
                                     [self.tgtLowerRghtX, self.tgtLowerRghtY]], self.effect)
        transf.transformImageOnto(self.targetIMG)

    def srcColortgtGrey(self):
        self.targetIMG = np.dstack((self.targetIMG, self.targetIMG, self.targetIMG))
        if self.srcSubregionIMG is None:
            transf = ColorTransformation(self.srcIMG)
        else:

            transf = ColorTransformation(self.srcSubregionIMG)

        transf.setupTransformation([[self.tgtUpLeftX, self.tgtUpLeftY],
                                     [self.tgtUpRghtX, self.tgtUpRghtY],
                                     [self.tgtLowerLeftX, self.tgtLowerLeftY],
                                     [self.tgtLowerRghtX, self.tgtLowerRghtY]], self.effect)
        transf.transformImageOnto(self.targetIMG)
    def srcColortgtColor(self):

        if self.srcSubregionIMG is None:
            transf = ColorTransformation(self.srcIMG)
        else:

            transf = ColorTransformation(self.srcSubregionIMG)

        transf.setupTransformation([[self.tgtUpLeftX, self.tgtUpLeftY],
                                     [self.tgtUpRghtX, self.tgtUpRghtY],
                                     [self.tgtLowerLeftX, self.tgtLowerLeftY],
                                     [self.tgtLowerRghtX, self.tgtLowerRghtY]], self.effect)
        transf.transformImageOnto(self.targetIMG)
    def transfPrs(self):
        if (not self.acqPtsPB.isChecked()) and (not self.acqPtsPB.isChecked()):
            self.getEffect()
            self.status = "Transformed State"
            self.acqPtsPB.setEnabled(False)
            self.acqPtsPB_extra.setEnabled(False)
            self.srcPt1LE.setEnabled(False)
            self.srcPt2LE.setEnabled(False)
            self.tgtPt1LE.setEnabled(False)
            self.tgtPt2LE.setEnabled(False)
            self.tgtPt3LE.setEnabled(False)
            self.tgtPt4LE.setEnabled(False)


            if self.srcType == "Grey" and self.tgtType == "Grey":
                self.srcGreytgtGrey()
            elif self.srcType == "Grey" and self.tgtType == "Color":
                self.srcGreytgtColor()
            elif self.srcType == "Color" and self.tgtType == "Grey":
                self.srcColortgtGrey()
            elif self.srcType == "Color" and self.tgtType == "Color":
                self.srcColortgtColor()
            else:
                raise ValueError








            msc.imsave("test.png", self.targetIMG)
            tmpFilePath = os.getcwd()+'/'+"test.png"

            scene = QtGui.QGraphicsScene(self)
            pix = QGraphicsPixmapItem(QtGui.QPixmap(tmpFilePath))
            scene.addItem(pix)
            self.tgtGraphicsView.setScene(scene)
            self.tgtGraphicsView.fitInView(scene.itemsBoundingRect(), Qt.KeepAspectRatio)


    def resetPrs(self):
        if self.status == "Transformed State":
            self.tgtGraphicsView.setEnabled(True)
            self.targetIMG = msc.imread(self.pretgtImgFP)

            scene = QtGui.QGraphicsScene(self)
            pix = QGraphicsPixmapItem(QtGui.QPixmap(self.pretgtImgFP))
            scene.addItem(pix)

            self.tgtGraphicsView.setScene(scene)
            self.tgtGraphicsView.fitInView(scene.itemsBoundingRect(), Qt.KeepAspectRatio)
            pix.mousePressEvent = self.tgtPixelSelect
            self.acqPtsPB.setChecked(False)
            self.acqPtsPB.setEnabled(True)
            self.acqPtsPB_extra.setEnabled(True)
            self.srcPt1LE.setEnabled(True)
            self.srcPt2LE.setEnabled(True)
            self.tgtPt1LE.setEnabled(True)
            self.tgtPt2LE.setEnabled(True)
            self.tgtPt3LE.setEnabled(True)
            self.tgtPt4LE.setEnabled(True)
        self.status = "Ready State"

    def savePrs(self):

        if self.targetIMG is not None:

            filename = self.saveData()
            if filename.endswith(".png"):
                msc.imsave(filename, self.targetIMG)
            else:
                msc.imsave(filename+".png", self.targetIMG)



class GeneralTransformation():
    def __init__(self, sourceImage, sourcePoints=None):
        self.srcIMG = sourceImage
        self.srcPts = sourcePoints
    def getSubregionIMG(self):
        if self.srcPts is None:
            return self.srcIMG
        else:
            return self.srcIMG[
            math.ceil(self.srcPts[0][1]):math.ceil(self.srcPts[1][1]),
            math.ceil(self.srcPts[0][0]):math.ceil(self.srcPts[1][0])]
            #self.srcIMG[
            #                       math.ceil(self.srcUpLeftY):math.ceil(self.srcLowerRghtY),
            #                       math.ceil(self.srcUpLeftX):math.ceil(self.srcLowerRghtX)]





if __name__ == '__main__':
    currentApp = QApplication(sys.argv)
    currentForm = HomographyAPP()

    currentForm.show()
    currentApp.exec_()
'''
1. save box
2. invalid white area
3. point integer float
4. debug
'''

