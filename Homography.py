import numpy as np
import scipy.misc as msc
from scipy.interpolate import RectBivariateSpline
from random import uniform
from enum import Enum
import math
class Effect(Enum):

        rotate90 = 0
        rotate180 = 1
        rotate270 = 2
        flipHorizontally = 3
        flipVertically = 4
        transpose = 5

class Homography:
    def __init__(self, **kwargs):
        if 'homographyMatrix' in kwargs:
            if len(kwargs['homographyMatrix']) == 3 \
                    and len(kwargs['homographyMatrix'][0]) == 3 \
                    and type(kwargs['homographyMatrix'][0][0]) is np.float64:

                self.forwardMatrix = kwargs['homographyMatrix']
                self.inverseMatrix = np.linalg.inv(kwargs['homographyMatrix'])
            #exception about not 3x3
            else:
                raise ValueError
        elif 'sourcePoints' in kwargs and 'targetPoints' in kwargs:
            if 'effect' in kwargs and kwargs['effect'] is not None:
                if len(kwargs['sourcePoints']) == 4 and len(kwargs['sourcePoints'][0]) == 2 and \
                                len(kwargs['targetPoints']) == 4 and len(kwargs['targetPoints'][0]) == 2:
                    if isinstance(kwargs['effect'],Effect):
                        self.forwardMatrix = self.computeHomography(kwargs['sourcePoints'],
                                                                    kwargs['targetPoints'], kwargs['effect'])
                        self.inverseMatrix = np.linalg.inv(self.forwardMatrix)
                    else:
                        raise TypeError
                else:
                    raise ValueError
            else:
                if len(kwargs['sourcePoints']) == 4 and len(kwargs['sourcePoints'][0]) == 2 and \
                                len(kwargs['targetPoints']) == 4 and len(kwargs['targetPoints'][0]) == 2 :
                    self.forwardMatrix = self.computeHomography(kwargs['sourcePoints'],
                                                                kwargs['targetPoints'])
                    self.inverseMatrix = np.linalg.inv(self.forwardMatrix)
                else:
                    raise ValueError
        else:
            raise ValueError
    def computeHomography(self, sourcePoints, targetPoints, effect=None):           #effect part unsolved
        tmpA = []
        tmpB = []
        for [x,y],[tx,ty] in zip(sourcePoints, targetPoints):
            tmpA.append([x, y, 1, 0, 0, 0, -tx*x, -tx*y])
            tmpA.append([0, 0, 0, x, y, 1, -ty*x, -ty*y])
            tmpB.append([tx])
            tmpB.append([ty])
        A = np.array([tmpA[0],tmpA[1],tmpA[2],tmpA[3],
                        tmpA[4],tmpA[5],tmpA[6],tmpA[7]],np.float64)
        B = np.array([tmpB[0],tmpB[1],tmpB[2],tmpB[3],
                        tmpB[4],tmpB[5],tmpB[6],tmpB[7]],np.float64)
        h = np.linalg.solve(A,B)
        H = np.array([
                         [h[0][0], h[1][0], h[2][0]],
                         [h[3][0], h[4][0], h[5][0]],
                         [h[6][0], h[7][0], 1]
                        ],np.float64)
        return H



class Transformation:
    def __init__(self, sourseImage, homography = None):
        self.sourseImage = sourseImage
        self.homography = homography
        self.bondingBox = None
        self.invHomo = None
        self.columns = 0
        self.rows = 0
        self.effect = None

    # def computeHomography(self, sourcePoints, targetPoints, effect=None):           #effect part unsolved
    #     tmpA = []
    #     tmpB = []
    #     for [x,y],[tx,ty] in zip(sourcePoints, targetPoints):
    #         tmpA.append([x, y, 1, 0, 0, 0, -tx*x, -tx*y])
    #         tmpA.append([0, 0, 0, x, y, 1, -ty*x, -ty*y])
    #         tmpB.append([tx])
    #         tmpB.append([ty])
    #     A = np.array([tmpA[0],tmpA[1],tmpA[2],tmpA[3],
    #                     tmpA[4],tmpA[5],tmpA[6],tmpA[7]],np.float64)
    #     B = np.array([tmpB[0],tmpB[1],tmpB[2],tmpB[3],
    #                     tmpB[4],tmpB[5],tmpB[6],tmpB[7]],np.float64)
    #     h = np.linalg.solve(A,B)
    #     H = np.array([
    #                      [h[0][0], h[1][0], h[2][0]],
    #                      [h[3][0], h[4][0], h[5][0]],
    #                      [h[6][0], h[7][0], 1]
    #                     ],np.float64)
    #     return H

    def getRange(self, targetPoints):
        xmin = None
        xmax = None
        ymin = None
        ymax = None
        for [x,y] in targetPoints:
            if xmin is None:
                xmin = x
                xmax = x
                ymin = y
                ymax = y
            else:
                if x < xmin:
                    xmin = x
                if x > xmax:
                    xmax = x
                if y < ymin:
                    ymin = y
                if y > ymax:
                    ymax = y
        #print((xmin, ymin, xmax, ymax))
        return (int(xmin), int(ymin), int(xmax), int(ymax))

    def setupTransformation(self, targetPoints, effect=None):
        self.effect = effect

        if self.effect is None:
            if self.homography is None:
                columns = len(self.sourseImage[0])
                rows = len(self.sourseImage)
                self.columns = columns
                self.rows = rows

                # self.homography = self.computeHomography([[0,0],[columns-1,0],[0,rows-1],[columns-1,rows-1]],
                #                                          targetPoints, effect)
                # self.invHomo = np.linalg.inv(self.homography)
                sourcePoints = np.array([[0,0],[columns-1,0],[0,rows-1],[columns-1,rows-1]])
                self.homography = Homography(sourcePoints=sourcePoints, targetPoints=targetPoints, effect=effect)
                self.invHomo = self.homography.inverseMatrix
                self.bondingBox = self.getRange(targetPoints)


            else:
                columns = len(self.sourseImage[0])
                rows = len(self.sourseImage)
                self.columns = columns
                self.rows = rows
                self.invHomo = self.homography.inverseMatrix
                self.bondingBox = self.getRange(targetPoints)
        else:

            columns = len(self.sourseImage[0])
            rows = len(self.sourseImage)
            self.columns = columns
            self.rows = rows
            sourcePoints = np.array([[0,0],[columns-1,0],[0,rows-1],[columns-1,rows-1]])
            if self.effect == Effect.rotate90:
                tpEffect = np.array([targetPoints[1],targetPoints[3],targetPoints[0], targetPoints[2]],np.float64)
            elif self.effect == Effect.rotate180:
                tpEffect = np.array([targetPoints[3],targetPoints[2],targetPoints[1], targetPoints[0]],np.float64)

            elif self.effect == Effect.rotate270:
                tpEffect = np.array([targetPoints[2],targetPoints[0],targetPoints[3], targetPoints[1]],np.float64)
            elif self.effect == Effect.flipHorizontally:
                tpEffect = np.array([targetPoints[2],targetPoints[3],targetPoints[0], targetPoints[1]],np.float64)
            elif self.effect == Effect.flipVertically:
                tpEffect = np.array([targetPoints[1],targetPoints[0],targetPoints[3], targetPoints[2]],np.float64)

            elif self.effect == Effect.transpose:
                tpEffect = np.array([targetPoints[0],targetPoints[2],targetPoints[1], targetPoints[3]],np.float64)
            else:
                raise ValueError
            sourcePoints = np.array([[0,0],[columns-1,0],[0,rows-1],[columns-1,rows-1]])
            self.homography = Homography(sourcePoints=sourcePoints, targetPoints=tpEffect, effect=effect)
            self.invHomo = self.homography.inverseMatrix
            self.bondingBox = self.getRange(tpEffect)
    def transformImageOnto(self, containerImage):
        bbArrayX = np.arange(self.bondingBox[0], self.bondingBox[2]+1)
        bbArrayY = np.arange(self.bondingBox[1], self.bondingBox[3]+1)
        row1 = np.tile(bbArrayX, self.bondingBox[3]-self.bondingBox[1]+1)
        row2 = np.repeat(bbArrayY, self.bondingBox[2] - self.bondingBox[0]+1)
        row3 = np.ones((self.bondingBox[3]-self.bondingBox[1]+1)*(self.bondingBox[2] - self.bondingBox[0]+1))

        bbMatrix = np.array([row1, row2, row3], np.float64)

        tmpMatrix = np.dot(self.invHomo, bbMatrix)
        srcPointMatrix = np.around(tmpMatrix/tmpMatrix[2], decimals=3)

        srcbbMatrix = np.array([bbMatrix[0], bbMatrix[1], srcPointMatrix[0], srcPointMatrix[1]], np.float64)
        transportMatrix = np.transpose(srcbbMatrix)

        transportMatrix = transportMatrix[transportMatrix[:,2] >= 0]
        transportMatrix = transportMatrix[transportMatrix[:,2] <= self.columns-1]
        transportMatrix = transportMatrix[transportMatrix[:,3] >= 0]
        srctgtValidMatrix = transportMatrix[transportMatrix[:,3] <= self.rows-1]







        x = np.arange(0, len(self.sourseImage[0]))
        y = np.arange(0, len(self.sourseImage))
        approximate = RectBivariateSpline(y, x, self.sourseImage,kx= 1,ky=1)


        #srctgtValidMatrix = np.concatenate((srctgtValidMatrix,aproxValues), axis=0)

        aprox = approximate.ev(srctgtValidMatrix[:,3], srctgtValidMatrix[:,2])
        srctgtValidMatrix = srctgtValidMatrix.astype(int)




        containerImage[srctgtValidMatrix[:,1], srctgtValidMatrix[:,0]] = np.round(aprox)
        return containerImage

    '''
    def transfPerform(a, invH, approx, column, row):
        tmpArray1 = np.array([[a[0]],[a[1]],[1]],np.float64)
        tmpArray2 = np.dot(invH,tmpArray1)
        tmpresult1 = tmpArray2/tmpArray2[2][0]
        ioriginal = np.round(tmpresult1[0][0],3)
        joriginal = np.round(tmpresult1[1][0],3)
        if ioriginal >= 0 and ioriginal <= column-1 \
                            and joriginal >= 0 and joriginal <= row-1 :
            return np.round(approx(joriginal,ioriginal))
        else:
            return 255



    def transformImageOnto(self, containerImage):

        if type(containerImage) is not np.ndarray:
            raise TypeError

        x = np.arange(0, len(self.sourseImage[0]))
        y = np.arange(0, len(self.sourseImage))
        approximate = RectBivariateSpline(y, x, self.sourseImage,kx= 1,ky=1)
        p = np.mgrid[self.bondingBox[1]:self.bondingBox[3],self.bondingBox[0]:self.bondingBox[2]].swapaxes(0,2).swapaxes(0,1)




        if self.bondingBox is None or self.invHomo is None:
            raise ValueError("bodingBox or invHomo not available")
        else:
            return np.apply_along_axis(self.transfPerform,2,p, self.invHomo, approximate, self.columns, self.rows)







    def transformImageOnto(self, containerImage):

        if type(containerImage) is not np.ndarray:
            raise TypeError

        x = np.arange(0, len(self.sourseImage[0]))
        y = np.arange(0, len(self.sourseImage))
        # for i in np.arange(0, len(self.sourseImage[0])):
        #     x.append(i)
        # for j in np.arange(0, len(self.sourseImage)):
        #     y.append(j)
        approximate = RectBivariateSpline(y, x, self.sourseImage,kx= 1,ky=1)




        if self.bondingBox is None or self.invHomo is None:
            raise ValueError("bodingBox or invHomo not available")
        else:

            for i in np.arange(self.bondingBox[0], self.bondingBox[2]+1):
                for j in np.arange(self.bondingBox[1], self.bondingBox[3]+1):

                    tmpres = np.dot(self.invHomo, [[i],[j],[1]])
                    result = tmpres/tmpres[2][0]

                    ioriginal = np.round(result[0][0],3)
                    joriginal = np.round(result[1][0],3)



                    if ioriginal >= 0 and ioriginal <= self.columns-1 \
                            and joriginal >= 0 and joriginal <= self.rows-1 :
                         containerImage[j][i] = np.round(approximate(joriginal,ioriginal))

                    else:
                            continue
            return containerImage
    '''







class ColorTransformation(Transformation):
    def __init__(self, sourceImage, homography = None):
        if type(sourceImage) is not np.ndarray:
            raise TypeError
        if type(sourceImage[0][0]) is not np.ndarray:
            raise ValueError
        if homography is not None and type(homography) is not Homography:
            raise TypeError
        Transformation.__init__(self,sourceImage,homography)






    def transformImageOnto(self, containerImage):

        bbArrayX = np.arange(self.bondingBox[0], self.bondingBox[2]+1)
        bbArrayY = np.arange(self.bondingBox[1], self.bondingBox[3]+1)
        row1 = np.tile(bbArrayX, self.bondingBox[3]-self.bondingBox[1]+1)
        row2 = np.repeat(bbArrayY, self.bondingBox[2] - self.bondingBox[0]+1)
        row3 = np.ones((self.bondingBox[3]-self.bondingBox[1]+1)*(self.bondingBox[2] - self.bondingBox[0]+1))

        bbMatrix = np.array([row1, row2, row3], np.float64)

        tmpMatrix = np.dot(self.invHomo, bbMatrix)
        srcPointMatrix = np.around(tmpMatrix/tmpMatrix[2], decimals=3)

        srcbbMatrix = np.array([bbMatrix[0], bbMatrix[1], srcPointMatrix[0], srcPointMatrix[1]], np.float64)
        transportMatrix = np.transpose(srcbbMatrix)

        transportMatrix = transportMatrix[transportMatrix[:,2] >= 0]
        transportMatrix = transportMatrix[transportMatrix[:,2] <= self.columns-1]
        transportMatrix = transportMatrix[transportMatrix[:,3] >= 0]
        srctgtValidMatrix = transportMatrix[transportMatrix[:,3] <= self.rows-1]







        x = np.arange(0, len(self.sourseImage[0]))
        y = np.arange(0, len(self.sourseImage))

        approximate1 = RectBivariateSpline(y, x, self.sourseImage[:,:,0],kx= 1,ky=1)
        approximate2= RectBivariateSpline(y, x, self.sourseImage[:,:,1],kx= 1,ky=1)
        approximate3 = RectBivariateSpline(y, x, self.sourseImage[:,:,2],kx= 1,ky=1)


        #srctgtValidMatrix = np.concatenate((srctgtValidMatrix,aproxValues), axis=0)

        aprox1 = approximate1.ev(srctgtValidMatrix[:,3], srctgtValidMatrix[:,2])
        aprox2 = approximate2.ev(srctgtValidMatrix[:,3], srctgtValidMatrix[:,2])
        aprox3 = approximate3.ev(srctgtValidMatrix[:,3], srctgtValidMatrix[:,2])

        aprox1.shape = (len(aprox1),)
        aprox2.shape = (len(aprox2),)
        aprox3.shape = (len(aprox3),)

        valueMatrix = np.array([aprox1, aprox2, aprox3])
        valueMatrix = np.transpose(valueMatrix)

        srctgtValidMatrix = srctgtValidMatrix.astype(int)




        containerImage[srctgtValidMatrix[:,1], srctgtValidMatrix[:,0]] = np.round(valueMatrix)
        return containerImage
        '''
    def transformImageOnto(self, containerImage):
        x = np.arange(0, len(self.sourseImage[0]))
        y = np.arange(0, len(self.sourseImage))

        approximate1 = RectBivariateSpline(y, x, self.sourseImage[:,:,0],kx= 1,ky=1)
        approximate2= RectBivariateSpline(y, x, self.sourseImage[:,:,1],kx= 1,ky=1)
        approximate3 = RectBivariateSpline(y, x, self.sourseImage[:,:,2],kx= 1,ky=1)

        if self.bondingBox is None or self.invHomo is None:
            raise ValueError("bodingBox or invHomo not available")
        else:
            flag = 0
            for i in range(self.bondingBox[0], self.bondingBox[2]+1):
                for j in range(self.bondingBox[1], self.bondingBox[3]+1):

                    tmpres = np.dot(self.invHomo, [[i],[j],[1]])
                    result = tmpres/tmpres[2][0]

                    ioriginal = np.round(result[0][0],3)
                    joriginal = np.round(result[1][0],3)


                    if ioriginal >= 0 and ioriginal <= self.columns-1 \
                            and joriginal >= 0 and joriginal <= self.rows-1 :
                        #containerImage[j][i] = approximate(joriginal,ioriginal)
                        tmpList = []
                        if flag == 0:
                            print(approximate1(joriginal,ioriginal))
                            print(approximate1(joriginal,ioriginal)[0][0])
                            flag = 1
                            raise ValueError
                        tmpList.append(round(approximate1(joriginal,ioriginal)[0][0]))
                        tmpList.append(round(approximate2(joriginal,ioriginal)[0][0]))
                        tmpList.append(round(approximate3(joriginal,ioriginal)[0][0]))

                        containerImage[j][i] = np.array(tmpList)


                    else:
                        continue
            return containerImage
        '''



class AdvancedTransformation():
    def __init__(self, sourceImage, v, h1, h2):
        if type(sourceImage) is not np.ndarray:
            raise TypeError
        if type(sourceImage[0][0]) is not np.ndarray:
            raise ValueError
        if len(sourceImage[0])%2 == 1:
            raise ValueError
        self.sourceImage = sourceImage
        self.v = v
        self.h1 = h1
        self.h2 = h2
        self.c = len(sourceImage[0])
        self.r = len(sourceImage)
        self.m1 = self.c/2 - 1
        self.m2 = self.c/2
        self.sourceIMG1 = np.hsplit(self.sourceImage,2)[0]
        self.sourceIMG2 = np.hsplit(self.sourceImage,2)[1]


    def applyEffectV(self):
        containerIMG = np.ones((self.r+self.v, self.c, 3))*255






        targetP1 = np.array([[0,0],[self.m1-self.h2,self.v],[self.h1,self.r-1],[self.m1,self.r-1+self.v]],np.float64)
        targetP2 = np.array([[self.m2+self.h2,self.v],[self.c-1,0],[self.m2,self.r-1+self.v],[self.c-1-self.h1,self.r-1]],np.float64)


        c1 = ColorTransformation(self.sourceIMG1)
        c1.setupTransformation(targetP1)
        c1.transformImageOnto(containerIMG)





        c2 = ColorTransformation(self.sourceIMG2)
        c2.setupTransformation(targetP2)
        c2.transformImageOnto(containerIMG)





        return containerIMG

    def applyEffectA(self):
        containerIMG = np.ones((self.r+self.v, self.c,3))*255
        targetP1 = np.array([[self.h1,self.v],[self.m1,0],[0,self.r-1+self.v],[self.m1-self.h2,self.r-1]],np.float64)
        targetP2 = np.array([[self.m2,0],[self.c-1-self.h1,self.v],[self.m2+self.h2,self.r-1],[self.c-1,self.r-1+self.v]],np.float64)


        c1 = ColorTransformation(self.sourceIMG1)
        c1.setupTransformation(targetP1)
        c1.transformImageOnto(containerIMG)





        c2 = ColorTransformation(self.sourceIMG2)
        c2.setupTransformation(targetP2)
        c2.transformImageOnto(containerIMG)





        return containerIMG
















if __name__ == '__main__':
    '''
    grid = np.indices((2,3))
    #print(grid)
    n = np.arange(5,55)
    n.shape = (5,5,2)
    #print(n)
    p = np.mgrid[0:51,0:51].swapaxes(0,2).swapaxes(0,1)
    #print(p)
    '''
    tp3 = np.array([[20,30],[200,20],[10,200],[180,260]],np.float64)
    targetIMG2 = msc.imread('White.png')
    colorIMG = msc.imread('phase1.jpg')
    c1 = ColorTransformation(colorIMG)
    c1.setupTransformation(tp3)
    result = c1.transformImageOnto(targetIMG2)
    msc.imsave('testcolorPerform.png', result)

    '''
    colorVIMG = msc.imread('Ring.png')


    Ad1 = AdvancedTransformation(colorVIMG, 50, 50, 50)
    resultV = Ad1.applyEffectV()
    msc.imsave('testcolorV222.png', resultV)
    resultA = Ad1.applyEffectA()
    msc.imsave('testcolorAAAA2222.png', resultA)




    nx,ny = (3,2)
    x = np.linspace(0,1,nx)
    y = np.linspace(0,1,ny)
    print(x)
    print(y)
    xv,yv = np.meshgrid(x,y)
    print(xv)
    print(yv)

    x = np.arange(-5,5,0.1)
    y = np.arange(-5,5,0.1)
    xx, yy = np.meshgrid(x,y)
    z = np.sin(xx**2 + yy**2)/(xx**2+yy**2)
    print(x)
    print(y)
    print(xx)
    print(yy)
    print(z)

    colorVIMG = msc.imread('Ring.png')

    containerIMG = msc.imread('blank.jpg')
    Ad1 = AdvancedTransformation(colorVIMG, 50, 50, 50)
    #resultV = Ad1.applyEffectV()
    #msc.imsave('testcolorV222.png', resultV)
    resultA = Ad1.applyEffectA(containerIMG=containerIMG)
    msc.imsave('testcolorAAAA2222.png', resultA)
    '''

    '''
    c1 = ColorTransformation(sourceIMG1)
    c1.setupTransformation(tp3)
    result1 = c1.transformImageOnto(containerIMG)
    msc.imsave('testcolorV.png', result1)
    '''




    #result1 = c1.transformImageOnto(self.containerIMG)
    '''
    print(len(colorVIMG[0])/2)
    sourceIMG1 = np.hsplit(colorVIMG,2)[0]

    sourceIMG2 = np.hsplit(colorVIMG,2)[1]

    print(len(sourceIMG2[0]))

    print(len(sourceIMG1[0]))

    msc.imsave('result1.png',sourceIMG1)


    msc.imsave('result2.png',sourceIMG2)
    '''
    '''
    tp3 = np.array([[20,30],[200,20],[10,200],[180,260]],np.float64)
    targetIMG2 = msc.imread('blank.jpg')
    colorIMG = msc.imread('phase1.jpg')
    c1 = ColorTransformation(colorIMG)
    c1.setupTransformation(tp3)
    result = c1.transformImageOnto(targetIMG2)
    msc.imsave('testcolor.png', result)

    x = np.arange(16.0).reshape(4, 4)
    print(x)
    print(np.hsplit(x,2))

    print(np.hsplit(x,2)[0])

    print(np.hsplit(x,2)[1])


    '''








    '''
    u = lambda : (uniform(1.0, 10.0), uniform(1.0, 10.0))
    s = np.array([u() for _ in range(4)])
    t = np.array([u() for _ in range(4)])

    colorIMG = msc.imread('phase1.jpg')
    img = np.ones([10, 10, 3], dtype=np.uint8)
    #img = np.ones([10, 10], dtype=np.uint8)
    print(type(img[0][0]))
    print(img[0][0])
    print(img)
    #s = np.array([u() for _ in range(3)])
    #print(s)
    #print(s)
    #print(t)

    #h = Homography(sourcePoints=s, targetPoints=t)
    #h2 = Homography(sourcePts=s, targetPoints=t)
    #print(h.forwardMatrix)
    #print(h.inverseMatrix)
    #print(Effect.rotate90)

    print(isinstance('rotate90',Effect))
    sourcePoints = np.array([[0, 0], [1919, 0], [0, 1079], [1919,  1079.0]])
    targetPoints = np.array([[600, 50], [1550, 500], [50, 400], [800, 1150.0]])
    h = Homography(sourcePoints=sourcePoints, targetPoints=targetPoints)
    print(h.forwardMatrix)
'''

'''
    sp1 = np.array([[0,0],[100,0],[0,100],[100,100]],np.float64)
    tp1 = np.array([[20,30],[70,20],[10,90],[60,80]],np.float64)
    h1 = Homography(sourcePoints=sp1,targetPoints=tp1)
    mat = np.array([[1.2, 2.3, 4.5], [9.0, 4.4, 5.5], [0.0, 0.0, 1.0]])
    print(type(mat[1][1]) == np.float64)
'''



    #print(h1.forwardMatrix)
    #print(h1.inverseMatrix)
'''
    sourceIMG = msc.imread('bw.jpg','L')
    targetIMG = msc.imread('dice.jpg','L')
    tp2 = np.array([[200,100],[700,200],[100,900],[600,800]],np.float64)

    colorIMG = msc.imread('phase1.jpg')
    '''
'''
    print(len(colorIMG))
    print(type(colorIMG))
    print(type(colorIMG[0][0]))
    print(colorIMG[0][6])
    print(colorIMG[20,30])
    print(type(colorIMG[20,30]))
    print(type(colorIMG[0][0]))
    print(colorIMG[20,30][0])
    print(np.array([31,40,5]),np.float64)
'''

'''
    print(type(colorIMG))
    x = []
    y = []
    for i in range(0,len(colorIMG[0])):
        x.append(i)
    for i in range(0,len(colorIMG)):
        y.append(i)
    print(len(x))
    print(len(y))
'''





    #approximate = RectBivariateSpline(y, x, colorIMG,kx= 1,ky=1)
    #print(approximate(3.5, 3.9))
'''
    blue = colorIMG[0:,:,]
    print(type(blue))
    msc.imsave('testblue.jpg',blue)
'''








    #print(len(sourceIMG))
    #print(len(sourceIMG[0]))
    #print(len(targetIMG))
    #print(len(targetIMG[0]))
    #print(type(sourceIMG))
'''
    transf1 = Transformation(sourceIMG)
    transf1.setupTransformation(tp2)
    transf1.transformImageOnto(targetIMG)

    #test1 = np.array([[1,2,3],[2,3,4],[4,5,6]],np.float64)
    #test2 = np.array([[1],[2],[3]],np.float64)
    #print(np.dot(test1,test2))
    #print(RectBivariateSpline([192.4256],[631.987], sourceIMG,kx= 1,ky=1))
    x = [0,1,2]
    y = [0,1]
    testIMG = np.array([[0,1,2],[1,2,3]],np.float64)
    approximate = RectBivariateSpline(y,x, testIMG,kx= 1,ky=1)
    print(approximate(0.5,0.5))
'''













