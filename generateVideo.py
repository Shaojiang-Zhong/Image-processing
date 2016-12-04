from Homography import ColorTransformation
import numpy as np
import scipy.misc as msc
import os



def generateVideo(folderName, videoName):
    absPath = os.getcwd()+'/'+folderName
    if not os.path.exists(absPath):
        os.makedirs(absPath)
    sourceIMG = msc.imread("dawn.png")
    #sourceIMG = msc.imread("haha.jpg")

    columns = len(sourceIMG[0])
    rows = len(sourceIMG)
    if columns%2 == 1:
        raise ValueError("bad input")
    sourceImgL = np.hsplit(sourceIMG,2)[0]
    sourceImgR = np.hsplit(sourceIMG,2)[1]
    c1 = np.linspace(start=0, stop=columns/2-1, num=60, dtype=int)
    c4 = c1 + columns/2
    c3 = np.linspace(start=columns/2-1, stop=0, num=60, dtype=int)
    c2 = c3 + columns/2

    r2 = np.linspace(start=0, stop=rows-1, num=60, dtype=int)
    r1 = np.linspace(start=rows-1, stop=0, num=60, dtype=int)


    os.chdir(absPath)
    counter = 0

    #first rotation, 0 degree to 90 degree


    for xc1, xc2, xc3, xc4, yr1, yr2 in zip(c1, c2, c3, c4, r1, r2):
        containerIMG = np.zeros((rows, columns, 3))
        tmpTargetP1 = np.array([[xc1, 0], [columns/2-1, yr2], [0, yr1], [xc3, rows-1]], np.int64)
        tmpTargetP2 = np.array([[columns/2, yr2], [xc2, 0], [xc4, rows-1], [columns-1, yr1]], np.int64)

        targetImageL = ColorTransformation(sourceImgL)
        targetImageL.setupTransformation(tmpTargetP1)
        targetImageL.transformImageOnto(containerIMG)





        targetImageR = ColorTransformation(sourceImgR)
        targetImageR.setupTransformation(tmpTargetP2)
        targetImageR.transformImageOnto(containerIMG)

        if counter < 10:

            msc.imsave("Frame"+"00"+str(counter)+".png", containerIMG)
        elif counter < 100:
            msc.imsave("Frame"+"0"+str(counter)+".png", containerIMG)
        else:
            msc.imsave("Frame"+str(counter)+".png", containerIMG)
        counter += 1

    #second rotation 90 degree to 180 degree
    for xc1, xc2, xc3, xc4, yr1, yr2 in zip(c1, c2, c3, c4, r1, r2):
        containerIMG = np.zeros((rows, columns, 3))
        tmpTargetP1 = np.array([[columns/2-1, yr2],  [xc3, rows-1],  [xc1, 0], [0, yr1]], np.int64)
        tmpTargetP2 = np.array([[xc4, rows-1],[columns/2, yr2], [columns-1, yr1], [xc2, 0]], np.int64)

        targetImageL = ColorTransformation(sourceImgL)
        targetImageL.setupTransformation(tmpTargetP1)
        targetImageL.transformImageOnto(containerIMG)





        targetImageR = ColorTransformation(sourceImgR)
        targetImageR.setupTransformation(tmpTargetP2)
        targetImageR.transformImageOnto(containerIMG)

        if counter < 10:

            msc.imsave("Frame"+"00"+str(counter)+".png", containerIMG)
        elif counter < 100:
            msc.imsave("Frame"+"0"+str(counter)+".png", containerIMG)
        else:
            msc.imsave("Frame"+str(counter)+".png", containerIMG)
        counter += 1

    #third rotation: 180 degree to 270 degree
    for xc1, xc2, xc3, xc4, yr1, yr2 in zip(c1, c2, c3, c4, r1, r2):
        containerIMG = np.zeros((rows, columns, 3))
        tmpTargetP1 = np.array([[xc3, rows-1], [0, yr1], [columns/2-1, yr2],  [xc1, 0]], np.int64)
        tmpTargetP2 = np.array([[columns-1, yr1],[xc4, rows-1],[xc2, 0], [columns/2, yr2]], np.int64)

        targetImageL = ColorTransformation(sourceImgL)
        targetImageL.setupTransformation(tmpTargetP1)
        targetImageL.transformImageOnto(containerIMG)





        targetImageR = ColorTransformation(sourceImgR)
        targetImageR.setupTransformation(tmpTargetP2)
        targetImageR.transformImageOnto(containerIMG)

        if counter < 10:

            msc.imsave("Frame"+"00"+str(counter)+".png", containerIMG)
        elif counter < 100:
            msc.imsave("Frame"+"0"+str(counter)+".png", containerIMG)
        else:
            msc.imsave("Frame"+str(counter)+".png", containerIMG)
        counter += 1
    #forth rotation: 270 degree to 360 degree
    for xc1, xc2, xc3, xc4, yr1, yr2 in zip(c1, c2, c3, c4, r1, r2):
        containerIMG = np.zeros((rows, columns, 3))
        tmpTargetP1 = np.array([[0, yr1],  [xc1, 0],[xc3, rows-1],  [columns/2-1, yr2]], np.int64)
        tmpTargetP2 = np.array([[xc2, 0],[columns-1, yr1], [columns/2, yr2], [xc4, rows-1]], np.int64)

        targetImageL = ColorTransformation(sourceImgL)
        targetImageL.setupTransformation(tmpTargetP1)
        targetImageL.transformImageOnto(containerIMG)





        targetImageR = ColorTransformation(sourceImgR)
        targetImageR.setupTransformation(tmpTargetP2)
        targetImageR.transformImageOnto(containerIMG)

        if counter < 10:

            msc.imsave("Frame"+"00"+str(counter)+".png", containerIMG)
        elif counter < 100:
            msc.imsave("Frame"+"0"+str(counter)+".png", containerIMG)
        else:
            msc.imsave("Frame"+str(counter)+".png", containerIMG)
        counter += 1

        #print("ffmpeg -r 30 -i \"{}/Frame%3d.png\" -vcodec libx264 \"../{}\"".format(folderName, videoName))
        os.system("cd ..")
        os.system('ffmpeg -r 30 -i \"{}/Frame%3d.png\" -vcodec libx264 .\"/{}\"'.format(folderName, videoName))













if __name__ == '__main__':
    generateVideo("folderFrame", "batman.mp4")



