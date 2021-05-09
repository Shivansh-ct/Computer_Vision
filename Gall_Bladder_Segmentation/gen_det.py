import os
import cv2 as cv
import json
from glob import glob
import numpy as np
import argparse

ddepth = cv.CV_16S

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluation Script for segmentation of Gall Bladder Images')
    parser.add_argument('-i', '--img_path', type=str, default='img', required=True, help="Path for the image folder")
    parser.add_argument('-d', '--det_path', type=str, default='det', required=True,help="Path for the detected masks folder")

    #args.img_path is the path to input image folder and args.det_path is the path to output image folder
    args = parser.parse_args()
    img_files = sorted(glob(os.path.join(args.img_path, "*jpg")))#loading all input image locations to 'img_files'

    y=0   #stores the image number of input currently in process
    for fimg in img_files:
        c=0
        for th in range(50,81, 1): #th is the threshold of the Binary Thresholding varying from 50 to 80 and selecting the closest best image it can
            src = cv.imread(fimg, cv.IMREAD_COLOR)#loads a BGR image
            src = cv.GaussianBlur(src, (53,53), 0) #applies gaussian kernel with kernel=53 on image
            src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)#converts BGR to grayscale image
            ret, src_gray = cv.threshold(src_gray, th, 255, cv.THRESH_BINARY) #applies binary thresholding on image where pixel>th are set to white and rest black

            dst = cv.cvtColor(src_gray, cv.COLOR_GRAY2RGB) #converts grayscale to RGB image
            dst = cv.Laplacian(dst, ddepth, ksize=5) #applies laplacian kernel with kernel=5 on image

            dst = cv.convertScaleAbs(dst) #converts to 8-bit pixel values incase datatype conversion happened to 16-bit
            src_gray = cv.cvtColor(dst, cv.COLOR_RGB2GRAY)#RGB to GrayScale conversion

            img = src_gray
            contours,hierarchy = cv.findContours(img, 1, 2) #contours stores the coordinate of all contours in image,
                                                    # contour[i] is the set of all points of ith contour

            k=0
            mask = np.zeros(img.shape[:2], dtype=img.dtype)
            for cnt in contours: #this for loop stores the area of max area contour in 'maxa' among all contours whose area is less than 400000, this removes extra large contours not comparable to gall bladder size
                area = cv.contourArea(cnt) #area of contour cnt
                if area<400000: #selecting those contours with area<400000
                    cv.drawContours(mask, [cnt], 0, (255), -1) 
                    if k==0:
                        maxa = area
                        k=1
                    elif maxa<area:
                            maxa=area

            k=0
            dst = mask
            for cnt in contours:
                area = cv.contourArea(cnt)
                if area<400000:
                    if area==maxa: #max area contour filled with white and rest black
                        cv.fillPoly(dst, pts=[cnt], color=(255,255,255))
                    else:
                        cv.fillPoly(dst, pts=[cnt], color=(0,0,0))

            dst = cv.convertScaleAbs(dst)

        #This if else represents Step-6) indicated in the report
            if c==0:
                pre = dst
                c = 1
            else:
                sum0 = np.sum(cv.bitwise_and(pre,dst)==255)
                sum1 = np.sum(cv.bitwise_or(pre,dst)==255)
                iou = float(sum0)/float(sum1) #sum represents IOU between current and previous image
                if iou<0.60:
                    break
                pre = dst


        cv.imwrite(args.det_path+'000'+str(y)+'.jpg', pre) #stores the output masks to destination
        y=y+1