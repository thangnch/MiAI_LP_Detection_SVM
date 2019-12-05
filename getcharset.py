import glob
import cv2
import os

image_path="data/charTrainset/"
write_path="data/"

for number in range(10):
    for img_org_path in glob.iglob(image_path + str(number) + '/*.jpg'):
        img = cv2.imread(img_org_path,0)
        img = cv2.resize(img,dsize=(30,60))
        #cv2.imshow("A", img)
        #cv2.waitKey()
        _, img = cv2.threshold(img,30,255,cv2.THRESH_BINARY)
        #cv2.imshow("B", img)
        #cv2.waitKey()
        img_org_path = os.path.basename(img_org_path)
        cv2.imwrite(write_path + str(number) + "/" + img_org_path, img)


for number in range(65,91):
    number = chr(number)
    print(image_path + str(number))
    if os.path.isdir(image_path + str(number)):
        for img_org_path in glob.iglob(image_path + str(number) + '/*.jpg'):
            img = cv2.imread(img_org_path,0)
            img = cv2.resize(img,dsize=(30,60))
            #cv2.imshow("A", img)
            #cv2.waitKey()
            _, img = cv2.threshold(img,30,255,cv2.THRESH_BINARY)
            #cv2.imshow("B", img)
            #cv2.waitKey()
            img_org_path = os.path.basename(img_org_path)
            if not os.path.isdir(write_path + str(ord(number)) ):
                os.mkdir(write_path + str(ord(number)) )
            cv2.imwrite(write_path + str(ord(number)) + "/" + img_org_path, img)
