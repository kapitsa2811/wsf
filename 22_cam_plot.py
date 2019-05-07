#-*- coding: utf-8 -*-

from keras.applications.resnet50 import preprocess_input
import numpy as np
import cv2

from src.model_builder import CamModelBuilder
from src.utils import plot_img, list_files
from keras.models import load_model

if __name__ == "__main__":

    expNo = str(2)
    indx=40
    #comment="newTabData_40.h5"
    modelName="newTabData_"+str(indx)+".h5"
    basePath = "C:\\Users\Lenovo\PycharmProjects\demo\\flask-keras-cnn-image-retrieval//publicationData//"
    modelPath=basePath+"//"+expNo+"//model//"+modelName
    results=basePath+"//"+expNo+"//results//"
    testData=basePath+"//"+expNo+"//test//"
    #testData="C:\\Users\Lenovo\PycharmProjects\wsl\publicationData\8\\train\\Table\\"
    # results=basePath+"//"+expNo+"//results1//"
    # testData=basePath+"//"+expNo+"//delMe//nonFocus//"

    detector = CamModelBuilder().get_cam_model(indx)
    #t="/home/kapitsa/PycharmProjects/MyOCRService/objectDetection/Weakly-Supervised-Text-Detection/backUP/paperModel"
    #detector.load_weights(".//backUP//weights.19-0.01.h5", by_name=True)

    detector.load_weights(modelPath, by_name=True)

    #hardPath="/home/kapitsa/PycharmProjects/MyOCRService/objectDetection/Weakly-Supervised-Text-Detection//"#backUP/paperModel//"
    #detector.load_weights(hardPath+"//weights.19-0.01.h5", by_name=True)
    detector.summary()
    imgs = list_files(testData)

    modelPath = basePath+"\2\model\\newTabData_40.h5"
    dumpPath = basePath+"\2\\delMe\\"
    #mod = load_model(modelPath)

    for i, img_path in enumerate(imgs):

        nm=img_path.split("test")[1]
        print(nm[1:])

        nm=nm[1:]
        temp=cv2.imread(img_path)

        # pixelUP = np.where(temp >= 3)
        #
        # countUP=len(pixelUP)
        # #sumUP=np.average(cam_map[pixelUP])
        #
        # #aboveSUM=np.where(cam_map >=sumUP)
        #
        # temp[pixelUP] = 255


        original_img = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
         
        #img = cv2.resize(255-original_img, (224, 224))
        img = cv2.resize(original_img, (224, 224))

        img = np.expand_dims(img, 0).astype(np.float64)
        cam_map = detector.predict(preprocess_input(img))
        #pred = mod.predict(255 - img)
        # print(pred[0][0])
        # print(pred[0][1])

        #cv2.imwrite(dumpPath + str(pred[0][0]) + "_" + str(pred[0][1]) + img_path,original_img)

        cam_map = cam_map[0, :, :, 1]
        cam_map1=cv2.cvtColor(cam_map, cv2.COLOR_GRAY2BGR)
        cam_map1 = cv2.resize(cam_map1, (original_img.shape[1], original_img.shape[0]))

        cam_map = cv2.resize(cam_map, (original_img.shape[1], original_img.shape[0]))
        cam_map1=cam_map1+original_img

        #plot_img(original_img, cam_map, show=False, save_filename="{}.png".format(i+1))

        #plot_img(i, original_img, cam_map, cam_map1, show=False, save_filename=".//predict//{}_0.png".format(i + 1))
        plot_img(nm,i,img_path,indx,results, original_img, cam_map, cam_map1, show=False, save_filename=".//predictVGG//{}_0.png".format(i + 1))
    
