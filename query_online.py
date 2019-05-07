# -*- coding: utf-8 -*-
# Author: yongyuan.name
from extract_cnn_vgg16_keras import VGGNet
from extract_cnn_vgg16_keras import Resnet

import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse


# ap = argparse.ArgumentParser()
# ap.add_argument("-query", required = True,
# 	help = "Path to query which contains image to be queried")
# ap.add_argument("-index", required = True,
# 	help = "Path to index")
# ap.add_argument("-result", required = True,
# 	help = "Path for output retrieved images")
# args = vars(ap.parse_args())


# read in indexed images' feature vectors and corresponding image names

#input1="C:\\Users\Lenovo\PycharmProjects\demo\\flask-keras-cnn-image-retrieval\database\\001_accordion_image_0001.jpg"
#input1="C:\\Users\Lenovo\PycharmProjects\wsl\publicationData\image\\graphs.pdf-28 - Copy.jpg"

input1="C:\\Users\Lenovo\PycharmProjects\demo\\flask-keras-cnn-image-retrieval\query\\IMG_5953.JPG"
filePath="C:\\Users\Lenovo\PycharmProjects\demo\\flask-keras-cnn-image-retrieval\\1\\2"
filePathRes="C:\\Users\Lenovo\PycharmProjects\demo\\flask-keras-cnn-image-retrieval\\1\\2_res"

h5f = h5py.File(filePath,'r')
# feats = h5f['dataset_1'][:]
feats = h5f['dataset_1'][:]
#print(feats)
imgNames = h5f['dataset_2'][:]
#print(imgNames)
h5f.close()

h5f = h5py.File(filePathRes,'r')
# feats = h5f['dataset_1'][:]
featsRes = h5f['dataset_1'][:]
#print(feats)
imgNamesRes = h5f['dataset_2'][:]
#print(imgNames)
h5f.close()

        
print("--------------------------------------------------")
print("               searching starts")
print("--------------------------------------------------")


queryPath="C:\\Users\Lenovo\PycharmProjects\demo\\flask-keras-cnn-image-retrieval\query\\"
# init VGGNet16 model
model = VGGNet()
modelRes = Resnet()

for q in os.listdir(queryPath):
    input1=queryPath+q

    #print("\n\t input=",q)
    # read and show query image
    queryDir = input1#args["query"]
    queryImg = mpimg.imread(queryDir)
    plt.title("Query Image")
    plt.imshow(queryImg)
    plt.pause(0.02)

    # extract query image's feature, compute simlarity score and sort
    queryVec = model.extract_feat(queryDir)
    queryVecRes = modelRes.extract_feat(queryDir)

    print("\n\t queryVec type=",type(queryVec),"\t queryVecRes type=",type(queryVecRes))
    print("\n\t queryVec shape=",queryVec.shape,"\t queryVecRes type=",queryVecRes.shape)


    scores = np.dot(queryVec, feats.T)
    rank_ID = np.argsort(scores)[::-1]
    rank_score = scores[rank_ID]

    print("\n\t scores =",scores.shape,"\t rank_ID ",rank_ID.shape,"\t rank_score =",rank_score.shape)
    print("\n\t featsRes shape=",featsRes.shape)
    scoresRes = np.dot(queryVecRes, featsRes.T)
    rank_IDRes = np.argsort(scoresRes)[::-1]
    rank_scoreRes = scoresRes[rank_IDRes]
    print("\n\t scoresRes =",scoresRes.shape)
    print("\t rank_ID ",rank_IDRes.shape,"\t rank_score =",rank_scoreRes.shape)


    # print ("\n\t rank_ID=",rank_ID)
    # print ("\n\t rank_score=",rank_score)


    # number of top retrieved images to show
    maxres = 5
    imlist = [imgNames[index] for i,index in enumerate(rank_ID[0:maxres])]
    #print("top %d images in order are: " %maxres, imlist)

    # show top #maxres retrieved result one by one

    dirPath="C:\\Users\\Lenovo\PycharmProjects\demo\\flask-keras-cnn-image-retrieval\\database\\"

    #dirPath="C:\\Users\\Lenovo\PycharmProjects\wsl\publicationData\image\\"

    for i,im in enumerate(imlist):
        image = mpimg.imread(dirPath+"/"+str(im, 'utf-8'))
        plt.title("vgg search output %d" %(i+1))
        plt.imshow(image)
        plt.pause(0.02)
        #plt.show()

    # number of top retrieved images to show
    maxres = 5
    imlist = [imgNamesRes[index] for i,index in enumerate(rank_IDRes[0:maxres])]
    #print("top %d images in order are: " %maxres, imlist)

    # show top #maxres retrieved result one by one

    dirPath="C:\\Users\\Lenovo\PycharmProjects\demo\\flask-keras-cnn-image-retrieval\\database\\"

    #dirPath="C:\\Users\\Lenovo\PycharmProjects\wsl\publicationData\image\\"

    for i,im in enumerate(imlist):
        image = mpimg.imread(dirPath+"/"+str(im, 'utf-8'))
        plt.title("resnet search output %d" %(i+1))
        plt.imshow(image)
        plt.pause(0.02)
        #plt.show()
