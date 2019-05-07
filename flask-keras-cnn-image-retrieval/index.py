# -*- coding: utf-8 -*-
# Author: yongyuan.name
import os
import h5py
import numpy as np
import argparse
from extract_cnn_vgg16_keras import VGGNet,Resnet
from keras.applications.resnet50 import ResNet50


# ap = argparse.ArgumentParser()
# ap.add_argument("-database", required = True,
# 	help = "Path to database which contains images to be indexed")
# ap.add_argument("-index", required = True,
# 	help = "Name of index file")
# args = vars(ap.parse_args())


'''
 Returns a list of filenames for all jpg images in a directory. 
'''
def get_imlist(path):
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]


'''
 Extract features and index the images
'''
if __name__ == "__main__":


    featsRes = []
    namesRes = []

    model = VGGNet()
    modelRes=Resnet()

    #print("\n\t model summary=",model)
   # print("\n\t modelRes summary=",ResNet50.summary())

    cwd=os.getcwd()+"//"
    #db = args["database"]

    db = cwd+"database"
    #db="C:\\Users\Lenovo\PycharmProjects\wsl\publicationData\image\\"

    img_list = get_imlist(db)
    
    print("--------------------------------------------------")
    print("         feature extraction starts")
    print("--------------------------------------------------")
    
    feats = []
    names = []

    for i, img_path in enumerate(img_list):
        norm_feat = model.extract_feat(img_path)
        norm_featRes=modelRes.extract_feat(img_path)

        print("\n\t norm_featRes shape=",norm_featRes.shape)

        img_name = os.path.split(img_path)[1]
        feats.append(norm_feat)
        names.append(img_name)

        featsRes.append(norm_featRes)
        namesRes.append(img_name)

        print("extracting feature from image No. %d , %d images in total" %((i+1), len(img_list)))

    feats = np.array(feats)
    featsRes = np.array(featsRes)
    # print(feats)
    # directory for storing extracted features
    output = cwd+'1//'+"2"#args["index"]
    
    print("--------------------------------------------------")
    print("      writing feature extraction results ...")
    print("--------------------------------------------------")


    h5f = h5py.File(output, 'w')
    h5f.create_dataset('dataset_1', data = feats)
    # h5f.create_dataset('dataset_2', data = names)
    h5f.create_dataset('dataset_2', data = np.string_(names))
    h5f.close()

    output = cwd + '1//' + "2_res"  # args["index"]

    print("--------------------------------------------------")
    print("      writing feature extraction results ...")
    print("--------------------------------------------------")

    h5f = h5py.File(output, 'w')
    h5f.create_dataset('dataset_1', data=featsRes)
    # h5f.create_dataset('dataset_2', data = names)
    h5f.create_dataset('dataset_2', data=np.string_(names))
    h5f.close()

