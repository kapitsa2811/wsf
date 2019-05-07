#-*- coding: utf-8 -*-
from src.model_builder import CamModelBuilder
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.applications.resnet50 import preprocess_input
from src.keras_utils import build_generator, create_callbacks
from shutil import copyfile

basePath = "C:\\Users\Lenovo\PycharmProjects\demo\\flask-keras-cnn-image-retrieval//publicationData//"
expNo = str(2)


if __name__ == "__main__":
    #indx="40_2"

    indx = "40"
    model_builder = CamModelBuilder()
    model = model_builder.get_cls_model(indx)
    model.summary()

#     fixed_layers = []
#     for layer in model.layers[:-6]:
#         layer.trainable = False
#         fixed_layers.append(layer.name)
#     print(fixed_layers)
 
    optimizer = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.005)
    model.compile(loss = 'categorical_crossentropy',
                  optimizer = optimizer,
                  metrics = ['accuracy'])


    comment="newTabData_"+str(indx)+".h5"

    train_generator = build_generator(basePath + "//" + expNo + "//train", preprocess_input, augment=True)
    '''
    train_generator,x,y = build_generator(basePath+"//"+expNo+"//train", preprocess_input, augment=True)

    X_train, X_val, Y_train, Y_val = train_test_split(x, y, test_size=0.2, random_state=7)

    train_model = model.fit(X_train, Y_train,
                            batch_size=64,
                            epochs=4,
                            verbose=1,
                            validation_data=(X_val, Y_val),shuffle=True)

    '''

    model.fit_generator(train_generator,
                        steps_per_epoch = 10, #len(train_generator),
                        callbacks = create_callbacks(comment),
                        epochs=50)

copyfile(comment, basePath+"//"+expNo+"//model//"+comment)