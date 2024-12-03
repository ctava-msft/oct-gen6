import cv2
import glob
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, Dropout
from keras.utils import normalize, to_categorical
import numpy as np
import os
from matplotlib import pyplot as plt
import random
import sys

np.set_printoptions(threshold=sys.maxsize)

def get_model():
    return multi_unet_model(n_classes=n_classes, IMG_HEIGHT=SIZE_Y, IMG_WIDTH=SIZE_X, IMG_CHANNELS=1)

def multi_unet_model(n_classes=4, IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=1):
    
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = inputs

    #Contraction path
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
     
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
     
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
     
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    
    #Expansive path 
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
     
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
     
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
     
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
     
    outputs = Conv2D(n_classes, (1, 1), activation='softmax')(c9)
     
    model = Model(inputs=[inputs], outputs=[outputs])
        
    return model

if __name__ == '__main__':
    #Resizing images
    SIZE_X = 640
    SIZE_Y = 640
    n_classes= 9 # Number of classes for segmentation

    TRAIN_PATH_X = '/home/khamituly/Documents/Diploma work/dataset_manual/temp_health/X'
    TRAIN_PATH_Y = '/home/khamituly/Documents/Diploma work/dataset_manual/temp_health/Y_test'

    train_ids_x = next(os.walk(TRAIN_PATH_X))[2]
    train_ids_y = next(os.walk(TRAIN_PATH_Y))[2]
    #Capture training image info as a list
    train_images = []

    for directory_path in glob.glob(TRAIN_PATH_X):
        for img_path in glob.glob(os.path.join(directory_path, "*.jpeg")):
            img = cv2.imread(img_path, 0)       
            img = cv2.resize(img, (SIZE_Y, SIZE_X))
            train_images.append(img)    

    #Convert list to array for machine learning processing        
    train_images = np.array(train_images)
    #Capture mask/label info as a list
    train_masks = [] 
    for directory_path in glob.glob(TRAIN_PATH_Y):
        for mask_path in glob.glob(os.path.join(directory_path, "*.jpeg")):
            mask = cv2.imread(mask_path, 0)       
            mask = cv2.resize(mask, (SIZE_Y, SIZE_X), interpolation = cv2.INTER_NEAREST)  #Otherwise ground truth changes due to interpolation
            train_masks.append(mask)
            
    #Convert list to array for machine learning processing          
    train_masks = np.array(train_masks)
    np.unique(train_masks)
    plt.figure(figsize=(40,100))
    for i in range(8):
        plt.subplot(14, 5, i+1)
        plt.imshow(train_images[i,:,:])
        plt.title("(Label: " + str(i) + ")")
    plt.show()  
    plt.figure(figsize=(40,100))
    for i in range(8):
        plt.subplot(14, 5, i+1)
        plt.imshow(train_masks[i,:,:])
        plt.title("(Label: " + str(i) + ")")
    plt.show()

    #Encode labels
    from sklearn.preprocessing import LabelEncoder
    labelencoder = LabelEncoder()
    n, h, w = train_masks.shape
    train_masks_reshaped = train_masks.reshape(-1,1)
    train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped)
    train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w)

    np.unique(train_masks_encoded_original_shape)

    train_images = np.expand_dims(train_images, axis=3)
    train_images = normalize(train_images, axis=1)

    train_masks_input = np.expand_dims(train_masks_encoded_original_shape, axis=3)

    #Create a subset of data for quick testing
    #Picking 10% for testing and remaining for training
    from sklearn.model_selection import train_test_split
    X1, X_test, y1, y_test = train_test_split(train_images, train_masks_input, test_size = 0.10, random_state = 0)

    #Further split training data t a smaller subset for quick testing of models
    X_train, X_do_not_use, y_train, y_do_not_use = train_test_split(X1, y1, test_size = 0.2, random_state = 0)

    print("Class values in the dataset are ... ", np.unique(y_train))  # 0 is the background/few unlabeled

    train_masks_cat = to_categorical(y_train, num_classes=n_classes)
    y_train_cat = train_masks_cat.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], n_classes))



    test_masks_cat = to_categorical(y_test, num_classes=n_classes)
    y_test_cat = test_masks_cat.reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2], n_classes))


    IMG_HEIGHT = X_train.shape[1]
    IMG_WIDTH  = X_train.shape[2]
    IMG_CHANNELS = X_train.shape[3]


    model = get_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    #If starting with pre-trained weights. 
    #model.load_weights('???.hdf5')

    history = model.fit(X_train, y_train_cat, 
                        batch_size = 16, 
                        verbose=1, 
                        epochs=200, 
                        validation_data=(X_test, y_test_cat), 
                        #class_weight=class_weights,
                        shuffle=False)
                        

    _, acc = model.evaluate(X_test, y_test_cat)
    print("Accuracy is = ", (acc * 100.0), "%")

    #plot the training and validation accuracy and loss at each epoch
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    acc = history.history['accuracy']
    val_acc = history.history['accuracy']

    plt.plot(epochs, acc, 'y', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
    plt.title('Training and validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    y_pred=model.predict(X_test)
    y_pred_argmax=np.argmax(y_pred, axis=3)

    #Using built in keras function
    from keras.metrics import MeanIoU
    IOU_keras = MeanIoU(num_classes=n_classes)  
    IOU_keras.update_state(y_test[:,:,:,0], y_pred_argmax)
    print("Mean IoU =", IOU_keras.result().numpy())

    ## This is example of IOU of 4 class, if you need provide corresponding amount of class 

    values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
    class1_IoU = values[0,0]/(values[0,0] + values[0,1] + values[0,2] + values[0,3] + values[1,0]+ values[2,0]+ values[3,0])
    class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[1,2] + values[1,3] + values[0,1]+ values[2,1]+ values[3,1])
    class3_IoU = values[2,2]/(values[2,2] + values[2,0] + values[2,1] + values[2,3] + values[0,2]+ values[1,2]+ values[3,2])
    class4_IoU = values[3,3]/(values[3,3] + values[3,0] + values[3,1] + values[3,2] + values[0,3]+ values[1,3]+ values[2,3])


    print("IoU for class1 is: ", class1_IoU)
    print("IoU for class2 is: ", class2_IoU)
    print("IoU for class3 is: ", class3_IoU)
    print("IoU for class4 is: ", class4_IoU)

    plt.imshow(train_images[0, :,:,0], cmap='gray')
    plt.imshow(train_masks[0], cmap='gray')