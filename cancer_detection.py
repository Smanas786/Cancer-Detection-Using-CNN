# -*- coding: utf-8 -*-
"""

The dataset was originally curated by Janowczyk and Madabhushi and Roa et al. but is
 available in public domain on Kaggle’s website.
"""

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python


# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import os
from glob import glob


os.chdir(r"C:\abdala_Phd_windsor_uni\COMP8590-1-R-2019F Machine Learning &Optimization\My ML Project 2019_2020\breast_cancer_project\breast cancer2") 

os.listdir('./IDC_regular_ps50_idx5')[:10]

base_dir = os.path.join('./IDC_regular_ps50_idx5')
print(os.listdir(base_dir))
print ("number of folder that we have is :" ,len(os.listdir(base_dir)))

#imagePatches = glob('./IDC_regular_ps50_idx5/**/*.png', recursive=True)
#Return a list of paths matching a pathname pattern
imagePatches = glob(base_dir+'/**/*.png', recursive=True)

for filename in imagePatches[0:10]:
    print(filename)

print (" the number of files that we have is :",len(imagePatches))

import cv2
import matplotlib.pylab as plt
image = cv2.imread(base_dir+'/10285/1/10285_idx5_x1151_y901_class1.png')

plt.figure(figsize=(16,16))
plt.imshow(image)

#Change from BGR (OpenCV format) to RGB (Matlab format) to fit Matlab output
#Similarly for BGR ----->  HSV, For HSV, Hue range is [0,179], Saturation range is [0,255] and Value range is [0,255].
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


# Plot Multiple Images
bunchOfImages = imagePatches
i_ = 0
plt.rcParams['figure.figsize'] = (10.0, 10.0)
plt.subplots_adjust(wspace=0, hspace=0)
for l in bunchOfImages[:100]:
    im = cv2.imread(l)
    im = cv2.resize(im, (50, 50)) 
    plt.subplot(10,10, i_+1) #.set_title(l)
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); plt.axis('off')
    i_ += 1
    
    import random
    
def randomImages(a):
    r = random.sample(a, 4)

    plt.figure(figsize=(16,16))
    plt.subplot(131)
    plt.imshow(cv2.imread(r[0]))
    plt.subplot(132)
    plt.imshow(cv2.imread(r[1]))
    plt.subplot(133)
    plt.imshow(cv2.imread(r[2])); 
    
randomImages(imagePatches)


#Preprocess data
import fnmatch # fnmatch() compares a single file name against a pattern and returns TRUE if they match else returns FALSE.
patternZero = '*class0.png'
patternOne = '*class1.png'
#saves the image file location of all images with file name 'class0' 
classZero = fnmatch.filter(imagePatches, patternZero)
#saves the image file location of all images with file name 'class1'
classOne = fnmatch.filter(imagePatches, patternOne)

print("IDC(-)\n\n",classZero[0:5],'\n')
print('the number of healthy images [benign ( 0/ )-no breast cancer] - class zero -',len(classZero))
print('\n**********************************************************\n\n')
#detection of invasive ductal carcinomas (IDC 1) : has a cancer 
print("IDC(+)\n\n",classOne[0:5])
print('the number of  images - class one [  malignant ( 1/ ) - indicating breast cancer was found in the patch] -',len(classOne))


# we want reading 90,000 images in x(predictores) y(label 0 or 1)
def proc_images(lowerIndex,upperIndex):
    """
    Returns two arrays: 
        x is an array of resized images
        y is an array of labels
    """ 
    x = []
    y = []
    WIDTH = 50
    HEIGHT = 50
    for img in imagePatches[lowerIndex:upperIndex]:
        full_size_image = cv2.imread(img)
        x.append(cv2.resize(full_size_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))
        if img in classZero:
            y.append(0)
        elif img in classOne:
            y.append(1)
        else:
            return
    return x,y

X,Y = proc_images(0,10000)

X1 = np.array(X)
X1.shape

df = pd.DataFrame()
df["images"]=X
df["labels"]=Y

X2=df["images"]
Y2=df["labels"]
type(X2)

X2=np.array(X2)
X2.shape

imgs0=[]
imgs1=[]
imgs0 = X2[Y2==0] # (0 = no IDC, 1 = IDC)
imgs1 = X2[Y2==1] 

#print statistics info 
def describeData(a,b): # b is the label 0 or 1
    print('Total number of images: {}'.format(len(a)))
    print('Number of IDC(-) [no cnacer] Images: {}'.format(np.sum(b==0)))
    print('Number of IDC(+) [has cancer] Images: {}'.format(np.sum(b==1)))
    print('Percentage of positive images: {:.2f}%'.format(100*np.mean(b)))
    print('Image shape (Width, Height, Channels): {}'.format(a[0].shape));
describeData(X2,Y2)

###############################################################################
#Deep Learning part

#Train and Test Set
import sklearn
from sklearn import model_selection
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, learning_curve, GridSearchCV
from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score

X=np.array(X)
X=X/255.0   # scalling the images (make all images with the same size) 

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

del X
del X1,X2
del Y2
del imgs0,imgs1

import gc # Manages memory used by Python objects
gc.collect() # the automatic garbage collector.


print ("the shape of X_train",X_train.shape)
print ("the shape of X_test",X_test.shape)

#print(df.head)
print(df['labels'].unique())
print (' the size of dataset [df] is :',np.shape(df))
dist = df['labels'].value_counts()
print("the count value of each label\n",dist)

import seaborn as sns
from scipy.misc import imresize, imread

#histograme for each count label (0,1)
sns.countplot(df['labels'])

del df
gc.collect()

#One Hot encoding¶
# Encode labels to hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
from keras.utils.np_utils import to_categorical
y_trainHot = to_categorical(Y_train, num_classes = 2)
y_testHot = to_categorical(Y_test, num_classes = 2)


import itertools
import keras
from keras import backend as K
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential, model_from_json
from keras.optimizers import SGD, RMSprop, Adam, Adagrad, Adadelta
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Conv2D, MaxPool2D, MaxPooling2D
%matplotlib inline

# Helper Functions
# Helper Functions  Learning Curves and Confusion Matrix

class MetricsCheckpoint(Callback):
    """Callback that saves metrics after each epoch"""
    def __init__(self, savepath):
        super(MetricsCheckpoint, self).__init__()
        self.savepath = savepath
        self.history = {}
    def on_epoch_end(self, epoch, logs=None):
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        np.save(self.savepath, self.history)

#forget this function
def plotKerasLearningCurve():
    plt.figure(figsize=(10,5))
    metrics = np.load('logs.npy')[()]
    filt = ['acc'] # try to add 'loss' to see the loss learning curve
    for k in filter(lambda x : np.any([kk in x for kk in filt]), metrics.keys()):
        l = np.array(metrics[k])
        plt.plot(l, c= 'r' if 'val' not in k else 'b', label='val' if 'val' in k else 'train')
        x = np.argmin(l) if 'loss' in k else np.argmax(l)
        y = l[x]
        plt.scatter(x,y, lw=0, alpha=0.25, s=100, c='r' if 'val' not in k else 'b')
        plt.text(x, y, '{} = {:.4f}'.format(x,y), size='15', color= 'r' if 'val' not in k else 'b')   
    plt.legend(loc=4)
    plt.axis([0, None, None, None]);
    plt.grid()
    plt.xlabel('Number of epochs')

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize = (5,5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def plot_learning_curve(history):
    plt.figure(figsize=(8,8))
    plt.subplot(1,2,1)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('./accuracy_curve.png')
    #plt.clf()
    # summarize history for loss
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('./loss_curve.png')
    
    
    
    batch_size = 128
    num_classes = 2
    epochs = 8
    img_rows,img_cols=50,50
    input_shape = (img_rows, img_cols, 3) #(50,50,3)
   stride_size = 2 # to define the strides (for first hidden layer)
    
    # Initialising the CNN
    model = Sequential()
    
    # Step 1 - Convolution

#filter :  feature Detectore ,
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',
                 input_shape=input_shape,strides=stride_size))
# Step 2 - Pooling  (class : 56 building a CNN step 5)
model.add(MaxPooling2D(pool_size = (2, 2))) #pool1

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu',strides=stride_size))
model.add(MaxPooling2D(pool_size=(2, 2))) #pool2

"""
model.add(Conv2D(128, kernel_size=(3, 3),activation = 'relu',
                      strides=stride_size))
model.add(MaxPooling2D(pool_size = (2, 2))) #pool3
"""

model.add(Dropout(0.25))


# Step 3 - Flattening
model.add(Flatten())

# Step 4 - Full connectio
model.add(Dense(128, activation='relu')) #hidden1
model.add(Dropout(0.5))

model.add(Dense(units = 512, activation = 'relu')) #hidden2


#model.add(Dense(num_classes, activation='softmax'))
model.add(Dense(num_classes, activation='sigmoid'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=0.0001), #Adadelta(),
              metrics=['accuracy'])
 # optimizer=Adam(lr=lr),

#number of layers 
print(len(model.layers)) 

# create a data generator to get the data from our folders and into Keras in an automated way.
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True)  # randomly flip images
    

a = X_train
b = y_trainHot
c = X_test
d = y_testHot
epochs =100

history = model.fit_generator(datagen.flow(a,b, batch_size=32),
                        steps_per_epoch=len(a) / 32, 
                              epochs=epochs,validation_data = [c, d],
                              callbacks = [MetricsCheckpoint('logs')])
                              """Callback that saves metrics after each epoch"""

"""
history_df = pd.DataFrame(history.history)
history_df[['loss', 'val_loss']].plot()

history_df = pd.DataFrame(history.history)
history_df[['acc', 'val_acc']].plot()
"""

y_pred = model.predict(c)

Y_pred_classes = np.argmax(y_pred,axis=1) 
Y_true = np.argmax(d,axis=1)

dict_characters = {0: 'IDC(-)', 1: 'IDC(+)'}

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
plot_confusion_matrix(confusion_mtx, classes = list(dict_characters.values())) 
plt.show()
""" or """
from sklearn import metrics
confusion_matrixx = metrics.confusion_matrix(y_true=Y_true, y_pred=Y_pred_classes)
print(confusion_matrixx )

plotKerasLearningCurve()
plt.show()  

plot_learning_curve(history)
plt.show()

del model
gc.collect()



from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import roc_curve
roc_log = roc_auc_score(np.argmax(y_testHot, axis=1), np.argmax(y_pred, axis=1))
false_positive_rate, true_positive_rate, threshold = roc_curve(np.argmax(y_testHot, axis=1), np.argmax(y_pred, axis=1))
area_under_curve = auc(false_positive_rate, true_positive_rate)

plt.plot([0, 1], [0, 1], 'r--')
plt.plot(false_positive_rate, true_positive_rate, label='AUC = {:.3f}'.format(area_under_curve))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
#plt.savefig(ROC_PLOT_FILE, bbox_inches='tight')
plt.close()
"""
https://towardsdatascience.com/predicting-invasive-ductal-carcinoma-using-convolutional-neural-network-cnn-in-keras-debb429de9a6
"""
tg_names=['ID-','ID+']
report = metrics.classification_report(Y_true, Y_pred_classes, target_names=tg_names)
print(report)   