import tensorflow as tf
from tensorflow.keras.utils import to_categorical

print('hello world ')

# ############################################################################
# ##Importing all lib
#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# #import tensorflow as tf
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,Dense,Flatten,Dropout

# ##################################################################################
data=[]
labels=[]
classes=43
cur_path=os.getcwd()
cur_path='E:\MISSION DATA SCIENTIST\GITHUB PROJECT\Traffice light classification'
print(cur_path)
#
for i in range(classes):
    path=os.path.join(cur_path,'train',str(i))
    images=os.listdir(path)

    for a in images:
        try :
            images=Image.open(path+'\\'+a)
            images=images.resize((30,30))
            images=np.array(images)
            data.append(images)
            labels.append(i)
        except:
            print('error loading images ')
print(type(data))
data=np.array(data)
labels=np.array(labels)

#######################3########################################
#preprocesssing the dataset

print(data.shape,labels.shape)

X_train,y_train,X_test,y_test=train_test_split(data,labels,test_size=0.2,random_state=42)

print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)

y_test=to_categorical(y_test)
y_train=to_categorical(y_train)

#####################################################################
#Building CNN Model

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(43, activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

epochs=15
history=model.fi(X_train,y_train,batch_size=32 ,epochs=epochs,validation_data=(X_test,y_test))

model.save('my_model.h5')

####################################################################################################################################

#plotting graphs for accuracy
plt.figure(0)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()
plt.figure(1)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

###################################################################################################################
#prediction

from sklearn.metrics import accuracy_score
y_test = pd.read_csv('\Traffice light classification\sTest.csv')
labels = y_test["ClassId"].values
imgs = y_test["Path"].values
data=[]
for img in imgs:
    image = Image.open(img)
    image = image.resize((30,30))
    data.append(np.array(image))
X_test=np.array(data)
pred = model.predict_classes(X_test)
#Accuracy with the test data
from sklearn.metrics import accuracy_score
print(accuracy_score(labels, pred))


