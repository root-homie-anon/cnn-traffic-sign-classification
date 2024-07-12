#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


# In[2]:


with open("./traffic-signs-data/train.p", mode='rb') as training_data: #data import
    train = pickle.load(training_data)
with open("./traffic-signs-data/valid.p", mode='rb') as validation_data:
    valid = pickle.load(validation_data)
with open("./traffic-signs-data/test.p", mode='rb') as testing_data:
    test = pickle.load(testing_data)


# In[3]:


x_train, y_train = train['features'], train['labels']
x_val, y_val = valid['features'], valid['labels']
x_test, y_test = test['features'], test['labels']


# In[4]:


x_train.shape


# In[5]:


y_train.shape


# In[6]:


x_val.shape


# In[7]:


y_val.shape


# In[8]:


x_test.shape


# In[9]:


y_test.shape


# In[10]:


#image exploration
i = 1000
plt.imshow(x_train[i])
y_train[i]


# In[11]:


i = 5000
plt.imshow(x_train[i])
y_train[i]


# In[12]:


i = 10000
plt.imshow(x_train[i])
y_train[i]


# In[13]:


#data preparation
from sklearn.utils import shuffle
x_train, y_train = shuffle(x_train, y_train)


# In[14]:


x_train_gray = np.sum(x_train/3, axis = 3, keepdims = True) #color to grayscale
x_test_gray = np.sum(x_test/3, axis = 3, keepdims = True)
x_val_gray = np.sum(x_val/3, axis = 3, keepdims = True)


# In[15]:


x_train_gray.shape


# In[16]:


x_val_gray.shape


# In[17]:


x_test_gray.shape


# In[18]:


#data normalization
x_train_gray_norm = x_train_gray/255
x_test_gray_norm = x_test_gray/255
x_val_gray_norm = x_val_gray/255


# In[19]:


x_train_gray_norm


# In[20]:


x_test_gray_norm


# In[21]:


x_val_gray_norm


# In[22]:


i = 600
plt.imshow(x_train_gray[i].squeeze(), cmap = 'gray')
plt.figure()
plt.imshow(x_train[i])
plt.figure()
plt.imshow(x_train_gray_norm[i].squeeze(), cmap = 'gray')


# In[23]:


i = 600
plt.imshow(x_test_gray[i].squeeze(), cmap = 'gray')
plt.figure()
plt.imshow(x_test[i])
plt.figure()
plt.imshow(x_test_gray_norm[i].squeeze(), cmap = 'gray')


# In[24]:


i = 600
plt.imshow(x_val_gray[i].squeeze(), cmap = 'gray')
plt.figure()
plt.imshow(x_val[i])
plt.figure()
plt.imshow(x_val_gray_norm[i].squeeze(), cmap = 'gray')


# In[25]:


#building the model

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

from sklearn.model_selection import train_test_split


# In[26]:


image_shape = x_train_gray[i].shape


# In[27]:


#model building
cnn_model = Sequential()

#layer 1 convolution
cnn_model.add(Conv2D(filters = 6, kernel_size = (5,5), activation = 'relu', input_shape = (32,32,1)))
cnn_model.add(AveragePooling2D(pool_size = (2,2), strides = (1,1), padding = 'valid'))

#layer 2 convulation
cnn_model.add(Conv2D(filters = 16, kernel_size = (5,5), activation = 'relu'))
cnn_model.add(AveragePooling2D(pool_size = (2,2), strides = (1,1), padding = 'valid'))

#flatten the network
cnn_model.add(Flatten())

#fully connected layer
cnn_model.add(Dense(units = 120, activation = 'relu'))
#fully connected layer
cnn_model.add(Dense(units = 84, activation = 'relu'))
#fully connected layer
cnn_model.add(Dense(units = 43, activation = 'softmax'))


# In[28]:


#complie the model
cnn_model.compile(loss = 'sparse_categorical_crossentropy', optimizer = Adam(learning_rate = 0.001), metrics = ['accuracy'])


# In[29]:


#train the model
history = cnn_model.fit(x_train_gray_norm, 
                        y_train, 
                        batch_size = 500, 
                        epochs = 50, 
                        verbose = 1, 
                        validation_data = (x_val_gray_norm, y_val))


# In[30]:


#model evaluation
score = cnn_model.evaluate(x_test_gray_norm, y_test, verbose =0)
print('Test Accuracy: {:.2f}%'.format(score[1]))


# In[31]:


history.history.keys()


# In[32]:


accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']


# In[33]:


epochs = range(len(accuracy))

plt.plot(epochs, accuracy, 'bo', label = 'Training Accuracy')
plt.plot(epochs, val_accuracy, 'b', label = 'Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()


# In[34]:


plt.plot(epochs, loss, 'ro', label = 'Training Loss')
plt.plot(epochs, val_loss, 'r', label = 'Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()


# In[41]:


predicted_classes = cnn_model.predict(x_test_gray_norm)
y_true = y_test
predicted_labels = predicted_classes.argmax(axis = 1)


# In[42]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, predicted_labels)
plt.figure(figsize = (30,30)) 
sns.heatmap(cm, annot = True)


# In[51]:


L = 7
W = 7

fix, axes = plt.subplots(L, W, figsize = (12,12))
axes = axes.ravel()

for i in np.arange(0,L*W):
    axes[i].imshow(x_test[i])
    axes[i].set_title('True = {}'.format(y_true[i]))
    axes[i].axis('off')
    
plt.subplots_adjust(wspace = 1)


# In[ ]:




