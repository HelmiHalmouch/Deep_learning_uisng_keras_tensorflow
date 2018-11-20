#import cifar10 dataset from keras 
from keras.datasets import cifar10
from keras.utils import np_utils
import h5py 

#create label from cifar dataset (look the wibsite of CIFAR ) 
labels  =['airplane', 'automobile','bird','cat', 'deer', 'dog', 'frog','horse','ship','truck'] 

#load dataset from cifar10
(X_train, y_train),(X_test, y_test) = cifar10.load_data()

#building training data 
new_X_train = X_train.astype('float32')
new_X_test = X_test.astype('float32')

#normalisation of the data between 0 and 1
new_X_train /=255
new_X_test /=255

#convert the lablet y_train and ytest in category 
new_Y_train = np_utils.to_categorical(y_train)
new_Y_test = np_utils.to_categorical(y_test)



#here we use a sequential model to build the model layer by layer 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten 
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.constraints import maxnorm

#start building the model 

model = Sequential()
'''
In Conv2D function :
32 because the size of  bdd image is 32
(3,3) is the size of the filter in the convolution
input_shape=(32,32,3) : here because 32 is the size of image 
and 3 is relative the the thre images Red, Green ,Bluein fact 
red, green, bleu = image.split()  
'''
#keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
model.add(Conv2D(32,(3,3),input_shape=(32,32,3), activation ='relu',padding='same',kernel_constraint=maxnorm(3))) 
model.add(MaxPooling2D(pool_size=(2,2)))
# befor adding layer we need to flatting the model 
model.add(Flatten())
#now we can add the danse layer 
model.add(Dense(512, activation='relu', kernel_constraint= maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))

#compile the mode 
model.compile(loss ='categorical_crossentropy', optimizer=SGD(lr=0.01), metrics =['accuracy'])


# train the model using model.fit with 10 epochs and 32 batch_size
model.fit(new_X_train, new_Y_train, epochs=10, batch_size=32)

#you need to save the trained model into a file using h5py
model.save('Trained_model.h5')

