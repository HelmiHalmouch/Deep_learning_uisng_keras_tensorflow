from PIL import Image 
from matplotlib import pyplot as plt 

#import cifar10 dataset from keras 
from keras.datasets import cifar10

#create label from cifar dataset (look the wibsite of CIFAR ) 
labels  =['airplane', 'automobile','bird','cat', 'deer', 'dog', 'frog','horse','ship','truck'] 

#load dataset from cifar10
(X_train, y_train),(X_test, y_test) = cifar10.load_data()

#display an image with index 5 for exemple 

index = 5
display_image = X_train[index]
display_label = y_train[index][0]

#show the image with matplotlib 
plt.imshow(display_image)
plt.show()

"""display_image_pathname= input('Enter image name :')
display_image = Image.open(display_image_pathname)
display_image.show()"""