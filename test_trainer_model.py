from PIL import Image 
import numpy as np 
from keras.models import load_model 
from keras_sequential_ascii import sequential_model_to_ascii_printout

labels  =['airplane', 'automobile','bird','cat', 'deer', 'dog', 'frog','horse','ship','truck'] 

model = load_model('Trained_model.h5')


sequential_model_to_ascii_printout(model )

input_image = input('Enter image file pathname:')
input_image = Image.open(input_image)

#resize the input image
input_image = input_image.resize((32,32),resample=Image.LANCZOS)

#convert image ito array 
image_array = np.array(input_image)
image_array = image_array.astype('float32')
image_array /= 255.0
image_array = image_array.reshape(1, 32, 32, 3)

#preduct the image 

answer = model.predict(image_array)
input_image.show()
print(labels[np.argmax(answer)])

print(np.argmax(answer))


