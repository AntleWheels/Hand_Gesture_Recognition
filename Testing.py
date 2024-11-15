from keras import preprocessing
from keras import models
import numpy as np

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = models.model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.keras")
print("Loaded model from disk")

def classify(img_file):
    img_name = img_file
    test_image = preprocessing.image.load_img(img_name, target_size=(256, 256), color_mode ='grayscale')
    test_image = preprocessing.image.img_to_array(test_image) #Converting the image to an array
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict(test_image)
    arr =np.array(result[0])
    print('Array',arr)
    max =np.amax(arr)
    max_prob = np.argmax(arr,axis=0) #getting the max probability
    max_prob =max_prob+1
    classes =['NONE','ONE','TWO','THREE','FOUR','FIVE']
    result = classes[max_prob - 1]
    print("Img_name :",img_name,"Predicted Gesture:",result)

import os
path = 'D:\Hand-Gesture-Recognition\check'
files =[]
# r =root ,f=files ,d=directories
for r, d, f in os.walk(path):
    for file in f:
        if '.png' in file:
            files.append(os.path.join(r, file))
for f in files:
    classify(f) 
