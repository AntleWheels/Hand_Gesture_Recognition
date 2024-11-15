from keras import models
from keras import layers
from keras import callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model = models.Sequential() 

#First Convolution Layer
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 1))) #32 neurons , 3x3 kernel ,activation function relu with the input shap of 150*150*1 and this one represents the grayscale image
model.add(layers.MaxPooling2D((2, 2))) #  Here we take the maximum value from 2x2 kernel

#Second Convolution Layer 
model.add(layers.Conv2D(64, (3, 3), activation='relu')) #64 Neurons
model.add(layers.MaxPooling2D((2, 2)))

#Third Convolution Layer    
model.add(layers.Conv2D(128, (3, 3), activation='relu')) #128 Neurons
model.add(layers.MaxPooling2D((2, 2)))

#Fourth Convolution Layer
model.add(layers.Conv2D(256, (3, 3), activation='relu')) #256 Neurons
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten()) #It is used to flatten the array

model.add(layers.Dense(units = 150 ,activation='relu')) #150 Neurons
model.add(layers.Dropout(0.25)) #Dropout is used to avoid overfitting

#Output Layer 
model.add(layers.Dense(units=6, activation ='softmax')) #Softmax is used to get the probability of each class ,We use Softmax since we have 6 classes

#Compile the model
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy']) #we use the Categorical Cross Entropy loss function since we use the Softmax activation function

#train the model
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range =12.,
    width_shift_range=0.2,
    height_shift_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
)

#Validate the datagen
val_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    'HandGestureDataset/train',
    target_size=(256, 256),
    color_mode ='grayscale',
    batch_size =8,
    classes =['NONE','ONE','TWO','THREE','FOUR','FIVE'],
    class_mode ='categorical',
)
val_set = val_datagen.flow_from_directory(
    'HandGestureDataset/train',
    target_size=(256, 256),
    color_mode ='grayscale',
    batch_size =8,
    classes =['NONE','ONE','TWO','THREE','FOUR','FIVE'],
    class_mode ='categorical',
)
 
callback_list =[ #Check the validation lose If there is no improvement then stops(monitor='val_loss')
    callbacks.EarlyStopping(monitor='val_loss',patience=10),
    callbacks.ModelCheckpoint('model.keras',monitor='val_loss',save_best_only=True, verbose=1)
]

steps_per_epoch = len(training_set) // training_set.batch_size
validation_steps = len(val_set) // val_set.batch_size
#Starting the training
model.fit(
    training_set,
    steps_per_epoch = steps_per_epoch, #batch size
    epochs = 50,
    validation_data = val_set,
    validation_steps = validation_steps,
    callbacks = callback_list
)

#Save the model
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.weights.h5")  
print('Saved model to disk')