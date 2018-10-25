import os
import glob
from PIL import Image
import tensorflow as tf
import h5py
import numpy as np
from keras.applications import imagenet_utils
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from keras.layers import Dense, GlobalAveragePooling2D
from keras.callbacks import TensorBoard
from keras.models import Model, load_model
from keras.optimizers import Adam
from datetime import datetime

#model = VGG16()

#print(model.summary())

# load an image from file
#image = load_img('Data/mug.jpg', target_size=(224, 224))

# convert the image pixels to a numpy array
#image = img_to_array(image)

# reshape data for the model
#image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

# prepare the image for the VGG model
#image = preprocess_input(image)

# predict the probability across all output classes
#yhat = model.predict(image)

# convert the probabilities to class labels
#label = decode_predictions(yhat)
# retrieve the most likely result, e.g. highest probability
#label = label[0][0]
# print the classification
#print('%s (%.2f%%)' % (label[1], label[2]*100))

IMG_WIDTH  = 224
IMG_HEIGHT = 224

NUM_EPOCHS = 1
BATCH_SIZE = 1
BATCH_SIZE_VAL = 1
FC_LAYER_SIZE = 1024

MODEL_DIRECTORY = "VGG16-plant.h5"
LOG_DIR = "logs"

def get_nb_files(directory):
	if not os.path.exists(directory):
		return 0
	count = 0
	for r, dirs, files in os.walk(directory):
		for dr in dirs:
			count += len(glob.glob(os.path.join(r, dr + "/*")))
	return count

def add_new_last_layer(model, num_classes):
  x = model.output
  x = GlobalAveragePooling2D()(x)
  x = Dense(FC_LAYER_SIZE, activation='relu')(x)
  print(num_classes)
  predictions = Dense(num_classes, activation='softmax')(x)
  new_model = Model(input=model.input, output=predictions)
  return new_model

def train(train_dir, val_dir):
	
	num_train_samples = get_nb_files(train_dir)
	#print(num_train_samples)
	num_classes = len(glob.glob(train_dir + "/*"))
	#print(num_classes)
	num_val_samples = get_nb_files(val_dir)

	train_datagen =  ImageDataGenerator(
		preprocessing_function=preprocess_input,
		rotation_range=30,
		width_shift_range=0.2,
		height_shift_range=0.2,
		shear_range=0.2,
		zoom_range=0.2,
		horizontal_flip=True
	)

	test_datagen = ImageDataGenerator(
		preprocessing_function=preprocess_input,
		rotation_range=30,
		width_shift_range=0.2,
		height_shift_range=0.2,
		shear_range=0.2,
		zoom_range=0.2,
		horizontal_flip=True)

	train_generator = train_datagen.flow_from_directory(
		train_dir,
		target_size=(IMG_WIDTH, IMG_HEIGHT),
		batch_size=BATCH_SIZE,
	)

	validation_generator = test_datagen.flow_from_directory(
		val_dir,
		target_size=(IMG_WIDTH, IMG_HEIGHT),
		batch_size=BATCH_SIZE_VAL,
	)

	base_model = VGG16(weights='imagenet', include_top=False)
	for layer in base_model.layers:
		layer.trainable = False

	model = None

	if os.path.exists(MODEL_DIRECTORY):
		model = load_model(MODEL_DIRECTORY)
	else:
		model = add_new_last_layer(base_model, num_classes)
		model.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), loss='categorical_crossentropy', metrics=['accuracy'])

	# print(model)
	# model.load_weights(MODEL_DIRECTORY, by_name=False)
	# print(model)	

	tensorboard = TensorBoard(log_dir="{}/{}".format(LOG_DIR, datetime.now().strftime('%Y%m%d-%H%M%S')))

	history_transfer_learning = model.fit_generator(
		train_generator,
		nb_epoch=NUM_EPOCHS,
		samples_per_epoch=num_train_samples,
		callbacks=[tensorboard],
		validation_data=validation_generator,
		nb_val_samples=num_val_samples,
		class_weight='auto')

	model.save(MODEL_DIRECTORY)

	#image = load_img('Data/Ca (1).JPG', target_size=(224, 224))
	#image = img_to_array(image)

	#x = np.expand_dims(image, axis=0)
	#x = preprocess_input(x)
	#features = model.predict(x)
	#print(features)
	#print(np.argmax(features))
	#label = imagenet_utils.decode_predictions(features)

	#image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

	#image = preprocess_input(image)	

	#yhat = model.predict(image)
	#label = decode_predictions(yhat)
	#for cls in train_generator.class_indices:
	#	print(cls+": "+preds[0][training_generator.class_indices[cls]])

	#print(yhat)
	#label = label[0][0]
	#print('%s (%.2f%%)' % (label[1], label[2]*100))

def main():
	train_dir = "Data/Tanaman/"
	val_dir = "Data/Tanaman_Validation/"
	train(train_dir, val_dir)

if __name__=="__main__":
	main()