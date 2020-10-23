import os, sys, cv2, tqdm
import numpy as np, tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers     import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers     import Dropout, Flatten, Dense, Input, Reshape
from tensorflow.keras.models     import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks  import ModelCheckpoint
from tensorflow.keras.utils      import to_categorical
from tensorflow.keras.preprocessing         import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input


class CNN():

    def __init__(self, input_shape, output_shape, load_path=None):

        self.model = None
        if load_path is not None:
            self.model = load_model(load_path)
        else:
            self.model =  self.build_classifier(input_shape, output_shape)

    def build_classifier(self, input_shape, output_shape, dropout=.8, base_nodes=512):

        base_model = ResNet50(weights='imagenet', include_top=False)

        net = base_model.output
        net = GlobalAveragePooling2D(input_shape=input_shape)(net)
        predictions = Dense(output_shape, activation='softmax')(net)

        model = Model(inputs=base_model.input, outputs=predictions)

        # i.e. freeze all convolutional ResNet50 layers
        for layer in base_model.layers:
            layer.trainable = False

        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

        return model

    def train(self, train_tensors, target_tensors, valid_train_tensors, valid_target_tensors, batch_size=32, epochs=100):

        datagen = image.ImageDataGenerator(
            width_shift_range=0.1,  # randomly shift images horizontally (10% of total width)
            height_shift_range=0.1, # randomly shift images vertically (10% of total height)
            horizontal_flip=True)   # randomly flip images horizontally

        # fit augmented image generator on data
        datagen.fit(train_tensors)

        checkpointer = ModelCheckpoint(filepath='weights.hdf5', verbose=1, save_best_only=True)

        self.model.fit_generator(datagen.flow(train_tensors, target_tensors, batch_size=batch_size),
                                validation_data=(valid_train_tensors, valid_target_tensors), 
                                steps_per_epoch=train_tensors.shape[0] // batch_size,
                                epochs=epochs, callbacks=[checkpointer], verbose=1)

    def predict(self, img_tensor):

        #img_tensor = image.img_to_array(img)
        # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
        img_tensor = np.expand_dims(img_tensor, axis=0)
        return self.model.predict(img_tensor)

def demo(limit=1000):

    from keras.datasets  import cifar10

    '''
        cifar10 categories:
        0: airplane
        1: automobile
        2: bird
        3: cat
        4: deer
        5: dog
        6: frog
        7: horse
        8: ship
        9: truck
    '''

    # x_train, x_test: uint8 array of RGB image data with shape (num_samples, 3, 32, 32)
    # y_train, y_test: uint8 array of category labels (integers in range 0-9) with shape (num_samples,)
    print('loading cifar images')
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = x_train[:limit]
    y_train = y_train[:limit]

    y_train = to_categorical(y_train, 10)
    y_test  = to_categorical(y_test,  10)

    print('upscaling images')
    x_train = np.array([cv2.resize(x, (0,0), fx=7, fy=7) for x in x_train])
    x_test  = np.array([cv2.resize(x, (0,0), fx=7, fy=7) for x in x_test])

    input_shape  = x_train.shape
    output_shape = y_train.shape

    print('Building CNN')
    load_path = None
    if os.path.isfile('./weights.hdf5'):
        load_path = './weights.hdf5'
    cnn = CNN(input_shape[-3:], output_shape[-1:][0], load_path=load_path)

    if not load_path:
        print('Training CNN')
        cnn.train(x_train, y_train, x_test, y_test, batch_size=32, epochs=100)

    print('showing off')
    categories = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    for _ in range(10):
        test_image = x_train[np.random.randint(1000)]
        image.array_to_img(test_image).show()

        prediction = cnn.predict(test_image)
        print('predicted the photo was a {} with {:2f}% certainty'.format(categories[np.argmax(prediction)], (np.amax(prediction)/np.sum(prediction))*100))

    return cnn

if __name__ == "__main__":

    trained_cnn = demo(limit=10000)

    valid_photo_extensions = ['.png', 'jpeg', '.jpg', '.gif', '.bmp'] #quick hack way to check if file is photo
    valid_path = False

    while not valid_path:
        # Ask user for image path
        img_path = input("Please provide a path to an image that you would like to classify: ")

        if img_path == 'exit':
            sys.exit(1)

        if not os.path.isfile(img_path) or not img_path[-4:] in valid_photo_extensions:
            print("Image path invalid. Make sure to include a path to a valid image file (png, jpg, gif)")

        img = cv2.imread(img_path)
        img = cv2.resize(img, (32,32))
        img = cv2.resize(img, (0,0), fx=7, fy=7)

        categories = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        prediction = trained_cnn.predict(img)
        print('predicted the photo was a {} with {:2f}% certainty'.format(categories[np.argmax(prediction)], (np.amax(prediction)/np.sum(prediction))*100))
