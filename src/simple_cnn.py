from keras.layers import Activation, Convolution2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential, load_model
from keras.applications.xception import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.utils import plot_model
import os
from glob import glob


def create_model(input_size, n_categories):
    """
    Create a simple baseline CNN

    Args:
        input_size (tuple(int, int, int)): 3-dimensional size of input to model
        n_categories (int): number of classification categories

    Returns:
        keras Sequential model: model with new head
        """

    nb_filters = 32
    kernel_size = (3, 3)
    pool_size = (2, 2)

    model = Sequential()
    # 2 convolutional layers followed by a pooling layer followed by dropout
    model.add(Convolution2D(48, (11,11),
                            padding='valid',
                            input_shape=input_size))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, kernel_size))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3,3)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, kernel_size))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3,3)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(nb_filters, kernel_size))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))

    model.add(Convolution2D(192, (3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_categories))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
    return model


if __name__ == '__main__':
    img_size = (200,200,3)
    target_size = (img_size[0],img_size[1])
    batch_size = 16
    num_epochs = 25
    train_path = '../data/train'
    validation_path = '../data/test'
    holdout_path = '../data/validation'
    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    holdout_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    train_generator = train_datagen.flow_from_directory(train_path,
                                                        target_size=target_size,
                                                        batch_size=batch_size)
    validation_generator = validation_datagen.flow_from_directory(validation_path,
                                                        target_size=target_size,
                                                        batch_size=batch_size)
    holdout_generator = holdout_datagen.flow_from_directory(holdout_path,
                                                            target_size=target_size,
                                                            batch_size=batch_size)
    model = create_model(img_size,7)

    # mcCallBack = ModelCheckpoint(filepath='./logs',monitor="val_loss",save_best_only=True)
    tbCallBack = TensorBoard(log_dir='../logs',
                            histogram_freq=0,
                            write_graph=True,
                            write_grads=True,
                            write_images=True)
    model.fit_generator(train_generator,
                        steps_per_epoch=15,
                        epochs=num_epochs,
                        validation_data=validation_generator,
                        validation_steps=5,
                        callbacks=[tbCallBack])
    #model.save('./best_model.h5')
    #best_model = load_model('./best_model.h5')
    #metrics = best_model.evaluate_generator(holdout_generator, steps=20)
    metrics =  model.evaluate_generator(holdout_generator, steps=20)
    print("loss: {}, accuracy: {}".format(metrics[0],metrics[1]))
    # plot_model(model,to_file='../images/cnn_arch.png')
