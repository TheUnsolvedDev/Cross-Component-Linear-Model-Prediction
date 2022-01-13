from signal import pthread_kill
import tensorflow as tf


def model():
    inputs = tf.keras.Input(shape=(64, 64, 1))
    inputs = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

    conv1_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(inputs)
    maxpool1_1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv1_1)

    conv1_2 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(maxpool1_1)
    maxpool1_2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv1_2)

    conv1_3 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(maxpool1_2)
    maxpool1_3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv1_3)

    conv2_1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(conv1_1)
    maxpool2_2 = tf.keras.layers.MaxPool2D(pool_size=(4, 4))(conv2_1)

    conv2_2 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(maxpool2_2)
    maxpool2_2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv2_2)

    conv3_1 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(conv1_1)
    maxpool3_1 = tf.keras.layers.MaxPool2D(pool_size=(8, 8))(conv3_1)

    cat = tf.keras.layers.concatenate([maxpool1_3, maxpool2_2, maxpool3_1])

    mid = tf.keras.layers.Flatten()(cat)
    
    mid = tf.keras.layers.Dense(units=16, activation='relu')(mid)
    mid = tf.keras.layers.Dropout(rate=0.5)(mid)
    outputs = tf.keras.layers.Dense(16, activation='sigmoid')(mid)
    # outputs = tf.keras.layers.Reshape((4, 4, 1))(outputs)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    model = model()
    model.summary()
    tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True)
    
    dat = tf.ones((1, 64, 64, 1))
    out = model(dat)
    print(out.shape)
    print(out)
