from signal import pthread_kill
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def Model():
    inputs1 = tf.keras.Input(shape=(64, 64, 1), name='IMAGE')
    inputs1 = tf.keras.layers.Lambda(lambda x: tf.multiply(tf.subtract(x, 128) / 255, 10))(inputs1)

    inputs2 = tf.keras.Input(shape=(1), name='QP')
    inputs2 = tf.keras.layers.Lambda(lambda x: (x*0.18)/51)(inputs2)

    conv1_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(inputs1)
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
    mid = tf.keras.layers.concatenate([mid, inputs2], axis=1)

    mid = tf.keras.layers.Dense(units=64, activation='relu', kernel_regularizer='l2', bias_regularizer='l2')(mid)
    mid = tf.keras.layers.Dense(units=16, activation='relu', kernel_regularizer='l2', bias_regularizer='l2')(mid)
    # mid = tf.keras.layers.Dropout(rate=0.5)(mid)

    mid1 = tf.keras.layers.Dense(units=64, activation='relu')(mid)

    outputs16 = tf.keras.layers.Dense(16, activation='sigmoid')(mid1)
    outputs16 = tf.keras.layers.Reshape((4, 4, 1))(outputs16)

    model = tf.keras.Model(inputs=[inputs1, inputs2], outputs=[outputs16])
    model.compile(optimizer='adam', loss=['binary_crossentropy'], metrics=['accuracy'])

    # model = tf.keras.Model(inputs=[inputs1,inputs2], outputs=outputs64)
    # model.compile(optimizer='adam', loss=['binary_crossentropy'], metrics=['accuracy'])
    return model


if __name__ == '__main__':
    model = Model()
    model.summary()
    tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True)

    dat = [tf.ones((3, 64, 64, 1)), tf.ones(3, 1)]
    out = model(dat)
    print(out)
