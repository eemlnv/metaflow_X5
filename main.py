from metaflow import FlowSpec, step, conda, S3, conda_base,\
                     resources, Flow, project, Parameter

from augment import *
from config import *
from model import *

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

@conda_base(libraries={'tensorflow': '2.4.0', 'imgaug': '0.4.0', 'numpy': '1.19.5'}, python='3.8')
class RandAugmentFlow(FlowSpec):
    @step
    def start(self):
        import tensorflow as tf
        tf.random.set_seed(42)
        import numpy as np

        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        print(f"Total training examples: {len(x_train)}")
        print(f"Total test examples: {len(x_test)}")

        self.x_train, self.y_train = x_train.astype("float32"), np.squeeze(y_train)
        self.x_test, self.y_test = x_test.astype("float32"), np.squeeze(y_test)

        get_training_model().summary()

        self.next(self.a, self.b)

    @step
    def a(self):
        import tensorflow as tf
        tf.random.set_seed(42)
        import numpy as np
        import matplotlib.pyplot as plt


        train_ds_rand = (
            tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
                .shuffle(BATCH_SIZE * 100)
                .batch(BATCH_SIZE)
                .map(
                lambda x, y: (tf.image.resize(x, (IMAGE_SIZE, IMAGE_SIZE)), y),
                num_parallel_calls=AUTO,
            )
                .map(
                lambda x, y: (tf.py_function(augment, [x], [tf.float32])[0], y),
                num_parallel_calls=AUTO,
            )
                .prefetch(AUTO)
        )

        sample_images, _ = next(iter(train_ds_rand))
        plt.figure(figsize=(10, 10))
        for i, image in enumerate(sample_images[:9]):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(image.numpy().astype("int"))
            plt.axis("off")
        # plt.show()

        test_ds = (
            tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test))
                .batch(BATCH_SIZE)
                .map(
                lambda x, y: (tf.image.resize(x, (IMAGE_SIZE, IMAGE_SIZE)), y),
                num_parallel_calls=AUTO,
            )
                .prefetch(AUTO)
        )

        rand_aug_model = get_training_model()
        rand_aug_model.load_weights("initial_weights.h5")
        rand_aug_model.compile(
            loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        rand_aug_model.fit(train_ds_rand, validation_data=test_ds, epochs=EPOCHS)
        _, self.test_acc = rand_aug_model.evaluate(test_ds)

        self.next(self.join)

    # @conda({'tensorflow': '2.4'})
    @step
    def b(self):
        import tensorflow as tf
        tf.random.set_seed(42)
        import numpy as np
        import matplotlib.pyplot as plt

        train_ds_simple = (
            tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
                .shuffle(BATCH_SIZE * 100)
                .batch(BATCH_SIZE)
                .map(lambda x, y: (simple_aug(x), y), num_parallel_calls=AUTO)
                .prefetch(AUTO)
        )

        sample_images, _ = next(iter(train_ds_simple))
        plt.figure(figsize=(10, 10))
        for i, image in enumerate(sample_images[:9]):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(image.numpy().astype("int"))
            plt.axis("off")
        # plt.show()

        test_ds = (
            tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test))
                .batch(BATCH_SIZE)
                .map(
                lambda x, y: (tf.image.resize(x, (IMAGE_SIZE, IMAGE_SIZE)), y),
                num_parallel_calls=AUTO,
            )
                .prefetch(AUTO)
        )

        simple_aug_model = get_training_model()
        simple_aug_model.load_weights("initial_weights.h5")
        simple_aug_model.compile(
            loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        simple_aug_model.fit(train_ds_simple, validation_data=test_ds, epochs=EPOCHS)
        _, self.test_acc = simple_aug_model.evaluate(test_ds)

        self.next(self.join)

    # @conda(disabled=True)
    @step
    def join(self, inputs):
        print("Test accuracy with RandAugment: {:.2f}%".format(inputs.a.test_acc * 100))
        print("Test accuracy with simple_aug: {:.2f}%".format(inputs.b.test_acc * 100))
        self.next(self.end)

    # @conda(disabled=True)
    @step
    def end(self):
        pass

if __name__ == '__main__':
    RandAugmentFlow()