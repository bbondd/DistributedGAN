import os
import numpy as np
from keras.preprocessing import image
import keras as k
import ray


class Data(object):
    def load_kejang(self):
        real_data = np.load('./kejang.npy')
        real_data = real_data.astype('float32')
        real_data = np.mean(real_data, axis=3)
        real_data = np.expand_dims(real_data, axis=3) / 255

        self.real_data = real_data

    def load_mnist(self):
        (real_data, _), (_, _) = k.datasets.mnist.load_data()
        real_data = real_data[:2000].astype('float32')
        real_data = np.expand_dims(real_data, axis=3) / 255

        self.real_data = real_data

    def __init__(self):
        self.real_data = None
        self.load_mnist()


class C(object):
    @staticmethod
    def random_layer(last_output, filter_size):
        layer = np.random.choice([k.layers.Conv2D, k.layers.DepthwiseConv2D, k.layers.Conv2DTranspose],
                                 p=[0.25, 0.25, 0.5])

        hyper_parameters = {
            'filters': filter_size,
            'kernel_size': (lambda x: [x, x])(np.random.randint(1, 4)),
            'strides': (lambda x: [x, x])(np.random.randint(1, 4)),
            'padding': 'valid',
            'dilation_rate': (lambda x: [x, x])(np.random.randint(1, 4)),
            'activation': np.random.choice(['relu']),
        }

        if layer == k.layers.Conv2D:
            hyper_parameters[np.random.choice(['dilation_rate'])] = [1, 1]
            hyper_parameters['padding'] = 'same'

        elif layer == k.layers.DepthwiseConv2D:
            hyper_parameters[np.random.choice(['dilation_rate'])] = [1, 1]
            hyper_parameters['padding'] = 'same'
            hyper_parameters.pop('filters')

        elif layer == k.layers.Conv2DTranspose:
            if last_output.shape[1] > C.image_width:
                hyper_parameters['strides'] = [1, 1]
            else:
                hyper_parameters['strides'] = [2, 2]
                hyper_parameters['dilation_rate'] = [1, 1]

        while True:
            try:
                layer = layer(**hyper_parameters)(last_output)
                break
            except ValueError:
                last_output = k.layers.ZeroPadding2D(data_format='channels_last')(last_output)

        layer = k.layers.BatchNormalization()(layer)
        return layer

    @staticmethod
    def make_fixed_discriminator():
        model_input = model_output = k.Input(shape=[C.image_width, C.image_width, 1])
        for i in range(4):
            model_output = k.layers.Conv2D(filters=(2 ** i) * 32,
                                           kernel_size=[5, 5],
                                           strides=[2, 2],
                                           padding='same',
                                           )(model_output)
            model_output = k.layers.BatchNormalization()(model_output)
            model_output = k.layers.LeakyReLU()(model_output)

        model_output = k.layers.Lambda(lambda x: x * 2)(model_output)

        model_output = k.layers.Flatten()(model_output)
        model_output = k.layers.Dense(units=1, activation='linear')(model_output)

        model = k.Model(inputs=model_input, outputs=model_output, name='discriminator')

        return model

    @staticmethod
    def make_random_discriminator():
        model_input = model_output = k.Input(shape=[C.image_width, C.image_width, 1])
        for i in range(C.discriminator_layer_size()):
            model_output = k.layers.Conv2D(filters=(2 ** i) * 32,
                                           kernel_size=[5, 5],
                                           strides=[2, 2],
                                           padding='same',
                                           )(model_output)
            model_output = k.layers.BatchNormalization()(model_output)
            model_output = k.layers.LeakyReLU()(model_output)

        model_output = k.layers.Lambda(lambda x: x * 2)(model_output)

        model_output = k.layers.Flatten()(model_output)
        model_output = k.layers.Dense(units=1, activation='linear')(model_output)

        model = k.Model(inputs=model_input, outputs=model_output, name='discriminator')

        return model

    @staticmethod
    def make_fixed_generator():
        model_output = model_input = k.Input(shape=[C.noise_dimension])
        model_output = k.layers.Dense(units=7 * 7 * 128)(model_output)
        model_output = k.layers.Reshape([7, 7, 128])(model_output)
        model_output = k.layers.Conv2DTranspose(filters=64,
                                                kernel_size=[5, 5],
                                                strides=[2, 2],
                                                padding='same',
                                                activation='relu'
                                                )(model_output)
        model_output = k.layers.BatchNormalization()(model_output)
        model_output = k.layers.Conv2DTranspose(filters=32,
                                                kernel_size=[5, 5],
                                                strides=[2, 2],
                                                padding='same',
                                                activation='relu'
                                                )(model_output)
        model_output = k.layers.BatchNormalization()(model_output)
        model_output = k.layers.Conv2DTranspose(filters=1,
                                                kernel_size=[5, 5],
                                                strides=[1, 1],
                                                padding='same',
                                                activation='tanh'
                                                )(model_output)
        model_output = k.layers.Lambda(lambda x: (x + 1) / 2)(model_output)

        return k.Model(inputs=model_input, outputs=model_output)

    @staticmethod
    def make_random_generator():
        model_input = model_output = k.Input(shape=[C.noise_dimension])
        model_output = k.layers.Dense(C.first_convolution_shape[0]
                                      * C.first_convolution_shape[1]
                                      * C.first_convolution_shape[2])(model_output)
        model_output = k.layers.Reshape(C.first_convolution_shape)(model_output)

        for i in reversed(range(C.generator_layer_size())):
            model_output = C.random_layer(model_output, 2 ** (i + 2))

        output_difference = k.Model(inputs=model_input, outputs=model_output).output_shape[1] - C.image_width
        if output_difference >= 0:
            model_output = k.layers.Conv2D(
                filters=1,
                kernel_size=[output_difference + 1, output_difference + 1],
                activation='tanh',
            )(model_output)
        else:
            model_output = k.layers.Conv2DTranspose(
                filters=1,
                kernel_size=[-output_difference + 1, -output_difference + 1],
                activation='tanh',
            )(model_output)

        model_output = k.layers.Lambda(lambda x: (x + 1) / 2)(model_output)

        model = k.Model(inputs=model_input, outputs=model_output, name='generator')

        return model

    @staticmethod
    def save_generator(generator: k.Model, directory_path, iteration_number):
        generator_path = directory_path + '/generator'
        try:
            os.makedirs(generator_path)
        except FileExistsError:
            pass

        generator.save_weights(generator_path + '/weights.h5')
        with open(generator_path + '/architecture.json', 'w') as f:
            f.write(generator.to_json())
        k.utils.plot_model(generator, generator_path + '/graph.png', show_shapes=True)

        image_path = directory_path + '/image'

        noise = np.random.uniform(-1, 1, [C.generate_image_size, C.noise_dimension])
        image_arrays = generator.predict(x=noise) * 255.0

        try:
            os.makedirs(image_path)
        except FileExistsError:
            pass

        for i in range(len(image_arrays)):
            image.save_img(x=image_arrays[i],
                           path=image_path + '/iteration%d num%d.jpg' % (iteration_number, i))

    @staticmethod
    def save_discriminator(discriminator: k.Model, directory_path):
        discriminator_path = directory_path + '/discriminator'
        try:
            os.makedirs(discriminator_path)
        except FileExistsError:
            pass

        discriminator.save_weights(discriminator_path + '/weights.h5')
        with open(discriminator_path + '/architecture.json', 'w') as f:
            f.write(discriminator.to_json())
        k.utils.plot_model(discriminator, discriminator_path + '/graph.png', show_shapes=True)

    @staticmethod
    def generator_layer_size():
        return np.random.randint(1, 2)

    @staticmethod
    def discriminator_layer_size():
        return np.random.randint(1, 2)

    image_width = 28
    noise_dimension = 128
    first_convolution_shape = np.array([4, 4, 128])
    gan_size = 3
    batch_size = 64
    learning_rate = 0.003

    path = './results'
    generate_image_size = 10


@ray.remote(num_gpus=1)
class Gan(object):
    def __init__(self, real_data, generator, discriminator, learning_rate):
        self.real_data = real_data
        optimizer = k.optimizers.Adam(learning_rate)

        discriminator.trainable = True
        discriminator.compile(optimizer=optimizer, loss='mse')

        discriminator.trainable = False
        adversarial = k.Model(inputs=generator.input, outputs=discriminator(generator.output))
        adversarial.compile(optimizer=optimizer, loss='mse')

        self.discriminator = discriminator
        self.generator = generator
        self.adversarial = adversarial

    def train_on_batch_index(self, batch_index):
        noise = np.random.uniform(-1, 1, [len(batch_index), C.noise_dimension]).astype('float32')
        fake_data = self.generator.predict(noise)
        real_data = self.real_data[batch_index]

        self.adversarial.train_on_batch(noise, np.ones([len(batch_index), 1]))
        self.discriminator.train_on_batch(real_data, np.ones([len(batch_index), 1]))
        self.discriminator.train_on_batch(fake_data, np.zeros([len(batch_index), 1]))

    def get_generator_weights(self):
        return self.generator.get_weights()

    def set_generator_weights(self, weights):
        self.generator.set_weights(weights)

    def get_discriminator_weights(self):
        return self.discriminator.get_weights()

    def set_discriminator_weights(self, weights):
        self.discriminator.set_weights(weights)


class DistributedGan(object):
    def __init__(self, distribution_size, real_data, generator, discriminator):
        self.discriminator = discriminator
        self.generator = generator
        self.gans = [Gan.remote(real_data[np.random.choice(len(real_data), len(real_data))],
                         k.models.clone_model(generator),
                         k.models.clone_model(discriminator),
                         C.learning_rate * distribution_size) for _ in range(distribution_size)]

    def get_generator_weights(self):
        weights_set = ray.get([gan.get_generator_weights.remote() for gan in self.gans])
        return np.mean(weights_set, axis=0)

    def set_generator_weights(self, weights):
        self.generator.set_weights(weights)
        ray.get([gan.set_generator_weights.remote(weights) for gan in self.gans])

    def get_discriminator_weights(self):
        weights_set = ray.get([gan.get_discriminator_weights.remote() for gan in self.gans])
        return np.mean(weights_set, axis=0)

    def set_discriminator_weights(self, weights):
        self.discriminator.set_weights(weights)
        ray.get([gan.set_discriminator_weights.remote(weights) for gan in self.gans])

    def train_on_batch_index(self, batch_index):
        ray.get([gan.train_on_batch_index.remote(batch_index) for gan in self.gans])
        self.set_generator_weights(self.get_generator_weights())
        self.set_discriminator_weights(self.get_discriminator_weights())

    def get_generator(self):
        return self.generator

    def get_discriminator(self):
        return self.discriminator


class MultiDistributedGan(object):
    def __init__(self, real_data, distribution_size, generators, discriminators):
        self.same_generator_gans_group = [[] for _ in range(len(generators))]
        self.same_discriminator_gans_group = [[] for _ in range(len(discriminators))]
        self.gans = []

        for i in range(len(generators)):
            for j in range(len(discriminators)):
                distributed_gan = DistributedGan(distribution_size,
                                                 real_data[np.random.choice(len(real_data), len(real_data))],
                                                 k.models.clone_model(generators[i]),
                                                 k.models.clone_model(discriminators[j]))

                self.gans.append(distributed_gan)
                self.same_generator_gans_group[i].append(distributed_gan)
                self.same_discriminator_gans_group[j].append(distributed_gan)

    def get_generators_weights(self):
        generators_weights = []
        for same_generator_gans in self.same_generator_gans_group:
            weights_set = [gan.get_generator_weights() for gan in same_generator_gans]
            generators_weights.append(np.mean(weights_set, axis=0))

        return generators_weights

    def set_generators_weights(self, generators_weights):
        for same_generator_gans, generator_weights in zip(self.same_generator_gans_group, generators_weights):
            [gan.set_generator_weights(generator_weights) for gan in same_generator_gans]

    def get_discriminators_weights(self):
        discriminators_weights = []
        for same_discriminator_gans in self.same_discriminator_gans_group:
            weights_set = []
            for same_discriminator_gan in same_discriminator_gans:
                weights_set.append(same_discriminator_gan.get_discriminator_weights())
            discriminators_weights.append(np.mean(np.array(weights_set), axis=0))

        return discriminators_weights

    def set_discriminators_weights(self, discriminators_weights):
        for same_discriminator_gans, discriminator_weights\
                in zip(self.same_discriminator_gans_group, discriminators_weights):
            [gan.set_discriminator_weights(discriminator_weights) for gan in same_discriminator_gans]

    def train_on_batch_index(self, batch_index):
        [gan.train_on_batch_index(batch_index) for gan in self.gans]
        self.set_generators_weights(self.get_generators_weights())
        self.set_discriminators_weights(self.get_discriminators_weights())

    def save(self, iteration_number):
        for i in range(len(self.same_generator_gans_group)):
            generator = self.same_generator_gans_group[i][0].get_generator()
            C.save_generator(generator, C.path + '/generator %d' % i, iteration_number)

        for i in range(len(self.same_discriminator_gans_group)):
            discriminator = self.same_discriminator_gans_group[i][0].get_discriminator()
            C.save_discriminator(discriminator, C.path + '/discriminator %d' % i)


def main():
    print('redis server :')
    ray.init(input())

    real_data = Data().real_data
    print('use fixed generator and discriminator for test? [y/n]')
    if input() == 'y':
        make_generator_function = C.make_fixed_generator
        make_discriminator_function = C.make_fixed_discriminator
    else:
        make_generator_function = C.make_random_generator
        make_discriminator_function = C.make_random_discriminator

    print('generator size :')
    generator_size = int(input())

    print('discriminator size :')
    discriminator_size = int(input())

    print('distribution size :')
    distribution_size = int(input())

    print('iteration size :')
    iteration_size = int(input())

    generators = [make_generator_function() for _ in range(generator_size)]
    discriminators = [make_discriminator_function() for _ in range(discriminator_size)]

    multi_distributed_gan = MultiDistributedGan(real_data, distribution_size, generators, discriminators)

    for i in range(iteration_size):
        if i % 10 == 0:
            multi_distributed_gan.save(i)
            print('iteration', i)

        batch_indexes = np.array_split(np.random.permutation(len(real_data)), int(len(real_data) / C.batch_size))
        for batch_index in batch_indexes:
            multi_distributed_gan.train_on_batch_index(batch_index)


main()



"""
ray start --head --redis-port=6379
watch -d -n 0.5 nvidia-smi
10.128.15.211:6379
192.168.227.143:6379
"""