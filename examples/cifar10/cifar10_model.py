import numpy as np
import lasagne as nn

from lasagne.layers import InputLayer, DenseLayer
from lasagne.nonlinearities import softmax

from sklearn.metrics import log_loss

from nolearn.lasagne import NeuralNet
from nolearn.lasagne.handlers import SaveWeights

from nolearn_utils.iterators import BatchIterator

from nolearn_utils.transformers import (
    RandomHorizontalFlipTransformer,
    RandomAffineTransformer
)

from nolearn_utils.handlers import (
    SaveTrainingHistory,
    PlotTrainingHistory,
    EarlyStopping
)


from nolearn_utils.layer_marco import (
    vgg
)


image_size = 32
batch_size = 1024
n_classes = 10


model_weights_fname = 'examples/cifar10/weights.pkl'
model_history_fname = 'examples/cifar10/history.csv'
model_graph_fname = 'examples/cifar10/history.png'


def get_conv_kwargs():
    leaky_alpha = 1 / 3.0
    glorot_gain = np.sqrt(2 / (1 + leaky_alpha ** 2))
    nonlinearity = nn.nonlinearities.LeakyRectify(leaky_alpha)
    W = nn.init.GlorotUniform(glorot_gain)

    return dict(
        W=W, nonlinearity=nonlinearity, pad='same'
    )


def get_layers():
    l = InputLayer(name='input', shape=(None, 3, image_size, image_size))
    # 32x32

    l = vgg(
        l, name='1', num_layers=3,
        num_filters=32, filter_size=3, downsample='maxpool',
        drop_p=0.1, bn=True,
        **get_conv_kwargs()
    )
    # 16x16

    l = vgg(
        l, name='2', num_layers=3,
        num_filters=64, filter_size=3, downsample='maxpool',
        drop_p=0.2, bn=True,
        **get_conv_kwargs()
    )
    # 8x8

    l = vgg(
        l, name='3', num_layers=3,
        num_filters=128, filter_size=3, downsample='maxpool',
        drop_p=0.3, bn=True,
        **get_conv_kwargs()
    )
    # 4x4

    l = nn.layers.DropoutLayer(l, name='gpdrop', p=0.5)
    l = nn.layers.GlobalPoolLayer(l, name='gp')

    l = DenseLayer(l, name='out', num_units=n_classes, nonlinearity=softmax)
    return l


def get_iterators():
    train_iterator = BatchIterator(
        batch_size=batch_size,
        worker_batch_size=batch_size // 2, n_workers=8,
        shuffle=True
    )

    train_iterator.add_transformer(
        RandomHorizontalFlipTransformer(p=0.5)
    )

    train_iterator.add_transformer(RandomAffineTransformer(
        p=1.0,
        scale=np.arange(-0.8, 1.25, 0.05),
        rotation=np.arange(-10, 11, 1),
        shear=np.arange(-0.1, 0.11, 0.01),
        translation_y=np.arange(-5, 6, 1),
        translation_x=np.arange(-5, 6, 1)
    ))

    test_iterator = BatchIterator(
        batch_size=batch_size,
        n_workers=1,
        shuffle=False
    )

    return train_iterator, test_iterator


def get_net(layers, train_iterator, test_iterator):
    save_weights = SaveWeights(
        model_weights_fname, only_best=True, pickle=False, verbose=True
    )
    save_training_history = SaveTrainingHistory(
        model_history_fname, output_format='csv'
    )
    plot_training_history = PlotTrainingHistory(model_graph_fname)
    early_stopping = EarlyStopping(patience=100)

    net = NeuralNet(
        layers=layers,

        regression=False,

        objective_loss_function=nn.objectives.categorical_crossentropy,
        objective_l2=1e-6,

        update=nn.updates.adam,
        update_learning_rate=1e-3,

        batch_iterator_train=train_iterator,
        batch_iterator_test=test_iterator,

        on_epoch_finished=[
            save_weights,
            save_training_history,
            plot_training_history,
            early_stopping,
        ],

        verbose=10,

        custom_scores=[
            ('logloss', log_loss)
        ],

        max_epochs=1000,
    )

    return net
