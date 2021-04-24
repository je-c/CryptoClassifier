import time, itertools
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix
from torchvision.utils import make_grid

def cm_plot(cm, classes, normalise=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """ 
    Outputs a confusion matrix based on model predictions
        * :param cm(np.ndarray): Confusion matrix
        * :param classes(list): List of class names
        * :param normalise(bool): Output proportional confusion matrix (optional) 
        * :param title(str): Plot title
        * :param cmap(plt.cm): Colour mapping for confusion matrix

    :return (NoneType): None
    """
    if normalise: cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j, i, 
            format(cm[i, j], fmt), 
            horizontalalignment="center", 
            color="white" if cm[i, j] > thresh else "black"
        )
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.tight_layout()

def accuracy_plot(history):
    """ 
    Outputs an accuracy plot based on accuracies obvserved in training
        * :param history(list): Epoch accuracies from training loop

    :return (NoneType): None
    """
    accuracies = [epoch['val_acc'] for epoch in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs')
  
def loss_plot(history):
    """ 
    Outputs a loss plot based on losses observed in training
        * :param history(list): Epoch losses from training loop

    :return (NoneType): None
    """
    training_losses = [epoch.get('train_loss') for epoch in history]
    validation_losses = [epoch['val_loss'] for epoch in history]
    plt.plot(training_losses, '-bx')
    plt.plot(validation_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs')

def learningRate_plot(history):
    """ 
    Outputs a learning rate plot based on learning rates selected in training
        * :param history(list): Epoch learning rates from training loop

    :return (NoneType): None
    """
    learningRates = np.concatenate([epoch.get('lrs', []) for epoch in history])
    plt.plot(learningRates)
    plt.xlabel('Batch no.')
    plt.ylabel('Learning rate')
    plt.title('Learning Rate vs. Batch no.')

def show_batch(dataLoader, rows):
    """ 
    Displays a single batch of data from the dataloader as a tile of images
        * :param dataLoader(torch.dataloader): Pytorch dataloader
        * :param rows(int): x-dimension of tile plot

    :return (NoneType): None
    """
    for images, labels in dataLoader:
        _, ax = plt.subplots(figsize=(12, 12))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images[:64], nrow=rows).permute(1, 2, 0))
        print(labels)
        break

def window_eda(data, columns, ext=1000):
    """ 
    Used in EDA to determine appropriate lag windows in auto-labelling of data, uses plotly interactive
    plotting methods
        * :param data(pd.DataFrame): Raw data
        * :param columns(list): Column names
        * :param ext(int): Examination window (optional)

    :return (NoneType): None
    """
    fig = make_subplots(specs=[[{'secondary_y': True}]])
    for col in columns:
        if col == 'close':
            fig.add_trace(
                go.Scatter(
                    x=data.index[-ext:],
                    y=data[col][-ext:],
                    line=dict(
                            shape='spline',
                            width=1.5
                    )
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=data.index[-ext:],
                    y=data[col][-ext:],
                    line=dict(
                            shape='spline',
                            width=1.5
                    )
                ),
                secondary_y=True
            )

    fig.show()

def performance_visualiser(targetLabels, predsMade, history, classNames):
    """ 
    Multi-plot performance summary
        * :param targetLabels(list): Ground truths
        * :param predsMade(list): Predicted labels
        * :param history(list): Stored learning rates/accuracies from model training
        * :param classNames(list): List of class names

    :return (NoneType): None
    """
    targetLabels = [x.item() for x in targetLabels]
    predsMade = [x.item() for x in predsMade]
    cm = confusion_matrix(targetLabels, predsMade)

    cm_plot(cm, classNames, normalize = True)
    loss_plot(history)
    learningRate_plot(history)
    accuracy_plot(history)
  