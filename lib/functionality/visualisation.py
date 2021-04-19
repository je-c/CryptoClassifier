import time
import pandas as pd
import numpy as np
import datetime as dt
import itertools
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix
from torchvision.utils import make_grid

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_accuracy(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs')
  
def plot_loss(history):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs')

def plot_learningRates(history):
    lrs = np.concatenate([x.get('lrs', []) for x in history])
    plt.plot(lrs)
    plt.xlabel('Batch no.')
    plt.ylabel('Learning rate')
    plt.title('Learning Rate vs. Batch no.')
  
def normed_cm(cm, labels):
  prc_cm = []
  for i in [0, 1, 2]:
    row = []
    for j in [0, 1, 2]:
      row.append(cm[i][j]/len(labels))
    prc_cm.append(row)

  return np.array(prc_cm)

def show_batch(dl, rows):
    for images, labels in dl:
        _, ax = plt.subplots(figsize=(12, 12))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(images[:64], nrow=rows).permute(1, 2, 0))
        print(labels)
        break

def window_eda(data, columns, ext):
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
    targetLabels = [x.item() for x in targetLabels]
    predsMade = [x.item() for x in predsMade]
    cm = confusion_matrix(targetLabels, predsMade)

    plot_confusion_matrix(cm, classNames, normalize = True)
    plot_losses(history)
    plot_lrs(history)
    plot_accuracies(history)