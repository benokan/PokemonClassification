import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from keras.models import load_model

log_list = [path for path in Path('histories').rglob('*.log')]


# print(log_list)


def plot_from_log(path, loss_or_acc):
    print("Opening -> ", path)

    df = pd.read_csv(path)

    textstrAcc = '\n'.join((
        r'Max Acc=%.2f' % (df['accuracy'].max(),),
        r'Max Val Acc=%.2f' % (df['val_accuracy'].max(),)
    ))

    textstrLoss = '\n'.join((
        r'Min Loss=%.2f' % (df['loss'].min(),),
        r'Min Val Loss=%.2f' % (df['val_loss'].min(),)
    ))

    if loss_or_acc == 'Accuracy':
        plt.plot(df['epoch'], df['accuracy'])
        plt.plot(df['epoch'], df['val_accuracy'])
        plt.gcf().text(0.1, 0.87, textstrAcc, fontsize=14)
        plt.show()
    elif loss_or_acc == 'Loss':
        plt.plot(df['epoch'], df['loss'])
        plt.plot(df['epoch'], df['val_loss'])
        plt.gcf().text(0.1, 0.87, textstrLoss, fontsize=14)
        plt.show()
    else:
        print("Pass 'Accuracy' or 'Loss' ")


plot_from_log(log_list[-4], 'Accuracy')


# loaded_model = load_model('histories/AugOnFly.h5')
# loaded_model.summary()
