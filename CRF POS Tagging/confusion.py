import sys
import itertools
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

real, predicted = [], []

def read_file(path: str, list):
    with open(path) as f:
        for line in f:
            line = line.strip().split("\t")[1].split();
            for word in line:
                list.append(word)
    f.close()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function is taken almost verbatim from scikit-learn.org
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.figure(figsize=(15, 11))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


read_file(sys.argv[1], predicted)
read_file('test-gold-labels.tsv', real)

cnf_matrix = confusion_matrix(real, predicted)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=sorted(set(real)), normalize=True,
                      title='Normalized ' + str(sys.argv[1]).strip('-')[:-16] + ' confusion matrix')

plt.savefig(str(sys.argv[1])[:-4])