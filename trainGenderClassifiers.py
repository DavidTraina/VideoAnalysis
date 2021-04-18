import numpy as np
import cv2
import os
import scipy.io
from joblib import dump
import argparse

from sklearn.model_selection import train_test_split
from sklearn.metrics         import confusion_matrix

from sklearn.ensemble       import AdaBoostClassifier
from sklearn.ensemble       import RandomForestClassifier
from sklearn.tree           import DecisionTreeClassifier
from sklearn.svm            import SVC
from sklearn.linear_model   import LogisticRegression
from sklearn.linear_model   import SGDClassifier 
from sklearn.naive_bayes    import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors      import KNeighborsClassifier

F_PATH = "/home/lawrence/420/project/train_data/female"
M_PATH = "/home/lawrence/420/project/train_data/male"
OUTPUT = None
H = 250
L = 250

ML = [
    AdaBoostClassifier(),
    RandomForestClassifier(),
    DecisionTreeClassifier(), 
    SVC(),
    LogisticRegression(),
    SGDClassifier(),
    GaussianNB(),
    KNeighborsClassifier(),
    MLPClassifier()
]

""" Trains nine classifiers and, if desired, saves the best model into a file.

Usage:
python3 trainGenderClassifiers.py --path *path to male/female training data* [--output *filename for model*]
"""
def main():
    # Get filenames
    f_files = [f.path for f in os.scandir(F_PATH) if f.path.endswith(".jpg")]
    m_files = [f.path for f in os.scandir(M_PATH) if f.path.endswith(".jpg")]

    # Let F = 0, M = 1 be the labels
    f_y = np.zeros(len(f_files))
    m_y = np.ones(len(m_files))
    y   = np.concatenate((f_y, m_y))

    # Vectorize each image into a numpy array.
    f_jpg = [cv2.imread(f) for f in f_files]
    f_jpg = [f.reshape(H*L*3) for f in f_jpg]
    m_jpg = [cv2.imread(f) for f in m_files]
    m_jpg = [f.reshape(H*L*3) for f in m_jpg]
    X_jpg = np.stack(f_jpg + m_jpg)

    # Vectorize image features into numpy array
    f_mat = [f[:-4] + ".mat" for f in f_files]
    f_mat = [scipy.io.loadmat(m) for m in f_mat]
    f_mat = [np.concatenate((m['x'], m['y'])).reshape(8) for m in f_mat]
    m_mat = [f[:-4] + ".mat" for f in m_files]
    m_mat = [scipy.io.loadmat(m) for m in m_mat]
    m_mat = [np.concatenate((m['x'], m['y'])).reshape(8) for m in m_mat]
    X_mat = np.stack(f_mat + m_mat)
    
    # Combine all image features
    X = np.concatenate((X_jpg, X_mat), axis=1)
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

    accuracies = []
    models = []
    for i, clf in enumerate(ML):
        print("Training classifier {}...".format(i+1))

        # Train, test, report accuracy
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        C = confusion_matrix(y_test, y_pred)
        accuracies.append(C.trace()/C.sum())

        print("Accuracy {:.2f}".format(accuracies[-1]))
        models.append(clf)
       
    if (OUTPUT != None):
        i = np.asarray(accuracies).argmax()
        print("Saving classifier {} as ".format(i+1) + OUTPUT)
        dump(models[i], OUTPUT)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="Path to male/female training data directories", type=str)
    parser.add_argument("--output", default=None, help="Output file for saving best classifier", type=str)

    args    = parser.parse_args()
    F_PATH  = args.path + "/female"
    M_PATH  = args.path + "/male"
    OUTPUT  = args.output

    main()