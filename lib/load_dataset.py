import cv2
import numpy as np
from multiprocessing import Pool, cpu_count
from sklearn.preprocessing import LabelEncoder

#convert str path to images
def __readimg__(im):
    img= cv2.imread(im)
    img = cv2.resize(img, (128, 128))
    return img

def __getimph__(X):
    X_img = []
    for x_class in X:
        print(X.index(x_class))
        with Pool(cpu_count()) as pool:
            results = pool.map_async(__readimg__, x_class)
            X_img.append(results.get())
    return X_img

#get X as str paths and y as the name of the folders label encoded
def getds(p):
    X = []
    y = []
    for i in p:
        if not i.is_file():
            x = list(map(str, i.glob("*")))
            X.append(x)
            y += [i.name] * len(x)
    le = LabelEncoder()
    y = le.fit_transform(y)
    X = __getimph__(X)

    X = np.array(X, dtype="int32")
    X = X / 255
    y = np.array(y)
    return X, y
