import csv
import numpy as np
from PIL import Image
import os
from scipy.misc import imresize

def load(n = 1000):
    images = []
    labels = []
    valence_arousal = []
    with open('data.csv') as file:
        cnt = 0
        reader = csv.reader(file, delimiter=',')

        for row in reader:
            if row[5] == '9' or row[5] == '10': continue
            
            path = "O:\\Documents\\AffectNet\\Manually_Annotated_compressed\\imgs\\" + row[0]
            if os.path.isfile(path):
                img = Image.open(path)
                img.load()
                data = np.asarray(img, dtype="float32")
                images.append(data)
                labels.append(np.array([int(row[5])]))
                valence_arousal.append(np.array([float(row[6]), float(row[7])]))
                cnt += 1
                if cnt == n: break
        
    return (np.asarray(images), np.asarray(labels), np.asarray(valence_arousal))