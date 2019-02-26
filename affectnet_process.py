import csv
import numpy as np
from PIL import Image
import os

with open('data.csv') as file:
    reader = csv.reader(file, delimiter=',')

    for row in reader:
        if row[5] == '9' or row[5] == '10': continue
        
        path = "O:\\Documents\\AffectNet\\Manually_Annotated_compressed\\part1\\Manually_Annotated_Images\\" + row[0]
        if os.path.isfile(path):
            img = Image.open(path)
            img.load()
            offset = int(row[1])
            size = int(row[3])
            img = img.crop((offset, offset, offset+size, offset+size))
            img.thumbnail((32, 32))
            img.save("O:\\Documents\\AffectNet\\Manually_Annotated_compressed\\imgs\\" + row[0])
            print("saved %s" % row[0])