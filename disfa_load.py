import os
import numpy as np
import cv2
from PIL import Image
import pandas as pd
from collections import defaultdict

au_ids = [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]
sn_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
def get_from_jpg(sn, frame):
            #if frame % 100 == 0: print("at frame %d" % frame)
            root = "O:\\Documents\\DISFA"
            path = "%s\\%d\\%d.jpg" % (root, sn + sn, frame)
            if os.path.isfile(path):
                img = Image.open(path)
                img.load()
                data = np.asarray(img, dtype="uint8")
                return data
            else:
                print("isfile failed {}: file broken?".format(path))
                return []

def load(n = 28, frames = 4844):
    sn_count = 1
    #au_vals = []
    #imgs_forward = []
    store = pd.HDFStore("disfa.h5")
    rowlist = []
    for i in sn_ids:
        sn = "SN0" + ("0" if i < 10 else "") + str(i)
        #au_single = []
        au_dict = defaultdict(list) #key is frame, value is list of au
        for au in au_ids:
            #data = []
            file = open("O:/Dropbox/ActionUnit_Labels/%s/%s_au%d.txt" % (sn, sn, au), "r")
            #frame_cnt = 0
            for line in file:
                #turn each frame value into 0-1 float and append to current AU's list
                activation = line.split(',')[1].strip()
                frame = int(line.split(',')[0])
                #if string != "": 
                    #data.append(float(string)/5)
                au_dict[frame-1].append(int(activation))
                if frame-1 == frames: break
                #frame_cnt += 1
                #if frame_cnt == frames: break
            
            #au_single.append(data)
        for frame_cnt in range(frames):
            img = get_from_jpg(i, frame_cnt)
            if len(img) == 0: continue
            row = {"sn_id": i, "frame": frame_cnt, "img": img}
            for au in range(len(au_ids)):
                row["au_{}".format(au_ids[au])] = au_dict[frame_cnt][au]
            rowlist.append(row)
        #df = pd.DataFrame(rowlist)
        #df.to_hdf('disfa.h5', 'disfa', format="table", append = True)
        sn_count += 1
        print("done with subject: {}".format(i))
        if sn_count > n: break
    df = pd.DataFrame(rowlist)
    df['sn_id'] = df.sn_id.astype('uint8')
    df['frame'] = df.frame.astype('uint16')
    for au in au_ids:
        df["au_{}".format(au)] = df["au_{}".format(au)].astype('uint8')
    store['disfa'] = df
    store.close()
    #return (np.asarray(imgs_forward), np.asarray(au_vals))
    #from disfa_load import load