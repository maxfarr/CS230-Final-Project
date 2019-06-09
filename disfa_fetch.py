import numpy as np
import pandas as pd
from multiprocessing import Pool
from functools import partial

au_ids = [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]
sn_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

def get_au(au, all_faces, id_faces):
    # sample up to 2000 nonzero frames
    # --------------------------------
    # find counts of each nonzero values for all sn_ids in fold
    nonzero = [all_faces.loc[all_faces["au_{}".format(au)].isin([v])] for v in range(1, 6)]

    fold = [[], [], []]

    # if 2000 or less, just take them all
    counts = [len(l) for l in nonzero]
    total_nonzero = sum(counts)
    print(total_nonzero)
    if total_nonzero <= 2000:
        fold.extend([[item.img.tolist(), list(item[1:13]), id_faces[item.sn_id].sample(1).img.tolist()[0]] for value in nonzero for item in value.itertuples()])
        for value in nonzero:
            for item in value.itertuples():
                fold[0].append(item.img.tolist())
                fold[1].append(list(item[1:13]))
                fold[2].append(id_faces[item.sn_id].sample(1).img.tolist()[0])
    else:
        # otherwise divide to get proportion and sample that many of each value
        counts = [x/float(total_nonzero) for x in counts]
        for i in range(len(counts)):
            if counts[i] == 0.0: continue
            items = nonzero[i].sample(int(counts[i] * 2000))
            for item in items.itertuples():
                fold[0].append(item.img.tolist())
                fold[1].append(list(item[1:13]))
                fold[2].append(id_faces[item.sn_id].sample(1).img.tolist()[0])

    del nonzero

    # sample 1000 zero frames
    #------------------------
    items = all_faces.loc[all_faces["au_{}".format(au)].isin([0])].sample(1000)
    print(len(items))

    for item in items.itertuples():
        fold.append([item.img.tolist(), list(item[1:13]), id_faces[item.sn_id].sample(1).img.tolist()[0]])
        fold[0].append(item.img.tolist())
        fold[1].append(list(item[1:13]))
        fold[2].append(id_faces[item.sn_id].sample(1).img.tolist()[0])
    print("finished au_{}".format(au))

    del items

    return fold

class Fetcher():
    def __init__(self):
        self.p = Pool(16)
        df = pd.HDFStore('disfa.h5')['disfa']
        cols = ['au_1', 'au_2', 'au_4', 'au_5', 'au_6', 'au_9', 'au_12', 'au_15', 'au_17', 'au_20', 'au_25', 'au_26', 'frame', 'img', 'sn_id']
        self.df = df[cols]

    def fetch(self):
        shuffled_ids = np.random.permutation(sn_ids)

        folds = []

        print("shuffled : {}".format(shuffled_ids))

        # create folds

        for i in range(4):
            print("creating fold {}".format(i+1))
            fold_ids = shuffled_ids[i * 7:min(27, (i+1)*7)]
            fold = [[], [], []]

            all_faces = self.df.loc[self.df["sn_id"].isin(fold_ids)]
            id_faces = {sn_id : all_faces.loc[all_faces["sn_id"].isin([sn_id])] for sn_id in fold_ids}

            results = self.p.map(partial(get_au, all_faces=all_faces, id_faces=id_faces), au_ids)
            for result in results:
                fold[0].extend(result[0])
                fold[1].extend(result[1])
                fold[2].extend(result[2])

            del all_faces
            folds.append([np.asarray(fold[0], dtype='float32')/255, np.asarray(fold[1], dtype='float32')/5, np.asarray(fold[2], dtype='float32')/255])
            del fold

        return folds