from __future__ import print_function, division

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise, Lambda
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D, Layer
from keras.layers import MaxPooling2D, merge, Add, Multiply, Concatenate, Conv2DTranspose, Conv2D, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import losses
from keras.utils import to_categorical, multi_gpu_model
import keras.backend as K

from datetime import datetime

au_ids = [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]
sn_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

class CDAEE():
    def __init__(self):
        self.img_rows = 32
        self.img_cols = 32
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100
        self.label_dim = 12
        self.label_shape = (self.label_dim,)
        self.batch_size = 64

        optimizer = Adam(0.0001)
        optimizer_d = Adam(0.000001)

        # Build and compile the discriminators
        self.disc_g = multi_gpu_model(self.build_disc_g(), 4)
        self.disc_g.name = "disc_g"

        self.disc_e = multi_gpu_model(self.build_disc_e(), 4)
        self.disc_e.name = "disc_e"

        n_disc_trainable = len(self.disc_g.trainable_weights)

        self.disc_g.compile(loss='binary_crossentropy',
            optimizer=optimizer_d,
            metrics=['accuracy'])

        self.disc_e.compile(loss='binary_crossentropy',
            optimizer=optimizer_d,
            metrics=['accuracy'])

        # Build the encoder / decoders
        self.encoder1 = self.build_enc_1()
        self.encoder1.name = "encoder1"
        self.encoder1.summary()
        self.encoder2 = self.build_enc_2()
        self.encoder2.name = "encoder2"
        self.encoder2.summary()
        self.decoder1 = self.build_dec_1()
        self.decoder1.name = "decoder1"
        self.decoder1.summary()
        self.decoder2 = self.build_dec_2()
        self.decoder2.name = "decoder2"
        self.decoder2.summary()
        
        self.adversarial_autoencoder = multi_gpu_model(self.build_stacked(), 4)
        self.adversarial_autoencoder.name = "aae"

        # adversarial_autoencoder model  (stacked generator and discriminator)
        self.adversarial_autoencoder.compile(loss=['mse', 'binary_crossentropy', 'binary_crossentropy'],
            loss_weights=[1, 0.01, 0.001],
            optimizer=optimizer,
            metrics=['accuracy'])

        print(self.disc_g.metrics_names)
        print(self.disc_e.metrics_names)
        print(self.adversarial_autoencoder.metrics_names)

        assert(len(self.disc_g._collected_trainable_weights) == n_disc_trainable)

    def build_stacked(self):
        
        img = Input(shape=self.img_shape)
        label = Input(shape=self.label_shape)

        self.disc_g.trainable = False

        self.disc_e.trainable = False

        for layer in self.disc_g.layers:
            layer.trainable = False

        for layer in self.disc_g.layers:
            layer.trainable = False

        pooled_repr = self.encoder1(img)
        encoded_repr = self.encoder2(pooled_repr)

        decoded_img = self.decoder1([encoded_repr, label])

        reconstructed_img = self.decoder2([decoded_img, pooled_repr])

        encoded_validity = self.disc_e(encoded_repr)
        reconst_validity = self.disc_g(reconstructed_img)

        return Model(inputs=[img, label], outputs=[reconstructed_img, encoded_validity, reconst_validity])

    def build_enc_1(self):
        # Encoder

        img = Input(shape=self.img_shape)

        conv1 = Conv2D(128, 5, input_shape=self.img_shape, padding='same')(img)
        conv1a = LeakyReLU(alpha=0.2)(conv1)

        pool1 = MaxPooling2D(2)(conv1a)
        
        return Model(img, pool1)

    def build_enc_2(self):
        # Encoder

        pooled_shape = (self.img_rows//2, self.img_cols//2, 128)
        img = Input(shape=pooled_shape)

        conv2 = Conv2D(256, 5, input_shape=pooled_shape, padding='same')(img)
        conv2a = LeakyReLU(alpha=0.2)(conv2)

        pool2 = MaxPooling2D(2)(conv2a)

        conv3 = Conv2D(256, 5, padding='same')(pool2)
        conv3a = LeakyReLU(alpha=0.2)(conv3)

        pool3 = MaxPooling2D(2)(conv3a)

        conv4 = Conv2D(512, 5, padding='same')(pool3)
        conv4a = LeakyReLU(alpha=0.2)(conv4)

        flatten = Flatten()(conv4a)

        mu = Dense(self.latent_dim)(flatten)
        log_var = Dense(self.latent_dim)(flatten)

        mapper = Lambda(lambda p: (p[0] + (K.random_normal(K.shape(p[0])) * K.exp(p[1] / 2))), output_shape=(self.latent_dim,))
        latent = mapper([mu, log_var])
        
        return Model(img, latent)

    def build_dec_1(self):

        latent = Input(shape=(self.latent_dim,))
        label = Input(shape=self.label_shape)

        concat = Concatenate(axis=-1)([label, latent])
        dense = Dense(4 * 4 * 512)(concat)
        reshape = Reshape((4, 4, 512))(dense)

        deconv1 = Conv2DTranspose(512, 5, padding='same')(reshape)
        deconv1a = LeakyReLU(alpha=0.2)(deconv1)
        upsamp1 = UpSampling2D(2)(deconv1a)

        deconv2 = Conv2DTranspose(256, 5, padding='same')(upsamp1)
        deconv2a = LeakyReLU(alpha=0.2)(deconv2)
        upsamp2 = UpSampling2D(2)(deconv2a)
        
        deconv3 = Conv2DTranspose(128, 5, padding='same', activation='tanh')(upsamp2)
        #deconv3a = LeakyReLU(alpha=0.2)(deconv3)

        return Model([latent, label], deconv3)

    def build_dec_2(self):

        decoded = Input(shape=(self.img_rows//2, self.img_cols//2, 128))
        residual = Input(shape=(self.img_rows//2, self.img_cols//2, 128))

        added = Add()([decoded, residual])
        upsamp = UpSampling2D(2)(added)
        deconv = Conv2DTranspose(self.channels, 5, padding='same', activation='sigmoid')(upsamp)
        #deconva = LeakyReLU(alpha=0.2)(deconv)

        return Model([decoded, residual], deconv)

    def build_disc_e(self, relu = False):

        model = Sequential()

        latent = Input(shape=(self.latent_dim, ))

        model.add(Dense(256, input_dim=self.latent_dim))
        if relu: model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        if relu: model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        if relu: model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation="sigmoid"))

        model.summary()

        validity = model(latent)

        return Model(latent, validity)

    def build_disc_g(self):

        model = Sequential()

        reconstructed = Input(shape=self.img_shape)

        model.add(Conv2D(32, 3, strides=2, padding='same', input_shape=self.img_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(MaxPooling2D(2))

        model.add(Conv2D(64, 5, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(MaxPooling2D(2))

        model.add(Conv2D(64, 3, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(MaxPooling2D(2))

        model.add(Conv2D(128, 3, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(MaxPooling2D(2))

        model.add(Flatten())
        model.add(Dense(1, activation="sigmoid"))

        model.summary()

        validity = model(reconstructed)

        return Model(reconstructed, validity)

    def train(self, epochs, batch_size=32, sample_interval=10):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S %p')

        df = pd.HDFStore('disfa.h5')['disfa']
        cols = ['au_1', 'au_2', 'au_4', 'au_5', 'au_6', 'au_9', 'au_12', 'au_15', 'au_17', 'au_20', 'au_25', 'au_26', 'frame', 'img', 'sn_id']
        df = df[cols]

        #list 0 is training, 1 is test
        dg_losses = [[], []]
        de_losses = [[], []]
        g_losses = [[], []]

        for epoch in range(epochs):
            shuffled_ids = np.random.permutation(sn_ids)

            folds = []

            print("shuffled : {}".format(shuffled_ids))

            # create folds

            for i in range(4):
                print("creating fold {}".format(i+1))
                fold_ids = shuffled_ids[i * 7:min(27, (i+1)*7)]
                fold = [[], [], []]

                all_faces = df.loc[df["sn_id"].isin(fold_ids)]
                id_faces = {sn_id : all_faces.loc[all_faces["sn_id"].isin([sn_id])] for sn_id in fold_ids}

                for au in au_ids:
                    # sample up to 2000 nonzero frames
                    # --------------------------------
                    # find counts of each nonzero values for all sn_ids in fold
                    nonzero = [all_faces.loc[all_faces["au_{}".format(au)].isin([v])] for v in range(1, 6)]

                    # if less than 2000, just take them all
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

                del all_faces
                folds.append([np.asarray(fold[0], dtype='float32')/255, np.asarray(fold[1], dtype='float32')/5, np.asarray(fold[2], dtype='float32')/255])
                del fold

            # train with folds
                
            for i, (Y_te, labels_te, X_te) in enumerate(folds):
                indices = [x for x in range(4) if x != i]
                X_tr = np.concatenate([folds[x][2] for x in indices])
                labels_tr = np.concatenate([folds[x][1] for x in indices])
                Y_tr = np.concatenate([folds[x][0] for x in indices])

                shuffled_ix = np.random.permutation(range(X_tr.shape[0]))
                X_tr = X_tr[shuffled_ix]
                labels_tr = labels_tr[shuffled_ix]
                Y_tr = Y_tr[shuffled_ix]

                shuffled_ix = np.random.permutation(range(X_te.shape[0]))
                X_te = X_te[shuffled_ix]
                labels_te = labels_te[shuffled_ix]
                Y_te = Y_te[shuffled_ix]

                

                print("--------------------------")
                print("training epoch {} fold {}".format(epoch, i))
                print("--------------------------")

                for batch_i in range(int(X_tr.shape[0]/batch_size)+1):
                    b_i_size = min(batch_size, X_tr.shape[0] - (batch_i * batch_size))
                    X_batch = X_tr[batch_i * batch_size : batch_i * batch_size + b_i_size]
                    Y_batch = Y_tr[batch_i * batch_size : batch_i * batch_size + b_i_size]
                    labels_batch = labels_tr[batch_i * batch_size : batch_i * batch_size + b_i_size]
                    latent_fake_pooled = self.encoder1.predict(X_batch)
                    latent_fake = self.encoder2.predict(latent_fake_pooled)
                    latent_real = np.random.normal(size=(b_i_size, self.latent_dim))

                    valid = np.ones((b_i_size, 1))
                    fake = np.zeros((b_i_size, 1))

                    decoded_fake = self.decoder1.predict([latent_fake, labels_batch])
                    reconstructed = self.decoder2.predict([decoded_fake, latent_fake_pooled])

                    dg_loss_real = self.disc_g.train_on_batch(X_batch, valid)
                    dg_loss_fake = self.disc_g.train_on_batch(reconstructed, fake)

                    dg_losses[0].append([dg_loss_real, dg_loss_fake])

                    de_loss_real = self.disc_e.train_on_batch(latent_real, valid)
                    de_loss_fake = self.disc_e.train_on_batch(latent_fake, fake)

                    de_losses[0].append([de_loss_real, de_loss_fake])

                    # train the generator
                    g_loss = self.adversarial_autoencoder.train_on_batch([X_batch, labels_batch], [Y_batch, fake, fake])

                    g_losses[0].append(g_loss)

                    # save to ./images
                    if batch_i % sample_interval == 0:
                        print("[ep {} fold {} batch {} gen] {}".format(epoch, i, batch_i, g_loss))
                        print("[ep {} fold {} batch {} d_g] real {} fake {}".format(epoch, i, batch_i, dg_loss_real, dg_loss_fake))
                        print("[ep {} fold {} batch {} d_e] real {} fake {}".format(epoch, i, batch_i, de_loss_real, de_loss_fake))
                        self.sample_images(epoch, i, batch_i, X_tr, timestamp)

                print("---------------------")
                print("----training over----")
                print("---------------------")

                # evaluate
                latent_fake_pooled = self.encoder1.predict(X_te)
                latent_fake = self.encoder2.predict(latent_fake_pooled)
                latent_real = np.random.normal(size=(X_te.shape[0], self.latent_dim))

                valid = np.ones((X_te.shape[0], 1))
                fake = np.zeros((X_te.shape[0], 1))

                decoded_fake = self.decoder1.predict([latent_fake, labels_te])
                reconstructed = self.decoder2.predict([decoded_fake, latent_fake_pooled])

                dg_l_r_te = self.disc_g.evaluate(X_te, valid)
                dg_l_f_te = self.disc_g.evaluate(reconstructed, fake)

                dg_losses[1].append([dg_l_r_te, dg_l_f_te])

                de_l_r_te = self.disc_e.evaluate(latent_real, valid)
                de_l_f_te = self.disc_e.evaluate(latent_fake, fake)

                de_losses[1].append([de_l_r_te, de_l_f_te])

                # train the generator
                g_loss_te = self.adversarial_autoencoder.evaluate([X_te, labels_te], [Y_te, fake, fake])

                g_losses[1].append(g_loss_te)

                print("[ep {} fold {} gen te] {}".format(epoch, i, g_loss))
                print("[ep {} fold {} d_g te] real {} fake {}".format(epoch, i, dg_loss_real, dg_loss_fake))
                print("[ep {} fold {} d_e te] real {} fake {}".format(epoch, i, de_loss_real, de_loss_fake))

        # save models

        self.disc_g.save(timestamp + '/disc_g.h5')
        self.disc_e.save(timestamp + '/disc_e.h5')
        self.adversarial_autoencoder.save(timestamp + '/aae.h5')

        # plot

        dg_tr = np.asarray(dg_losses[0])
        dg_te = np.asarray(dg_losses[1])

        de_tr = np.asarray(de_losses[0])
        de_te = np.asarray(de_losses[1])

        g_tr = np.asarray(g_losses[0])
        g_te = np.asarray(g_losses[1])

        Y_tr_len = range(len(g_tr.shape[0]))
        Y_te_len = range(len(g_te.shape[0]))

        # tr

        plt.plot(dg_tr[:,0,0], Y_tr_len, "--m", dg_tr[:,1,0], Y_tr_len, "-m", de_tr[:,0,0], Y_tr_len, "--c", de_tr[:,1,0], Y_tr_len, "-c")
        plt.savefig(timestamp + "/losses_d_tr.png")
        plt.close()

        plt.plot(g_tr[:,1], Y_tr_len, "-y", g_tr[:,2], Y_tr_len, "-c", g_tr[:,3], Y_tr_len, "-m", g_tr[:,0], Y_tr_len, "-r")
        plt.savefig(timestamp + "/losses_g_tr.png")
        plt.close()

        plt.plot(dg_tr[:,0,1], Y_tr_len, "--m", dg_tr[:,1,1], Y_tr_len, "-m", de_tr[:,0,1], Y_tr_len, "--c", de_tr[:,1,1], Y_tr_len, "-c")
        plt.savefig(timestamp + "/acc_d_tr.png")
        plt.close()

        plt.plot(g_tr[:,4], Y_tr_len, "-y", g_tr[:,5], Y_tr_len, "-c", g_tr[:,6], Y_tr_len, "-m")
        plt.savefig(timestamp + "/acc_g_tr.png")
        plt.close()

        # te

        plt.plot(dg_te[:,0,0], Y_te_len, "--m", dg_te[:,1,0], Y_te_len, "-m", de_te[:,0,0], Y_te_len, "--c", de_te[:,1,0], Y_te_len, "-c")
        plt.savefig(timestamp + "/losses_d_te.png")
        plt.close()

        plt.plot(g_te[:,1], Y_te_len, "-y", g_te[:,2], Y_te_len, "-c", g_te[:,3], Y_te_len, "-m", g_te[:,0], Y_te_len, "-r")
        plt.savefig(timestamp + "/losses_g_te.png")
        plt.close()

        plt.plot(dg_te[:,0,1], Y_te_len, "--m", dg_te[:,1,1], Y_te_len, "-m", de_te[:,0,1], Y_te_len, "--c", de_te[:,1,1], Y_te_len, "-c")
        plt.savefig(timestamp + "/acc_d_te.png")
        plt.close()

        plt.plot(g_te[:,4], Y_te_len, "-y", g_te[:,5], Y_te_len, "-c", g_te[:,6], Y_te_len, "-m")
        plt.savefig(timestamp + "/acc_g_te.png")
        plt.close()

        del folds
            

    def sample_images(self, epoch, fold, batch, imgs, timestamp):
        r, c = 6, 6

        for img in range(3):
            fig, axs = plt.subplots(r, c+1)
            cnt = 0
            for i in range(r):
                axs[i, 0].axis('off')
                for j in range(1,c+1):
                    pooled = self.encoder1.predict(np.expand_dims(imgs[img], axis=0))
                    z = self.encoder2.predict(pooled)

                    au_ids = [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]
                    
                    label = ([0.0, 0.0, float(j - 1)/c, 0.0, 0.0, 0.0, float(i)/r, 0.0, 0.0, 0.0, 0.0, 0.0]
                    if img == 0 else ([0.0, float(j - 1)/c, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(i)/r, float(i)/r]
                    if img == 1 else [0.0, 0.0, float(i)/r, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(j - 1)/c, 0.0]))
                    decoded = self.decoder1.predict([z, np.array([label])])
                    reconstructed = self.decoder2.predict([decoded, pooled])
                    axs[i,j].imshow(reconstructed[0])
                    axs[i,j].axis('off')
                    cnt += 1
            
            axs[0,0].imshow(imgs[img])
            axs[0,0].axis('off')
            fig.savefig(timestamp + "/epoch_%d_fold_%d_batch_%d_img_%d.png" % (epoch, fold, batch, img))
            plt.close()

        


if __name__ == '__main__':
    aae = CDAEE()
    aae.train(epochs=40, batch_size=32, sample_interval=10)