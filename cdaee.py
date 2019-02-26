from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise, Lambda
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D, Layer
from keras.layers import MaxPooling2D, merge, Add, Multiply, Concatenate, Conv2DTranspose, Conv2D, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import losses
from keras.utils import to_categorical
from affectnet_load import load
import keras.backend as K

import matplotlib.pyplot as plt

import numpy as np

class CDAEE():
    def __init__(self):
        self.img_rows = 32
        self.img_cols = 32
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100
        self.label_dim = 2
        self.label_shape = (self.label_dim,)
        self.batch_size = 128

        optimizer = Adam(0.000001, 0.5, clipnorm=3.)
        optimizer_d = Adam(0.000001, 0.5)

        # build and compile the discriminator
        self.disc_g = self.build_disc_g()
        self.disc_g.name = "discriminator"

        n_disc_trainable = len(self.disc_g.trainable_weights)

        self.disc_g.compile(loss='binary_crossentropy',
            optimizer=optimizer_d,
            metrics=['accuracy'])

        # build the encoders / decoders
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
        
        self.adversarial_autoencoder = self.build_stacked()
        self.adversarial_autoencoder.name = "aae"

        # adversarial_autoencoder model  (stacked generator and discriminator)
        self.adversarial_autoencoder.compile(loss=['mse', 'kullback_leibler_divergence', 'binary_crossentropy', 'mse'],
            loss_weights=[.75, 0.01, 0.001, 0.45],
            optimizer=optimizer,
            metrics=['accuracy'])

        assert(len(self.disc_g._collected_trainable_weights) == n_disc_trainable)

    def build_stacked(self):
        
        img = Input(shape=self.img_shape)
        label = Input(shape=self.label_shape)

        for layer in self.disc_g.layers:
            layer.trainable = False

        pooled_repr = self.encoder1(img)
        encoded_repr = self.encoder2(pooled_repr)

        decoded_img = self.decoder1([encoded_repr, label])

        reconstructed_img = self.decoder2([decoded_img, pooled_repr])

        reconst_validity = self.disc_g(reconstructed_img)

        return Model([img, label], [reconstructed_img, encoded_repr, reconst_validity, reconstructed_img])

    def build_enc_1(self):

        img = Input(shape=self.img_shape)

        conv1 = Conv2D(128, 3, input_shape=self.img_shape, padding='same')(img)
        conv1a = LeakyReLU(alpha=0.2)(conv1)
        
        return Model(img, conv1a)

    def build_enc_2(self):

        pooled_shape = (self.img_rows, self.img_cols, 128)
        img = Input(shape=pooled_shape)

        conv2 = Conv2D(256, 3, input_shape=pooled_shape, padding='same')(img)
        conv2a = LeakyReLU(alpha=0.2)(conv2)

        flatten = Flatten()(conv2a)

        mu = Dense(self.latent_dim)(flatten)
        log_var = Dense(self.latent_dim)(flatten)

        mapper = Lambda(lambda p: p[0] + K.random_normal(K.shape(p[0])) * K.exp(p[1] / 2), output_shape=(self.latent_dim,))
        latent = mapper([mu, log_var])
        
        return Model(img, latent)

    def build_dec_1(self):

        latent = Input(shape=(self.latent_dim,))
        label = Input(shape=self.label_shape)

        concat = Concatenate(axis=-1)([label, latent])
        dense = Dense(32 * 32 * 256)(concat)
        reshape = Reshape((32, 32, 256))(dense)

        deconv2 = Conv2DTranspose(256, 3, padding='same')(reshape)
        batchnorm1 = BatchNormalization()(deconv2)
        deconv2a = Activation("relu")(batchnorm1)
        
        deconv3 = Conv2DTranspose(128, 3, padding='same')(deconv2a)
        batchnorm2 = BatchNormalization()(deconv3)
        deconv2a = Activation("relu")(batchnorm2)

        return Model([latent, label], deconv3)

    def build_dec_2(self):

        decoded = Input(shape=(self.img_rows, self.img_cols, 128))
        residual = Input(shape=(self.img_rows, self.img_cols, 128))

        added = Add()([decoded, residual])
        deconv = Conv2DTranspose(self.channels, 3, padding='same', activation='sigmoid')(added)

        return Model([decoded, residual], deconv)

    def build_disc_e(self):

        model = Sequential()

        latent = Input(shape=(self.latent_dim, ))

        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation="sigmoid"))

        model.summary()

        validity = model(latent)

        return Model(latent, validity)

    def build_disc_g(self):

        model = Sequential()

        reconstructed = Input(shape=self.img_shape)

        model.add(Conv2D(32, 3, strides=2, padding='same', input_shape=self.img_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, 3, padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, 3, padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation="sigmoid"))

        model.summary()

        validity = model(reconstructed)

        return Model(reconstructed, validity)

    def train(self, epochs, batch_size=128, sample_interval=50):

        # get facial data
        (X_train, labels_train, val_aro_train) = load(80000)

        X_train /= 255

        # ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # train discriminator

            # select random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            target_idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]
            labels = labels_train[idx]
            val_aro = val_aro_train[idx]

            target_imgs = X_train[target_idx]
            target_labels = labels_train[target_idx]
            target_val_aro = val_aro_train[target_idx]

            latent_fake_pooled = self.encoder1.predict(imgs)
            latent_fake = self.encoder2.predict(latent_fake_pooled)
            latent_real = np.random.normal(size=(batch_size, self.latent_dim))

            decoded_fake = self.decoder1.predict([latent_fake, target_val_aro])
            reconstructed = self.decoder2.predict([decoded_fake, latent_fake_pooled])

            dg_loss_real = self.disc_g.train_on_batch(imgs, valid)
            dg_loss_fake = self.disc_g.train_on_batch(reconstructed, fake)
            dg_loss = 0.5 * np.add(dg_loss_real, dg_loss_fake)

            # train generator
            g_loss = self.adversarial_autoencoder.train_on_batch([imgs, target_val_aro], [target_imgs, latent_real, valid, imgs])

            # plot
            print ("%d [total loss: %f] [DE loss: %f, acc: %.2f%%] [DG loss: %f, acc: %.2f%%] [G loss: %f, mse new: %.2f%%, mse old: %.2f%%]" % (epoch, g_loss[0], g_loss[2], 100*g_loss[6], dg_loss[0], 100*dg_loss[1], g_loss[1], 100*g_loss[5], 100*g_loss[8]))

            # save images
            if epoch % sample_interval == 0:
                self.sample_images(epoch, imgs, labels, target_labels)

    def sample_images(self, epoch, imgs, labels, target_labels):
        r, c = 7, 7

        for img in range(5):
            fig, axs = plt.subplots(r, c+1)
            cnt = 0
            for i in range(r):
                axs[i, 0].axis('off')
                for j in range(1,c+1):
                    pooled = self.encoder1.predict(np.expand_dims(imgs[img], axis=0))
                    z = self.encoder2.predict(pooled)
                    decoded = self.decoder1.predict([z, np.array([[float(i - 3)/3, float(j - 4)/3]])])
                    reconstructed = self.decoder2.predict([decoded, pooled])
                    axs[i,j].imshow(reconstructed[0])
                    axs[i,j].axis('off')
                    cnt += 1
            
            axs[3,0].imshow(imgs[img])
            axs[3,0].axis('off')
            fig.savefig("images/epoch_%d_img_%d.png" % (epoch, img))

        plt.close()


if __name__ == '__main__':
    aae = CDAEE()
    aae.train(epochs=20000, batch_size=32, sample_interval=200)