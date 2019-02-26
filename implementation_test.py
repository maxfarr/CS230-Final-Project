import cdaee

autoencoder = cdaee.CDAEE()
autoencoder.train(epochs=20000, batch_size=128, sample_interval=200)