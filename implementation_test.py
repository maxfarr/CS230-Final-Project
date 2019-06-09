import ZHOUSHIFORREAL

autoencoder = ZHOUSHIFORREAL.CDAEE()
autoencoder.train(epochs=40, batch_size=32, sample_interval=10)