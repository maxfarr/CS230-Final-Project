if __name__ == "__main__":
    import ZHOUSHIFORREAL

    autoencoder = ZHOUSHIFORREAL.CDAEE()
    autoencoder.train(epochs=40, batch_size=32, sample_interval=20)