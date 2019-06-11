if __name__ == "__main__":
    import zhoushi

    autoencoder = zhoushi.CDAEE()
    autoencoder.train(epochs=25, batch_size=32, sample_interval=20)