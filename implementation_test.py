if __name__ == "__main__":
    import disfa_fetch
    import ZHOUSHIFORREAL

    f = disfa_fetch.Fetcher()

    autoencoder = ZHOUSHIFORREAL.CDAEE()
    autoencoder.train(f, 40, batch_size=32, sample_interval=10)