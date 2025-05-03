class UNetConfig:

    model_prefix = 'unet_best_model'
    unet_in_channels = 5
    unet_out_channels = 1
    unet_init_features = 32
    unet_depth = 5
    unet_bilinear = True

    # --- Training params ---
    n_epochs = 100
    learning_rate = 1e-4
    weight_decay = 1e-5
    plot_every_n_epochs = 5
