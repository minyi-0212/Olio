def config():
    cfg = {
        'epochs': 99000,
        'LOG_INTERVAL': 300,
        'update_lr': 3300,
        'ouput_path': 'output',
        'content_filepath': './content',
        'style_filepath': './style',
        'content_batch_size': 2,
        'style_size': 3,
        'img_size': 256,
        'content_weight': 1,
        'style_weight': 100,
        'tv_weight': 1,
        'debug': 0,
        'lr': 0.000005,
        'T': 2,
        'lambda': 1,
        'GPU': 1,
        'begin_epochs': 0,
    }
    return cfg