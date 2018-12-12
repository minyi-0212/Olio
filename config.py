def config():
    cfg = {
        'epochs': 50000,
        'LOG_INTERVAL': 2000,
        'update_lr': 5000,
        'ouput_path': 'output',
        'content_filepath': './content',
        'style_filepath': './style',
        'content_batch_size': 2,
        'style_size': 3,
        'img_size': 128,
        'debug': 0,
        'lr': 0.01,
        'T': 2,
        'lambda': 1,
        'GPU': 1,
        'begin_epochs': 0,
    }
    return cfg