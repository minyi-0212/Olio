def config():
    cfg = {
        'epochs' : 10000,
        'LOG_INTERVAL' : 1000,
        'content_filepath' : './content',
        'style_filepath' : './style',
        'content_batch_size' : 4,
        'style_size' : 5,
        'img_size' : 64,
        'debug' : 0,
        'lr' : 8*1e-5,
        'momentum' : 0.3,
        'T' : 2,
        'lambda' : 1,
        'GPU' : 1,
        'ouput_path' : 'output',
    }
    return cfg