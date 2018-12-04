
def config():
    cfg = {
        'epochs' : 10000,
        'LOG_INTERVAL' : 200,
        'content_filepath' : './content',
        'style_filepath' : './style',
        'content_batch_size' : 5,
        'style_size' : 5,
        'img_size' : 64,
        'debug' : 0,
        'lr' : 4*1e-5,
        'momentum' : 0.3,
        'T' : 2,
        'lambda' : 1,
        'GPU' : 0,
    }
    return cfg