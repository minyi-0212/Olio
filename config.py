
def config():
    cfg = {
        'epochs' : 10,
        'LOG_INTERVAL' : 200,
        'content_filepath' : './content',
        'style_filepath' : './style',
        'style_size' : 10,
        'debug' : 0,
        'lr' : 1e-5,
        'momentum' : 0.09,
        'T' : 2,
        'lambda' : 1,
        'GPU' : 0,
    }
    return cfg