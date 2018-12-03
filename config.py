
def config():
    cfg = {
        'epochs' : 3000,
        'LOG_INTERVAL' : 200,
        'content_filepath' : './content',
        'style_filepath' : './style',
        'style_size' : 5,
        'debug' : 0,
        'lr' : 4*1e-7,
        'momentum' : 0.3,
        'T' : 2,
        'lambda' : 1,
        'GPU' : 0,
    }
    return cfg