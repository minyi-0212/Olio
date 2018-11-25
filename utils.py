import pylab
import matplotlib.pyplot as plt

def show_file(filename):
    p = plt.subplot()
    p.set_title("image file: %5s" % filename)
    plt.imshow(plt.imread(filename))
    pylab.show()