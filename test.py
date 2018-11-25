import torch
from torch.autograd import Variable
from network import StyleBankNet
from dot import  make_dot

def test_net():
    x = Variable(torch.randn(1,3,20,20))#change 12 to the channel number of network input
    net = StyleBankNet()
    print(net)
    # y = net(x)
    # g = make_dot(y)
    # g.view()


from data_load import ImageLoader
from utils import *

def test_data_load():
    image_list = ImageLoader("./content")
    filename = image_list.get_image_filename(0)
    show_file(filename)

if __name__ == '__main__':
    test_net()
    # test_data_load()