''' Code copied from /deep/group/gen-eval/model-training/src/GAN_models/ '''

import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, opt, disc=False):
        super(Decoder, self).__init__()
        self.num_channel = opt.nc
        self.b_size = opt.b_size
        self.h = opt.h
        self.disc = disc
        self.t_act = opt.tanh
        self.scale_size = opt.scale_size

        self.l0 = nn.Linear(self.h, 8*8*self.num_channel)
        self.l1 = nn.Conv2d(self.num_channel, self.num_channel, 3, 1, 1)
        self.l2 = nn.Conv2d(self.num_channel, self.num_channel, 3, 1, 1)
        self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.l3 = nn.Conv2d(self.num_channel, self.num_channel, 3, 1, 1)
        self.l4 = nn.Conv2d(self.num_channel, self.num_channel, 3, 1, 1)
        self.up2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.l5 = nn.Conv2d(self.num_channel, self.num_channel, 3, 1, 1)
        self.l6 = nn.Conv2d(self.num_channel, self.num_channel, 3, 1, 1)
        self.up3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.l7 = nn.Conv2d(self.num_channel, self.num_channel, 3, 1, 1)
        self.l8 = nn.Conv2d(self.num_channel, self.num_channel, 3, 1, 1)
        if self.scale_size == 128:
            self.up4 = nn.UpsamplingNearest2d(scale_factor=2)
            self.l10 = nn.Conv2d(self.num_channel, self.num_channel, 3, 1, 1)
            self.l11 = nn.Conv2d(self.num_channel, self.num_channel, 3, 1, 1)
        elif self.scale_size == 256:
            self.up4 = nn.UpsamplingNearest2d(scale_factor=2)
            self.l10 = nn.Conv2d(self.num_channel, self.num_channel, 3, 1, 1)
            self.l11 = nn.Conv2d(self.num_channel, self.num_channel, 3, 1, 1)

            self.up5 = nn.UpsamplingNearest2d(scale_factor=2)
            self.l12 = nn.Conv2d(self.num_channel, self.num_channel, 3, 1, 1)
            self.l13 = nn.Conv2d(self.num_channel, self.num_channel, 3, 1, 1)
        elif self.scale_size == 512:
            self.up4 = nn.UpsamplingNearest2d(scale_factor=2)
            self.l10 = nn.Conv2d(self.num_channel, self.num_channel, 3, 1, 1)
            self.l11 = nn.Conv2d(self.num_channel, self.num_channel, 3, 1, 1)

            self.up5 = nn.UpsamplingNearest2d(scale_factor=2)
            self.l12 = nn.Conv2d(self.num_channel, self.num_channel, 3, 1, 1)
            self.l13 = nn.Conv2d(self.num_channel, self.num_channel, 3, 1, 1)

            self.up6 = nn.UpsamplingNearest2d(scale_factor=2)
            self.l14 = nn.Conv2d(self.num_channel, self.num_channel, 3, 1, 1)
            self.l15 = nn.Conv2d(self.num_channel, self.num_channel, 3, 1, 1)
        elif self.scale_size == 1024:
            self.up4 = nn.UpsamplingNearest2d(scale_factor=2)
            self.l10 = nn.Conv2d(self.num_channel, self.num_channel, 3, 1, 1)
            self.l11 = nn.Conv2d(self.num_channel, self.num_channel, 3, 1, 1)

            self.up5 = nn.UpsamplingNearest2d(scale_factor=2)
            self.l12 = nn.Conv2d(self.num_channel, self.num_channel, 3, 1, 1)
            self.l13 = nn.Conv2d(self.num_channel, self.num_channel, 3, 1, 1)

            self.up6 = nn.UpsamplingNearest2d(scale_factor=2)
            self.l14 = nn.Conv2d(self.num_channel, self.num_channel, 3, 1, 1)
            self.l15 = nn.Conv2d(self.num_channel, self.num_channel, 3, 1, 1)

            self.up7 = nn.UpsamplingNearest2d(scale_factor=2)
            self.l16 = nn.Conv2d(self.num_channel, self.num_channel, 3, 1, 1)
            self.l17 = nn.Conv2d(self.num_channel, self.num_channel, 3, 1, 1)
        self.l9 = nn.Conv2d(self.num_channel, 3, 3, 1, 1)



    def forward(self, input, batch_size=None):
        if not batch_size:
            batch_size = self.b_size

        x = self.l0(input)
        #x = x.view(self.b_size, self.num_channel,8, 8)
        x = x.view(-1, self.num_channel,8, 8)

        x = F.elu(self.l1(x), True)
        x = F.elu(self.l2(x), True)
        x = self.up1(x)

        x = F.elu(self.l3(x), True)
        x = F.elu(self.l4(x), True)
        x = self.up2(x)
        x = F.elu(self.l5(x), True)
        x = F.elu(self.l6(x), True)
        x = self.up3(x)
        x = F.elu(self.l7(x), True)
        x = F.elu(self.l8(x), True)
        if self.scale_size == 128:
            x = self.up4(x)
            x = F.elu(self.l10(x))
            x = F.elu(self.l11(x))
        elif self.scale_size == 256:
            x = self.up4(x)
            x = F.elu(self.l10(x))
            x = F.elu(self.l11(x))

            x = self.up5(x)
            x = F.elu(self.l12(x))
            x = F.elu(self.l13(x))
        elif self.scale_size == 512:
            x = self.up4(x)
            x = F.elu(self.l10(x))
            x = F.elu(self.l11(x))

            x = self.up5(x)
            x = F.elu(self.l12(x))
            x = F.elu(self.l13(x))

            x = self.up6(x)
            x = F.elu(self.l14(x))
            x = F.elu(self.l15(x))
        elif self.scale_size == 1024:
            x = self.up4(x)
            x = F.elu(self.l10(x))
            x = F.elu(self.l11(x))

            x = self.up5(x)
            x = F.elu(self.l12(x))
            x = F.elu(self.l13(x))

            x = self.up6(x)
            x = F.elu(self.l14(x))
            x = F.elu(self.l15(x))

            x = self.up7(x)
            x = F.elu(self.l16(x))
            x = F.elu(self.l17(x))
        x = self.l9(x)
        #if not self.disc:
        #if self.scale_size != 128:# and self.t_act:
        x = F.tanh(x)
        return x

class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        self.num_channel = opt.nc
        self.h = opt.h
        self.b_size = opt.b_size
        self.scale_size = opt.scale_size
        self.l0 = nn.Conv2d(3, self.num_channel, 3, 1, 1)
        self.l1 = nn.Conv2d(self.num_channel, self.num_channel, 3, 1, 1)
        self.l2 = nn.Conv2d(self.num_channel, self.num_channel, 3, 1, 1)
        self.down1 = nn.Conv2d(self.num_channel, self.num_channel, 1, 1, 0)
        self.pool1 = nn.AvgPool2d(2, 2)

        self.l3 = nn.Conv2d(self.num_channel, self.num_channel, 3, 1, 1)
        self.l4 = nn.Conv2d(self.num_channel, self.num_channel, 3, 1, 1)
        self.down2 = nn.Conv2d(self.num_channel, 2*self.num_channel, 1, 1, 0)
        self.pool2 = nn.AvgPool2d(2, 2)

        self.l5 = nn.Conv2d(2*self.num_channel, 2*self.num_channel, 3, 1, 1)
        self.l6 = nn.Conv2d(2*self.num_channel, 2*self.num_channel, 3, 1, 1)
        self.down3 = nn.Conv2d(2*self.num_channel, 3*self.num_channel, 1, 1, 0)
        self.pool3 = nn.AvgPool2d(2, 2)

        if self.scale_size == 64:
            self.l7 = nn.Conv2d(3*self.num_channel, 3*self.num_channel, 3, 1, 1)
            self.l8 = nn.Conv2d(3*self.num_channel, 3*self.num_channel, 3, 1, 1)
            self.l9 = nn.Linear(8*8*3*self.num_channel, 64)
        elif self.scale_size == 128:
            self.l7 = nn.Conv2d(3*self.num_channel, 3*self.num_channel, 3, 1, 1)
            self.l8 = nn.Conv2d(3*self.num_channel, 3*self.num_channel, 3, 1, 1)
            self.down4 = nn.Conv2d(3*self.num_channel, 4*self.num_channel, 1, 1, 0)
            self.pool4 = nn.AvgPool2d(2, 2)

            self.l9 = nn.Conv2d(4*self.num_channel, 4*self.num_channel, 3, 1, 1)
            self.l11 = nn.Conv2d(4*self.num_channel, 4*self.num_channel, 3, 1, 1)
            self.l12 = nn.Linear(8*8*4*self.num_channel, self.h)

        elif self.scale_size == 256:
            self.l7 = nn.Conv2d(3*self.num_channel, 3*self.num_channel, 3, 1, 1)
            self.l8 = nn.Conv2d(3*self.num_channel, 3*self.num_channel, 3, 1, 1)
            self.down4 = nn.Conv2d(3*self.num_channel, 4*self.num_channel, 1, 1, 0)
            self.pool4 = nn.AvgPool2d(2, 2)

            self.l9 = nn.Conv2d(4*self.num_channel, 4*self.num_channel, 3, 1, 1)
            self.l10 = nn.Conv2d(4*self.num_channel, 4*self.num_channel, 3, 1, 1)
            self.down5 = nn.Conv2d(4*self.num_channel, 5*self.num_channel, 1, 1, 0)
            self.pool5 = nn.AvgPool2d(2, 2)

            self.l11 = nn.Conv2d(5*self.num_channel, 5*self.num_channel, 3, 1, 1)
            self.l12 = nn.Conv2d(5*self.num_channel, 5*self.num_channel, 3, 1, 1)
            self.l13 = nn.Linear(8*8*5*self.num_channel, self.h)

        elif self.scale_size == 512:
            self.l7 = nn.Conv2d(3*self.num_channel, 3*self.num_channel, 3, 1, 1)
            self.l8 = nn.Conv2d(3*self.num_channel, 3*self.num_channel, 3, 1, 1)
            self.down4 = nn.Conv2d(3*self.num_channel, 4*self.num_channel, 1, 1, 0)
            self.pool4 = nn.AvgPool2d(2, 2)

            self.l9 = nn.Conv2d(4*self.num_channel, 4*self.num_channel, 3, 1, 1)
            self.l10 = nn.Conv2d(4*self.num_channel, 4*self.num_channel, 3, 1, 1)
            self.down5 = nn.Conv2d(4*self.num_channel, 5*self.num_channel, 1, 1, 0)
            self.pool5 = nn.AvgPool2d(2, 2)

            self.l11 = nn.Conv2d(5*self.num_channel, 5*self.num_channel, 3, 1, 1)
            self.l12 = nn.Conv2d(5*self.num_channel, 5*self.num_channel, 3, 1, 1)
            self.down6 = nn.Conv2d(5*self.num_channel, 6*self.num_channel, 1, 1, 0)
            self.pool6 = nn.AvgPool2d(2, 2)

            self.l13 = nn.Conv2d(6*self.num_channel, 6*self.num_channel, 3, 1, 1)
            self.l14 = nn.Conv2d(6*self.num_channel, 6*self.num_channel, 3, 1, 1)
            self.l15 = nn.Linear(8*8*6*self.num_channel, self.h)

        elif self.scale_size == 1024:
            self.l7 = nn.Conv2d(3*self.num_channel, 3*self.num_channel, 3, 1, 1)
            self.l8 = nn.Conv2d(3*self.num_channel, 3*self.num_channel, 3, 1, 1)
            self.down4 = nn.Conv2d(3*self.num_channel, 4*self.num_channel, 1, 1, 0)
            self.pool4 = nn.AvgPool2d(2, 2)

            self.l9 = nn.Conv2d(4*self.num_channel, 4*self.num_channel, 3, 1, 1)
            self.l10 = nn.Conv2d(4*self.num_channel, 4*self.num_channel, 3, 1, 1)
            self.down5 = nn.Conv2d(4*self.num_channel, 5*self.num_channel, 1, 1, 0)
            self.pool5 = nn.AvgPool2d(2, 2)

            self.l11 = nn.Conv2d(5*self.num_channel, 5*self.num_channel, 3, 1, 1)
            self.l12 = nn.Conv2d(5*self.num_channel, 5*self.num_channel, 3, 1, 1)
            self.down6 = nn.Conv2d(5*self.num_channel, 6*self.num_channel, 1, 1, 0)
            self.pool6 = nn.AvgPool2d(2, 2)

            self.l13 = nn.Conv2d(6*self.num_channel, 6*self.num_channel, 3, 1, 1)
            self.l14 = nn.Conv2d(6*self.num_channel, 6*self.num_channel, 3, 1, 1)
            self.down7 = nn.Conv2d(6*self.num_channel, 7*self.num_channel, 1, 1, 0)
            self.pool7 = nn.AvgPool2d(2, 2)

            self.l15 = nn.Conv2d(7*self.num_channel, 7*self.num_channel, 3, 1, 1)
            self.l16 = nn.Conv2d(7*self.num_channel, 7*self.num_channel, 3, 1, 1)
            self.l17 = nn.Linear(8*8*7*self.num_channel, self.h)




    '''def forward(self, input):
        x = F.elu(self.l0(input), True)
        x = F.elu(self.l1(x), True)
        x = F.elu(self.l2(x), True)
        x = self.down1(x)
        x = self.pool1(x)

        x = F.elu(self.l3(x), True)
        x = F.elu(self.l4(x), True)
        x = self.pool2(self.down2(x))

        x = F.elu(self.l5(x), True)
        x = F.elu(self.l6(x), True)
        x = self.pool3(self.down3(x))

        if self.scale_size == 64:
            x = F.elu(self.l7(x), True)
            x = F.elu(self.l8(x), True)
            x = x.view(self.b_size, 8*8*3*self.num_channel)
            x = self.l9(x)
        else:
            x = F.elu(self.l7(x), True)
            x = F.elu(self.l8(x), True)
            x = self.down4(x)
            x = self.pool4(x)
            x = F.elu(self.l9(x), True)
            x = F.elu(self.l11(x), True)
            x = x.view(self.b_size, 8*8*4*self.num_channel)
            x = F.elu(self.l12(x), True)

        return x'''

    def forward(self, input):
        #import pdb;pdb.set_trace()
        x = F.elu(self.l0(input), True)
        x = F.elu(self.l1(x), True)
        x = F.elu(self.l2(x), True)
        x = self.down1(x)
        x = self.pool1(x)

        x = F.elu(self.l3(x), True)
        x = F.elu(self.l4(x), True)
        x = self.pool2(self.down2(x))

        x = F.elu(self.l5(x), True)
        x = F.elu(self.l6(x), True)
        x = self.pool3(self.down3(x))

        #here x is (b, 192, 16, 16) for 128 input

        if self.scale_size == 64:
            x = F.elu(self.l7(x), True)
            x = F.elu(self.l8(x), True)
            #x = x.view(self.b_size, 8*8*3*self.num_channel)
            x = x.view(-1, 8*8*3*self.num_channel)
            x = self.l9(x)
        elif self.scale_size == 128:
            x = F.elu(self.l7(x), True)
            x = F.elu(self.l8(x), True)
            x = self.down4(x)
            x = self.pool4(x)
            x = F.elu(self.l9(x), True)
            x = F.elu(self.l11(x), True)
            #here x is (b, 256, 8, 8)
            #x = x.view(self.b_size, 8*8*4*self.num_channel)
            x = x.view(-1, 8*8*4*self.num_channel)
            x = F.elu(self.l12(x), True)
            #here x is (b, h=64)
        elif self.scale_size == 256:
            x = F.elu(self.l7(x), True)
            x = F.elu(self.l8(x), True)
            x = self.down4(x)
            x = self.pool4(x)

            x = F.elu(self.l9(x), True)
            x = F.elu(self.l10(x), True)
            x = self.down5(x)
            x = self.pool5(x)

            x = F.elu(self.l11(x), True)
            x = F.elu(self.l12(x), True)
            #x = x.view(self.b_size, 8*8*5*self.num_channel)
            x = x.view(-1, 8*8*5*self.num_channel)
            x = F.elu(self.l13(x), True)
        elif self.scale_size == 512:
            x = F.elu(self.l7(x), True)
            x = F.elu(self.l8(x), True)
            x = self.down4(x)
            x = self.pool4(x)

            x = F.elu(self.l9(x), True)
            x = F.elu(self.l10(x), True)
            x = self.down5(x)
            x = self.pool5(x)

            x = F.elu(self.l11(x), True)
            x = F.elu(self.l12(x), True)
            x = self.down6(x)
            x = self.pool6(x)

            x = F.elu(self.l13(x), True)
            x = F.elu(self.l14(x), True)
            #x = x.view(self.b_size, 8*8*6*self.num_channel)
            x = x.view(-1, 8*8*6*self.num_channel)
            x = F.elu(self.l15(x), True)

        elif self.scale_size == 1024:
            x = F.elu(self.l7(x), True)
            x = F.elu(self.l8(x), True)
            x = self.down4(x)
            x = self.pool4(x)

            x = F.elu(self.l9(x), True)
            x = F.elu(self.l10(x), True)
            x = self.down5(x)
            x = self.pool5(x)

            x = F.elu(self.l11(x), True)
            x = F.elu(self.l12(x), True)
            x = self.down6(x)
            x = self.pool6(x)

            x = F.elu(self.l13(x), True)
            x = F.elu(self.l14(x), True)
            x = self.down7(x)
            x = self.pool7(x)

            x = F.elu(self.l15(x), True)
            x = F.elu(self.l16(x), True)
            #x = x.view(self.b_size, 8*8*7*self.num_channel)
            x = x.view(-1, 8*8*7*self.num_channel)
            x = F.elu(self.l17(x), True)



        return x

class Discriminator(nn.Module):
    def __init__(self, nc):
        super(Discriminator, self).__init__()
        self.enc = Encoder(nc)
        self.dec = Decoder(nc, True)
    def forward(self, input):
        return self.dec(self.enc(input))


def weights_init(self, m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class _Loss(nn.Module):

    def __init__(self, size_average=True):
        super(_Loss, self).__init__()
        self.size_average = size_average

    def forward(self, input, target):
        # this won't still solve the problem
        # which means gradient will not flow through target
        # _assert_no_grad(target)
        backend_fn = getattr(self._backend, type(self).__name__)
        return backend_fn(self.size_average)(input, target)

class L1Loss(_Loss):
    r"""Creates a criterion that measures the mean absolute value of the
    element-wise difference between input `x` and target `y`:

    :math:`{loss}(x, y)  = 1/n \sum |x_i - y_i|`

    `x` and `y` arbitrary shapes with a total of `n` elements each.

    The sum operation still operates over all the elements, and divides by `n`.

    The division by `n` can be avoided if one sets the constructor argument `sizeAverage=False`
    """
    pass


