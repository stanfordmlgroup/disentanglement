''' Conversion script for celeba StyleGAN weights from Tensorflow to PyTorch.
Not directly used anymore, but kept in case it is needed in future.

Adapted from
https://github.com/lernapparat/lernapparat/blob/master/style_gan/pytorch_style_gan.ipynb
'''

from model import *
from pdb import set_trace as st
import collections

from matplotlib import pyplot
import torchvision

'''
I removed the dnnlib library to clean up the repo
since we don't need to convert weights anymore.
If needed in future just redownload from NVlabs/StyleGAN website.

import dnnlib, dnnlib.tflib
'''

TF_CELEBA_64 = '/deep/u/fredlu/coord_trans/celeba_gan/StyleGAN/network-snapshot-016085.pkl'
TORCH_OUT_PT = '/deep/u/fredlu/coord_trans/celeba_gan/StyleGAN/stylegan_celeba_64.pt'
TORCH_OUT_GS_PT = '/deep/u/fredlu/coord_trans/celeba_gan/StyleGAN/stylegan_celeba_64_g_all.pt'


g_all = nn.Sequential(OrderedDict([
    ('g_mapping', G_mapping()),
    #('truncation', Truncation(avg_latent)),
    ('g_synthesis', G_synthesis(
        dlatent_size=512, resolution=64, blur_filter=None))
]))


# this can be run to get the weights, but you need the reference implementation and weights
if 0:
    dnnlib.tflib.init_tf()

    weights = pickle.load(open(TF_CELEBA_64, 'rb'))
    weights_pt = [collections.OrderedDict([(k, torch.from_numpy(v.value().eval())) for k,v in w.trainables.items()]) for w in weights]
    torch.save(weights_pt, TORCH_OUT_PT)

# then on the PyTorch side run
state_G, state_D, state_Gs = torch.load(TORCH_OUT_PT)

def key_translate(k):
    k = k.lower().split('/')
    if k[0] == 'g_synthesis':
        if not k[1].startswith('torgb'):
            k.insert(1, 'blocks')
        k = '.'.join(k)
        k = (k.replace('const.const','const').replace('const.bias','bias').replace('const.stylemod','epi1.style_mod.lin')
                .replace('const.noise.weight','epi1.top_epi.noise.weight')
                .replace('conv.noise.weight','epi2.top_epi.noise.weight')
                .replace('conv.stylemod','epi2.style_mod.lin')
                .replace('conv0_up.noise.weight', 'epi1.top_epi.noise.weight')
                .replace('conv0_up.stylemod','epi1.style_mod.lin')
                .replace('conv1.noise.weight', 'epi2.top_epi.noise.weight')
                .replace('conv1.stylemod','epi2.style_mod.lin')
                .replace('torgb_lod0','torgb'))
    else:
        k = '.'.join(k)
    return k

def weight_translate(k, w):
    k = key_translate(k)
    if k.endswith('.weight'):
        if w.dim() == 2:
            w = w.t()
        elif w.dim() == 1:
            pass
        else:
            assert w.dim() == 4
            w = w.permute(3, 2, 0, 1)
    return w

# we delete the useless torgb filters
# NB: ^ that is as stated in the jupyter notebook. The implementation does not have torgb.
# results are fine for 1024px and there are no filter mismatches for 64px.
param_dict = {key_translate(k) : weight_translate(k, v) for k,v in state_Gs.items() if 'torgb_lod' not in key_translate(k)}
if 1:
    sd_shapes = {k : v.shape for k,v in g_all.state_dict().items()}
    param_shapes = {k : v.shape for k,v in param_dict.items() }

    print(len(sd_shapes))
    print(len(param_shapes))
    for k in list(sd_shapes)+list(param_shapes):
        pds = param_shapes.get(k)
        sds = sd_shapes.get(k)
        if pds is None:
            print ("sd only", k, sds)
        elif sds is None:
            print ("pd only", k, pds)
        elif sds != pds:
            print ("mismatch!", k, pds, sds)


g_all.load_state_dict(param_dict, strict=False) # needed for the blur kernels
torch.save(g_all.state_dict(), TORCH_OUT_GS_PT)


def test_image_gen():
    ''' refer to debug_gan.py instead for image sampling. '''
    g_all.load_state_dict(
        torch.load('/deep/u/fredlu/coord_trans/celeba_gan/StyleGAN/karras2019stylegan-ffhq-1024x1024.for_g_all.pt'))

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    g_all.eval()
    g_all.to(device)

    torch.manual_seed(20)
    nb_rows = 2
    nb_cols = 5
    nb_samples = nb_rows * nb_cols
    latents = torch.randn(nb_samples, 512, device=device)
    with torch.no_grad():
        imgs = g_all(latents)
        imgs = (imgs.clamp(-1, 1) + 1) / 2.0 # normalization to 0..1 range
    imgs = imgs.cpu()

    vutils.save_image(imgs.data, './test.png', nrow=2, normalize=True)
