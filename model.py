import os
import glob
import numpy as np
from PIL import Image
import torch
import runway
from options.test_options import TestOptions
from models.models import create_model
from data.base_dataset import get_params, get_transform
import util.util as util


checkpoints = [c.replace('checkpoints/', '') for c in glob.glob('checkpoints/*')]


def get_model_args(model_name):
    file_name = os.path.join(os.path.join('checkpoints/', model_name), 'opt.txt')
    with open(file_name) as f:
        content = f.readlines()
    args_all = [ l.split('\n')[0].split(': ') for l in content[1:-1] ]
    args = {}
    for a in args_all:
        args[a[0]] = a[1]
    return args


@runway.setup(options={"model_name": runway.category(choices=checkpoints, default=checkpoints[0]), 'which_epoch': runway.number })
def setup(options):
    global opt
    model_name = options['model_name']
    which_epoch = options['which_epoch']
    opt = TestOptions().parse(save=False)
    args = get_model_args(model_name)
    opt.nThreads = 1
    opt.batchSize = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.name = model_name
    opt.resize_or_crop = args['resize_or_crop']
    opt.use_features = eval(args['no_instance'])
    opt.no_instance = eval(args['no_instance'])
    opt.label_nc = int(args['label_nc'])
    opt.ngf = int(args['ngf'])
    opt.ndf = int(args['ndf'])
    opt.which_epoch = which_epoch if which_epoch > 0 else opt.which_epoch
    model = create_model(opt)
    return model


@runway.command('convert', inputs={'image': runway.image}, outputs={'output': runway.image})
def convert(model, inp):
    img = np.array(inp['image'])
    h, w = img.shape[0:2]
    img = Image.fromarray(img.astype('uint8'))
    params = get_params(opt, (w, h))
    transform_label = get_transform(opt, params, method=Image.NEAREST, normalize=False)
    label_tensor = transform_label(img)
    label_tensor = label_tensor.unsqueeze(0)
    output = model.inference(label_tensor, None)
    torch.cuda.synchronize()
    output = util.tensor2im(output.data[0])
    output = Image.fromarray(output)
    return dict(output=output)


if __name__ == '__main__':
    runway.run()

