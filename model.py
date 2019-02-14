import numpy as np
from PIL import Image
import torch
from runway import RunwayModel
from options.test_options import TestOptions
from models.models import create_model
from data.base_dataset import get_params, get_transform
import util.util as util

pix2pixhd = RunwayModel()

@pix2pixhd.setup
def setup():
    model_name = 'landscapes_1_80'
    global opt
    opt = TestOptions().parse(save=False)
    opt.nThreads = 1
    opt.batchSize = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.name = model_name
    opt.resize_or_crop = 'none'
    opt.use_features = False
    opt.no_instance = True
    opt.label_nc = 0
    model = create_model(opt)
    return model

@pix2pixhd.command('convert', inputs={'image': 'image'}, outputs={'output': 'image'})
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
    pix2pixhd.run()
