

import numpy as np
import torch
import os
import yaml
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate
from lama.saicinpainting.evaluation.utils import move_to_device
from lama.saicinpainting.training.trainers import load_checkpoint


def load_model():
    model_path = './lama/big-lama'
    device = torch.device("cpu")

    train_config_path = os.path.join(model_path, 'config.yaml')
    with open(train_config_path, 'r') as f:
        train_config = OmegaConf.create(yaml.safe_load(f))

    train_config.training_model.predict_only = True
    train_config.visualizer.kind = 'noop'

    # out_ext = predict_config.get('out_ext', '.png')

    checkpoint_path = os.path.join(model_path,
                                   'models',
                                   'best.ckpt')
    model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
    model.freeze()
    return model


def ceil_modulo(x, mod):
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod

def pad_img_to_modulo(img, mod):
    channels, height, width = img.shape
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)
    return np.pad(img, ((0, 0), (0, out_height - height), (0, out_width - width)), mode='symmetric')

def load_image(img, mode='RGB', return_orig=False):
    # img = np.array(Image.open(fname).convert(mode))
    if img.ndim == 3:
        img = np.transpose(img, (2, 0, 1))
    out_img = img.astype('float32') / 255
    if return_orig:
        return out_img, img
    else:
        return out_img

def extend_image(img, mask, model):
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    device = torch.device("cpu")

    img = load_image(img)
    mask = load_image(mask)

    item = dict(image=img, mask=mask)
    pad_out_to_modulo = 8
    item['unpad_to_size'] = item['image'].shape[1:]
    item['image'] = pad_img_to_modulo(item['image'], pad_out_to_modulo)
    item['mask'] = pad_img_to_modulo(item['mask'], pad_out_to_modulo)

    batch = default_collate([item])

    with torch.no_grad():
        batch = move_to_device(batch, device)
        batch['mask'] = (batch['mask'] > 0) * 1
        batch = model(batch)
        cur_res = batch['inpainted'][0].permute(1, 2, 0).detach().cpu().numpy()
        unpad_to_size = batch.get('unpad_to_size', None)
        if unpad_to_size is not None:
            orig_height, orig_width = unpad_to_size
            cur_res = cur_res[:orig_height, :orig_width]

    return cur_res