import glob
import torch
import random
import os.path
import argparse
import numpy as np
import watermarklab as wl
from numpy import ndarray
from typing import List, Any
from models.nets import Model
from PIL import Image, ImageOps
from compressor.rdh import CustomRDH
from watermarklab.laboratories import WLab
from watermarklab.utils.data import DataLoader
from compressor.utils_compressors import TensorCoder
from watermarklab.noiselayers.testdistortions import *
from watermarklab.utils.basemodel import BaseWatermarkModel, Result, BaseDataset, NoiseModelWithFactors


class DRRW(BaseWatermarkModel):
    def __init__(self, device, img_size, channel_dim, bit_length, min_size, k, fc, model_save_path: str, level_bits_len,
                 freq_bits_len, modelname: str, height_end=5, compress_mode="a"):
        super().__init__(bit_length, img_size, modelname)
        self.device = device
        self.bit_length = bit_length
        torch.set_default_dtype(torch.float64)
        self.model = Model(img_size, channel_dim, bit_length, k, min_size, fc)
        self.model.load_model(model_save_path)
        self.model.to(self.device)
        self.model.eval()
        self.compress_mode = compress_mode
        self.rdh = CustomRDH((img_size, img_size, channel_dim), height_end)
        self.tensorcoder = TensorCoder((img_size, img_size, channel_dim), (1, bit_length), level_bits_len,
                                       freq_bits_len)

    def embed(self, cover_list: List[Any], secrets: List[List]) -> Result:
        _cover_tensor = torch.as_tensor(np.stack(cover_list)).permute(0, 3, 1, 2) / 255.
        _secrets_tensor = torch.as_tensor(secrets) / 1.
        with torch.no_grad():
            secret_tensor = _secrets_tensor.to(self.device)
            cover_tensor = _cover_tensor.to(self.device)
            cover_tensor = cover_tensor.to(torch.float64)
            secret_tensor = secret_tensor.to(torch.float64)
            stego, drop_z = self.model(cover_tensor, secret_tensor, True, False)
            stego_255 = torch.round(stego * 255.)
            drop_z_round = torch.round(drop_z)
        stego_list = []
        for i in range(stego_255.shape[0]):
            clip_stego, aux_bits_tuple = self.tensorcoder.compress(stego_255[i].unsqueeze(0),
                                                                   drop_z_round[i].unsqueeze(0),
                                                                   mode=self.compress_mode)
            data_list, drop_z_bits, overflow_bits = aux_bits_tuple
            _, rw_stego_img = self.rdh.embed(clip_stego, data_list)
            stego_list.append(rw_stego_img)
        res = Result(stego_img=stego_list)
        return res

    def extract(self, stego_list: List[ndarray]) -> Result:
        _stego_tensor = torch.as_tensor(np.stack(stego_list)).permute(0, 3, 1, 2) / 255.
        with torch.no_grad():
            stego_tensor = _stego_tensor.to(self.device)
            z_tensor = torch.randn(size=(len(stego_tensor), self.bit_length)).to(self.device)
            stego_tensor = stego_tensor.to(torch.float64)
            z_tensor = z_tensor.to(torch.float64)
            _, ext_secrets = self.model(stego_tensor, z_tensor, True, True)
            ext_secrets = torch.round(torch.clip(ext_secrets, 0, 1))
        secret_list = []
        for i in range(ext_secrets.shape[0]):
            secret = ext_secrets[i].cpu().detach().numpy().astype(int).tolist()
            secret_list.append(secret)
        res = Result(ext_bits=secret_list)
        return res

    def recover(self, stego_list: List[ndarray]) -> Result:
        pass


class Mydataloader(BaseDataset):
    def __init__(self, root_path: str, im_size: int, bit_length, iter_num: int):
        super().__init__(iter_num)
        self.root_path = root_path
        self.bit_length = bit_length
        self.im_size = im_size
        self.covers = []
        self.load_paths()

    def load_paths(self):
        self.covers = glob.glob(os.path.join(self.root_path, '*.png'), recursive=True)

    def load_cover_secret(self, index: int):
        cover = Image.open(self.covers[index])
        cover = np.float32(ImageOps.fit(cover, (self.im_size, self.im_size)))
        random.seed(index)
        secret = [random.randint(0, 1) for _ in range(self.bit_length)]
        return cover, secret

    def get_num_covers(self):
        return len(self.covers)


if __name__ == '__main__':
    testnoisemodels = [
        NoiseModelWithFactors(noisemodel=Jpeg(), noisename="Jpeg Compression", factors=[10, 30, 50, 70, 90],
                              factorsymbol="$Q_f$"),
        NoiseModelWithFactors(noisemodel=SaltPepperNoise(), noisename="Salt&Pepper Noise",
                              factors=[0.1, 0.3, 0.5, 0.7, 0.9], factorsymbol="$p$"),
        NoiseModelWithFactors(noisemodel=GaussianNoise(), noisename="Gaussian Noise",
                              factors=[0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45], factorsymbol="$\sigma$"),
        NoiseModelWithFactors(noisemodel=GaussianBlur(), noisename="Gaussian Blur",
                              factors=[3.0, 4.0, 5.0, 6.0, 7.0], factorsymbol="$\sigma$"),
        NoiseModelWithFactors(noisemodel=MedianFilter(), noisename="Median Filter", factors=[11, 13, 15, 17, 19],
                              factorsymbol="$w$"),
        NoiseModelWithFactors(noisemodel=Dropout(), noisename="Dropout", factors=[0.1, 0.3, 0.5, 0.7, 0.9],
                              factorsymbol="$p$"),
    ]

    wlab = WLab("save_new/realflow_compare", noise_models=testnoisemodels)

    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--channel_dim', type=int, default=3)
    parser.add_argument('--bit_length', type=int, default=64)
    parser.add_argument('--min_size', type=int, default=16)
    parser.add_argument('--iter_num', type=int, default=5)
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--level_bits_len', type=int, default=10)
    parser.add_argument('--freq_bits_len', type=int, default=10)
    parser.add_argument('--fc', type=bool, default=False)
    parser.add_argument('--model_name', type=str, default="DRRW-R")
    parser.add_argument('--model_save_path', type=str, default=r"/")
    parser.add_argument('--dataset_path', type=str, default=r"/")
    parser.add_argument('--device', type=str, default=r"cuda:4")
    parser.add_argument('--seed', type=int, default=99)
    args = parser.parse_args()
    drrw_r = DRRW(args.device, args.img_size, args.channel_dim, args.bit_length, args.min_size, args.k,
                  args.fc, args.model_save_path, args.level_bits_len, args.freq_bits_len, args.model_name)
    mydataset = Mydataloader(args.dataset_path, args.img_size, args.bit_length, args.iter_num)
    dataloader = DataLoader(mydataset, args.batchsize)
    result = wlab.test(drrw_r, dataloader)
    wl.plot_robustness([result], r"save/draw/")
