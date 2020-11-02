import os
from multiprocessing import Pool
from typing import Union

import torch
import torchvision.transforms as tf
from PIL import Image
from torchvision.utils import save_image

from model import Generator


class Stylizer:
    def __init__(self, g_src: str, domain: str = 'b'):
        model_data = torch.load(g_src, map_location='cpu')
        self.generator = Generator()
        self.generator.load_state_dict(model_data['generator_%s' % domain])
        self.generator.eval()

        self.transforms = tf.Compose([tf.ToTensor(),
                                      tf.Normalize((.5, .5, .5), (.5, .5, .5))])

    def render(self, src: str) -> torch.Tensor:
        img = self.imread(src)
        with torch.no_grad():
            return self.generator(img.unsqueeze_(0) if len(img.size()) == 3 else img)

    def imread(self, src: str) -> torch.Tensor:
        img = Image.open(src).convert('RGB')
        return self.transforms(img)

    @staticmethod
    def save(img: torch.Tensor, file_name: str) -> None:
        save_image(img, file_name, padding=0)
        print('[Saved] %s' % file_name)

    def render_image(self, src: Union[str, tuple, list], dst: str, num_workers: int = 1) -> None:
        if isinstance(src, str):
            src = [src]

        with Pool(processes=num_workers) as p:
            styled_img = p.map(self.render, src)

        styled_name = [os.path.join(dst, os.path.basename(s)) for s in src]
        with Pool(processes=num_workers) as p:
            p.starmap(self.save, zip(styled_img, styled_name))
