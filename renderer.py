import os
from typing import Union, Tuple, Dict

import torch
import torch.multiprocessing as mp
import torchvision.transforms as tf
from PIL import Image
from torch.utils.data import TensorDataset, DataLoader
from torchvision.io import read_video, write_video
from torchvision.utils import save_image

from model import Generator


class Stylizer:
    def __init__(self, g_src: str, domain: str = 'b', device='cpu'):
        model_data = torch.load(g_src, map_location=device)
        self.generator = Generator().to(device)
        self.generator.load_state_dict(model_data['generator_%s' % domain])
        self.generator.eval()

        self.image_transforms = tf.Compose([tf.ToTensor(),
                                            tf.Normalize((.5, .5, .5), (.5, .5, .5))])
        self.video_transforms = tf.Compose([tf.Lambda(self.norm),
                                            tf.Normalize((.5, .5, .5), (.5, .5, .5))])

        self.device = device

    @staticmethod
    def norm(x):
        return x.permute((0, 3, 1, 2)) / 255.

    def render(self, src: Union[str, torch.Tensor]) -> torch.Tensor:
        if isinstance(src, str):
            img = self.fetch_image(src)
        else:
            img = src
        img = img.unsqueeze_(0) if len(img.size()) == 3 else img
        with torch.no_grad():
            return self.generator(img.to(self.device))

    def fetch_image(self, src: str) -> torch.Tensor:
        img = Image.open(src).convert('RGB')
        return self.image_transforms(img)

    def fetch_video(self, src: str) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, str]]:
        video, audio, info = read_video(src)
        video = self.video_transforms(video)
        return video, audio, info

    @staticmethod
    def save(img: torch.Tensor, file_name: str) -> None:
        save_image(img, file_name, padding=0)
        print('[Saved] %s' % file_name)

    def render_image(self, src: Union[str, tuple, list], dst: str, num_workers: int = 1) -> None:
        if isinstance(src, str):
            src = [src]

        with mp.Pool(processes=min(num_workers, len(src))) as p:
            styled_img = p.map(self.render, src)

        for img, s in zip(styled_img, src):
            styled_name = os.path.join(dst, os.path.basename(s))
            self.save(img, styled_name)

    def render_video(self, src: str, dst: str, batch_size: int = 1, num_workers: int = 1) -> None:
        video, _, info = self.fetch_video(src)

        dataset = TensorDataset(video)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        styled_video = torch.zeros_like(video)
        styled_name = os.path.join(dst, os.path.basename(src))
        with torch.no_grad():
            for i, batch in enumerate(loader):
                styled_video[i*batch_size:(i+1)*batch_size] = self.render(batch[0]).cpu()
                print('%d/%d' % (i, len(loader)))

        styled_video = styled_video.permute((0, 2, 3, 1)) * 127.5 + 127.5
        write_video(styled_name, styled_video, float(info['video_fps']))
