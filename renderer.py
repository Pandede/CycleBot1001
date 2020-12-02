import os
from typing import Union, Dict, Tuple, List

import numpy as np
import torch
import torch.multiprocessing as mp
import torchvision.transforms as tf
from PIL import Image
from torch.utils.data import TensorDataset, DataLoader
from torchvision.io import read_video, write_video
from torchvision.utils import save_image


class Stylizer:
    def __init__(self, g_src: str, domain: str = 'b', device='cpu'):
        model_data = torch.load(g_src, map_location=device)
        self.generator = model_data['generator_%s' % domain].to(device)
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
                styled_video[i * batch_size:(i + 1) * batch_size] = self.render(batch[0]).cpu()
                print('%d/%d' % (i, len(loader)))

        styled_video = styled_video.permute((0, 2, 3, 1)) * 127.5 + 127.5
        write_video(styled_name, styled_video, float(info['video_fps']))


class Slice:
    @staticmethod
    def apply(images: Tuple[np.ndarray, ...],
              breakpoints: Union[np.ndarray, Tuple[int], List[int]],
              axis: int = 1) -> np.ndarray:
        assert len(images) == len(breakpoints) - 1, 'The (number of breakpoints) must equal to (number of images - 1)'
        h, w, c = images[0].shape
        assert all(i.shape == (h, w, c) for i in images), 'All images must have identical dimensions'

        result = np.empty_like(images[0])
        pts = [breakpoints[i:i + 2] for i in range(len(breakpoints) - 1)]

        for i, (p0, p1) in enumerate(pts):
            fill_axis = (slice(None), slice(p0, p1)) if axis else (slice(p0, p1), slice(None))
            result[fill_axis] = images[i][fill_axis]

        return result


class GradientSlice:
    @classmethod
    def apply(cls,
              images: Tuple[np.ndarray, ...],
              breakpoints: Union[np.ndarray, Tuple[int], List[int]],
              axis: int = 1) -> np.ndarray:
        assert len(images) == len(breakpoints) // 2, 'The (number of breakpoints) must equal to (number of images * 2)'
        h, w, c = images[0].shape
        assert all(i.shape == (h, w, c) for i in images), 'All images must have identical dimensions'

        result = np.empty_like(images[0])
        pts = [breakpoints[i:i + 2] for i in range(len(breakpoints) - 1)]

        for i, (p0, p1) in enumerate(pts[::2]):
            fill_axis = (slice(None), slice(p0, p1)) if axis else (slice(p0, p1), slice(None))
            result[fill_axis] = images[i][fill_axis]

        for i, (p0, p1) in enumerate(pts[1::2]):
            fill_axis = (slice(None), slice(p0, p1)) if axis else (slice(p0, p1), slice(None))
            result[fill_axis] = cls.blend(images[i][fill_axis],
                                          images[i + 1][fill_axis],
                                          axis)

        return result

    @staticmethod
    def blend(img1: np.ndarray, img2: np.ndarray, axis: int) -> np.ndarray:
        assert img1.shape == img2.shape
        h, w, _ = img1.shape
        g = np.linspace(0, 1, w if axis else h)
        subscripts = 'ijk,%s->ijk' % ('j' if axis else 'i')
        return np.einsum(subscripts, img1, 1 - g) + np.einsum(subscripts, img2, g)
