import os
import time
from functools import partial
from hashlib import md5
from io import BytesIO
from multiprocessing import Pool
from typing import Union

import requests
from PIL import Image


class ImageStreamer:
    def __init__(self, size: Union[int, tuple, None] = None):
        if isinstance(size, int):
            size = (size, size)
        assert len(size) == 2 or size is None, 'Invalid size'
        self.size = size

    def download(self, src: str, dst: str, delay: float = 1.) -> None:
        response = requests.get(src)
        print('Downloading from %s [%d]' % (src, response.status_code), end='')
        if response.ok:
            content = response.content
            random_name = md5(content).hexdigest()
            path = os.path.join(dst, '%s.jpg' % random_name)
            if self.size is None:
                with open(path, 'wb') as f:
                    f.write(content)
            else:
                image = Image.open(BytesIO(content)).resize(self.size)
                image.save(path)
            print('[%s Success]' % random_name)
        else:
            print('[Failed]')
        time.sleep(delay)

    def parallel_download(self, src_list: Union[list, tuple], dst: str, num_workers: int = 1, delay: float = 1.) -> None:
        with Pool(processes=num_workers) as p:
            p.map(partial(self.download, dst=dst, delay=delay), src_list)
