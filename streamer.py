import os
import time
from functools import partial
from hashlib import md5
from multiprocessing import Pool
from typing import Union

import requests


class ImageStreamer:
    @staticmethod
    def download(src: str, dst: str):
        response = requests.get(src)
        print('Downloading from %s [%d]' % (src, response.status_code), end='')
        if response.ok:
            content = response.content
            random_name = md5(content).hexdigest()
            with open(os.path.join(dst, '%s.jpg' % random_name), 'wb') as f:
                f.write(content)
                print('[%s Success]' % random_name)
        else:
            print('[Failed]')
        time.sleep(1)

    def parallel_download(self, src_list: Union[list, tuple], dst: str, num_workers: int = 1):
        p = Pool(processes=num_workers)
        p.map(partial(self.download, dst=dst), src_list)
        p.close()
        p.join()