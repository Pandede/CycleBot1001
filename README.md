# CycleBot1001
Transfer the painting style between real-life and Van Gogh's by using CycleGAN. \
The instagram account: http://www.instagram.com/cyclebot1001/

## Preparation
1. Prepare the images of 2 domain for training, saved as following structure:
```
+-- Data
|   +-- domain_a
|   |   +-- 0001.png
|   |   +-- 0002.png
|   |   +-- ...
|   +-- domain_b
|   |   +-- 0001.png
|   |   +-- 0002.png
|   |   +-- ...
```
For the convenience, you may download the **image URLs** from the following sources, including Van Gogh's painting *(Domain A)* and landmarks in real life *(Domain B)*:

| Name     | Size   | Type | Link   |
|----------|--------|------|--------|
| Van Gogh | ~107MB | CSV  | [Google Drive](https://drive.google.com/file/d/1K51AtogQdSMnNEV5irpE1IAAVd-1DVCR/view?usp=sharing) |
| Landmark | ~119KB | CSV  | [Google Drive](https://drive.google.com/file/d/1s4jAnVWdJ_vzbPWv83dx_v_YQS_RL4Kh/view?usp=sharing) |

Then, execute the following code in python console, for example, downloading images of Van Gogh *(Domain A)*:
```python
import pandas as pd
from streamer import ImageStreamer

src_list = pd.read_csv('./van_gogh.csv')['ImageURL']
s = ImageStreamer(1024)
s.parallel_download(src_list, './Data/domain_a/', num_workers=8, delay=1.) 
```
- The argument in `ImageStreamer` denoted the image size of image. It can be `tuple`, `int` or `None`: 
    - the image will be resized in (1024, 256) if the argument is `tuple(1024, 256)`
    - the image will be resized in (1024, 1024) if the argument is `int(1024)`
    - the image will not be resized if the argment is `None`
- `num_workers` denoted the number of threads that requesting the content simultaneously, \
- `delay` denoted the time interval (in sec) between each request. The usage of `delay` avoids the `Error code 429: Too many requests`.

2. Edit `config.ini`
```ini
[default]
# Number of training epochs
epoch = 1000
# Save the model each n epochs
save_per_epoch = 20
# Sampling size of generating image each epoch
sample_size = 6

# The image size of generating immage
img_size = 1024
# The image size for random cropped
crop_size = 256
# The number of image channels (should be always RGB)
img_channel = 3
# Batch size
batch_size = 5

[path]
# The folder path of images of Domain A
domain_A_src = ./Data/domain_a/
# The folder path of images of Domain B
domain_B_src = ./Data/domain_b/
# The folder path of saved model
model_src = ./Pickle/
# The folder path for saving the generated samples
sample_src = ./Sample/ab/

[cuda]
# The device string (training on CPU if device=cpu or GPU if device=cuda)
device = cuda:0

[loss]
# The lambda of cycle consistency loss for Domain A
lambda_a = 10.0
# The lambda of cycle consistency loss for Domain B
lambda_b = 10.0
# The lambda of identity loss for both domains
lambda_idt = 0.5

[wgan]
# The clipping limit of weight (WGAN)
weight_clipping_limit = 0.01
```

## Run
`python main.py`

## Evaluation
**Images** or **Videos** can be rendered by `Stylizer` in `renderer.py`:
```python
from renderer import Stylizer

# Render a single image
file_name = './sample.jpg'
stylizer = Stylizer('./Pickle/cyclegan.pkl', domain='b', device='cuda:0')
stylizer.render_image(file_name, './Output/', num_workers=1)

# Render all images in folder
from glob import glob
file_name = glob('./Sample/*.jpg')
stylizer.render_image(file_name, './Output/', num_workers=8)

# Render video (Beta)
file_name = './sample.mp4'
stylizer.render_video(file_name, './Output/', num_workers=8)
```

## Result
Several filters are implemented in `rendered.py` for combining the original and rendered images
| Image              | Description                            | Example                                                                                                             |
|--------------------|----------------------------------------|---------------------------------------------------------------------------------------------------------------------|
| Original           | The genuine image                      | <img src="https://github.com/Pandede/CycleBot1001/blob/master/assets/heidelberg_castle.jpg" width="300px">          |
| Rendered           | The image which rendered by GAN        | <img src="https://github.com/Pandede/CycleBot1001/blob/master/assets/styled_heidelberg_castle.jpg" width="300px">   |
| **Slice**          | Combine the images simply              | <img src="https://github.com/Pandede/CycleBot1001/blob/master/assets/slice_heidelberg_castle.jpg" width="300px">    |
| **Gradient Slice** | Combine the images with gradient blend | <img src="https://github.com/Pandede/CycleBot1001/blob/master/assets/gradient_heidelberg_castle.jpg" width="300px"> |
