```
conda create -n thesis python=3.7

pip install torch==1.8.0+cpu torchvision==0.9.0+cpu torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html

# segmentation model
pip install opencv-python
pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install scikit-image

# inpainting model 
pip install tensorflow==1.15
pip install git+https://github.com/JiahuiYu/neuralgym

# generative model 
pip install dall-e

# scene recognition model
pip install imgaug

pip install -e .
```
