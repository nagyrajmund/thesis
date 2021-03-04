```
conda create -n thesis python=3.7

conda install pytorch torchvision torchaudio cpuonly -c pytorch

python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.7/index.html

pip install opencv-python

python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

pip install tensorflow==1.15
pip install git+https://github.com/JiahuiYu/neuralgym
```
