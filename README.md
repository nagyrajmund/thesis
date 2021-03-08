# thesis
Ongoing thesis project on improving integrated gradients with latent interpolation


# Installation notes
- Install dependencies as described in [](DEPENDENCIES.MD)
- Download pretrained inpainting models with gdown (`pip install gdown`):
  ```
  gdown https://drive.google.com/uc?id=1dyPD2hx0JTmMuHYa32j-pu--MXqgLgMy -O utils/deepfill_checkpoint
  gdown https://drive.google.com/uc?id=1z9dbEAzr5lmlCewixevFMTVBmNuSNAgK -O utils/deepfill_checkpoint
  gdown https://drive.google.com/uc?id=1ExY4hlx0DjVElqJlki57la3Qxu40uhgd -O utils/deepfill_checkpoint
  gdown https://drive.google.com/uc?id=1C7kPxqrpNpQF7B2UAKd_GvhUMk0prRdV -O utils/deepfill_checkpoint
  ```
- Download the pretrained dVAE from DALL-E:
  ```
  wget https://cdn.openai.com/dall-e/encoder.pkl -P utils/dall_e_checkpoint/
  wget https://cdn.openai.com/dall-e/decoder.pkl -P utils/dall_e_checkpoint/
  ```
