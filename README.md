# thesis
Ongoing thesis project on improving integrated gradients with latent interpolation

***
# TODO
- Implement IG for the places classifier
- Measure the difference between the explanations for the original image and the baseline
- Implement the insertion/deletion metric + visualizaiton
- Implement the latent version of IG
- Compare latent IG with original IG on the full dataset using the previous metrics
  - Check the numerical stability of the integral approximation
  - Random segmentations as baseline
  - Great circle interpolation



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

