# Pic2Word (CVPR2023)

This is an open source implementation of [Pic2Word](https://arxiv.org/pdf/2302.03084.pdf). This is not an
officially supported Google product.


## Data

### Training Data
We utilize [Conceptual Captions URLs](https://ai.google.com/research/ConceptualCaptions/download) to train a model. 
See [open_clip](https://github.com/mlfoundations/open_clip) to see the process of getting the dataset. 

The training data directory has to be in the root of this repo, and should be structured like below.
```bash
  cc_data
    ├── train ## training image diretories.
    └── val ## validation image directories.
  cc
    ├── Train_GCC-training_output.csv ## training data list
    └── Validation_GCC-1.1.0-Validation_output.csv ## validation data list
```

### Test Data
See [README](data/README.md) to prepare test dataset.

## Training

### Install dependencies
See [open_clip](https://github.com/mlfoundations/open_clip) for the details of installation. 
The same environment should be usable in this repo.
setenv.sh is the script we used to set-up the environment in virtualenv. 

Also run below to add directory to pythonpath:
```bash
. env3/bin/activate
export PYTHONPATH="$PYTHONPATH:$PWD/src"
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
```
### Pre-trained model
The model is available in [GoogleDrive](https://drive.google.com/file/d/1IxRi2Cj81RxMu0ViT4q4nkfyjbSHm1dF/view?usp=sharing).

### Sample running code for training:

```bash
python -u src/main.py \
    --save-frequency 1 \
    --train-data="cc/Train_GCC-training_output.csv"  \
    --warmup 10000 \
    --batch-size=128 \
    --lr=1e-4 \
    --wd=0.1 \
    --epochs=30 \
    --workers=8 \
    --openai-pretrained \
    --model ViT-L/14
```

### Sample evaluation only:

Evaluation on COCO, ImageNet, or CIRR.
```bash
python src/eval_retrieval.py \
    --openai-pretrained \
    --resume /path/to/checkpoints \
    --eval-mode $data_name \ ## replace with coco, imgnet, or cirr
    --gpu $gpu_id
    --model ViT-L/14
```

Evaluation on fashion-iq (shirt or dress or toptee)
```bash
python src/eval_retrieval.py \
    --openai-pretrained \
    --resume /path/to/checkpoints \
    --eval-mode fashion \
    --source $cloth_type \ ## replace with shirt or dress or toptee
    --gpu $gpu_id
    --model ViT-L/14
```

### Demo:

Evaluation on COCO, ImageNet, or CIRR.

```bash
python src/demo.py \
    --openai-pretrained \
    --resume /path/to/checkpoints \
    --retrieval-data $data_name \ ## Choose from coco, imgnet, cirr, dress, shirt, toptee.
    --query_file "path_img1,path_img2,path_img3..." \ ## query images
    --prompts "prompt1,prompt2,..." \ #prompts. Use * to indicate the token to be replaced with an image token. e.g., "a sketch of *"
    --demo-out $path_demo \ # directory to generate html file and image directory.
    --gpu $gpu_id
    --model ViT-L/14
```
This demo will generate a directory which includes html file and an image directory. Download the directory and open html to see results.

## Citing

If you found this repository useful, please consider citing:

```bibtex
@article{saito2023pic2word,
  title={Pic2Word: Mapping Pictures to Words for Zero-shot Composed Image Retrieval},
  author={Saito, Kuniaki and Sohn, Kihyuk and Zhang, Xiang and Li, Chun-Liang and Lee, Chen-Yu and Saenko, Kate and Pfister, Tomas},
  journal={CVPR},
  year={2023}
}

```
