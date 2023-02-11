# Open Vocabulary Object Detection

This repository provides an implementation of the CVPR 2021 oral paper [Open-Vocabulary Object Detection Using Captions](https://arxiv.org/pdf/2011.10678.pdf). The code is built on top of Facebook's [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark). We have also partially used some code from Facebook's [ViLbert](https://github.com/facebookresearch/vilbert-multi-task) and HuggingFace's [transformers](https://github.com/huggingface/transformers). We appreciate the work of everyone involved in those invaluable projects.

![alt text](demo/example.jpg)

## Jupyter notebook demo

We provide a simple demo that creates a side-by-side video of a regular Faster R-CNN vs. our open-vocabulary detector. To run, just open any of the notebooks inside the [`demo`](demo) folder.

## Installation

Check [INSTALL.md](INSTALL.md) for installation instructions. For the demo to create the video output, it might be necessary to build OpenCV from source instead of installing using pip.

## Perform multimedia self-supervised pre-training on COCO captions dataset

For the following examples to work, you need to download the COCO dataset.
We recommend to symlink the path to the coco dataset to [`datasets/`](datasets). Refer to [`path_catalog.py`](maskrcnn_benchmark/config/path_catalog.py) for the names of the required files. After setting up the dataset, run:

```bash
python -m torch.distributed.launch --nproc_per_node=8 tools/train_net.py --config-file configs/mmss_v07.yaml --skip-test OUTPUT_DIR ~/runs/vltrain/121
```

## Perform fine-tuning (or training from scratch) on COCO object detection dataset

For the zero-shot experiment to work, you need to first create a new annotation json using [this notebook](ipynb/003.ipynb). Then run:

```bash
python -m torch.distributed.launch --nproc_per_node=8 tools/train_net.py --config-file configs/zeroshot_v06.yaml OUTPUT_DIR ~/runs/maskrcnn/130
```

## Evaluation
You can evaluate using a similar command as above, by running [`tools/test_net.py`](tools/test_net.py) and providing the right checkpoint path to `MODEL.WEIGHT`


## Pretrained Models

Our best model is available for download [here](https://drive.google.com/file/d/1lP7LFRCPKIbKjXK_n6xCDn80I7WEH3-h/view?usp=sharing), and has been trained using [this config](configs/zeroshot_v06.yaml).

The pretrained model before fine-tuning on object detection can be found [here](https://drive.google.com/file/d/1mFnAZVnn2NT2Ys841EPOMaQ6jnvFXPWJ/view?usp=sharing), and has been trained using [this config](configs/mmss_v07.yaml).


**Note:** If the links are broken, please [contact me directly](mailto:alireza@cs.columbia.edu).

## Additional Notes

We did not test all the functionality of `maskrcnn_benchmark` under the zero-shot settings, such as instance segmentation, or feature pyramid network. Anything besides the provided config files may not work.

Created and maintained by [Alireza Zareian](https://www.linkedin.com/in/alzareian/).
