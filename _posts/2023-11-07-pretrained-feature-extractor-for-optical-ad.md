---
layout: post
title: Using pre-trained feature extractor for optical anomaly detection
date: 2023-11-07 20:02 +0100
tags: [anomaly_detection, python, image_analysis, pytorch, optical_inspection]
toc: true
---


## Introduction

In this post we will look at an approach to detect anomalies in images. The goal is that it should be applicable to common automatic optical inspection scenarios in manufacturing like for example in the semiconductor industry.
Often for optical inspection in manufacturing companies can provide plenty of example images for products without any anomalies (normal samples). However, as a lot of effort is put into optimizing the manufacturing processes, example images of defect products (anomalies) are scarce.
Furthermore, it is often difficult to predict in advance what kind of defects may appear. This makes common supervised image classification or segmentation approaches unfeasible.
We will address this scenario with an anomaly detection approach which uses only normal samples for training and is able to detect any deviations from the normal case on a pixel level.
Previous research in this direction has demonstrated a high effectiveness of features extracted from Deep Learning models pre-trained on the ImageNet dataset. See for example the SOTA approaches on the [Anomaly Detection on MVTec AD benchmark](https://paperswithcode.com/sota/anomaly-detection-on-mvtec-ad). The approach here is in particular inspired by the papers:

- [SPADE - Sub-Image Anomaly Detection with Deep Pyramid Correspondences](https://paperswithcode.com/paper/sub-image-anomaly-detection-with-deep-pyramid)
- [Gaussian-AD - Modeling the Distribution of Normal Data in Pre-Trained Deep Features for Anomaly Detection](https://paperswithcode.com/paper/modeling-the-distribution-of-normal-data-in)
- [PaDiM - a Patch Distribution Modeling Framework for Anomaly Detection and Localization](https://paperswithcode.com/paper/padim-a-patch-distribution-modeling-framework)
- [PatchCore - Towards Total Recall in Industrial Anomaly Detection](https://paperswithcode.com/paper/towards-total-recall-in-industrial-anomaly)

To follow along with the post you can use the [corresponding notebook in the github repo](https://github.com/r-mart/blog-posts/blob/a3dd5e3914ebb3e157976a7eaca8e075416741ab/posts/pretrained_feature_extractor_for_optical_AD.ipynb)

## Dataset

Like in the previous post, we will use the [MVTec anomaly detection dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad) which you can download from the website.
The dataset contains 15 different categories. For the examples in this post we will use the 'Metal Nut' category.

Here is a normal example without anomaly

{% include pretrained-feature-extractor-for-optical-ad/normal_image.html %}

and in contrast an anomalous example

{% include pretrained-feature-extractor-for-optical-ad/defect_image.html %}


## Feature Extraction

Like in the PaDim or PatchCore paper we are going to extract features for each image patch of the training set using a neural network architecture for vision tasks pre-trained on the ImageNet dataset. The patch size is determined by our choice for the network layer. Earlier layers in the network will in general yield smaller patch sizes.<br>
To do the feature extraction we use the PyTorch `feature_extraction` package [based on Torch FX](https://pytorch.org/blog/FX-feature-extraction-torchvision/).
The goal of this post is to demonstrate the principle rather than optimizing our approach to the dataset. Hence, we will simplify some steps compared to the papers.

For the backbone we pick the [ConvNeXt architecture](https://arxiv.org/abs/2201.03545).

```python
from torchvision.models import get_model

backbone = get_model("convnext_base", weights="DEFAULT")
```

In the papers, features from several layers were combined. To keep it simple, we will use only one layer.
To see the available layer names for feature extraction you can use

```python
from torchvision.models.feature_extraction import get_graph_node_names

train_nodes, eval_nodes = get_graph_node_names(backbone)
```

Looking at `train_nodes` or `eval_nodes`, you will see that ConvNeXt base has 7 main feature blocks. If you just want to pick the last node of a block, the feature_extraction module allows you to use truncated node names. We will use `'features.3'` to get the last node of all the `features.3.x.ops` nodes. We choose layer 3 as a compromise between having expressive high-level features but still a somewhat high spatial feature map resolution.

```python
from torchvision.models.feature_extraction import create_feature_extractor

layer_names = ["features.3"]
feature_extractor = create_feature_extractor(backbone, return_nodes=layer_names)
```

As in the papers, we fix the weights to the pre-trained ImageNet weights. Hence, we can turn off gradient computation to save memory

```python
for param in feature_extractor.parameters():
    param.requires_grad = False
```

We follow the PatchCore paper to apply an average pooling layer to the features extracted for each layer. This should give the extracted features more context from their local neighborhood. The motivation is that sometimes by looking at a single patch it is impossible to determine whether a structure is an anomaly. If, however, you also look at the surrounding patches it often becomes more clear.


```python
import torch
import torch.nn as nn


class PatchCoreModel(nn.Module):
    def __init__(self, feature_extractor):
        super().__init__()

        self.feature_extractor = feature_extractor
        self.patch_layer = torch.nn.AvgPool2d(3, 1, 1)

    def forward(self, x):
        feature_dict = self.feature_extractor(x)
        for k, v in feature_dict.items():
            feature_dict[k] = self.patch_layer(v)

        return feature_dict


feature_extractor=PatchCoreModel(feature_extractor)
```

To simplify experimenting with different configurations, we use a Config object. 

In theory, we could look up the number of features and the spatial feature reduction factor from the model source code or paper. However, it is easier to determine later. We therefore set it to 'None' for now

```python
class Config:
    img_shape = (224, 224)  # height, width
    batch_size = 4
    num_workers: int = 2  # adjust to the number of processing cores you want to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    red_factor = None  # spatial reduction factor (equivalent to patch size)
    n_feats = None  # number of features (depends on the chosen layer)
```

To save the features we will follow the memory bank approach from PatchCore. We save the extracted features into a large array without linking them to the original patch location. This means our approach becomes more robust to rotations, translations and other spatial variations of the objects in the dataset. The disadvantage is that the number of feature vectors we have to compare each patch to becomes quite large. Therefore, to make a simple nearest neighbor lookup feasible, further steps to reduce the memory bank size are necessary. We will get around this by choosing a different anomaly detection approach later.

To prepare the data for the feature extractor we create a pyTorch Dataset object

```python
import os
import random
from typing import Optiona
from PIL import Imagel
import numpy as np
from torch.utils.data import Dataset
import albumentations as A


class TrainDataset(Dataset):
    def __init__(
        self,
        data_path: os.PathLike,
        transforms: Optional[A.Compose] = None,
        N_train: Optional[int] = None,
    ):
        super(TrainDataset).__init__()

        self.img_paths = list(data_path.iterdir())
        self.transforms = transforms

        if N_train is not None and len(self.img_paths) > N_train:
            self.img_paths = random.sample(self.img_paths, N_train)

    def __getitem__(self, index: int):
        img_path = self.img_paths[index]

        img = Image.open(img_path)
        img = img.convert("RGB")
        img = np.array(img)

        if self.transforms:
            img = self.transforms(image=img)["image"]

        return img

    def __len__(self) -> int:
        return len(self.img_paths)
```

As we are using a backbone network pre-trained on ImageNet, we need to apply the same normalization transformations as for the original backbone training

```python
from albumentations.pytorch import ToTensorV2
from pathlib import Path

data_path = Path("../data/raw/metal_nut")
train_path = data_path / "train/good"

default_transforms = A.Compose(
    [
        A.Resize(Config.img_shape[0], Config.img_shape[1]),
        A.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0
        ),
        ToTensorV2(),
    ]
)

train_ds = TrainDataset(train_path, transforms=default_transforms)
```

Afterwards, we create the DataLoader object to feed the data to the feature extractor

```python
from torch.utils.data import DataLoader

train_dl = DataLoader(
    train_ds,
    batch_size=Config.batch_size,
    shuffle=False,
    num_workers=Config.num_workers,
)
```

With the data loader defined, we can now find the number of features and spatial reduction factor simply by running the feature extraction 

```python
imgs = next(iter(train_dl))
feats_shapes = []

for layer_name in layer_names:
    feats_shapes.append(feature_extractor(imgs)[layer_name].shape)

Config.n_feats = sum([fs[1] for fs in feats_shapes])
Config.red_factor = Config.img_shape[0] // feats_shapes[0][2]

print("n feats:", Config.n_feats)
print("red factor:", Config.red_factor)
```

    n feats: 256
    red factor: 8


We create a function containing the logic to call the feature extractor with a batch of images and collect the resulting features:

```python
def get_features(imgs, extractor, cfg):
    imgs = imgs.to(cfg.device)

    with torch.no_grad():
        feature_dict = extractor(imgs)

    layers = list(feature_dict.keys())

    feats = feature_dict[layers[0]]
    feats = feats.cpu().numpy()
    feats = np.transpose(feats, (0, 2, 3, 1))
    feats = feats.reshape(-1, cfg.n_feats)

    return feats
```

Finally, we can put everything together to compute the feature memory bank


```python
import math

h, w = Config.img_shape[:2]
h_layer = math.ceil(h / Config.red_factor)
w_layer = math.ceil(w / Config.red_factor)

memory_bank_size = len(train_ds) * h_layer * w_layer
memory_bank = np.empty((memory_bank_size, Config.n_feats), dtype=np.float32)

feature_extractor = feature_extractor.to(Config.device)

i_mem = 0

for i, imgs in enumerate(train_dl):
    n_samples = imgs.shape[0]

    feats = get_features(imgs, feature_extractor, Config)
    memory_bank[i_mem : i_mem + feats.shape[0]] = feats
    i_mem += feats.shape[0]
```

Printing the memory bank shape, we see that it contains almost 200k feature vectors.


```python
print(memory_bank.shape)
# (172480, 256)
```

In the next step we will compute anomaly scores for each patch of a test image by comparing its feature vector with the feature vectors in the memory bank.

## Anomaly Detection

For the anomaly detection part, we extract the features of a target image with the same model as before. Afterwards, we apply an off-the-shelf anomaly detection algorithm from the [Python Outlier Detection (PyOD) library](https://github.com/yzhao062/pyod).

Side remark: I will use the terms anomaly detection and outlier detection interchangeably.

After some experiments, the [LUNAR outlier detection method](https://arxiv.org/abs/2112.05355) proved to have a good performance with reasonable processing time.
Fitting the anomaly detection model on this large set of feature vectors still takes a couple of minutes


```python
from pyod.models.lunar import LUNAR

clf = LUNAR(n_neighbours=5)
clf.fit(memory_bank)
```

Afterwards we pick a defect image from the training data and extract its features in the same way as before


```python
img_path = data_path / "test/bent/000.png"
seg_path = data_path / "ground_truth/bent/000_mask.png"

img = Image.open(img_path)

img_np = np.array(img)
img_t = default_transforms(image=img_np)["image"]
img_t = torch.unsqueeze(img_t, 0)

test_feats = get_features(img_t, feature_extractor, Config)
```

To get an anomaly score map, we reshape the features to first match the image patch locations and eventually resize it to the original image size

```python
import cv2
from scipy.ndimage import gaussian_filter

ano_scores = clf.decision_function(test_feats)
score_patches = np.expand_dims(ano_scores, 0)
score_patches = score_patches.reshape(h_layer, w_layer)

anomaly_map = cv2.resize(score_patches, (img.width, img.height))

# apply Gaussian blur to smooth out possible resizing artifacts
anomaly_map = gaussian_filter(anomaly_map, sigma=4)

# make anomaly scores start at 0
anomaly_map = anomaly_map - anomaly_map.min()
```

This allows us to overlay the anomaly score map with the original defect image and to compare with the ground truth annotation. See the [github repo](https://github.com/r-mart/blog-posts) for the plotting function code


```python
import bokeh
from bokeh.plotting import show
from src.visualization.image import (
    plot_img_rgba,
    add_seg_on_img,
    add_score_map_on_img,
)

seg = Image.open(seg_path)
seg = np.array(seg)

p_img = plot_img_rgba(img, title="Image with ground truth annotation")
p_img = add_seg_on_img(p_img, seg)
p_ano = plot_img_rgba(img, title="Image with prediction")
p_ano = add_score_map_on_img(p_ano, anomaly_map, alpha=0.6)
p = bokeh.layouts.row(p_img, p_ano)
show(p)
```

{% include pretrained-feature-extractor-for-optical-ad/defect_image_with_gt_and_predictions.html %}

And indeed, we can see how the area with the highest anomaly scores correspond to the marked ground-truth defect annotation.

### Putting everything together


```python
class AnomalyDetector:
    def __init__(self, transforms, feature_extractor, clf, cfg) -> None:
        self.transforms = transforms
        self.feature_extractor = feature_extractor.to(cfg.device)
        self.clf = clf
        self.cfg = cfg

        self.h_layer = math.ceil(cfg.img_shape[0] / cfg.red_factor)
        self.w_layer = math.ceil(cfg.img_shape[1] / cfg.red_factor)

    def __call__(self, img: Image.Image) -> np.ndarray:
        img_np = np.array(img)
        img_t = self.transforms(image=img_np)["image"]
        img_t = torch.unsqueeze(img_t, 0)

        feats = get_features(img_t, self.feature_extractor, self.cfg)

        ano_scores = self.clf.decision_function(feats)
        score_patches = np.expand_dims(ano_scores, 0)
        score_patches = score_patches.reshape(self.h_layer, self.w_layer)

        anomaly_map = cv2.resize(score_patches, (img.width, img.height))

        # apply Gaussian blur to smooth out possible resizing artifacts
        anomaly_map = gaussian_filter(anomaly_map, sigma=4)
        anomaly_map = anomaly_map - anomaly_map.min()

        return anomaly_map
```

Let's test our new anomaly detector on different defect images


```python
detector = AnomalyDetector(default_transforms, feature_extractor, clf, Config)

img_path = next((data_path / "test/scratch").iterdir())
seg_path = (
    data_path
    / "ground_truth"
    / img_path.parent.name
    / f"{img_path.stem}_mask{img_path.suffix}"
)

img = Image.open(img_path)
seg = Image.open(seg_path)
seg = np.array(seg)

anomaly_map = detector(img)

p_img = plot_img_rgba(img)
p_img = add_seg_on_img(p_img, seg)
p_ano = plot_img_rgba(img)
p_ano = add_score_map_on_img(p_ano, anomaly_map, alpha=0.6)
p = bokeh.layouts.row(p_img, p_ano)

show(p)
```

{% include pretrained-feature-extractor-for-optical-ad/defect_image_2_with_gt_and_predictions.html %}

For comparison, we test it also on unseen good images


```python
defect_max_score = anomaly_map.max()
img_path = next((data_path / "test/good").iterdir())

img = Image.open(img_path)

anomaly_map = detector(img)

p_img = plot_img_rgba(img)
p_ano = plot_img_rgba(img)
p_ano = add_score_map_on_img(p_ano, anomaly_map, alpha=0.6, max_score=defect_max_score)
p = bokeh.layouts.row(p_img, p_ano)

show(p)
```

{% include pretrained-feature-extractor-for-optical-ad/normal_image_with_predictions.html %}


Note that for plotting we set the same upper limit anomaly score as for the defect image before. This visualizes more intuitively that the anomaly scores are a lot lower than before and more or less evenly distributed over the image.

## Validation

To quantify how well this approach works over the whole test dataset, we make anomaly score predictions over all test images and compare with the provided ground-truth annotations using the area under receiver operating characteristic curve (AUROC) metric. See [Receiver operating characteristic](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) for more details.

Like in the 'training' phase, we first create a pyTorch Dataset. As the predictor class can handle already native python image objects, we don't necessarily need a DataLoader. The DataLoader would allow us to speed up the process by using batches but for this blog post we will keep it simple.


```python
class ValidationDataset(Dataset):
    def __init__(
        self,
        data_path: os.PathLike,
        gt_path: os.PathLike,
    ):
        super(ValidationDataset).__init__()

        self.img_paths = list()
        self.gt_paths = list()

        gt_class_paths = list(data_path.iterdir())

        for p in gt_class_paths:
            for img_path in p.iterdir():
                self.img_paths.append(img_path)
                self.gt_paths.append(
                    gt_path / p.name / f"{img_path.stem}_mask{img_path.suffix}"
                )

    def __getitem__(self, index: int):
        img_path = self.img_paths[index]
        gt_path = self.gt_paths[index]

        img = Image.open(img_path)
        img = img.convert("RGB")

        if not gt_path.exists():
            # there are no gt annotations for good cases -> all 0
            gt = np.zeros((img.height, img.width))
        else:
            gt = Image.open(gt_path)
            gt = gt.convert("L")
            gt = np.array(gt)
            gt = gt / 255

        return img, gt

    def __len__(self) -> int:
        return len(self.img_paths)


val_path = data_path / "test"
gt_path = data_path / "ground_truth"

val_ds = ValidationDataset(val_path, gt_path)
```

With that in place, we can loop through the validation dataset and store ground truth and anomaly score predictions


```python
img, gt = val_ds[0]

pred_size = len(val_ds) * img.height * img.width
preds_pix = np.empty(pred_size, dtype=np.float32)
gts_pix = np.empty(pred_size, dtype=np.int32)
preds_img = np.empty(len(val_ds), dtype=np.float32)
gts_img = np.empty(len(val_ds), dtype=np.int32)

i_pix = 0

for i in range(len(val_ds)):
    img, gt = val_ds[i]
    gt = gt.astype(np.int32)

    anomaly_map = detector(img)
    n_pix = anomaly_map.shape[0] * anomaly_map.shape[1]

    preds_pix[i_pix : i_pix + n_pix] = anomaly_map.reshape((-1,))
    gts_pix[i_pix : i_pix + n_pix] = gt.reshape((-1,))

    # use max score of the map as image-level anomaly score
    preds_img[i] = anomaly_map.max()
    # for good images gt will be all zero, for defect images max will be 1
    gts_img[i] = gt.max()

    i_pix += n_pix
```

The AUROC score is computed using the ground truth values and prediction scores. We compute it first for the whole image. Here, if an image contains an anomaly anywhere the ground-truth annotation is '1' for the whole image, otherwise '0'.
To predict a single score from the anomaly maps we simply used the maximum anomaly score of the map.


```python
from sklearn.metrics import roc_curve, auc

fpr_img, tpr_img, thresholds_img = roc_curve(gts_img, preds_img)
auroc_img = auc(fpr_img, tpr_img)

print(f"image-wise AUROC: {auroc_img:.5f}")
# image-wise AUROC: 0.99609
```


```python
from bokeh.plotting import figure

p = figure(
    title=f"ROC curve for image-wise prediction (area = {auroc_img:.5f})",
    x_axis_label="False Positive Rate",
    y_axis_label="True Positive Rate",
)
p.line(fpr_img, tpr_img, line_width=2)
show(p)
```

{% include pretrained-feature-extractor-for-optical-ad/ROC_curve.html %}


Afterwards, we compute the pixel-wise AUROC score:


```python
fpr_pix, tpr_pix, thresholds_pix = roc_curve(gts_pix, preds_pix)
auroc_pix = auc(fpr_pix, tpr_pix)

print(f"pixel-wise AUROC: {auroc_pix:.5f}")
# pixel-wise AUROC: 0.99015
```


We see that both results are pretty high and comparable to recent results on the [Anomaly Detection on MVTec AD benchmark](https://paperswithcode.com/sota/anomaly-detection-on-mvtec-ad).

Note however that the benchmark takes the average score for all 15 dataset categories while here we only considered the metal nut category.

## Conclusion

We demonstrated a simple approach for image anomaly detection that reaches results comparable to the SOTA on the MVTec AD dataset.
This approach uses only normal data samples and doesn't require a conventional Deep Learning training pipeline, just a 'memorizing' of features. It can therefore be easily applied in practice, even without a powerful workstation. The main assumption is that the images are fairly similar to natural images (as this is what ImageNet was originally trained for). Furthermore, the complexity should be similar to the images in the MVTec AD dataset, i.e. single centered objects without much variation in background or images completely covered by textures.
