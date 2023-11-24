---
layout: post
title: Analyzing features from pre-trained DINOv2 model for anomaly detection 
date: 2023-11-23 23:15 +0100
tags: [anomaly_detection, python, image_analysis, pytorch, optical_inspection]
toc: true
math: true
---


## Introduction

In this post we will classify images into normal or anomalous images. This is a common task in automatic optical inspection in the manufacturing industry. 
We consider the scenario where plenty of example images without any anomalies (normal samples) can be provided but example images with defects (anomalies) are rare.

We therefore address this scenario with an anomaly detection approach that learns the distribution of normal samples during training. During inference it then scores target images based on how well they fit to the learned distribution. This approach gives one score for the whole image instead of determining pixel-wise anomaly maps (see the [previous blog post](https://r-mart.github.io/posts/pretrained-feature-extractor-for-optical-ad/) for the latter).

Previous research has demonstrated a high effectiveness of features from Deep Learning models pre-trained on the ImageNet dataset. See for example the SOTA approaches on the [Anomaly Detection on MVTec AD benchmark](https://paperswithcode.com/sota/anomaly-detection-on-mvtec-ad). This post is in particular inspired by the paper [Gaussian-AD - Modeling the Distribution of Normal Data in Pre-Trained Deep Features for Anomaly Detection](https://paperswithcode.com/paper/modeling-the-distribution-of-normal-data-in) which I will refer to as the 'Gaussian AD paper'.

To follow along with the post you can use the [corresponding notebook in the github repo](https://github.com/r-mart/blog-posts/blob/86602c0ba190956396c6bba36346da7e253c6ee0/posts/analyzing_pretrained_DINOv2_features_for_AD.ipynb)


## Dataset

Like in the previous posts, we use the [MVTec anomaly detection dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad) which can be downloaded from the website.
This time we consider the 'wood' category. The following figure shows example images from the test set for each of the 6 classes

{% include analyzing-pretrained-dinov2-features-for-ad/examples_images.html %}

The 'good' class has no defects. The other classes each show a different type of anomaly. The training data only contains images of the 'good' class.

## Feature Extraction

In this approach we will extract one feature vector for each image in the training and test data. First, we define a class to hold the configs which consists of quite few options this time


```python
import torch

class Config:
    model_name = 'facebook/dinov2-small'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_feats = None # number of features (depends on the chosen layer)
```

In the original Gaussian AD paper, features from individual layers have been extracted and combined. Since the release of the paper, however, a couple of powerful feature extraction architectures have been published.
We are going to use the [Hugging Face implementation of the DINOv2 model](https://huggingface.co/docs/transformers/main/model_doc/dinov2) as it claims to be a stable 'all-purpose' feature extractor.

We load a pretrained model from the `Dinov2Model` class from Hugging Face. See [Hugging Face models](https://huggingface.co/models?sort=created&search=facebook%2Fdinov2) for a list of available models.
The `AutoImageProcessor` class loads the corresponding image pre-processing steps for the given model name. As we are not going to finetune any weights, we turn of gradients for the model parameter.


```python
from transformers import AutoImageProcessor, Dinov2Model

model = Dinov2Model.from_pretrained(Config.model_name)
image_processor = AutoImageProcessor.from_pretrained(Config.model_name)

for param in model.parameters():
    param.requires_grad = False
```

To get a feature vector for each image from this model we simply access the output after the last pooling layer


```python
def get_features(imgs, extractor, cfg):
    imgs = imgs.to(cfg.device)

    with torch.no_grad():
        feats = extractor(**imgs).pooler_output

    feats = feats.cpu().numpy()

    return feats
```

### Training Data

In the pyTorch `Dataset` class for loading the data we just load the images and apply the image processor corresponding to our model


```python
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(
        self,
        data_path: os.PathLike,
        processor = None,
    ):
        super(TrainDataset).__init__()

        self.img_paths = list(data_path.iterdir())
        self.processor = processor

    def __getitem__(self, index: int):
        img_path = self.img_paths[index]

        img = Image.open(img_path)
        img = img.convert("RGB")
        img = np.array(img)

        if self.processor:
            img = self.processor(img, return_tensors="pt")

        return img

    def __len__(self) -> int:
        return len(self.img_paths)
```


```python
from pathlib import Path

data_path = Path("../data/raw/wood")
train_path = data_path / "train/good"
train_ds = TrainDataset(train_path, processor=image_processor)
```

To sanity check the loading code and also determine the size of the extracted feature vector we run it for one image


```python
model = model.to(Config.device)
imgs = train_ds[0]

feats = get_features(imgs, model, Config)
Config.n_feats = feats.shape[1]

print("Feature shape:", feats.shape)
# Feature shape: (1, 384)
```

Now we are ready to extract and store all features for the training data by feeding the images to the model


```python
train_features = np.empty((len(train_ds), Config.n_feats), dtype=np.float32)
model = model.to(Config.device)

for i in range(len(train_ds)):
    imgs = train_ds[i]
    feats = get_features(imgs, model, Config)
    train_features[i] = feats

print("Train features shape:", train_features.shape)
# Train features shape: (247, 384)
```

Note: for a larger dataset I would recommend to use a pytorch 'DataLoader' class to use batch processing to speed up the feature extraction

### Test Data

We also extract and store the features for the test data. Compared to the training data, the `Dataset` class needs more logic to find the various defect classes in the subfolders and return a label representing the defect class for each image


```python
class TestDataset(Dataset):
    def __init__(
        self,
        data_path: os.PathLike,
        gt_path: os.PathLike,
        processor=None,
    ):
        super(TestDataset).__init__()

        self.img_paths = list()
        self.gt_paths = list()

        gt_class_paths = list(data_path.iterdir())
        self.gt_class_name_to_label_map = {
            p.name: i for i, p in enumerate(gt_class_paths)
        }

        for p in gt_class_paths:
            for img_path in p.iterdir():
                self.img_paths.append(img_path)
                self.gt_paths.append(
                    gt_path / p.name / f"{img_path.stem}_mask{img_path.suffix}"
                )
        self.processor = processor

    def __getitem__(self, index: int):
        img_path = self.img_paths[index]
        gt_path = self.gt_paths[index]

        label = self.gt_class_name_to_label_map[gt_path.parent.name]

        img = Image.open(img_path)
        img = img.convert("RGB")
        img = np.array(img)

        if self.processor:
            img = self.processor(img, return_tensors="pt")

        return img, label

    def __len__(self) -> int:
        return len(self.img_paths)
```


```python
test_path = data_path / "test"
gt_path = data_path / "ground_truth"

test_ds = TestDataset(test_path, gt_path, processor=image_processor)
```

For later use, we save the label for the 'good' class and a mapping from label to the class name


```python
good_label = test_ds.gt_class_name_to_label_map['good']
label_to_name_map = {v: k for k, v in test_ds.gt_class_name_to_label_map.items()}
```

With the test dataset class defined, we can now extract the features and corresponding label for each image in the test set


```python
test_features = np.empty((len(test_ds), Config.n_feats), dtype=np.float32)
test_labels = np.zeros((len(test_ds)), dtype=np.uint32)

model = model.to(Config.device)

for i in range(len(test_ds)):
    imgs, label = test_ds[i]

    feats = get_features(imgs, model, Config)

    test_features[i] = feats
    test_labels[i] = label

print("Test features shape:", test_features.shape)
print("Test labels shape:", test_labels.shape)
# Test features shape: (79, 384)
# Test labels shape: (79,)
```

The labels (class names) will be mostly relevant for visualization. As our goal is to distinguish normal from anomalous images, we save the 'ground truth' values simply as '0' for normal (good) image and '1' for anomaly (any other class)


```python
ano_gt = (test_labels != good_label).astype(np.int32)
```

## Anomaly Detection

After having the features for all images and the corresponding labels for the test images stored, we can get to the actual anomaly detection. We follow the Gaussian AD paper in fitting a multivariate Gaussian distribution to the extracted feature vectors of the training data. A multivariate Gaussian is parameterized by a mean vector and a covariance matrix. We therefore have to fit both to the training data. While the mean vector is simple, for the covariance matrix we use the [Ledoit Wolf Estimator](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.LedoitWolf.html).

We use the [Mahalanobis distance](https://en.wikipedia.org/wiki/Mahalanobis_distance) from the Gaussian distribution as anomaly score. It is basically the distance from the mean of the Gaussian which also takes the different variances and covariances for each direction in feature space into account.


```python
from sklearn.covariance import LedoitWolf


class GaussianAD:
    def __init__(self):
        self.mean = None
        self.lw_cov = None
        self.lw_prec = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        lw_cov = LedoitWolf().fit(X)

        self.lw_cov = lw_cov
        self.lw_prec = lw_cov.precision_

    def decision_function(self, X):
        return mahalanobis_distance(X, self.mean, self.lw_prec)


def mahalanobis_distance(
    values: np.ndarray, mean: np.ndarray, inv_covariance: np.ndarray
) -> np.ndarray:
    """Compute the batched mahalanobis distance.
    values is a batch of feature vectors.
    mean is either the mean of the distribution to compare, or a second
    batch of feature vectors.
    inv_covariance is the inverse covariance of the target distribution.
    """
    assert values.ndim == 2
    assert 1 <= mean.ndim <= 2
    assert len(inv_covariance.shape) == 2
    assert values.shape[1] == mean.shape[-1]
    assert mean.shape[-1] == inv_covariance.shape[0]
    assert inv_covariance.shape[0] == inv_covariance.shape[1]

    if mean.ndim == 1:  # Distribution mean.
        mean = np.expand_dims(mean, 0)
    x_mu = values - mean  # batch x features
    # Same as dist = x_mu.t() * inv_covariance * x_mu batch wise
    dist = np.einsum("im,mn,in->i", x_mu, inv_covariance, x_mu)
    return np.sqrt(dist)
```

With that in place we fit the multivariate Gaussian to the training data


```python
clf = GaussianAD()
clf.fit(train_features)
```

and consequently score the test data using the Mahalanobis distance


```python
ano_scores = clf.decision_function(test_features)
```

To measure the performance of this approach over the whole test dataset, we use the area under [receiver operating characteristic](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) curve (AUROC) metric.


```python
from sklearn.metrics import roc_curve, auc

fpr_img, tpr_img, thresholds_img = roc_curve(ano_gt, ano_scores)
auroc_img = auc(fpr_img, tpr_img)

print(f"Image-wise Anomaly Detection AUROC: {auroc_img:.5f}")
# Image-wise Anomaly Detection AUROC: 0.97544
```

This already gives a decent score compared to the [Anomaly Detection on MVTec AD benchmark](https://paperswithcode.com/sota/anomaly-detection-on-mvtec-ad). (The comparison is not completely fair, as the benchmark score consists of the average for all MVTec AD categories, but it can still serve as an orientation)

The Gaussian AD paper went one step further and analyzed the extracted features to get an intuition why they work so well. Inspired by the paper, we will look at the [principal components](https://en.wikipedia.org/wiki/Principal_component_analysis) of the training feature vectors in the next section.

### Principal Component Analysis (PCA)

We start by fitting a PCA model to the training features to find how much variance is 'explained' by the principal components (feature eigen vectors corresponding to largest variance).<br>
In particular, we save the indices for the principal components up until 90% and 99% of the variance of the training data


```python
from sklearn.decomposition import PCA

X_train = train_features
pca = PCA(n_components=None).fit(X_train)

variance_thresholds = [0.9, 0.99]
variances = pca.explained_variance_ratio_.cumsum()

i_comp_thresholds = []
for variance_threshold in variance_thresholds:
    i_comp_thresholds.append((variances > variance_threshold).argmax())

print("Dimension of feature space:", X_train.shape[1])
for i in range(len(variance_thresholds)):
    print(f"The first {i_comp_thresholds[i]} features explain {variance_thresholds[i]*100}% of variance")

# Dimension of feature space: 384
# The first 29 features explain 90.0% of variance
# The first 111 features explain 99.0% of variance
```

We are going to use this information to perform a standard PCA dimensionality reduction in which we keep the components explaining most of the variance. <br>
However, we will also consider a 'negative PCA' dimensionality reduction in which we keep the components explaining the least variance. Note that we perform this reduction on the test data (with defect images) while the PCA components have been fitted to the training data (with good images only).


```python
X_test = test_features
y = ano_gt
y_label = test_labels

# Normal PCA
pca_comps = pca.components_[: i_comp_thresholds[0]]
X_pca = np.matmul(X_test, pca_comps.T)

# Negative PCA
npca_comps = pca.components_[i_comp_thresholds[1] :]
X_npca = np.matmul(X_test, npca_comps.T)

print("Test data shape after reduction with standard PCA:", X_pca.shape)
print("Test data shape after reduction with negative PCA:", X_npca.shape)
# Test data shape after reduction with standard PCA: (79, 29)
# Test data shape after reduction with negative PCA: (79, 136)
```

To get an intuition how these lower dimensional feature vectors behave we visualize them by mapping onto the 3 dimensional space using [Uniform Manifold Approximation and Projection (UMAP)](https://umap-learn.readthedocs.io/en/latest/)


```python
import umap

n_dim = 3

umap_for_all = umap.UMAP(n_components=n_dim)
X_all_embed = umap_for_all.fit_transform(X_test)

umap_for_pca = umap.UMAP(n_components=n_dim)
X_pca_embed = umap_for_pca.fit_transform(X_pca)

umap_for_npca = umap.UMAP(n_components=n_dim)
X_npca_embed = umap_for_npca.fit_transform(X_npca)
```

To see the plot function definitions and to be able to interactively rotate the following plots, see the [corresponding notebook](https://github.com/r-mart/blog-posts/blob/86602c0ba190956396c6bba36346da7e253c6ee0/posts/analyzing_pretrained_DINOv2_features_for_AD.ipynb)

![pca all features](/assets/img/analyzing-pretrained-dinov2-features-for-ad/pca_all_features.png)

The UMAP embedding for the complete feature vectors shows that there is a tendency for the different image classes to be clustered. This explains why the anomaly detection before was quite good.

One has to be cautious though to interpret the result of such low-dimensional embeddings. The original feature space is 384 dimensional in our case. It is well possible that the features are separated in such high dimensional spaces while a 2D or 3D embedding shows them as mixed up. In general, the rule of thumb is:

$$ \textrm{separated cluster in embedding} \Rightarrow \textrm{separated cluster in high-dim space} $$
$$ \textrm{mixed samples in embedding} \nRightarrow  \textrm{mixed samples in high-dim space} $$

![pca high-var features](/assets/img/analyzing-pretrained-dinov2-features-for-ad/pca_high_var_features.png)

Looking at the embeddings for the features that have been reduced using standard PCA (high variance components), we see that there is still a separation but the clusters are less clear. Note in particular how wide the 'good' samples are separated. This is what this feature selection is optimized for. However, in between the other classes are partly mixed in. 


![pca low-var features](/assets/img/analyzing-pretrained-dinov2-features-for-ad/pca_low_var_features.png)

For the features that have been reduced using 'negative PCA' (low variance components), compared to the previous plots, in particular the good samples form a tighter cluster now. 
Nevertheless, it seems to do an astonishingly good job in clustering the different classes which should also lead to a good anomaly detection performance.

To confirm that, we will again determine the anomaly detection performance using the AUROC metric, but this time using only the high variance PCA components (up until 99% of variance)


```python
variance_thresholds = [0.99]
i_comp_thresholds = []
for variance_threshold in variance_thresholds:
    i_comp_thresholds.append((variances > variance_threshold).argmax())
```


```python
# Normal PCA
pca_comps = pca.components_[: i_comp_thresholds[0]]

train_features_pca = np.matmul(train_features, pca_comps.T)
test_features_pca = np.matmul(test_features, pca_comps.T)

print("PCA training features shape", train_features_pca.shape)
print("PCA test features shape", test_features_pca.shape)
# PCA training features shape (247, 111)
# PCA test features shape (79, 111)

clf_pca = GaussianAD()
clf_pca.fit(train_features_pca)
```

```python
ano_scores_pca = clf_pca.decision_function(test_features_pca)

fpr_img, tpr_img, thresholds_img = roc_curve(ano_gt, ano_scores_pca)
auroc_img = auc(fpr_img, tpr_img)

print(f"PCA reduction, image-wise AUROC: {auroc_img:.5f}")
# PCA reduction, image-wise AUROC: 0.91667
```


afterwards we do the same for the low-variance components (all components explaining the remaining 1% of variance).


```python
# Negative PCA
npca_comps = pca.components_[i_comp_thresholds[0]:]

train_features_npca = np.matmul(train_features, npca_comps.T)
test_features_npca = np.matmul(test_features, npca_comps.T)

print("NPCA training features shape", train_features_npca.shape)
print("NPCA test features shape", test_features_npca.shape)
# NPCA training features shape (247, 136)
# NPCA test features shape (79, 136)

clf_npca = GaussianAD()
clf_npca.fit(train_features_npca)
```

```python
ano_scores_npca = clf_npca.decision_function(test_features_npca)

fpr_img, tpr_img, thresholds_img = roc_curve(ano_gt, ano_scores_npca)
auroc_img = auc(fpr_img, tpr_img)

print(f"NPCA reduction, image-wise AUROC: {auroc_img:.5f}")
# NPCA reduction, image-wise AUROC: 0.97982
```

And indeed, the scores seem to confirm the visual impression that the low variance components do a better job in separating the 'good' and defect images. 
This is somewhat counterintuitive as one would expect that the the high variance components approximate the training data a lot better. After all, they represent 99% of the variance.
It seems like the components used to differentiate individual images of the normal data are different than the components used to differentiate normal from anomalous images. 
The authors of the Gaussian AD paper made the same observation and hypothesized that this explains why using very general feature extractors like models trained on Imagenet perform a lot better than training or even finetuning a model on the training data.
The training data consists only of good images. A model trained on that data learns features to distinguish them. However, as we have seen, those features are less useful in distinguishing normal from anomalous images. Hence, a model trained on this data doesn't learn the necessary features to perform anomaly detection. Even when finetuning on the training data, one risks unlearning those features used to differentiate anomalous images and replacing them with high variance features for normal data only. 
Therefore, according to this argumentation, it is best to simply use an all-purpose feature extractor with frozen weights for anomaly detection. 


## Conclusion

We demonstrated a simple approach for image anomaly detection that consists of storing features from the DINOv2 all-purpose feature extractor and fitting a multivariate Gaussian distribution on it. Afterwards, the distance from the Gaussian can be used to detect anomalies in features extracted from test images with a high accuracy. The approach uses only normal data samples and can therefore be easily applied in practice where defect images are hard to get. Furthermore, by analyzing the extracted features, we observed that feature combinations used to separate anomalous images from normal ones show little variance in the normal training data. This could be an explanation for the good performance of fixed feature extractor versus training on the data.
