---
layout: post
title: Fourier Transform for detecting defects on images with regular patterns
date: 2023-09-23 11:38 +0200
tags: [python, image_analysis]
---

<!-- 
Introduction
Theoretical Background
Method
Experiment
Result 
Discussion
Summary
 -->

## Problem Description

You have images from a production line which you want to inspect for any defects or anomalies. Defects are very rare and you don't know in advance what kind of defects may appear. Hence, it is impossible to gather data for a Deep Learning based classification approach.  On the other hand, your images exhibit a regular pattern which the defects are breaking.

## Example Dataset

To reproduce this setting we will use the [MVTec anomaly detection dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad). You can download the dataset from the website. 
The dataset contains 15 different categories. Five of them are texture categories which fit to our use case because they exhibit a regular pattern. For this post we will look at the 'Grid' category in particular.

Here is a normal image without defect

{% include ft-image-defect-detection/normal_image.html %}

And in comparison, a defect image

{% include ft-image-defect-detection/defect_image.html %}

We can see that the defects break the otherwise quite regular grid pattern.

## Method

Using Fourier Transform to find defects in images with regular patterns is motivated by the following properties. Transforming an extended regular pattern in position space, like a single sine wave, leads to a narrow peak in frequency domain

{% include ft-image-defect-detection/FT_of_wave.html %}

In contrast, transforming a narrow peak in position space, leads to an evenly distributed signal in frequency domain

{% include ft-image-defect-detection/FT_of_peak.html %}

Hence, applied to images, patterns will lead to individual peaks in frequency domain while locally-confined defects will lead to a spread-out noise. 
This should make it simple to separate the two signals using a threshold in frequency domain.

So let's try this on our defect image from above. To follow along with the code samples you will need a python virtual environment setup with the used libraries, the data downloaded locally and the `data_path` variable adjusted if necessary. See the [github repo](https://github.com/r-mart/blog-posts) for the full source code, including the plotting functions.<br>

We start by converting the image into frequency domain

```python
from pathlib import Path
from PIL import Image, ImageChops
import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift

data_path = Path("../data/raw/grid")

img_path = data_path / "test/broken/000.png"

img = Image.open(img_path)
img = img.convert("L")

# prepare image
img_np = np.array(img)
img_np = img_np / 255.0

# transform to Fourier space
f = fft2(img_np)
fshift = fftshift(f)

# frequency magnitudes
mag_img = np.log(np.abs(fshift))
```

Plotting the frequency magnitudes gives us

{% include ft-image-defect-detection/FT_of_defect_image.html %}

If our intuition is correct, the bright frequency peaks in the center should encode the regular grid pattern of the image while the spread out noise encodes the defects. Let's mask the frequency peaks by applying a threshold

```python
import cv2

mag_thresh = 0.65 # relative to max value
max_val = mag_img.max()
thresh_val = mag_img.min() + mag_thresh * (max_val - mag_img.min())
ret, mag_img_mask = cv2.threshold(mag_img, thresh_val, 1.0, cv2.THRESH_BINARY)
```

The resulting `mag_img_mask` looks like

{% include ft-image-defect-detection/FT_mag_peak_masked.html %}

The mask allows us to separate the peaks from the rest and transform the modified frequencies back to image domain

```python
# masking
mag_img_mask = mag_img_mask.astype(bool)
fshift_proc = fshift * mag_img_mask

# transform back
f_ishift = ifftshift(fshift_proc)
img_proc = ifft2(f_ishift)
img_proc = np.abs(img_proc)

# convert to image
img_proc = (img_proc * 255.0).astype(np.uint8)
img_proc = Image.fromarray(np.uint8(img_proc))
```

The resulting image compared with the original image is shown below

{% include ft-image-defect-detection/defect_image_reconstructed.html %} 

And indeed, we managed to spirit away the defects! However, the image borders are a bit blurred now. This is because the Fourier Transform assumes in theory and infinitely long input signal but our image has finite dimensions. We will deal with any side effects caused by this in the post processing.

The image difference highlights the defects

```python
diff = ImageChops.difference(img, img_proc)
```

{% include ft-image-defect-detection/defect_image_difference.html %}

An alternative approach to get to the same result is to invert the masking and keep only the non-peak areas of the frequency magnitude map

```python
# invert mask
mag_img_mask_inv = ~mag_img_mask

# masking
fshift_proc = fshift * mag_img_mask_inv

# transform back
f_ishift = ifftshift(fshift_proc)
img_proc = ifft2(f_ishift)
img_proc = np.abs(img_proc)

# convert to image
img_proc = (img_proc * 255.0).astype(np.uint8)
img_proc = Image.fromarray(np.uint8(img_proc))
```

This yields directly the highlighted defects

{% include ft-image-defect-detection/defect_image_invert_reconstructed.html %}

This is because we now only kept the part of the frequency signal encoding the defect. As this is a bit simpler, we will use this approach to put everything together in one function

```python
def ft_extract_anomalies(img: Image.Image, mag_thresh: float = 0.5) -> Image.Image:

    # prepare image
    img_np = np.array(img)
    img_np = img_np / 255.0

    # transform to Fourier space
    f = fft2(img_np)
    fshift = fftshift(f)

    # frequency magnitudes
    mag_img = np.log(np.abs(fshift))

    # thresholding
    max_val = mag_img.max()
    thresh_val = mag_img.min() + mag_thresh * (max_val - mag_img.min())
    ret, mag_img_mask = cv2.threshold(mag_img, thresh_val, 1.0, cv2.THRESH_BINARY)

    # masking
    mag_img_mask = mag_img_mask.astype(bool)
    mag_img_mask_inv = ~mag_img_mask
    fshift_proc = fshift * mag_img_mask_inv

    # transform back
    f_ishift = ifftshift(fshift_proc)
    img_proc = ifft2(f_ishift)
    img_proc = np.abs(img_proc)

    # convert to image
    img_proc = (img_proc * 255.0).astype(np.uint8)
    img_proc = Image.fromarray(img_proc)

    return img_proc
```

With that we can obtain the last result from the input image using

```python
img_path = data_path / "test/broken/000.png"
img = Image.open(img_path)
img = img.convert("L")

img_proc = ft_extract_anomalies(img, mag_thresh=0.65)
```

So far we have highlighted the defects but we still haven't properly localized them using bounding boxes. We will do this in the following post processing section.

## Post Processing

To process the highlighted defect areas we first threshold the image

```python
img_proc_np = np.array(img_proc)

max_val = img_proc_np.max()
thresh_val = np.percentile(img_proc_np, 99)
ret, img_thresh = cv2.threshold(img_proc_np, int(thresh_val), 1.0, cv2.THRESH_BINARY)
img_thresh = img_thresh > 0
```

giving us 

{% include ft-image-defect-detection/defects_thresholded.html %}

We can see the defects in the mask. However, due to their structure they don't appear as one connected region. Furthermore, due to the blurring at the borders we see some noise there. We can fix both by using morphological dilation and clearing everything touching the image borders. Afterwards, we assign unique labels to each connected region

```python
from skimage import morphology
from skimage import measure
from skimage import segmentation

# combine neighboring mask regions
img_morph = morphology.binary_dilation(img_thresh, np.ones([7,7]))

# remove artifacts due to blurring at the edges
img_morph = segmentation.clear_border(img_morph)

# assign label to each connected region
img_lab = measure.label(img_morph) 
```

The resulting label image looks like

{% include ft-image-defect-detection/defects_label_map.html %}

By hovering over the big regions we see that they got assigned the same label (meaning they are connected). Afterwards, we filter out small regions by an area threshold an save the bounding boxes of the remaining regions

```python
regions = measure.regionprops(img_lab)
# filter out small regions
area_thresh = img_proc_np.shape[0] * img_proc_np.shape[1] * 0.001
defects_bboxes = [reg.bbox for reg in regions if reg.area >= area_thresh]
```

The resulting bounding boxes match the defects

{% include ft-image-defect-detection/defects_bboxes.html %}

Putting everything together, the post processing becomes

```python
def find_bounding_boxes(
    img: Image.Image,
    perc_thresh: int = 99,
    area_thresh: float = 0.001,
    dilation_size: int = 7,
) -> list:
    img_np = np.array(img)

    max_val = img_np.max()
    thresh_val = np.percentile(img_np, perc_thresh)
    ret, img_thresh = cv2.threshold(img_np, int(thresh_val), 1.0, cv2.THRESH_BINARY)
    img_thresh = img_thresh > 0

    # combine neighboring mask regions
    img_morph = morphology.binary_dilation(
        img_thresh, np.ones([dilation_size, dilation_size])
    )

    # remove artifacts due to blurring at the edges
    img_morph = segmentation.clear_border(img_morph)

    # assign label to each connected region
    img_lab = measure.label(img_morph)

    # filter out small regions
    regions = measure.regionprops(img_lab)
    area_thresh = img_proc_np.shape[0] * img_proc_np.shape[1] * area_thresh
    bboxes = [reg.bbox for reg in regions if reg.area >= area_thresh]

    return bboxes
```

## Defect Detection

Now we have all components to build a Fourier Transform based defect detection

```python
def ft_defect_detection(
    img: Image.Image,
    mag_thresh: float = 0.5,
    perc_thresh: int = 99,
    area_thresh: float = 0.001,
    dilation_size: int = 7,
) -> list:
    
    img_proc = ft_extract_anomalies(img, mag_thresh)
    bboxes = find_bounding_boxes(img_proc, perc_thresh, area_thresh, dilation_size)

    return bboxes
```

To see it in action, we iterate through the defect images and show the results

```python
# iterator through defect images
path_it = (data_path / "test/broken/").iterdir()

img_path = next(path_it)

img = Image.open(img_path)
img = img.convert("L")

defect_bboxes = ft_defect_detection(img, mag_thresh=0.65)

p = plot_img_rgba(img)
p = add_bboxes_on_plot(p, defect_bboxes)
show(p)
```

{% include ft-image-defect-detection/defects_example.html %}

{% include ft-image-defect-detection/defects_example_2.html %}

Use the [notebook in the github repo](https://github.com/r-mart/blog-posts/blob/7c11226f29447b73100eba344b6ae820feb650cd/posts/ft_for_defects_in_regular_structures.ipynb) to interactively apply the method to more examples.

## Summary

We demonstrated a simple approach to detect defects in images based on Fourier Transforms. The method is limited to the specific use case of images with strong regular patterns and locally confined defects. Defects which follow a pattern themselves would break our approach. It is also vulnerable to inhomogeneous lighting or vignetting effects. Furthermore, there are a couple of parameters like `mag_thresh` or `perc_thresh` which you will likely have to adjust to your specific use case. This requires knowledge and makes it hard to use the method in plug and play scenarios.
However, the advantage in contrast to Machine Learning based approaches is that it doesn't require a large training dataset and expensive model training. It also runs fast on CPUs without large memory footprint.


