# Model Card for Relative Depth Estimation Model

<!-- Provide a quick summary of what the model is/does. -->

**Model Summary:** A human-centric monocular depth estimation model to predict relative depth maps from a single image.
The images can come from various sources, such as photos, videos, or webcams and the model has been trained to cope with general conditions, such as different lighting and backgrounds.


## Model Details

### Model Description

Monocular relative depth estimation model.

- **Developed by:** DAViD's authors
- **Funded by:** Microsoft
- **Model type:** DPT-based neural network (released as ONNX)
- **License:** MIT

### Model Sources

- **Repository:** aka.ms/DAViD
- **Paper:**  Saleh et al., DAViD: Data-efficient and Accurate Vision Models from Synthetic Data, ICCV 2025

## Uses


### Direct Use
Estimating relative depth from a single image of human.

### Out-of-Scope Use

The models are trained for estimating relative depth maps for humans in the image and not suitable for other objects.

## Bias, Risks, and Limitations

The model is trained purely on __human-centric__ synthetic data, limited by its assets and scene variations.  As for all human-centric computer vision, the models we
train and demonstrate in this work could have lower accuracy for some demographic groups. We find that our use of
synthetic data helps in addressing any lack of fairness we
discover in model evaluations, given the precise control we
have over the training data distribution. Nevertheless, there
are aspects of human diversity that are not yet represented
by our datasets (see Sec. 6 of the paper), and there may also be lack of
fairness that we have not yet discovered in evaluations.

### Recommendations

Outputs should not be used as sole input for safety-critical applications (e.g., autonomous driving, medical use) without further validation. Further evaluation and dataset-specific adaptation are recommended before deployment in new domains.


## How to Get Started with the Model

See the [`demo.py`](../demo.py) for an example usage of the model.

## Training Details

### Training Data

See [`README`](../README.md) for accessing the training data.

### Training Procedure

See the paper.

#### Training Hyperparameters


    - Model backbone: [vitl16/vitb16]_384
    - Pretrained backbone: True
    - Input size: 512x512 px.
    - Input format: the input image is expected to be BGR with floats in the range [0, 1]
    - Output format: Relative depth map (1, 512, 512)

See the paper for more information.

#### Speed and Size

| Model         | Version | Size (MB) | MACs         |
|---------------|---------|-----------|--------------|
| Relative Depth Estimation   | Base    | 428.42    | 172.4B       |
|               | Large   | 1314.13   | 331.5B       |

## Evaluation


### Testing Data, Factors & Metrics

#### Testing Data


We evaluate our depth model on [Goliath](https://github.com/facebookresearch/goliath) and [Hi4D](https://github.com/yifeiyin04/Hi4D) datasets. See the paper for further details.

#### Metrics


To evaluate depth estimation models, we report the mean absolute value of the relative depth (AbsRel) and the root mean square error (RMSE), following standard practice.

**Absolute Relative Error (AbsRel)**: The mean of the element-wise ratio between the absolute difference of predicted and ground truth depth, and the ground truth depth itself. Computed after aligning the prediction to the target using a scale and shift transformation. Lower is better.

**Root Mean Square Error (RMSE)**: The square root of the mean squared difference between predicted and ground truth depth values, after applying optimal scale and shift alignment. This metric penalizes larger errors more heavily. Lower is better.

### Results

<table style="width:100%; border-collapse: collapse; font-size: small;">
  <caption style="caption-side: top; font-weight: bold; margin-bottom: 0.5em;">
    Depth estimation on Goliath and Hi4D dataset.
  </caption>
  <thead>
    <tr>
      <th rowspan="3" style="border-bottom: 2px solid #000;">Method</th>
      <th rowspan="3" style="border-bottom: 2px solid #000;">GFLOPS</th>
      <th rowspan="3" style="border-bottom: 2px solid #000;">Params</th>
      <th colspan="2" style="border-bottom: 1px solid #999; text-align:center;">Goliath-Face</th>
      <th colspan="2" style="border-bottom: 1px solid #999; text-align:center;">Goliath-UpperBody</th>
      <th colspan="2" style="border-bottom: 1px solid #999; text-align:center;">Goliath-FullBody</th>
      <th colspan="2" style="border-bottom: 1px solid #999; text-align:center;">Hi4D</th>
      <th colspan="2" style="border-bottom: 1px solid #999; text-align:center;">Averaged over all</th>
    </tr>
    <tr>
      <th style="border-bottom: 1px solid #ccc; text-align:center;">RMSE ↓</th>
      <th style="border-bottom: 1px solid #ccc; text-align:center;">AbsRel ↓</th>
      <th style="border-bottom: 1px solid #ccc; text-align:center;">RMSE ↓</th>
      <th style="border-bottom: 1px solid #ccc; text-align:center;">AbsRel ↓</th>
      <th style="border-bottom: 1px solid #ccc; text-align:center;">RMSE ↓</th>
      <th style="border-bottom: 1px solid #ccc; text-align:center;">AbsRel ↓</th>
      <th style="border-bottom: 1px solid #ccc; text-align:center;">RMSE ↓</th>
      <th style="border-bottom: 1px solid #ccc; text-align:center;">AbsRel ↓</th>
      <th style="border-bottom: 1px solid #ccc; text-align:center;">RMSE ↓</th>
      <th style="border-bottom: 1px solid #ccc; text-align:center;">AbsRel ↓</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Ours-Base</strong></td>
      <td style="text-align:center;">344</td>
      <td style="text-align:center;">0.12B</td>
      <td style="text-align:center;">0.142</td>
      <td style="text-align:center;">0.009</td>
      <td style="text-align:center;">0.316</td>
      <td style="text-align:center;">0.009</td>
      <td style="text-align:center;">0.376</td>
      <td style="text-align:center;">0.010</td>
      <td style="text-align:center;">0.085</td>
      <td style="text-align:center;">0.024</td>
      <td style="text-align:center;">0.212</td>
      <td style="text-align:center;">0.014</td>
    </tr>
    <tr>
      <td><strong>Ours-Large</strong></td>
      <td style="text-align:center;">663</td>
      <td style="text-align:center;">0.34B</td>
      <td style="text-align:center;">0.140</td>
      <td style="text-align:center;">0.009</td>
      <td style="text-align:center;">0.283</td>
      <td style="text-align:center;">0.008</td>
      <td style="text-align:center;">0.334</td>
      <td style="text-align:center;">0.009</td>
      <td style="text-align:center;">0.072</td>
      <td style="text-align:center;">0.019</td>
      <td style="text-align:center;">0.191</td>
      <td style="text-align:center;">0.012</td>
    </tr>
  </tbody>
</table>

## Environmental Impact


Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

These are the numbers for training the _Large_ version of the model:

- **Hardware Type:** NVIDIA A100x4 for training
- **Hours used:** 500 GPU-hours
- **Cloud Provider:** Microsoft Azure
- **Compute Region:** westeurope
- **Carbon Emitted:**  71.25 kg CO_2 eq.

## Citation

**BibTeX:**

```
@inproceedings{saleh2025david,
    title={{DAViD}: Data-efficient and Accurate Vision Models from Synthetic Data},
    author={Saleh, Fatemeh and Aliakbarian, Sadegh and Hewitt, Charlie and Petikam, Lohit and Xiao, Xian and Criminisi, Antonio and Cashman, Thomas J. and Baltru{\v{s}}aitis, Tadas},
    booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    year={2025},
    month={October}
}
```
