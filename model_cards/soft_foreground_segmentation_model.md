# Model Card for Soft Foreground Segmentation Model


**Model Summary:** 
A soft foreground segmentation model for segmenting foreground human subjects in an image.
The images can come from various sources, such as photos, videos, or webcams and the model has been trained to cope with general conditions, such as different lighting and backgrounds.


## Model Details

### Model Description

Soft foreground segmentation.

- **Developed by:** DAViD's authors
- **Funded by:** Microsoft
- **Model type:** DPT-based neural network (released as ONNX)
- **License:** [MIT](../licenses/LICENSE-MIT.txt)

### Model Sources

- **Repository:** [aka.ms/DAViD]()
- **Paper:**  [Saleh et al., DAViD: Data-efficient and Accurate Vision Models from Synthetic Data, ICCV 2025]()

## Uses

### Direct Use
Segmenting the human foreground from a single image (soft mask).

### Out-of-Scope Use

The models are intended for segmenting prominent human subjects within images and are unsuitable for crowded scenes or other objects.

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
    - Output format: Soft foreground mask (1, 1, 512, 512)

See the paper for more information.

#### Speed and Size

| Model         | Version | Size (MB) | MACs         |
|---------------|---------|-----------|--------------|
| Soft Foreground segmentation   | Base    | 428.42    | 172.4B       |
|               | Large   | 1314.13   | 331.5B       |

## Evaluation


### Testing Data, Factors & Metrics

#### Testing Data


We evaluate our soft foreground segmentation model on [PhotoMatte85](https://grail.cs.washington.edu/projects/background-matting-v2/#/datasets), [PPM-100](https://github.com/ZHKKKe/PPM), and [P3M](https://github.com/JizhiziLi/P3M) datasets. See the paper for further details.

#### Metrics


To evaluate soft foreground segmentation model, we report the following metrics:

**SAD (Sum of Absolute Differences)**: Measures the total absolute difference between the predicted and ground truth masks, scaled by a factor of 1000. Lower values indicate better alignment between predictions and ground truth.

**MSE (Mean Squared Error)**: Computes the average squared difference per pixel between the predicted and ground truth masks. Lower values reflect better prediction accuracy.

**SAD (Trimap)**: Similar to SAD but calculated only within the uncertain regions defined by the trimap (i.e., where the trimap value equals 128). Lower values are better.


### Results

<table>
  <thead>
    <tr>
      <th rowspan="2">Model</th>
      <th colspan="2">PhotoMatte85</th>
      <th colspan="1">PPM-100</th>
      <th colspan="2">P3M-NP</th>
      <th colspan="2">P3M-P</th>
    </tr>
    <tr>
      <th>SAD ↓</th>
      <th>MSE ↓</th>
      <th>SAD ↓</th>
      <th>SAD ↓</th>
      <th>SAD-T ↓</th>
      <th>SAD ↓</th>
      <th>SAD-T ↓</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>DAViD-Base</b></td>
      <td>7.97</td>
      <td>0.0017</td>
      <td>90.86</td>
      <td>14.88</td>
      <td>10.31</td>
      <td>17.92</td>
      <td>9.66</td>
    </tr>
    <tr>
      <td><b>DAViD-Large</b></td>
      <td>5.85</td>
      <td>0.0009</td>
      <td>78.17</td>
      <td>14.83</td>
      <td>10.23</td>
      <td>12.65</td>
      <td>9.19</td>
    </tr>
  </tbody>
</table>



## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

These are the numbers for training the _Large_ version of the model:
- **Hardware Type:** NVIDIA A100x4
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
