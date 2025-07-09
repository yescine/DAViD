# Model Card for Surface Normal Estimation Model


**Model Summary:** 
A human-centric surface normal estimation model to predict normal maps from images.
The images can come from various sources, such as photos, videos, or webcams and the model has been trained to cope with general conditions, such as different lighting and backgrounds.


## Model Details

### Model Description

Surface normal estimation model.

- **Developed by:** DAViD's authors
- **Funded by:** Microsoft
- **Model type:** DPT-based neural network (released as ONNX)
- **License:** MIT

### Model Sources

- **Repository:** [aka.ms/DAViD](aka.ms/DAViD)
- **Paper:**  [Saleh et al., DAViD: Data-efficient and Accurate Vision Models from Synthetic Data, ICCV 2025]()

## Uses

### Direct Use
Estimating surface normal from a single image of human.

### Out-of-Scope Use

The models are specifically trained for human subjects in images and are not suitable for other object categories.

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
    - Output format: Surface normal map (1, 3, 512, 512)

See the paper for more information.

#### Speed and Size

| Model         | Version | Size (MB) | MACs         |
|---------------|---------|-----------|--------------|
| Surface Normal Estimation   | Base    | 428.42    | 172.4B       |
|               | Large   | 1314.13   | 331.5B       |

## Evaluation


### Testing Data, Factors & Metrics

#### Testing Data


We evaluate our surface normal model on [Goliath](https://github.com/facebookresearch/goliath) and [Hi4D](https://github.com/yifeiyin04/Hi4D) datasets. See the paper for further details.

#### Metrics


To evaluate surface normal estimation model, we report the following metrics:

**Mean Angular Error**: The average angle (in degrees) between predicted and ground truth surface normal vectors, computed per pixel within the evaluation mask. Lower is better.

**Median Angular Error**: The median of per-pixel angular errors, providing a robust measure of the deviation between predicted and ground truth normals. Lower is better.

**Percentage within t°**: The percentage of pixels whose angular error falls below a threshold of t degrees, for t ∈ {11.25°, 22.5°, 30°}. This metric captures the proportion of accurate predictions. Higher is better.

### Results
<table style="width:100%; border-collapse: collapse; font-size: small;">
  <caption style="caption-side: top; font-weight: bold; margin-bottom: 0.5em;">
    Surface normal estimation results on Goliath and Hi4D. 
  </caption>
  <thead>
    <tr>
      <th rowspan="3" style="border-bottom: 2px solid #000;">Method</th>
      <th colspan="3" style="border-bottom: 1px solid #999; text-align:center;">Goliath-Face</th>
      <th colspan="3" style="border-bottom: 1px solid #999; text-align:center;">Goliath-UpperBody</th>
      <th colspan="3" style="border-bottom: 1px solid #999; text-align:center;">Goliath-FullBody</th>
      <th colspan="3" style="border-bottom: 1px solid #999; text-align:center;">Hi4D</th>
    </tr>
    <tr>
      <th colspan="2" style="border-bottom: 1px solid #ccc; text-align:center;">Angular Error (°) ↓</th>
      <th style="border-bottom: 1px solid #ccc; text-align:center;">% Within t° ↑</th>
      <th colspan="2" style="border-bottom: 1px solid #ccc; text-align:center;">Angular Error (°) ↓</th>
      <th style="border-bottom: 1px solid #ccc; text-align:center;">% Within t° ↑</th>
      <th colspan="2" style="border-bottom: 1px solid #ccc; text-align:center;">Angular Error (°) ↓</th>
      <th style="border-bottom: 1px solid #ccc; text-align:center;">% Within t° ↑</th>
      <th colspan="2" style="border-bottom: 1px solid #ccc; text-align:center;">Angular Error (°) ↓</th>
      <th style="border-bottom: 1px solid #ccc; text-align:center;">% Within t° ↑</th>
    </tr>
    <tr>
      <th style="text-align:center;">Mean</th>
      <th style="text-align:center;">Median</th>
      <th style="text-align:center;">11.25° / 22.5° / 30°</th>
      <th style="text-align:center;">Mean</th>
      <th style="text-align:center;">Median</th>
      <th style="text-align:center;">11.25° / 22.5° / 30°</th>
      <th style="text-align:center;">Mean</th>
      <th style="text-align:center;">Median</th>
      <th style="text-align:center;">11.25° / 22.5° / 30°</th>
      <th style="text-align:center;">Mean</th>
      <th style="text-align:center;">Median</th>
      <th style="text-align:center;">11.25° / 22.5° / 30°</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>DAViD-Base</strong></td>
      <td style="text-align:center;">17.33</td>
      <td style="text-align:center;">12.36</td>
      <td style="text-align:center;">47.7 / 75.9 / 84.5</td>
      <td style="text-align:center;">14.10</td>
      <td style="text-align:center;">11.32</td>
      <td style="text-align:center;">50.3 / 83.9 / 91.8</td>
      <td style="text-align:center;">14.60</td>
      <td style="text-align:center;">11.79</td>
      <td style="text-align:center;">48.1 / 82.3 / 91.1</td>
      <td style="text-align:center;">15.72</td>
      <td style="text-align:center;">12.95</td>
      <td style="text-align:center;">43.2 / 78.7 / 89.2</td>
    </tr>
    <tr>
      <td><strong>DAViD-Large</strong></td>
      <td style="text-align:center;">17.15</td>
      <td style="text-align:center;">12.19</td>
      <td style="text-align:center;">48.4 / 76.3 / 84.7</td>
      <td style="text-align:center;">13.96</td>
      <td style="text-align:center;">11.23</td>
      <td style="text-align:center;">50.7 / 84.2 / 92.1</td>
      <td style="text-align:center;">14.60</td>
      <td style="text-align:center;">11.66</td>
      <td style="text-align:center;">48.7 / 82.2 / 90.8</td>
      <td style="text-align:center;">15.37</td>
      <td style="text-align:center;">12.51</td>
      <td style="text-align:center;">45.1 / 79.7 / 89.6</td>
    </tr>
  </tbody>
</table>



## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

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
