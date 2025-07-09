# Model Card for Relative Depth Estimation Model

**Model Summary:** A human-centric multi-task model to predict relative depth maps, surface normal maps, and soft foreground masks from a single image.
The images can come from various sources, such as photos, videos, or webcams and the model has been trained to cope with general conditions, such as different lighting and backgrounds.


## Model Details

### Model Description

Multi-task model for estimating relative depth, surfance normals, and soft foreground segmentation.

- **Developed by:** DAViD's authors
- **Funded by:** Microsoft
- **Model type:** DPT-based neural network (released as ONNX)
- **License:** [MIT](../LICENSE-MIT.txt)

### Model Sources

- **Repository:** [aka.ms/DAViD](aka.ms/DAViD)
- **Paper:**  [Saleh et al., DAViD: Data-efficient and Accurate Vision Models from Synthetic Data, ICCV 2025]()

## Uses


### Direct Use
Estimating relative depth, surface normals, and soft foreground mask from a single image of human.

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


    - Model backbone: vitl16_384
    - Pretrained backbone: True
    - Input size: 512x512 px.
    - Input format: the input image is expected to be BGR with floats in the range [0, 1]
    - Output format: 
      1. Relative depth map (1, 512, 512)
      2. Surface normal map (1, 3, 512, 512)
      3. Soft foreground mask (1, 1, 512, 512)

See the paper for more information.

#### Speed and Size

| Model         | Version | Size (MB) | MACs         |
|---------------|---------|-----------|--------------| 
|  Multi task model             | Large   | 1316.66   | 389.9B       |

## Evaluation


### Testing Data, Factors & Metrics

#### Testing Data


We evaluate our multi-task model for 3 tasks of depth estimation, surface normal estimation and soft foreground segmentation. For depth and normal estimation, we evaluate our model on [Goliath](https://github.com/facebookresearch/goliath) and [Hi4D](https://github.com/yifeiyin04/Hi4D) datasets. For soft foreground segmentation, we evaluate our model on [PhotoMatte85](https://grail.cs.washington.edu/projects/background-matting-v2/#/datasets), [PPM-100](https://github.com/ZHKKKe/PPM), and [P3M](https://github.com/JizhiziLi/P3M). See the paper for further details.

#### Metrics


To evaluate depth estimation, we report the mean absolute value of the relative depth (AbsRel) and the root mean square error (RMSE), following standard practice.

To evaluate surface normal estimation, we report Mean/Median Angular Error and the percentage of pixels whose angular error falls below a threshold of t degrees, for t ∈ {11.25°, 22.5°, 30°}.

To evaluate soft foreground segmentation, we report SAD (Sum of Absolute Differences), MSE (Mean Squared Error), and SAD (Trimap).

For more details, please refer to the paper or the model cards of the individual models.

### Results

<table>
  <thead>
    <tr>
      <th rowspan="2">Setting</th>
      <th rowspan="2">Params</th>
      <th colspan="4">Depth</th>
      <th colspan="4">Surface Normal</th>
      <th colspan="3">Matting</th>
    </tr>
    <tr>
      <th>Goliath RMSE ↓</th>
      <th>Goliath AbsRel ↓</th>
      <th>Hi4D RMSE ↓</th>
      <th>Hi4D AbsRel ↓</th>
      <th>Goliath MAE(°) ↓</th>
      <th>Goliath %W 30° ↑</th>
      <th>Hi4D MAE(°) ↓</th>
      <th>Hi4D %W 30° ↑</th>
      <th>PPM-100 SAD ↓</th>
      <th>PhotoMatte85 SAD ↓</th>
      <th>PhotoMatte85 MSE ↓</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>DAViD-Multi-task-Large</td>
      <td>1× 0.35B</td>
      <td>0.270</td>
      <td>0.009</td>
      <td>0.078</td>
      <td>0.021</td>
      <td>15.27</td>
      <td>89.12</td>
      <td>15.61</td>
      <td>89.48</td>
      <td>66.08</td>
      <td>5.40</td>
      <td>0.0008</td>
    </tr>
  </tbody>
</table>


## Environmental Impact


Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

These are the numbers for training the _Large_ version of the model:

- **Hardware Type:** NVIDIA A100x4 for training
- **Hours used:** 528 GPU-hours
- **Cloud Provider:** Microsoft Azure
- **Compute Region:** westeurope
- **Carbon Emitted:**  75.24 kg CO_2 eq.

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
