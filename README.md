# DAViD: Data-efficient and Accurate Vision Models from Synthetic Data

The repo accompanies the ICCV 2025 paper [DAViD: Data-efficient and Accurate Vision Models from Synthetic Data](https://microsoft.github.io/DAViD) and contains instructions for downloading and using the SynthHuman dataset introduced in the paper.

## 📊 The SynthHuman Dataset

TODO

## 🔓 Released Models

We release models for the following tasks:
<table>
  <thead>
    <tr>
      <th>Task</th>
      <th>Version</th>
      <th>Download</th>
      <th>Model Card</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2">Soft Foreground Segmentation</td>
      <td>Base</td>
      <td><a href="">Download</a></td>
      <td rowspan="2"><a href="./model_cards/soft_foreground_segmentation_model.md">Model Card</a></td>
    </tr>
    <tr>
      <td>Large</td>
      <td><a href="">Download</a></td>
    </tr>
    <tr>
      <td rowspan="2">Relative Depth Estimation</td>
      <td>Base</td>
      <td><a href="">Download</a></td>
      <td rowspan="2"><a href="./model_cards/depth_model.md">Model Card</a></td>
    </tr>
    <tr>
      <td>Large</td>
      <td><a href="">Download</a></td>
    </tr>
    <tr>
      <td rowspan="2">Surface Normal Estimation</td>
      <td>Base</td>
      <td><a href="">Download</a></td>
      <td rowspan="2"><a href="./model_cards/surface_normal_model.md">Model Card</a></td>
    </tr>
    <tr>
      <td>Large</td>
      <td><a href="">Download</a></td>
    </tr>
    <tr>
      <td rowspan="2">Multi-Task Model</td>
      <td>Base</td>
      <td><a href="./models/multitask_base.onnx">Download</a></td>
      <td rowspan="2"><a href="./model_cards/multi_task_model.md">Model Card</a></td>
    </tr>
    <tr>
      <td>Large</td>
      <td><a href="">Download</a></td>
    </tr>
  </tbody>
</table>     


## 🚀 Run the Demo

This demo supports running:

- Relative depth estimation
- Soft foreground segmentation
- Surface normal estimation

To install the requirements for running demo:
```bash
pip install -r requirement.txt
```

You can use either run:

1. A multi-task model that performs all tasks simultaneously

```bash
python demo.py \
  --image path/to/input.jpg \
  --multitask-model models/multitask.onnx
```
2. Or using individual models

```bash
python demo.py \
  --image path/to/input.jpg \
  --depth-model models/depth.onnx \
  --foreground-model models/foreground.onnx \
  --normal-model models/normal.onnx
```

🧠 **Notes:**
- The script expects ONNX models. Ensure the model paths are correct.
- If both multi-task and individual models are provided, results from both will be shown and compared.
- Foreground masks are used for improved visualization of depth and normals.

Here is an example output image after running the demo:

![](img/demo_result.png)



## Citation

If you use the SynthHuman Dataset in your research, please cite the following:

```
@inproceedings{saleh2025david,
    title={{DAViD}: Data-efficient and Accurate Vision Models from Synthetic Data},
    author={Saleh, Fatemeh and Aliakbarian, Sadegh and Hewitt, Charlie and Petikam, Lohit and Xiao, Xian and Criminisi, Antonio and Cashman, Thomas J. and Baltru{\v{s}}aitis, Tadas},
    booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    year={2025},
    month={October}
}
```
