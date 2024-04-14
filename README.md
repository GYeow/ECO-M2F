# Efficient Transformer Encoders for Mask2Former-style models

Official implement of ECO-M2F

[[arXiv]()]

## Installation
- Our project is developed on Detectron2. Please follow the official installation [instructions](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md), or the following instructions.
```
# create new environment
conda create -n eco_m2f python=3.8
conda activate eco_m2f

# install pytorch
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 cudatoolkit=11.1 -c pytorch

# install Detectron2 from a local clone
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
```

## Datasets
- COCO
- Cityscapes

## Evaluation with pretrained weights

## Model Zoo
### Results on COCO
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Configs</th>
<th valign="bottom">PQ</th>
<th valign="bottom">mIoU_p</th>
<th valign="bottom">AP_p</th>
<th valign="bottom">Total GLOPs</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->
<!-- ROW: 00302_model_0109999 -->
 <tr><td align="left"><a href="./configs/00302.yaml">$\beta=0.0005$</a></td>
<td align="center">52.06</td>
<td align="center">62.76</td>
<td align="center">41.51</td>
<td align="center">202.39</td>
<td align="center"><a href="https://drive.google.com/file/d/1XFEBSMgnWHYVdNF7w5Zo6HbeSWIfc5fG/view?usp=drive_link">model</a></td>
</tr>
<!-- ROW: 00297_model_0009999 -->
 <tr><td align="left"><a href="./configs/00297.yaml">$\beta=0.02$</a></td>
<td align="center">50.79</td>
<td align="center">62.25</td>
<td align="center">39.71</td>
<td align="center">181.64</td>
<td align="center"><a href="https://drive.google.com/file/d/1z3r9tzZIUXqQ_cOPiXiR83VAg6QjWJ44/view?usp=drive_link">model</a></td>
</tr>
</tbody></table>

### Results on Cityscapes
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Configs</th>
<th valign="bottom">PQ</th>
<th valign="bottom">mIoU_p</th>
<th valign="bottom">AP_p</th>
<th valign="bottom">Total GLOPs</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->
<!-- ROW: 00308_model_0013999 -->
 <tr><td align="left"><a href="./configs/00308.yaml">$\beta=0.003$</a></td>
<td align="center">64.18</td>
<td align="center">80.49</td>
<td align="center">39.64</td>
<td align="center">507.51</td>
<td align="center"><a href="https://drive.google.com/file/d/1AZxFyGTz4pFZuchSmTK7Dj2sHLNHz9Ve/view?usp=drive_link">model</a></td>
</tr>
<!-- ROW: 00284_model_0042499 -->
 <tr><td align="left"><a href="./configs/00284.yaml">$\beta=0.01$</a></td>
<td align="center">62.09</td>
<td align="center">79.58</td>
<td align="center">36.04</td>
<td align="center">439.67</td>
<td align="center"><a href="https://drive.google.com/file/d/1FAJ0s5VpL-YJB97_TENrvoXQU5v_V40H/view?usp=drive_link">model</a></td>
</tr>
</tbody></table>


## Acknowledgement

Code is largely based on Mask2Former (https://github.com/facebookresearch/Mask2Former).