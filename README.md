# MTEV-Stereo
"Enhancing Stereo Matching in Edge and Textureless Regions via Multi-Level Texture Encoding and Dual-Frequency Feature Separation"

<img width="920" height="356" alt="image" src="https://github.com/user-attachments/assets/09ce4637-0a7d-44af-a0d6-7d87c4cafa4c" />

Comparison with state-of-the-art methods on the SceneFlow and KITTI dataset. Our method has achieved better performance.

<img width="922" height="190" alt="image" src="https://github.com/user-attachments/assets/a65d0373-5606-481a-a304-29c8751f0812" />

Compared with the IGEV algorithm, it can be seen that IGEV tends to blur image edges and produce mismatches in textureless regions. The method proposed in this paper effectively addresses the issue of insufficient performance in textureless and weak-texture areas.

<img width="921" height="396" alt="image" src="https://github.com/user-attachments/assets/20c4160e-29ad-41e8-a254-ef5b92bc4b3e" />



DEMO

You can demo a trained model on pairs of images. To predict stereo for demo-imgs directory, run
```Shell
python demo_imgs.py --restore_ckpt ./pretrained_models/sceneflow.pth --left_imgs './demo-imgs/*/im0.png' --right_imgs './demo-imgs/*/im1.png'
```
You can switch to your own test data directory, or place your own pairs of test images in ./demo-imgs.

Evaluation
```Shell
python evaluate_stereo.py --restore_ckpt ./pretrained_models/sceneflow.pth --dataset sceneflow
python evaluate_stereo.py --restore_ckpt ./pretrained_models/sceneflow.pth --dataset middlebury_H
```
Training

To train MTEV-Stereo on Scene Flow or KITTI, run
```Shell
python train_stereo.py --train_datasets sceneflow
```
or
```Shell
python train_stereo.py --train_datasets kitti --restore_ckpt ./pretrained_models/sceneflow.pth
```
Submission to the KITTI benchmark, run
```Shell
python save_disp.py
```
