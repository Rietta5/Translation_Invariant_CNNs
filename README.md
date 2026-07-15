# 🖼️ Translation Invariant CNNs

Official implementation of the paper:
**"Parameter-Efficient Architectural Modifications for Translation-Invariant CNNs"**  
Nuria Alabau-Bosque, Jorge Vila-Tomás, Paula Daudén-Oliver, Valero Laparra, and Jesús Malo.  
[[ArXiv](https://arxiv.org/abs/2604.27870)] [[DOI: 10.48550/arXiv.2604.27870](https://doi.org/10.48550/arXiv.2604.27870)]

---

## 📝 About the Paper

Standard Convolutional Neural Networks (CNNs) exhibit a startling fragility: even a single-pixel shift can drastically degrade performance due to their reliance on spatially dependent fully connected layers. In this work, we resolve this vulnerability by proposing a lightweight **'Online Architecture'** strategy. By strategically inserting **Global Average Pooling (GAP)** layers, we effectively decouple feature recognition from spatial location. 

Using VGG-16 as a case study, we demonstrate:
- **98% reduction** in trainable parameters (from 5.2M to 82K).
- **2x improvement** in translational robustness while maintaining competitive accuracy on ImageNet.
- **Superior generalization** when integrated into perceptual metrics like LPIPS.

## ✨ Key Features

This repository provides the scripts for training and evaluating standard vs. modified (GAP-based) architectures to assess their invariance to geometric transformations.

- **Training Pipelines**: Automated scripts for ResNet and VGG models on MNIST and ImageNet.
- **Invariance Benchmarking**: Comprehensive evaluation tools for rotation, scaling, and translation robustness.
- **GAP Architectures**: Implementations of models using Global Average Pooling for enhanced spatial invariance.
- **Perceptual Metrics**: Training and evaluation of the LPIPS metric using invariant backbones.
- **Utilities**: Robust image manipulation tools (rotation, translation, scaling) and dataset loaders in `utils.py`.

## 📂 Repository Structure

- `00_Entrenamiento_*.py`: Training scripts for MNIST.
- `01_Entrenamiento_*.py`: Training scripts for ImageNet.
- `02_TestModelos_*.py`: Evaluation scripts for MNIST models under transformations.
- `03_TestModelos_*.py`: Evaluation scripts for ImageNet models under transformations.
- `05_Entrenamiento_LPIPSVGGGAP_TID08.py`: Training a modified LPIPS metric on the TID2008 dataset.
- `utils.py`: Common helper functions for data loading and image processing.

## ⚙️ Installation

Ensure you have Python 3.8+ and install the dependencies using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

## 🚀 Usage Examples

### 🏋️ Training a Model
```bash
python 00_Entrenamiento_ResNet_MNIST.py
```

### 🧪 Testing Invariance
To evaluate a model's performance under various scaling factors, rotations, or translations:
```bash
python 02_TestModelos_ResNet_MNIST.py
```
This script loads the weights, applies transformations, and saves results as `.pkl` files.

### 🛠️ Image Manipulation
```python
from utils import rotar, escalar, trasladar_MNIST

# Rotate an image by 45 degrees
rotated_img = rotar(image_data, rotacion=45)

# Scale a dataset
scaled_dataset = escalar(dataset, escala=1.2, size=128)
```

## 📊 Datasets

The project supports:
- **MNIST / CIFAR-10**: Loaded via Keras.
- **ImageNet**: Expected in a local `./Imagenet` folder.
- **TID2008 / TID2013**: Used for perceptual metric experiments.

## 🎓 Citation

If you find this work useful, please cite our paper:

```bibtex
@misc{alabaubosque2026parameterefficient,
      title={Parameter-Efficient Architectural Modifications for Translation-Invariant CNNs}, 
      author={Nuria Alabau-Bosque and Jorge Vila-Tomás and Paula Daudén-Oliver and Valero Laparra and Jesús Malo},
      year={2026},
      eprint={2604.27870},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2604.27870}, 
}
```

## 📜 License

This project is licensed under the terms provided in the `LICENSE` file.
