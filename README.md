# MAU-Net: MRI Brain Tumour Segmentation

Implementation of "MRI Brain Tumour Segmentation Using Multiscale Attention U-Net" by Chen et al. (2024)

**Optimized for RTX 2060 8GB GPU**

## Quick Start

### 1. Setup Environment
```bash
# Create an virtual environment
python -m venv myenv
# Or
python3 -m venv myenv

# Active your virtual environment
# Windows
.\myenv\Scripts\activate
# Mac and Linux
source myenv/bin/activate

# Or manually install
pip install -r requirements.txt
```

### 2. Extract the dataset
```bash
# Reassembel dataset zip
# Mac or Linux
cat BraTs_2020.zip.part.0* > "BraTs 2020.zip"
# windows
copy /b BraTs_2020.zip.part.* "BraTs 2020.zip"

# Extract
# Mac or Linux
unzip "BraTs 2020.zip"
# Windows
tar -xf "BraTs 2020.zip"
```

### 3. Start Jupyter Notebook
```bash
jupyter notebook
```

### 4. Open the Main Notebook
- Open `MAU_Net_Brain_Tumor_Segmentation.ipynb`
- Select kernel: `mau_net or Name of your virtual environment`

## Project Structure

```
AI700Project/
├── MAU_Net_Brain_Tumor_Segmentation.ipynb  # Main notebook
├── setup.py                                # Setup script
├── requirements.txt                        # Dependencies
├── research paper.pdf                      # Original paper
└── BraTs 2020/                            # Dataset directory
    ├── BraTS2020_TrainingData/
    └── BraTS2020_ValidationData/
```

## Memory Optimizations for RTX 2060 8GB

- **Input Size**: 96×96×96 (reduced from 128×128×128)
- **Batch Size**: 1 (reduced from 4)
- **Mixed Precision**: FP16 training enabled
- **Gradient Checkpointing**: Enabled for CPM module
- **Reduced Channels**: Optimized channel dimensions

## Model Architecture

### Key Components:
1. **Mixed Depth-wise Convolution (MDConv)**: Multi-scale feature extraction
2. **Context Pyramid Module (CPM)**: Attention mechanism for global context
3. **Self-Ensemble**: Feature fusion from multiple decoder levels

### Expected Performance (BraTS 2020):
- **ET (Enhancing Tumor)**: ~78.5% DSC
- **WT (Whole Tumor)**: ~90.2% DSC  
- **TC (Tumor Core)**: ~82.8% DSC

## Training Configuration

```python
config = {
    'learning_rate': 0.001,
    'batch_size': 1,
    'num_epochs': 100,
    'crop_size': (96, 96, 96),
    'optimizer': 'Adam',
    'loss': 'Combined CE + Dice (α=0.25)'
}
```

## Usage

### Option 1: Jupyter Notebook (Recommended)
1. Open `MAU_Net_Brain_Tumor_Segmentation.ipynb`
2. Run cells sequentially
3. Monitor training progress with plots

## Dataset Format

The BraTS dataset should have this structure:
```
BraTS2020_TrainingData/
├── BraTS20_Training_001/
│   ├── BraTS20_Training_001_flair.nii.gz
│   ├── BraTS20_Training_001_t1.nii.gz
│   ├── BraTS20_Training_001_t1ce.nii.gz
│   ├── BraTS20_Training_001_t2.nii.gz
│   └── BraTS20_Training_001_seg.nii.gz
└── ...
```

## Segmentation Labels

- **Label 0**: Background
- **Label 1**: Necrotic and non-enhancing tumor core
- **Label 2**: Peritumoral edema  
- **Label 4**: GD-enhancing tumor (converted to 3 in model)

## Evaluation Regions

- **WT (Whole Tumor)**: Labels 1, 2, 4
- **TC (Tumor Core)**: Labels 1, 4
- **ET (Enhancing Tumor)**: Label 4

## Troubleshooting

### Out of Memory Errors
- Reduce batch size to 1
- Reduce input size to (80, 80, 80)
- Enable gradient checkpointing for all modules

### Slow Training
- Ensure CUDA is properly installed
- Use mixed precision training (FP16)
- Reduce number of workers in DataLoader

### Dataset Issues
- Verify BraTS data structure
- Check file paths in dataset.py
- Ensure all modalities are present

### Check the results in tensorboard
- Run Cell 8 in `MAU_Net_Brain_Tumor_Segmentation.ipynb` and run the command
```bash
tensorboard --logdir=runs
```
And open http://localhost:6006 on your browser

## Citation

```bibtex
@article{chen2024mri,
  title={MRI Brain Tumour Segmentation Using Multiscale Attention U-Net},
  author={Chen, Bonian and He, Tao and Wang, Weizhuo and Han, Yutong and Zhang, Jianxin and Bobek, Samo and Zabukovsek, Simona Sternad},
  journal={Informatica},
  volume={35},
  number={4},
  pages={751--774},
  year={2024}
}
```
