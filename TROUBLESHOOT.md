# ERRORS
## 1. `from  vispr.dataset.pap_dataset import PAPDataset ModuleNotFoundError: No module named 'vispr'`
### Solution:
set PYTHONPATH for a single session From the repo root, set PYTHONPATH to the repo root and run the script normally:
```powershell
cd 'C:\Users\mihir\Documents\PhD\Workspace\Dev\vpa'
$env:PYTHONPATH = (Get-Location).Path
python .\vispr\tools\scripts\train_torch.py `
    --infile train_list.txt `
    --valfile val_list.txt `
    --arch resnet50 `
    --pretrained `
    --epochs 50 `
    --batch-size 32 `
    --lr 0.001 `
    --num-classes 68 `
    --save-path .\checkpoints\model_final.pth `
    --device cuda
```
This puts the repository root onto sys.path so Python can find the `vispr` package.