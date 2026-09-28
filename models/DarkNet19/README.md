# DarkNet19

The included result files are the outputs associated with this model in the study.

## Configuration

- Epochs: 25
- Batch size: 32
- Optimizer: Adam
- Learning rate: 1e-4
- Scheduler: StepLR(step_size=7, gamma=0.1) in the cleaned script
- Loss: CrossEntropyLoss
- Pretrained weights: No
- Preprocessing: Resize(224×224) + ImageNet normalization
- Final test accuracy: 97.12%
- Reported manuscript total training energy: 325.25 kJ

Run example:

```bash
python train.py --train-dir /path/to/train --test-dir /path/to/test --output-dir results_new
```
