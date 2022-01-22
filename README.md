# Cross-Component-Linear-Model-Prediction

## Intra-Prediction:

- Documents and links:
    - [A deep convolutional neural network approach for complexity reduction on intra-mode HEVC](./docs/ADeepConvolutionalNeuralNetworkApproachForComplexityReductionOnIntra-modeHEVC.pdf)
    - [CPH CU partition dataset](https://www.dropbox.com/sh/eo5dc3h27t41etl/AAADvFKoc5nYcZw6KO9XNycZa?dl=0)

### Database for Inra-mode:

It is served for CU Partition of HEVC at Intra-mode: the CPH-Intra database. First, 2000 images at resolution 4928×3264 are selected from Raw Images Dataset (RAISE). These 2000 images are randomly divided into training (1700 images), validation (100 images) and test (200 images) sets. Furthermore, each set is equally divided into four subsets: one subset is with original resolution and the other three subsets are down-sampled to be 2880×1920, 1536×1024 and 768×512. As such, this CPH-Intra database contains images at different resolutions. This ensures sufficient and diverse training data for learning to predict CU partition.

## Abbreviations used:
- CTU: Coding Tree Unit
- QP: Quantization Parameter
- CU: Codign Unit
- HEVC: High Efficiency Video Coding
- RAISE: Raw Images Dataset
- CPH: CU partition of HEVC

## Depth Prediction Result:

| Objective | Accuracy |
|----------|-------------:|
| Train | 87.85 |
| Validation | 86.5 |
| Test | 85.5 |

```text
work in progress...
```
# SMILE IN PAIN :'):

Apply intra pred with cuda(cuda_version=11.6) 

