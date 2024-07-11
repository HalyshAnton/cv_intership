# Text classification
Classifier for printed and handwritten text

## Model Architecture
Model consists of 2 parts:
* pre-trained feature extractor(vgg19_bn)
* 1-layer classifier head with dropout


Training params
* loss - bce
* optimizer - Adam with lr=1e-3
* scheduler - ReduceLROnPlateau

## Results
Training results for 10 epochs and cross-validation with 3 folds
![](results/loss.png)

![](results/accuracy.png)

![](results/confusion_matrix.png)

![](results/classification_report.png)
