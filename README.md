# An implementation of the Binarized Neural Networks

## Dataset:
MNIST dataset
## Method:
There were two kinds of models named binarized neural networks (1-bit) and full-precision model (16-bits). Each model has three building blocks.    
## Results:
1. Loss curve

![loss](https://github.com/grassking100/bnn/blob/9394d46f364604f14286853d43f1360345731fb8/resource/loss.jpeg)

2. Macro F1 curve

![macro_f1](https://github.com/grassking100/bnn/blob/9394d46f364604f14286853d43f1360345731fb8/resource/macro_f1.jpeg)

3. Test confusion matrix of full-precision model 

![full_confusion](https://github.com/grassking100/bnn/blob/9394d46f364604f14286853d43f1360345731fb8/resource/test_full_confusion_matrix.jpeg)

4. Test confusion matrix of binarized neural networks

![binary_confusion](https://github.com/grassking100/bnn/blob/9394d46f364604f14286853d43f1360345731fb8/resource/test_binary_confusion_matrix.jpeg)

5. Test result

|Model|Epoch|Macro F1|Loss|
|:---:|:---:|:------:|:--:|
|Full-precision model|33|0.991|0.0382|
|Binarized neural networks|35|0.978|0.2589

6. Visualization of some kernels of the models

|Model|Kernel 0|Kernel 1|Kernel 2|
|:---:|:--:|:---:|:------:|
|Full-precision model|![f0](https://github.com/grassking100/bnn/blob/9394d46f364604f14286853d43f1360345731fb8/resource/full_0.jpeg)|![f1](https://github.com/grassking100/bnn/blob/9394d46f364604f14286853d43f1360345731fb8/resource/full_1.jpeg)|![f2](https://github.com/grassking100/bnn/blob/9394d46f364604f14286853d43f1360345731fb8/resource/full_2.jpeg)
|Binarized neural networks|![b0](https://github.com/grassking100/bnn/blob/9394d46f364604f14286853d43f1360345731fb8/resource/binary_0.jpeg)|![b1](https://github.com/grassking100/bnn/blob/9394d46f364604f14286853d43f1360345731fb8/resource/binary_1.jpeg)|![b2](https://github.com/grassking100/bnn/blob/9394d46f364604f14286853d43f1360345731fb8/resource/binary_2.jpeg)

7. Weights

- Full-precision model: [link](https://github.com/grassking100/bnn/blob/9394d46f364604f14286853d43f1360345731fb8/resource/full_model.pth)

- Binarized neural networks: [link](https://github.com/grassking100/bnn/blob/9394d46f364604f14286853d43f1360345731fb8/resource/binary_model.pth)

## References:
Simons, Taylor, and Dah-Jye Lee. "A review of binarized neural networks." Electronics 8.6 (2019): 661.
