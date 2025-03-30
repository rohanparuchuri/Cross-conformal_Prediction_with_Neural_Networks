# Cross-conformal_Prediction_with_Neural_Networks
## Introduction:
The implementation of cross-conformal predictors is the same as cross- validation. Like cross-validation, the training set is divided into k-folds, and iteratively each fold will be considered as a calibration set and the remaining folds will be considered as proper training sets. During each iteration, the non-conformity scores of calibration samples are calculated the same as they were done with split conformal predictions. This makes cross-conformal predictors, a better alternative to inductive conformal predictors in terms of the reliability of prediction sets. However, they are not as good as split conformal predictions when it comes to computational efficiency, but they are much better than transductive conformal predictors, as the underlying model only needs to be trained as many as the number of folds. Cross-conformal predictors can further be divided into different variants based on the way how data is split. One of those variants is the jackknife+ conformal predictor. Jackknife+ is the extension of the Jackknife method, which uses a leave-one-out approach. While implementing Jackknife, during each iteration, the underlying model is trained on the entire training set except one and this one sample is considered a calibration set. So, the underlying model needs to be trained as many times as the number of samples. The model will be trained on the entire dataset after completion of all the iterations, and make predictions on the test set. But jackknife+ makes predictions on the test set during the training iterations, along with the leave- one-out sample. This makes Jackknife+, a special case of CV+ conformal predictor. However, training the underlying model as many times as the number of training samples requires a lot more computational capacity than it is needed for implementing standard cross-conformal predictors.
## Background:
Even though inductive conformal predictors are computationally very efficient, they don’t use the entire training dataset. So, to tackle this issue and to make use of the whole training set, the training set Z is divided into k non-empty sets.
Z = (Z1,Z2,………,ZK)
After dividing the training set into k-folds, iteratively each fold will be considered as calibration set and the remaining folds will used as the proper training set. The p-value for each label is calculated by averaging the sum of all the ranks over the training set. This enables the model to learn from the whole dataset and the model only needs to be trained as many times as the number of folds. This makes cross-conformal prediction a better alternative over transductive conformal predictor, in terms of computational efficiency and inductive conformal predictor in terms of providing reliable prediction sets. In this project, two variations of cross-conformal predictors were implemented. The first one is the standard cross-conformal predictor and the second one is implemented based on the jackknife+ method.
1. Standard cross-conformal predictor: - standard cross-conformal predictor is implemented by dividing the training set into k-folds. The value of k is usually 5 or 1o.
2. Jackknife+ cross-conformal predictor:- The jackknife+ cross-conformal predictor is implemented based on the jackknife method. Jackknife is a leave- one-out method, where every training sample will be considered as a fold.
## Dataset:
The Boston Housing prices dataset is also from the UCL machine learning repository. This dataset is used for the regression task of finding out the median values of houses based on various features. The dataset consists of 506 samples with 13 input features and a target variable, 404 samples were used for training and the remaining 102 were used for testing. The 13 input features are
CRIM : Per capita crime rate
ZN: Proportion of residential land zone
INDUS: Proportion of non-retail business acres
CHAS: value is 1 if the land touches Charles River and 0 otherwise
NOX: Nitric Oxide concentration
RM: Average number of rooms
AGE: Proportion of houses built before 1940
DIS: Distance to 5 Boston employment centres
RA: accessibility to highways
TAX: Property tax rate per $10,000
PTRATIO: student-teacher ratio per town
B: Proportion of black residents per town
LSTAT: Percentage of low-economic status population
The target variable is the median value of the houses in thousands of dollars based on the above features.
## Implementation:
The programming language used for implementing cross-conformal prediction framework is Python 3.12.4, and the libraries used are Tensorflow, Pandas, Numpy, Keraas-Tuner, Scikit-learn, Matplotlib.
## References:
1] Algorithmic Learning in a Random World, 2005. . Springer-Verlag, New York. https://doi.org/10.1007/b106715
2] Barber, R.F., Candes, E.J., Ramdas, A., Tibshirani, R.J., 2019. Predictive inference with the jackknife+. https://doi.org/10.48550/ARXIV.1905.02928
3] Duch, W., Iliadis, L.S., Diamantaras, K.I., 2010. Artificial Neural Networks - ICANN 2010: 20th International Conference, Thessaloniki, Greece, September 15-18, 2010, Proceedings, Part I, Lecture Notes in Computer Science. Springer Berlin Heidelberg Springer e-books, Berlin, Heidelberg.
4] Papadopoulos, H., Vovk, V., Gammermam, A., 2007. Conformal Prediction with Neural Networks, in: 19th IEEE International Conference on Tools with Artificial Intelligence(ICTAI 2007). Presented at the 19th IEEE International Conference on Tools with Artificial Intelligence(ICTAI 2007), IEEE, Patras, Greece, pp. 388–395. https://doi.org/10.1109/ICTAI.2007.47
5] Vovk, V., 2015. Cross-conformal predictors. Ann. Math. Artif. Intell. 74, 9–28. https://doi.org/10.1007/s10472-013-9368-4
6] Fontana, Matteo, Gianluca Zeni, and Simone Vantini. “Conformal Prediction: A Unified Review of Theory and New Challenges.” Bernoulli
29, no. 1 (February 1, 2023). https://doi.org/10.3150/21-BEJ1447. Tools in Artificial Intelligence. Erscheinungsort nicht ermittelbar: IntechOpen, 2008.
