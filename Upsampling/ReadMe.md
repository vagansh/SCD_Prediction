This folder explores two algorithms for UpSampling the data and training the model on these re-sampled datasets.

Two methods are used: 

a.	Synthetic Minority Oversampling for Numerical and Categorical (SMOTENC): generate new samples by interpolation. SMOTENC is an extension of SMOTE 
algorithm for which categorical data are treated differently. 
b.	Adaptive Synthetic Algorithm (ADASYN) generates samples next to the original samples which are wrongly classified using a k-Nearest Neighbors 
classifier. 
