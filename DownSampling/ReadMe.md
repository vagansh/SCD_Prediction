This folder contains three downsampling methods used in the training set. 
These were: 

1. Cluster Centroids.
2. Edited Nearest Neighbors.
3. Tomek Links.


Details about them are provided below: 

1. Cluster Centroids: is a prototype generation algorithm that generates a new set S’ where |S’| < |S| and S’ does not belong to S. Thus, this technique 
reduces the number of samples in the targeted classes but the remaining samples are generated- and not selected-from the original dataset. Cluster 
Centroids makes use of K-means to reduce the number of samples. Therefore each class is synthesized with centroids of the K-means method instead of 
the original samples. 

2. Tomek Links: this method detects the Tomek’s links between two samples of different class x and y, is defined such that for any sample z: 
D(x,y) < D(x,z) and D(x,y) < D(y,z), where D(.) is the distance between the two samples. In some other words, a Tomek’s link exist if the two samples are
the nearest neighbors of each other. 


3. Edited Nearest Neighbors: applies a nearest-neighbors algorithm and “edit” the dataset by removing samples which do not agree “enough” with their 
neighborhood. For each sample in the class to be under-sampled, the nearest neighbors are computed and if the selection criterion is not fulfilled, 
the sample is removed. 


Documentation Link: 
https://imbalanced-learn.org/stable/references/under_sampling.html

