# SciKit-Learn Regression Models and PCA
By August Semrau Andersen.

This project is an entry into the Kaggle competition 'House Prices: Advanced Regression Techniques'.  
https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview  
The competition consists of a regression task of predicting the sales price of houses based on 79 different explanatory variables.

The intent with the project is to display proficiency in feature engineering of data and using the SciKit-Learn package for a regression task.  
Further, Principal Component Analysis is used for garnering a better understanding of the data.


### Scripts
The following scripts are used for completing the competition.
 
1. **dataLoader.py** which loads .csv data and uses scikit-learn (sklearn) for preprocessing. 
2. **pca.py** is used for investigating the data visually using principal component analysis.
3. **models.py** contains some non-tuned sklearn regression models, some tuned models.
4. **predictions.py** is used for printing predictions to .csv format for entry in Kaggle-competition.



### PCA
The PCA is used for visualizing the data which originally has way too many dimensions (79) to properly plot.  
The first PCA plot uses 2 principal components and distinguishes houses that are priced over or under the mean house price.
Second PCA plot uses the same 2 principal components and then a range of price-intervals instead of just over/under mean.


### Models and their Accuracy
Below is a short description of each model used and which accuracy they yielded.  
Accuracy is measured in Root Mean Squared Logarithmic Error (lower is better).

- Baseline (mean house price): 0.42577

- Linear regression model: 






### TODO
I want to get into top 10%, as to why the next two steps will be:  

- Comprehensive Feature Engineering of the data.
- Ensemble models for potential big boost in performance.

#### Thank you for reading, I hope it was interesting
