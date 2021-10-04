# Ames Housing Regression Challenge

The "House Price - Advanced Regression Techniques" is a Kaggle competition where users developed data processing, feature engineering, and advanced regression in order to predict price of a house in Ames, IA, from a multivariate data set. 

From Kaggle: "Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. But this playground competition's dataset proves that much more influences price negotiations than the number of bedrooms or a white-picket fence. With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home."
***

**Feature Engineering**
Clearly define the training ("train") and testing ("test") data set. Perform data preprocessing, where we select features with the highest correlation and therefore the most relevant to the regression modeling. Encode categorical features, remove colinearity, correct for skew data and outliers, and prepare for our regression models. The preprocessing choice is not as important to get higher accuracy, but it is an essential step to understand our models through Exploratory Data Analysis, to run our models, and to get acceptable predictions.
Inferential machine learning included PCA, tSNE, K-Means clustering, and Local Outlier Factors to further analyze the dataset.
<br/><img src="https://github.com/misaelmmorales/Ames-Housing-Regression-challenge/blob/main/images/corr_kmeans.png">

**Regression Modeling Experiments**:
Use these preprocessed data and key features to predict the (log10) SalePrice ("target") of the Housing dataset from the test set. While we used several metrics to understand the performance of our models, for the purpose of presenting a consistent metric, we will compute the cv_rmse on every model. The models used include: Linear Regression, Lasso/Ridge/ElasticNet Regression, Gradient Boosting, Random Forest, Support Vector Machines, Artificial Neural Network; and Stacked Regression.

Comparing all models: 
<br/><img src="https://github.com/misaelmmorales/Ames-Housing-Regression-challenge/blob/main/images/mse_regression_comparison.png" width="475"> <img src="https://github.com/misaelmmorales/Ames-Housing-Regression-challenge/blob/main/images/reg_preds_histogram.png" width="475">

**Stacked Regression**
Stacking (short for stacked generalization) is an ensemble method which uses trivial functions (such as hard voting) to aggregate the predictions of all predictors in an ensemble. Each of the primary predictors predicts a different value and then the final predictor (called a blender or a meta learner) takes these predictions as inputs and makes the final prediction. To avoid the tedious work to find the best hyperparameters for every base model and for the meta-model manually we used Grid Search and Random Search to automatize fine tuning. For the project purposes splitting our training set into 10 folds gave us the best result.
![alt text](https://github.com/misaelmmorales/Ames-Housing-Regression-challenge/blob/main/images/StackedReg.png)

***
Remarks:
- After our data preprocessing techniques, most regression algorithms will provide good predictions for the log10(SalePrice) target variable using the most high-correlated feature variables.
- Fully-Connected Neural Network with nonlinear activations provides really good MSE (since we are optimizing with this loss function - but simpler regression methods like regularized regression provide a better distribution of the test predictions.
- Further sensitivity analysis and hyperparameter tuning can provide improved testing accuracy compared to the standard techniques deployed.

Conclusion:
- The SalePrice target variable from the Ames Housing data set can be easily predicted after detailed data preprocessing and advanced regression techniques.
- Preprocessing and data wrangling is crucial for exploiting the full potential of all features.
- Most regression algorithms will work to provide acceptable predictions.
- Regularization helps in further discriminating the most important predictor features.
