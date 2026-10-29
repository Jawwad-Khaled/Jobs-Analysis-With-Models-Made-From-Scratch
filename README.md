# Jobs Analysis

## Overview

In this project, I have created models from scratch using only NumPy as I wanted to understand the math behind them and their architecture. I have created linear and logistic regression with ridge regularization, decision trees, random forests, gradient boosting, and neural networks. After creating these models, I applied them to a job postings dataset. Here, I want to talk about which features matter the most and also how the models compare to each other.

## Preprocessing

The job posting dataset comes from Luke Barousse's SQL tutorial for data analytics and Power BI tutorial. The 2023 postings came from the SQL tutorial while the 2024 postings came from the Power BI tutorial. The datasets also had multiple tables so I needed to combine them in a way that was useful for this project. Most of the features happened to be categorical so one-hot and multi-hot encodings were used extensively. I also removed every job posting where the yearly salary was not included which was almost 97% of them. The final data table ended up being 38360 rows and 74 columns.

## Hypothesis Testing

I have run 3 two sample Welch's  t-tests. All of them had p-values of 0 to 6 decimal places so the results are quite significant. The results of the t-tests show that remote jobs are associated with higher pay than non-remote jobs, jobs with health insurance are associated with higher pay than jobs without health insurance, and jobs that do not require a degree pay are associated with lower pay than  jobs that do require one.

## Regression And Classification

I have used both regression and classification here. For regression, the target variable was the yearly salary while for classification, it was whether the yearly salary was above the median. 

For regression, the models I used were linear regression with ridge regularization, regression trees, random forests with regression trees, gradient boosting with regression trees, and a neural network that uses mean squared error loss and the ReLU activation function. The performance metrics used were RMSE and R squared.

For classification, the models I used were logistic regression with ridge regularization, classification trees, random forests with classification trees, gradient boosting with regression trees, and a neural network that uses binary cross entropy loss and the sigmoid activation function. The performance metric used was accuracy.

5 fold cross validation was used to tune hyperparameters and a separate testing set was used to calculate performance metrics. All hyperparameters were tuned before the performance metrics were calculated. Model agnostic feature importances such as permutation and marginal importance are used for all models.

## Model Performances For Regression

The model performances for regression are shown in the table below.

|       Model       |    RMSE   | R Squared |
|:-----------------:|:---------:|:---------:|
| Linear Regression | 38415.173 |   0.307   |
|  Regression Tree  | 37570.210 |   0.338   |
|   Random Forest   | 34157.628 |   0.452   |
| Gradient Boosting | 36058.565 |   0.402   |
|   Neural Network  | 37639.071 |   0.348   |

The model with the best performance is the random forest and the model with the worst performance is linear regression with ridge regularization. The linear model performs the worst which suggests evidence of nonlinearity and interactions between the features. Random forests performs the best even beating out gradient boosting. This is probably due to gradient boosting chasing after the noise in the dataset. This is a job postings dataset so it is expected to be noisy and since random forests average out many predictions while decreasing variance, it does the best on this dataset.

## Model Performances For Classification

The model performances for classification are shown in the table below.

|        Model        | Accuracy |
|:-------------------:|:--------:|
| Logistic Regression |  73.450  |
| Classification Tree |  75.220  |
|    Random Forest    |  79.220  |
|  Gradient Boosting  |  76.740  |
|    Neural Network   |  75.880  |

The model with the best performance is the random forest and the model with the worst performance is logistic regression with ridge regularization. Again, logistic regression performing the worst suggests nonlinearity and interactions between the features. My explanation for random forests winning is the same as above.

## Feature Importances

The feature that almost consistently gets the highest permuatation importance is job title. It only lost to the job website in the classification gradient boosting model. Job website was the next most important feature. After that, we have a couple of skills that have the highest permutation imporatances such as Excel and Python, whether or not the job has health insurance, and the job country. The features with the least permutation importance include when the job was posted, whether the job requires a degree, whether the job is remote, the job schedule type, and the rest of the skills. It makes sense for job schedule type to not have much importance as over 97% of the postings are full-time.

Out of job titles, more technical job titles such as senior data scientist and senior data engineer have large positive marginal importances while more entry level ones like data analyst and business analyst have large negative marginal importances. Having health insurance increases marginal importance by quite a bit as well. Skills found more commonly in entry level jobs such as Excel and Power BI have quite negative marginal importances while skills found in more technical jobs such as Python or PyTorch have quite positive marginal importances. A job schedule type of intern has a massive negative marginal importance which is expected.