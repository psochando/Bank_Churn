# Bank Customer Churn (incomplete)
Prediction on whether bank customers will cancel their accounts or not.

## Context and goal
It is typically more costly for a bank (at least in economic terms) to acquire new customers compared to retaining the ones they already have. Therefore, it's not surprising that banks are interested in identifying which of their customers have a high probability of canceling their accounts, so they can direct their efforts towards preventing this from happening.

The goal, therefore, is to develop a model capable of classifying the bank's customers (based on the characteristics collected in the provided dataset) into "cancels their account" (1) and "does not cancel their account" (0).

**Note**: We have chosen the "recall" metric as the primary metric for evaluating model performance, as we are particularly interested in maximizing the rate of predicted "true positives."

## Data
The following link contains all the information about the dataset used: <br>
https://www.kaggle.com/datasets/mathchi/churn-for-bank-customers