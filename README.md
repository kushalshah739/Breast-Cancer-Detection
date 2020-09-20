# Breast-Cancer-Detection

Introduction: 

Breast cancer is cancer that forms in the cells of the breasts. Breast cancer can occur in both men and women, but it's far more common in women. Breast cancer starts in the cells of the mammary gland. Breast cancer survival rates have increased, and the number of deaths associated with this disease is steadily declining, largely due to factors such as earlier detection, a new personalized approach to treatment and a better understanding of the disease.

Overview:

- In 2019 an estimated 26,900 Canadian women were diagnosed with breast cancer and 5,000 died of it.
- Breast cancer accounts for approximately 25% of new cases of cancer and 13% of all cancer deaths in Canadian women.
- 1 in 8 women are expected to develop breast cancer during her lifetime and 1 in 33 will die of it.

Architecture and Training:

- Extracted the raw dataset from the internet that has different biological attributes such as radius mean, concavity, perimeter and many more.
- Performed data analysis looking for null, invalid and outlier data points that would affect the accuracy of the model.
- Derived insights by executing exploratory data analysis on the dataset by plotting various attributes and correlating Diagnosis attribute with others.
- Trained and deployed an Extreme Gradient Boost (XGBoost) model to classify unseen data as Malignant (M) or Benign (B).  
- Using Python as the primary languages, I compared all machine learning models including Logic Regression, Naive Bayes, K-Neighbour Classifier, and deployed the XGBoost algorithm   out of all.
- Tuned the hyperparameters to increase the accuracy of the XGBoost model from 97% to 99%.
