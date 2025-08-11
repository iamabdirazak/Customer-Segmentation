# 🛍️ Customer Segmentation using Clustering (Mall Customers Dataset)

## 📌 Overview

This project applies **unsupervised machine learning** techniques to segment mall customers into distinct groups based on their demographic and spending patterns.  
Unlike classification or regression, clustering does not use labels — it groups similar customers together purely based on patterns in the data.

**Problem Type:** Unsupervised Machine Learning — Clustering  
**Goal:** Identify distinct customer segments for targeted marketing strategies.

---

## 🗂 Dataset

- **Source:** [Mall Customers Dataset on Kaggle](https://www.kaggle.com/datasets/singhmaninder/mall-customer-dataset)
- **Size:** `XYZ` rows × `ABC` columns _(replace XYZ/ABC after inspection)_
- **Features:**
  - CustomerID _(removed for modeling)_
  - Gender _(categorical)_
  - Age _(numeric)_
  - Annual Income (k$) _(numeric)_
  - Spending Score (1-100) _(numeric)_

---

## 🔍 Project Pipeline

1. **Data Mining**

   - Loaded dataset from Kaggle.
   - Simulated real-world messiness with missing values & duplicates.
   - Verified dataset shape and column types.

2. **Data Cleaning**

   - Missing values handled via mean/mode imputation.
   - Removed duplicates.
   - Identified and treated outliers.

3. **Exploratory Data Analysis (EDA)**

   - Visualized numerical distributions.
   - Analyzed categorical distributions.
   - Explored relationships between features.
   - **Key EDA Observations:**
     - _(Add insights from your notebook output here)_

4. **Data Transformation**

   - One-hot encoded categorical features.
   - Scaled numerical features using StandardScaler.
   - Ensured all features were numeric for clustering.

5. **Feature Engineering**

   - Created new features (if applicable).
   - Justified each engineered feature based on dataset needs.
   - **New Feature(s) Added:**
     - _(List and explain feature names here)_

6. **Modeling**

   - Used **KMeans clustering** from scikit-learn.
   - Determined optimal `k` using the Elbow Method.
   - Trained KMeans model on scaled features.

7. **Evaluation**

   - Computed **Silhouette Score**: _(Insert score here)_
   - Cluster profiles generated:
     - **Cluster 0:** _(Insert characteristics)_
     - **Cluster 1:** _(Insert characteristics)_
     - **Cluster 2:** _(Insert characteristics)_

8. **Model Saving**
   - Saved processed dataset as `mall_customers_clustered.csv`.
   - Saved KMeans model as `kmeans_mall_customers.pkl`.
   - Saved scaler as `scaler.pkl`.

---

## 📊 Visualizations

- Age distribution
- Income distribution
- Spending Score distribution
- Cluster visualizations in 2D space

_(Insert charts or reference images here)_

---

## 💡 Insights & Recommendations

- _(Insert your business/marketing recommendations based on clusters here)_
  ![Alt text](customer_clustering_chart.png)

---

## 🛠 Technologies Used

- Python 3.x
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- joblib

---

## 🚀 How to Run

1. Install dependencies:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn joblib
   ```

2. Download dataset
   kaggle datasets download -d singhmaninder/mall-customer-dataset
   unzip mall-customer-dataset.zip

3. Run the Jupyter Notebook step-by-step.

4. Use the saved model (kmeans_mall_customers.pkl) for predictions.

## 📅 Author & Date

Author: Abdirazak Mubarak
Date: Aug 11, 2025
