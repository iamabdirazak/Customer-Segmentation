# 🛍️ Customer Segmentation using Clustering (Mall Customers Dataset)

## 📌 Overview

This project applies **unsupervised machine learning** techniques to segment mall customers into distinct groups based on their demographic and spending patterns.  
Unlike classification or regression, clustering does not use labels — it groups similar customers together purely based on patterns in the data.

![image alt](https://github.com/iamabdirazak/Customer-Segmentation/blob/main/customer_clustering%20_chart.png?raw=true)

**Problem Type:** Unsupervised Machine Learning — Clustering  
**Goal:** Identify distinct customer segments for targeted marketing strategies.

---

## 🗂 Dataset

- **Source:** [Mall Customers Dataset on Kaggle](https://www.kaggle.com/datasets/singhmaninder/mall-customer-dataset)
- **Size:** `198` rows × `9` columns
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

4. **Data Transformation**

   - One-hot encoded categorical features.
   - Scaled numerical features using StandardScaler.
   - Ensured all features were numeric for clustering.

   Scaling is crucial for clustering algorithms because they are sensitive to the scale of the data. If features are on different scales, those with larger ranges can disproportionately influence the distance calculations, leading to biased clustering results. Standardizing features ensures that each feature contributes equally to the distance metrics used in clustering algorithms like K-Means.

5. **Feature Engineering**

   - Created new features (if applicable).
   - Justified each engineered feature based on dataset needs.
   - **New Feature(s) Added:**

     - Income_Spending_Ratio: Annual Income/Spending Score
     - Age Group: young, adult, midAge, senior

     The new feature 'Income_Spending_Ratio' provides a more nuanced view of customer behavior by combining income and spending score. It helps identify customers who may have high income but low spending, or vice versa, which can be crucial for targeted marketing strategies and segmentation.

6. **Modeling**

   - Used **KMeans clustering** from scikit-learn.
   - Determined optimal `k` using the Elbow Method.
   - Trained KMeans model on scaled features.

7. **Evaluation**

   - Computed **Silhouette Score**: _(Insert score here)_
   - Cluster profiles generated:
     - **Cluster 0:** Low Income, Low Spending
     - **Cluster 1:** High Income, High Spending
     - **Cluster 2:** Medium Income, Medium Spending
     - **Cluster 3:** Low Income, High Spending
     - **Cluster 4:** High Income, Low Spending

The Silhouette Score is used to evaluate the quality of clusters formed by a clustering algorithm. It measures how similar an object is to its own cluster compared to other clusters. The score ranges from -1 to 1, where a score close to 1 indicates that the object is well clustered, a score close to 0 indicates that the object is on or very close to the decision boundary between two neighboring clusters, and a score close to -1 indicates that the object may have been assigned to the wrong cluster. A higher Silhouette Score generally indicates better-defined clusters.

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

---

## 💡 Insights & Recommendations

![image alt](https://github.com/iamabdirazak/Customer-Segmentation/blob/main/customer_clustering%20_chart.png?raw=true)

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

## Author

Abdirazak Mubarak

## Date

Aug 11, 2025
