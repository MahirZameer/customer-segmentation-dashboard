# Customer Segmentation Dashboard using K-Means

This project is an interactive customer segmentation dashboard built with Python, Streamlit, Pandas, Plotly, and Scikit-learn.

The app allows a mall owner or business user to upload customer data, train a K-Means clustering model, visualise customer segments, assign new customers to existing clusters, and generate targeted marketing recommendations.

## Features

- Upload a customer CSV dataset
- Use the included sample Mall Customers dataset
- Preview and summarise the dataset
- Select two numeric features for clustering
- Use the Elbow Method to support the choice of K
- Train a K-Means clustering model
- View an interactive Plotly cluster visualisation
- View cluster-level business summaries
- Add a new customer manually
- Assign the new customer to the nearest trained cluster
- Generate a recommended marketing action
- Download clustered results and newly assigned customers

## Dataset

The included sample dataset contains:

- CustomerID
- Gender
- Age
- Annual Income (k$)
- Spending Score (1-100)

The default clustering features are:

- Annual Income (k$)
- Spending Score (1-100)

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- Scikit-learn

## How K-Means is Used

K-Means groups customers based on similarity in selected numeric features. The selected features are scaled using `StandardScaler` before training because K-Means is distance-based.

When a new customer is added, their values are scaled using the same scaler and assigned to the nearest trained cluster using `model.predict()`.

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## How to Deploy on Streamlit Cloud

1. Push this project to GitHub
2. Go to Streamlit Community Cloud
3. Click **New app**
4. Choose your GitHub repository
5. Set the main file path as `app.py`
6. Click **Deploy**

## Business Value

The app helps business owners identify different customer groups and make better marketing decisions. For example, premium customers can receive VIP offers, high-potential customers can receive personalised campaigns, and budget-sensitive customers can receive discount-focused promotions.