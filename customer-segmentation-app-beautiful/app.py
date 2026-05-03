import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="🛍️",
    layout="wide"
)


# -----------------------------
# Custom CSS
# -----------------------------
st.markdown(
    """
    <style>
    .main {
        background-color: #0E1117;
    }

    .hero-card {
        background: linear-gradient(135deg, #1F2937 0%, #111827 100%);
        padding: 2rem;
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 1.5rem;
    }

    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        line-height: 1.1;
        margin-bottom: 0.8rem;
        color: #FFFFFF;
    }

    .hero-subtitle {
        font-size: 1.1rem;
        color: #D1D5DB;
        max-width: 950px;
    }

    .metric-card {
        background-color: #111827;
        padding: 1.3rem;
        border-radius: 18px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
    }

    .metric-label {
        color: #9CA3AF;
        font-size: 0.9rem;
        margin-bottom: 0.3rem;
    }

    .metric-value {
        color: #FFFFFF;
        font-size: 1.8rem;
        font-weight: 700;
    }

    .section-card {
        background-color: #111827;
        padding: 1.5rem;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 1.2rem;
    }

    .cluster-card {
        background-color: #111827;
        padding: 1.2rem;
        border-radius: 18px;
        border-left: 5px solid #38BDF8;
        margin-bottom: 1rem;
    }

    .cluster-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 0.4rem;
    }

    .cluster-text {
        color: #D1D5DB;
        font-size: 0.98rem;
    }

    .success-card {
        background: linear-gradient(135deg, #064E3B 0%, #022C22 100%);
        padding: 1.4rem;
        border-radius: 18px;
        border: 1px solid rgba(16, 185, 129, 0.35);
        margin-top: 1rem;
    }

    .warning-note {
        background-color: #312E1D;
        color: #FDE68A;
        padding: 1rem;
        border-radius: 14px;
        border: 1px solid rgba(250, 204, 21, 0.25);
    }

    div[data-testid="stSidebar"] {
        background-color: #111827;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #111827;
        border-radius: 14px;
        padding: 10px 18px;
        color: #D1D5DB;
    }

    .stTabs [aria-selected="true"] {
        background-color: #2563EB;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------------
# Helper functions
# -----------------------------
@st.cache_data
def load_default_data():
    return pd.read_csv("data/Mall.csv")


def get_numeric_columns(dataframe):
    return dataframe.select_dtypes(include=np.number).columns.tolist()


def create_elbow_figure(X_scaled):
    wcss = []

    for k in range(1, 11):
        model = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
        model.fit(X_scaled)
        wcss.append(model.inertia_)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(1, 11)),
            y=wcss,
            mode="lines+markers",
            marker=dict(size=9),
            line=dict(width=3),
            name="WCSS"
        )
    )

    fig.update_layout(
        title="Elbow Method",
        xaxis_title="Number of Clusters (K)",
        yaxis_title="WCSS / Inertia",
        template="plotly_dark",
        height=460,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    return fig


def describe_level(value, low_cutoff, high_cutoff):
    if value < low_cutoff:
        return "Low"
    elif value > high_cutoff:
        return "High"
    else:
        return "Medium"


def create_cluster_profiles(df, features):
    profile = df.groupby("Cluster")[features].agg(["mean", "min", "max", "count"])

    # Flatten column names
    profile.columns = [
        f"{feature}_{stat}" for feature, stat in profile.columns
    ]
    profile = profile.reset_index()

    # Add simple mean columns for easier display
    for feature in features:
        mean_col = f"{feature}_mean"
        low_cutoff = df[feature].quantile(0.33)
        high_cutoff = df[feature].quantile(0.67)

        profile[f"{feature} Level"] = profile[mean_col].apply(
            lambda x: describe_level(x, low_cutoff, high_cutoff)
        )

    return profile


def get_marketing_recommendation(cluster_row, feature1, feature2):
    f1_level = cluster_row[f"{feature1} Level"]
    f2_level = cluster_row[f"{feature2} Level"]

    feature_names = {feature1.lower(), feature2.lower()}

    uses_income_spending = (
        any("income" in name for name in feature_names)
        and any("spending" in name for name in feature_names)
    )

    if uses_income_spending:
        income_feature = feature1 if "income" in feature1.lower() else feature2
        spending_feature = feature1 if "spending" in feature1.lower() else feature2

        income_level = cluster_row[f"{income_feature} Level"]
        spending_level = cluster_row[f"{spending_feature} Level"]

        if income_level == "High" and spending_level == "High":
            return "Premium segment: offer VIP deals, exclusive launches, premium loyalty tiers, and personalised shopping experiences."
        elif income_level == "High" and spending_level in ["Low", "Medium"]:
            return "High-potential segment: use premium product trials, personalised offers, and improved service touchpoints to increase spending."
        elif income_level in ["Low", "Medium"] and spending_level == "High":
            return "Loyal value-driven segment: offer loyalty points, bundles, cashback-style promotions, and repeat-purchase rewards."
        elif income_level == "Low" and spending_level == "Low":
            return "Budget-sensitive segment: promote discounts, affordable product bundles, and value-for-money campaigns."
        else:
            return "Mainstream segment: use seasonal promotions, limited-time offers, and category-based recommendations."

    return (
        f"This group has {f1_level.lower()} {feature1} and {f2_level.lower()} {feature2}. "
        "Use targeted campaigns based on the shared behaviour of this segment."
    )


def make_cluster_label(row, feature1, feature2):
    return f"{row[f'{feature1} Level']} {feature1}, {row[f'{feature2} Level']} {feature2}"


def build_new_customer_id(df):
    if "CustomerID" in df.columns:
        try:
            return int(df["CustomerID"].max()) + 1
        except Exception:
            return len(df) + 1
    return len(df) + 1


# -----------------------------
# Hero section
# -----------------------------
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">🛍️ Customer Segmentation Dashboard</div>
        <div class="hero-subtitle">
            Train a K-Means clustering model, understand customer groups, assign new customers to segments,
            and generate marketing actions from customer income and spending behaviour.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.title("⚙️ Controls")

data_source = st.sidebar.radio(
    "Data source",
    ["Use sample Mall dataset", "Upload my own CSV"]
)

uploaded_file = None
if data_source == "Upload my own CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if data_source == "Use sample Mall dataset":
    df = load_default_data()
else:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.info("Upload a CSV file from the sidebar to begin.")
        st.stop()


numeric_columns = get_numeric_columns(df)

if len(numeric_columns) < 2:
    st.error("The dataset must contain at least two numeric columns for clustering.")
    st.stop()

default_features = []
for col in ["Annual Income (k$)", "Spending Score (1-100)"]:
    if col in numeric_columns:
        default_features.append(col)

if len(default_features) < 2:
    default_features = numeric_columns[:2]

features = st.sidebar.multiselect(
    "Select exactly 2 numeric features",
    numeric_columns,
    default=default_features
)

if len(features) != 2:
    st.warning("Please select exactly 2 numeric features from the sidebar.")
    st.stop()

k = st.sidebar.slider("Number of clusters (K)", min_value=2, max_value=10, value=5)

st.sidebar.markdown("---")
st.sidebar.info(
    "Tip: For the sample dataset, Annual Income and Spending Score usually give the clearest customer segments."
)


# -----------------------------
# Model training
# -----------------------------
X = df[features].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)

df_clustered = df.copy()
df_clustered["Cluster"] = model.fit_predict(X_scaled)

centers_original = scaler.inverse_transform(model.cluster_centers_)

cluster_profile = create_cluster_profiles(df_clustered, features)
cluster_profile["Segment Description"] = cluster_profile.apply(
    lambda row: make_cluster_label(row, features[0], features[1]),
    axis=1
)
cluster_profile["Recommended Action"] = cluster_profile.apply(
    lambda row: get_marketing_recommendation(row, features[0], features[1]),
    axis=1
)


# -----------------------------
# KPI cards
# -----------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Total Customers</div>
            <div class="metric-value">{df_clustered.shape[0]}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Selected Clusters</div>
            <div class="metric-value">{k}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Feature 1</div>
            <div class="metric-value">{features[0]}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Feature 2</div>
            <div class="metric-value">{features[1]}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📄 Data", "📈 Model", "🎨 Clusters", "➕ New Customer", "⬇️ Download"]
)


# -----------------------------
# Tab 1: Data
# -----------------------------
with tab1:
    st.subheader("Dataset Preview")

    col_a, col_b = st.columns([2, 1])

    with col_a:
        st.dataframe(df.head(20), use_container_width=True)

    with col_b:
        st.markdown("### Dataset Summary")
        st.write(f"**Rows:** {df.shape[0]}")
        st.write(f"**Columns:** {df.shape[1]}")
        st.write(f"**Missing values:** {int(df.isna().sum().sum())}")
        st.write("**Numeric columns:**")
        st.write(", ".join(numeric_columns))

    with st.expander("Show full summary statistics"):
        st.dataframe(df.describe(), use_container_width=True)


# -----------------------------
# Tab 2: Model
# -----------------------------
with tab2:
    st.subheader("Elbow Method")

    st.markdown(
        """
        The elbow method helps decide a reasonable number of clusters. 
        Look for the point where the line starts to bend or flatten. 
        For the standard mall customer dataset, 5 clusters is commonly a good starting point.
        """
    )

    elbow_fig = create_elbow_figure(X_scaled)
    st.plotly_chart(elbow_fig, use_container_width=True)

    st.markdown(
        f"""
        <div class="warning-note">
        Current model is trained using <b>K = {k}</b> and the features 
        <b>{features[0]}</b> and <b>{features[1]}</b>.
        </div>
        """,
        unsafe_allow_html=True
    )


# -----------------------------
# Tab 3: Clusters
# -----------------------------
with tab3:
    st.subheader("Interactive Cluster Visualisation")

    fig = px.scatter(
        df_clustered,
        x=features[0],
        y=features[1],
        color=df_clustered["Cluster"].astype(str),
        hover_data=df_clustered.columns,
        title="Customer Segments",
        template="plotly_dark",
        labels={"color": "Cluster"}
    )

    fig.add_trace(
        go.Scatter(
            x=centers_original[:, 0],
            y=centers_original[:, 1],
            mode="markers",
            marker=dict(symbol="x", size=18, line=dict(width=3)),
            name="Cluster centres"
        )
    )

    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cluster Business Report")

    display_cols = [
        "Cluster",
        f"{features[0]}_mean",
        f"{features[1]}_mean",
        f"{features[0]} Level",
        f"{features[1]} Level",
        "Segment Description",
        "Recommended Action"
    ]

    st.dataframe(
        cluster_profile[display_cols],
        use_container_width=True
    )

    st.markdown("### Segment Cards")

    for _, row in cluster_profile.iterrows():
        cluster_id = int(row["Cluster"])
        count_col = f"{features[0]}_count"
        customer_count = int(row[count_col])

        st.markdown(
            f"""
            <div class="cluster-card">
                <div class="cluster-title">Cluster {cluster_id}: {row["Segment Description"]}</div>
                <div class="cluster-text">
                    Customers in this segment: <b>{customer_count}</b><br>
                    Average {features[0]}: <b>{row[f"{features[0]}_mean"]:.2f}</b><br>
                    Average {features[1]}: <b>{row[f"{features[1]}_mean"]:.2f}</b><br><br>
                    <b>Recommended action:</b> {row["Recommended Action"]}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


# -----------------------------
# Tab 4: New Customer
# -----------------------------
with tab4:
    st.subheader("Assign a New Customer to a Cluster")

    st.write(
        "Enter a new customer's details. The app will scale the values using the same scaler from training, "
        "then assign the customer to the nearest K-Means cluster."
    )

    with st.form("new_customer_form"):
        col_left, col_right = st.columns(2)

        with col_left:
            customer_name = st.text_input("Customer Name")
            customer_email = st.text_input("Customer Email")
            gender_value = None

            if "Gender" in df.columns:
                gender_options = sorted(df["Gender"].dropna().unique().tolist())
                gender_value = st.selectbox("Gender", gender_options)

        with col_right:
            value_1 = st.number_input(
                features[0],
                min_value=float(df[features[0]].min()),
                max_value=float(df[features[0]].max()),
                value=float(df[features[0]].mean())
            )

            value_2 = st.number_input(
                features[1],
                min_value=float(df[features[1]].min()),
                max_value=float(df[features[1]].max()),
                value=float(df[features[1]].mean())
            )

            age_value = None
            if "Age" in df.columns and "Age" not in features:
                age_value = st.number_input(
                    "Age",
                    min_value=int(df["Age"].min()),
                    max_value=int(df["Age"].max()),
                    value=int(df["Age"].mean())
                )

        submitted = st.form_submit_button("Assign Customer Cluster")

    if submitted:
        new_customer = pd.DataFrame([[value_1, value_2]], columns=features)
        new_customer_scaled = scaler.transform(new_customer)
        predicted_cluster = int(model.predict(new_customer_scaled)[0])

        selected_cluster_profile = cluster_profile[
            cluster_profile["Cluster"] == predicted_cluster
        ].iloc[0]

        recommendation = selected_cluster_profile["Recommended Action"]
        segment_description = selected_cluster_profile["Segment Description"]

        st.markdown(
            f"""
            <div class="success-card">
                <h3>Customer Assigned Successfully</h3>
                <p><b>{customer_name if customer_name else "New customer"}</b> has been assigned to 
                <b>Cluster {predicted_cluster}</b>.</p>
                <p><b>Segment:</b> {segment_description}</p>
                <p><b>Recommended Action:</b> {recommendation}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        new_record = {
            "CustomerID": build_new_customer_id(df_clustered),
            "Customer Name": customer_name,
            "Customer Email": customer_email,
            features[0]: value_1,
            features[1]: value_2,
            "Assigned Cluster": predicted_cluster,
            "Segment Description": segment_description,
            "Recommended Action": recommendation
        }

        if gender_value is not None:
            new_record["Gender"] = gender_value

        if age_value is not None:
            new_record["Age"] = age_value

        if "new_customers" not in st.session_state:
            st.session_state.new_customers = []

        st.session_state.new_customers.append(new_record)

    if "new_customers" in st.session_state and len(st.session_state.new_customers) > 0:
        st.markdown("### Newly Assigned Customers")
        new_customers_df = pd.DataFrame(st.session_state.new_customers)
        st.dataframe(new_customers_df, use_container_width=True)


# -----------------------------
# Tab 5: Download
# -----------------------------
with tab5:
    st.subheader("Download Results")

    st.write("Download the clustered dataset after the K-Means model has assigned clusters.")

    csv_data = df_clustered.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Clustered Dataset",
        data=csv_data,
        file_name="clustered_customers.csv",
        mime="text/csv"
    )

    if "new_customers" in st.session_state and len(st.session_state.new_customers) > 0:
        new_customers_df = pd.DataFrame(st.session_state.new_customers)
        new_customers_csv = new_customers_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Newly Assigned Customers",
            data=new_customers_csv,
            file_name="newly_assigned_customers.csv",
            mime="text/csv"
        )