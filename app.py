import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    classification_report, accuracy_score
)
from sklearn.feature_extraction.text import TfidfVectorizer

@st.cache_data
def load_data():
    df = pd.read_csv('synthetic_defect_data_mindgate_500_updated.csv')
    df['Reported Date'] = pd.to_datetime(df['Reported Date'], errors='coerce')
    df['Closed Date'] = pd.to_datetime(df['Closed Date'], errors='coerce')
    df['Estimated Fix Days'] = (df['Closed Date'] - df['Reported Date']).dt.days
    df.dropna(subset=['Description', 'Module Name', 'Priority', 'Estimated Fix Days',
                      'Assigned To', 'Severity', 'Root Cause', 'Reopen Count'], inplace=True)
    df = df[df['Estimated Fix Days'] >= 0]
    cap = df['Estimated Fix Days'].quantile(0.95)
    df = df[df['Estimated Fix Days'] <= cap]
    df['Reported Month'] = df['Reported Date'].dt.month
    df['Reported Weekday'] = df['Reported Date'].dt.weekday
    df['Description'] = df['Description'].apply(lambda x: re.sub(r'[^a-zA-Z0-9 ]', ' ', str(x).lower()))
    df['Description'] = df['Description'].str.replace(r'\s+', ' ', regex=True).str.strip()
    return df

df = load_data()
shared_metrics = {}

tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
desc_vecs_full = tfidf.fit_transform(df['Description']).toarray()

le_pri_full = LabelEncoder()
le_mod_full = LabelEncoder()
le_sev = LabelEncoder()
le_root = LabelEncoder()

X_struct_full = pd.DataFrame({
    'Priority': le_pri_full.fit_transform(df['Priority']),
    'Module': le_mod_full.fit_transform(df['Module Name']),
    'Severity': le_sev.fit_transform(df['Severity']),
    'Root Cause': le_root.fit_transform(df['Root Cause']),
    'Reopen Count': df['Reopen Count'].astype(int),
    'Month': df['Reported Month'],
    'Weekday': df['Reported Weekday']
})

desc_df = pd.DataFrame(desc_vecs_full, columns=[f"tfidf_{i}" for i in range(desc_vecs_full.shape[1])])
X_full = pd.concat([desc_df, X_struct_full.reset_index(drop=True)], axis=1)
y_full = df['Estimated Fix Days'].apply(np.log1p)

X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42
)

rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
rf_model.fit(X_train_rf, y_train_rf)

y_pred_rf = np.expm1(rf_model.predict(X_test_rf))
y_test_actual_rf = np.expm1(y_test_rf)

shared_metrics['mae'] = mean_absolute_error(y_test_actual_rf, y_pred_rf)
shared_metrics['mse'] = mean_squared_error(y_test_actual_rf, y_pred_rf)
shared_metrics['rmse'] = np.sqrt(shared_metrics['mse'])
shared_metrics['r2'] = r2_score(y_test_actual_rf, y_pred_rf)

model_full = rf_model

user_tab, admin_tab = st.tabs(["User Panel", "Admin Panel"])

with user_tab:
    st.title("🛠️ Defect Fix Time Predictor")
    module_filter = st.sidebar.selectbox("📁 Filter by Module", ["All"] + sorted(df['Module Name'].unique().tolist()))
    df_user = df if module_filter == "All" else df[df['Module Name'] == module_filter]

    le_priority = le_pri_full
    le_module = le_mod_full
    le_sev_user = le_sev
    le_root_user = le_root

    st.subheader("📋 Enter New Defect Details")
    desc = st.text_area("📝 Description of the defect")
    priority = st.selectbox("⚠️ Priority", df_user['Priority'].unique())
    module = st.selectbox("📦 Module", df_user['Module Name'].unique())
    severity = st.selectbox("🚨 Severity", df_user['Severity'].unique())
    root = st.selectbox("🧬 Root Cause", df_user['Root Cause'].unique())
    reopen = st.number_input("🔁 Reopen Count", min_value=0, step=1)

    if st.button("🔍 Predict Fix Time"):
        if not desc.strip():
            st.error("Please enter a valid description.")
        else:
            desc_clean = re.sub(r'[^a-zA-Z0-9 ]', ' ', str(desc).lower())
            desc_clean = re.sub(r'\s+', ' ', desc_clean).strip()
            desc_input = tfidf.transform([desc_clean]).toarray()
            struct_input = pd.DataFrame([{
                'Priority': le_priority.transform([priority])[0],
                'Module': le_module.transform([module])[0],
                'Severity': le_sev_user.transform([severity])[0],
                'Root Cause': le_root_user.transform([root])[0],
                'Reopen Count': reopen,
                'Month': pd.Timestamp.now().month,
                'Weekday': pd.Timestamp.now().weekday()
            }])
            final_input = pd.concat([pd.DataFrame(desc_input, columns=desc_df.columns), struct_input], axis=1)
            pred_days = np.expm1(model_full.predict(final_input)[0])
            st.success(f"🕒 Estimated fix time: **{round(pred_days, 2)} days**")
            st.metric("⏱️ Hours to Deployment", f"{round(pred_days * 8, 1)} hrs")

with admin_tab:
    st.title("🔒 Admin Panel")
    username = st.text_input("👤 Username")
    password = st.text_input("🔑 Password", type="password")

    if username == "admin" and password == "1234":
        st.success("Welcome, admin!")

        desc_vecs_admin = tfidf.transform(df['Description']).toarray()
        le_assignee = LabelEncoder()
        X_struct_admin = pd.DataFrame({
            'Priority': le_pri_full.transform(df['Priority']),
            'Module': le_mod_full.transform(df['Module Name'])
        })
        X_admin = np.concatenate([desc_vecs_admin, X_struct_admin.values], axis=1)
        y_admin = le_assignee.fit_transform(df['Assigned To'])

        X_train_admin, X_test_admin, y_train_admin, y_test_admin = train_test_split(
            X_admin, y_admin, test_size=0.2, random_state=42
        )
        clf = RandomForestClassifier()
        clf.fit(X_train_admin, y_train_admin)
        y_pred_admin = clf.predict(X_test_admin)

        st.subheader("📈 Assignee Classification Report")
        st.text(classification_report(y_test_admin, y_pred_admin, target_names=le_assignee.classes_))
        st.write(f"**Accuracy:** {accuracy_score(y_test_admin, y_pred_admin):.2f}")

        st.subheader("🕒 Fix Time Predictor Metrics")
        st.write(f"**MAE:** {shared_metrics['mae']:.2f} days")
        st.write(f"**MSE:** {shared_metrics['mse']:.2f} days²")
        st.write(f"**RMSE:** {shared_metrics['rmse']:.2f} days")
        st.write(f"**R² Score:** {shared_metrics['r2']:.2f}")
    elif username or password:
        st.error("Invalid credentials. Access denied.")
