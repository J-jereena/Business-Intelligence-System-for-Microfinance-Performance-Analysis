import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# ===============================
# PAGE SETUP
# ===============================
st.set_page_config(page_title="AI Microfinance Dashboard", layout="wide")
st.markdown("<h1 style='text-align:center;'>🏦 AI Powered Microfinance Loan Dashboard</h1>", unsafe_allow_html=True)

# ===============================
# FILE UPLOAD
# ===============================
uploaded_file = st.file_uploader("📂 Upload Microfinance_Final_Loan_Master.csv", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    st.success("✅ File Uploaded Successfully")

    # ===============================
    # SHOW DATASET
    # ===============================
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    # ===============================
    # CONVERT DATE IF EXISTS
    # ===============================
    date_col = None
    for col in df.columns:
        if "Date" in col:
            date_col = col
            break

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df["Year"] = df[date_col].dt.year
        df["Date_Ordinal"] = df[date_col].map(pd.Timestamp.toordinal)
    else:
        df["Year"] = range(1, len(df)+1)
        df["Date_Ordinal"] = range(1, len(df)+1)

    # ===============================
    # NUMERIC CONVERSION
    # ===============================
    numeric_cols = ["LoanAmount", "AmountPaid", "DueAmount", "DelayDays", "Age"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # ===============================
    # SIDEBAR FILTERS
    # ===============================
    st.sidebar.header("🔍 Filter Data")

    if "LoanType" in df.columns:
        loan_filter = st.sidebar.multiselect("Loan Type", df["LoanType"].dropna().unique(), default=df["LoanType"].dropna().unique())
        df = df[df["LoanType"].isin(loan_filter)]

    if "Status" in df.columns:
        status_filter = st.sidebar.multiselect("Loan Status", df["Status"].dropna().unique(), default=df["Status"].dropna().unique())
        df = df[df["Status"].isin(status_filter)]

    if "Gender" in df.columns:
        gender_filter = st.sidebar.multiselect("Gender", df["Gender"].dropna().unique(), default=df["Gender"].dropna().unique())
        df = df[df["Gender"].isin(gender_filter)]

    # ===============================
    # KPI CARDS
    # ===============================
    total_loans = len(df)
    total_loan_amount = df["LoanAmount"].sum() if "LoanAmount" in df.columns else 0
    total_paid = df["AmountPaid"].sum() if "AmountPaid" in df.columns else 0

    if "Status" in df.columns:
        default_rate = (len(df[df["Status"]=="Default"]) / len(df)) * 100
    else:
        default_rate = 0

    st.markdown("## 📌 Dashboard Summary")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Loans", total_loans)
    c2.metric("Total Loan Amount", f"{total_loan_amount:,.0f}")
    c3.metric("Total Paid", f"{total_paid:,.0f}")
    c4.metric("Default Rate", f"{default_rate:.2f}%")

    st.markdown("---")

    # ===============================
    # GRAPH 1 LOAN TREND
    # ===============================
    g1, g2 = st.columns(2)

    with g1:
        st.subheader("📈 Loan Amount Trend")
        if "LoanAmount" in df.columns:
            yearly = df.groupby("Year")["LoanAmount"].sum()
            fig1, ax1 = plt.subplots(figsize=(7,4))
            ax1.plot(yearly.index, yearly.values, marker='o')
            ax1.grid(True)
            st.pyplot(fig1)

    with g2:
        st.subheader("📊 Loan Status Distribution")
        if "Status" in df.columns:
            status_data = df["Status"].value_counts()
            fig2, ax2 = plt.subplots(figsize=(7,4))
            ax2.bar(status_data.index, status_data.values)
            st.pyplot(fig2)

    # ===============================
    # GRAPH 2
    # ===============================
    g3, g4 = st.columns(2)

    with g3:
        st.subheader("👨‍👩‍👧 Gender Analysis")
        if "Gender" in df.columns:
            gender_data = df["Gender"].value_counts()
            fig3, ax3 = plt.subplots(figsize=(7,4))
            ax3.pie(gender_data.values, labels=gender_data.index, autopct='%1.1f%%')
            st.pyplot(fig3)

    with g4:
        st.subheader("💵 Repayment Analysis")
        if "AmountPaid" in df.columns:
            repay = df.groupby("Year")["AmountPaid"].sum()
            fig4, ax4 = plt.subplots(figsize=(7,4))
            ax4.plot(repay.index, repay.values, marker='o')
            ax4.grid(True)
            st.pyplot(fig4)

    st.markdown("---")

    # ===============================
    # AI PREDICTION
    # ===============================
    st.subheader("🤖 AI Loan Prediction")

    if "LoanAmount" in df.columns:

        X = df[["Date_Ordinal"]]
        y = df["LoanAmount"]

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        y_pred = model.predict(X)
        accuracy = r2_score(y, y_pred)

        future_x = pd.DataFrame({"Date_Ordinal": range(df["Date_Ordinal"].max()+1, df["Date_Ordinal"].max()+13)})
        future_pred = model.predict(future_x)

        fig5, ax5 = plt.subplots(figsize=(10,4))
        ax5.plot(df["Date_Ordinal"], y, label="Actual Loan Amount")
        ax5.plot(future_x["Date_Ordinal"], future_pred, linestyle='dashed', label="Predicted Loan Amount")
        ax5.legend()
        ax5.grid(True)
        st.pyplot(fig5)

        st.success(f"✅ AI Model Accuracy: {accuracy:.2f}")

    # ===============================
    # FINAL TABLE
    # ===============================
    st.subheader("📋 Full Filtered Dataset")
    st.dataframe(df)

else:
    st.warning("👆 Please upload Microfinance_Final_Loan_Master.csv")