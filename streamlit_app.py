import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols
import streamlit as st

st.set_page_config(page_title="Irasse Construction Company Data Analysis", layout="wide")

st.title("Construction Project Analysis Dashboard")

# ---- Upload files ----
st.sidebar.header("Upload Your CSV Files")
site_progress_file = st.sidebar.file_uploader("Site Progress CSV", type="csv")
site_log_file = st.sidebar.file_uploader("Site Log CSV", type="csv")
payroll_file = st.sidebar.file_uploader("Payroll CSV", type="csv")
spend_file = st.sidebar.file_uploader("Current Spend CSV", type="csv")
shrinkage_file = st.sidebar.file_uploader("Shrinkage CSV", type="csv")

if all([site_progress_file, site_log_file, payroll_file, spend_file, shrinkage_file]):

    # ---- Load data ----
    site_progress = pd.read_csv(site_progress_file)
    site_log = pd.read_csv(site_log_file)
    payroll = pd.read_csv(payroll_file)
    spend = pd.read_csv(spend_file)
    shrinkage = pd.read_csv(shrinkage_file)

    # ---- Feature Engineering ----
    site_progress["Schedule_Variance_Days"] = site_progress["Planned_Days"] - site_progress["Actual_Days"]

    site_log["Utilisation_%"] = (site_log["Used_Hours"] / site_log["Available_Hours"]) * 100
    util_avg = site_log.groupby("Project_ID")["Utilisation_%"].mean().reset_index()
    util_avg.rename(columns={"Utilisation_%": "Project_Utilisation_AvgPct"}, inplace=True)

    master = site_progress.merge(util_avg, on="Project_ID")
    master = master.merge(spend[["Project_ID", "Total_Spend_$"]], on="Project_ID")
    master = master.merge(shrinkage[["Project_ID", "Total_Shrinkage_$"]], on="Project_ID")

    st.subheader("📊 Master Dataset")
    st.dataframe(master)

    # ---- Correlation Heatmap ----
    st.subheader("Correlation Heatmap between attributes")
    corr_cols = ["Schedule_Variance_Days", "Project_Utilisation_AvgPct", "Total_Spend_$", "Total_Shrinkage_$"]
    corr = master[corr_cols].corr()

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # ---- Regression Model ----
    st.subheader(" Regression Model  : Predict Spend from Shrinkage")
    master = master.rename(columns={"Total_Spend_$": "Total_Spend", "Total_Shrinkage_$": "Total_Shrinkage"})
    reg_model = smf.ols("Total_Spend ~ Total_Shrinkage", data=master).fit()
    st.text(reg_model.summary())

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.regplot(x="Total_Shrinkage", y="Total_Spend", data=master, ax=ax2)
    st.pyplot(fig2)

    # ---- Controlled Regression ----
    st.subheader(" Controlled Regression Model (Shrinkage ~ SV + Spend)")
    causal_model = ols("Total_Shrinkage ~ Schedule_Variance_Days + Total_Spend", data=master).fit()
    st.text(causal_model.summary())

    # ---- Partial Correlation ----
    resid_y = sm.OLS(master["Total_Shrinkage"], sm.add_constant(master["Total_Spend"])).fit().resid
    resid_x = sm.OLS(master["Schedule_Variance_Days"], sm.add_constant(master["Total_Spend"])).fit().resid
    partial_corr_value = np.corrcoef(resid_x, resid_y)[0, 1]

    st.markdown(f"**Partial correlation between Schedule Variance and Shrinkage (controlling for Spend):** `{partial_corr_value:.4f}`")

else:
    st.info(" upload all 5 CSV files in the sidebar to begin.")
