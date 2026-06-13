import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score

import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ⚡ PASTE THIS INSTEAD (Optimized & Cached):

@st.cache_data
def load_and_clean_everything():
    file_name = 'vacc-kid.xlsx'
    years_we_want = ['iso3', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023']
    years = ['2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023']
    
    # Load basic sheets safely inside memory
    df_dtp1 = pd.read_excel(file_name, sheet_name='DTP1')[years_we_want]
    df_dtp3 = pd.read_excel(file_name, sheet_name='DTP3')[years_we_want]
    
    # Create the internal df_merged
    df_merged = pd.merge(df_dtp1, df_dtp3, on='iso3', suffixes=('_dtp1', '_dtp3'))
    
    # Calculate Gaps
    for year in years:
        df_merged[f"{year}_gap"] = df_merged[f"{year}_dtp1"] - df_merged[f"{year}_dtp3"]

    # 1. Melt DTP1 Columns into rows
    dtp1_columns = ['iso3'] + [f"{year}_dtp1" for year in years]
    df_dtp1_long = pd.melt(df_merged[dtp1_columns], id_vars=['iso3'], var_name='Year', value_name='DTP1_Coverage')
    df_dtp1_long['Year'] = df_dtp1_long['Year'].str.replace('_dtp1', '').astype(int)

    # 2. Melt DTP3 Columns into rows
    dtp3_columns = ['iso3'] + [f"{year}_dtp3" for year in years]
    df_dtp3_long = pd.melt(df_merged[dtp3_columns], id_vars=['iso3'], var_name='Year', value_name='DTP3_Coverage')
    df_dtp3_long['Year'] = df_dtp3_long['Year'].str.replace('_dtp3', '').astype(int)

    # 3. Melt Gap Columns into rows
    gap_columns = ['iso3'] + [f"{year}_gap" for year in years]
    df_gap_wide = df_merged[gap_columns]
    df_gap_long = pd.melt(df_gap_wide, id_vars=['iso3'], var_name='Year', value_name='Gap')
    df_gap_long['Year'] = df_gap_long['Year'].str.replace('_gap', '').astype(int)

    # 4. Now merge everything together safely inside the function
    df_final = pd.merge(df_dtp1_long, df_dtp3_long, on=['iso3', 'Year'])
    df_final = pd.merge(df_final, df_gap_long, on=['iso3', 'Year'])

    # 5. Load/Create Meta Information
    try:
        df_metadata = pd.read_csv('countries.csv')
    except FileNotFoundError:
        unique_countries = df_final['iso3'].unique()
        df_metadata = pd.DataFrame({
            'iso3': unique_countries,
            'Region': ['Eastern Mediterranean' if c == 'MAR' else 'Africa' for c in unique_countries],
            'Income_Group': ['EMRO' if c == 'MAR' else 'AFRO' for c in unique_countries]
        })

    # 6. Build the final Master Dataset 
    df_master = pd.merge(df_final, df_metadata, on='iso3', how='left')
    
    # 7. Standardize strings and convert to categorical numeric codes
    df_master['Region'] = df_master['Region'].astype(str).str.strip()
    df_master['Income_Group'] = df_master['Income_Group'].astype(str).str.strip()
    
    df_master['Region_Code'] = df_master['Region'].astype('category').cat.codes
    df_master['Subregion_code'] = df_master['Income_Group'].astype('category').cat.codes
    
    return df_master

# 🚀 Fire the completely consolidated data engine
df_master = load_and_clean_everything()

# Now set up your model inputs cleanly with zero warnings!
X = df_master[['Year', 'DTP1_Coverage', 'Subregion_code', 'Region_Code']]


y = df_master['DTP3_Coverage'] 

print("Master Dataset built perfectly! Ready for training pipeline.")
print(X.head())


# ==========================================
# 1. Cached Machine Learning Training Pipeline
# ==========================================

@st.cache_resource
def train_or_load_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
    )

    print("🧠 Training Neural Network Architecture... Please wait...")
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    joblib.dump(pipeline, 'vaccine_model_pipeline.pkl')
    return pipeline, mae, r2

model_pipeline, mae, r2 = train_or_load_model(X, y)

print("Model pipeline verified successfully! 🚀")
print(f"📉 Average error rate (MAE): {mae:.2f}%")
print(f"🎯 Prediction accuracy (R2 Score): {r2 * 100:.2f}%")


# ==========================================
# 2. Production-Grade Sidebar Interface 🔮
# ==========================================

st.sidebar.header("Prediction Settings 🔮")

selected_year = st.sidebar.slider("Select Target Year:", 2027, 2035, 2030)
dtp1_input = st.sidebar.number_input("Expected DTP1 Coverage (%):", min_value=0, max_value=100, value=99)

valid_regions = ['Africa', 'Americas', 'Eastern Mediterranean', 'Europe', 'South-East Asia', 'Western Pacific']
valid_incomes = ['Low income', 'Lower middle income', 'Upper middle income', 'High income']

df_master['Region'] = df_master['Region'].astype(str).str.strip()
df_master['Income_Group'] = df_master['Income_Group'].astype(str).str.strip()

# --- Region Dropdown Engine ---
region_df = df_master[df_master['Region'].isin(valid_regions)][['Region', 'Region_Code']].drop_duplicates().dropna()

region_options = {}
for _, row in region_df.iterrows():
    try:
        region_options[row['Region']] = int(float(row['Region_Code']))
    except:
        continue

if not region_options:
    region_options = {r: i for i, r in enumerate(valid_regions)}

selected_region = st.sidebar.selectbox("Select Region:", list(region_options.keys()))
region_code = region_options[selected_region]


# --- Income Group Dropdown Engine ---
subregion_df = df_master[df_master['Income_Group'].isin(valid_incomes)][['Subregion_code', 'Income_Group']].drop_duplicates().dropna()

subregion_options = {}
for _, row in subregion_df.iterrows():
    try:
        subregion_options[row['Income_Group']] = int(float(row['Subregion_code']))
    except:
        continue

if not subregion_options:
    subregion_options = {inc: i for i, inc in enumerate(valid_incomes)}

selected_subregion = st.sidebar.selectbox("Select Sub-Region / Income Group:", list(subregion_options.keys()))
sub_region_code = subregion_options[selected_subregion]
import pandas as pd


# ==========================================
# 1. STREAMLIT MANDATORY CONFIGURATION (Must be First!)
# ==========================================
st.set_page_config(page_title="Health AI - Vaccination Forecast", layout="wide")

st.title("🌍 Early Warning System: Vaccination Coverage Predictor")
st.subheader("Developed by: Health Data Scientist - Morocco 🇲🇦")
st.info("""
🔬 **What is this?** This platform uses **Advanced Machine Learning** to look at historic healthcare data and forecast DTP3 vaccine coverage for the year 2030.

💡 **Why it matters?** Instead of waiting for vaccination rates to drop, this AI-driven system acts as an **Early Warning Radar**—pointing global health organizations directly toward countries that will need supply-chain or financial support before the deadline.
""")

st.markdown("---")
# ==========================================
# 2. CACHED GLOBAL RISK ANALYTICS (Runs Only Once!)
# ==========================================
@st.cache_data
def run_global_2030_assessment(_model_pipeline, _df_master):
    """Loops through all countries once to build the 2030 Risk Watchlist without causing dashboard lag."""
    all_countries = _df_master['iso3'].unique()
    future_predictions = []

    for country in all_countries:
        country_data = _df_master[_df_master['iso3'] == country].iloc[-1]

        future_input = pd.DataFrame({
            'Year': [2030],
            'DTP1_Coverage': [country_data['DTP1_Coverage']], 
            'Subregion_code': [country_data['Subregion_code']],
            'Region_Code': [country_data['Region_Code']]
        })
        pred = _model_pipeline.predict(future_input)[0]
        future_predictions.append({'Country': country, 'Predicted_DTP3': pred})

    results = pd.DataFrame(future_predictions)
    
    def classify_risk(coverage):
        if coverage < 50: return 'Critical'
        elif 50 <= coverage < 65: return 'At Risk'
        elif 65 <= coverage < 80: return 'watchlist pre-warning'
        else: return 'safe'
        
    results['Category'] = results['Predicted_DTP3'].apply(classify_risk)
    return results.sort_values(by='Predicted_DTP3', ascending=True)

# Load your high-performance trained model pipeline cleanly
model = joblib.load('vaccine_model_pipeline.pkl')

# Generate your 2030 dataset immediately behind the scenes (cached)
results_2030 = run_global_2030_assessment(model, df_master)

# ==========================================
# 3. BACKGROUND TERMINAL LOGS (For Your Console Analytics)
# ==========================================
morocco_data = df_master[df_master['iso3'] == 'MAR'].iloc[0]
sandbox_input = pd.DataFrame({
    'Year': [2027],
    'DTP1_Coverage': [99],
    'Subregion_code': [morocco_data['Subregion_code']],
    'Region_Code': [morocco_data['Region_Code']]    
})
sandbox_pred = model.predict(sandbox_input)[0]

print("\n🇲🇦 --- AI expectations for Morocco in 2027 --- 🇲🇦")
print(f"Target year: 2027 | Baseline First dose coverage: 99%")
print(f"🎯 Expected third dose coverage (DTP3): {sandbox_pred:.2f}%")

print("\n📊 Health Risk Assessment Summary for 2030 (Console Verification):")
print(results_2030.head(15)) 

# ==========================================
# 4. INTERACTIVE USER PREDICTION LOGIC
# ==========================================
if st.sidebar.button("Run AI Prediction 🚀"):
    input_data = pd.DataFrame({
        'Year': [selected_year],
        'DTP1_Coverage': [dtp1_input],
        'Subregion_code': [sub_region_code],
        'Region_Code': [region_code]
    })

    # Generate prediction using your clean unified pipeline model
    prediction = model.predict(input_data)[0]
    
    # Calculate the dropout gap between DTP1 and DTP3
    gap = dtp1_input - prediction

    st.success("✅ AI Analysis Complete!")
    
   with col1:
        st.metric(label="Predicted DTP3 Coverage", value=f"{prediction:.2f}%")
   with col2:
        st.metric(label="Predicted Dropout Gap", value=f"{gap:.2f}%")
         
    
   if prediction >= 80:
        st.success("✅ The predicted DTP3 coverage is within acceptable ranges.")
   elif 50 <= prediction < 80:
        st.warning("⚠️ Warning: Moderate coverage. Risk of under-vaccination detected.")
   else:
        st.error("🚨 Critical: Very low DTP3 coverage predicted! Immediate intervention required.")

# ==========================================
# 5. GLOBAL RISK WATCHLIST VIEW FOR JUDGES
# ==========================================
st.write("### 🚨 Global Risk Watchlist Preview (2030 Horizon)")
st.dataframe(results_2030.head(15), use_container_width=True)

# Footer
st.markdown("---")



# ==========================================
# 1. CACHED AI EXPLAINER ENGINE (SHAP)
# ==========================================

@st.cache_data
def calculate_shap_assets(_model_pipeline, _X_train):
    """
    Pre-computes SHAP values once upon startup to prevent dashboard freeze.
    Extracts the model from the pipeline to compute weights safely.
    """
    # Extract the actual trained model from the pipeline steps
    trained_model = _model_pipeline.named_steps['mlpregressor']
    scaler = _model_pipeline.named_steps['standardscaler']
    
    # Scale X_train so it matches what the Neural Network expects internally
    X_train_scaled = pd.DataFrame(scaler.transform(_X_train), columns=_X_train.columns)
    
    # Summarize background data for performance speed
    background_data = shap.kmeans(X_train_scaled, 10)
    explainer = shap.KernelExplainer(trained_model.predict, background_data)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(X_train_scaled)
    return explainer, shap_values, X_train_scaled


# Isolate features and target splits for proper scaling tracking
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Generate assets instantly and safely behind the scenes
explainer, shap_values, X_train_scaled = calculate_shap_assets(model, X_train)


# ==========================================
# 2. INTERACTIVE VISUALIZATION TABS FOR THE UI
# ==========================================

st.write("## 📊 Advanced Risk Analytics & Explainable AI (SHAP)")

tab1, tab2, tab3 = st.tabs(["📉 Global Risk Distribution", "🧠 Global AI Drivers", "🇲🇦 Country Specific Logic"])

with tab1:
    st.write("### 2030 Country Dropout Risk Allocation")
    
    
    results_2030['Category'] = results_2030['Predicted_DTP3'].apply(
        lambda x: 'Critical' if x < 50 else ('At Risk' if x < 70 else ('Watchlist' if x < 80 else 'Safe'))
    )
    
    df_filtered = results_2030[results_2030['Category'] != 'Safe']
    counts = df_filtered['Category'].value_counts().reindex(['Critical', 'At Risk', 'Watchlist']).fillna(0)

     
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        x=counts.index, 
        y=counts.values, 
        hue=counts.index,
        palette={'Critical': '#d62728', 'At Risk': '#ff7f0e', 'Watchlist': '#66c2a5'},
        legend=False,
        ax=ax
    )
    
    ax.set_title('Global Health Strategy: 2030 DTP3 Dropout Risk Assessment', fontsize=16, pad=15)
    ax.set_ylabel('Number of Countries', fontsize=15)
    ax.set_xlabel('Risk Category', fontsize=15)

    for i, v in enumerate(counts.values):
        ax.text(i, v + 0.2, str(int(v)), ha='center', fontsize=15, fontweight='bold')
        
    plt.tight_layout()
    st.pyplot(fig) 
    plt.close(fig)
    
with tab2:
    st.write("### Global Feature Importance (What the AI Looks For)")
    st.info("This chart displays how much weight the Neural Network assigns to each variable globally.")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    shap.summary_plot(shap_values, X_train_scaled, plot_type="bar", show=False)
    plt.title('Neural Network Global Feature Importance (SHAP Overview)', fontsize=14, pad=15)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


with tab3:
    st.write("### Local Model Reasoning Explainer")
    
    # Dynamic Dropdown allowing judges to select a feature profile to analyze live
    available_indices = list(range(len(X_train)))
    selected_idx = st.selectbox(
        "Select Data Sample Row Index to Decipher:", 
        options=available_indices,
        format_func=lambda idx: f"Data Sample Record #{idx} (DTP1 Target Baseline: {X_train.iloc[idx]['DTP1_Coverage']:.1f}%)"
    )
    
    st.write(f"#### Deciphering AI Decisions for Record #{selected_idx}")
    
    # Rebuild the specific prediction contribution waterfall matrix
    country_explanation = shap.Explanation(
        values=shap_values[selected_idx],
        base_values=explainer.expected_value,
        data=X_train_scaled.iloc[selected_idx],
        feature_names=X_train.columns
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    shap.plots.waterfall(country_explanation, show=False)
    plt.title(f'AI Prediction Vector Analysis (Sample Baseline)', fontsize=14, pad=15)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)



