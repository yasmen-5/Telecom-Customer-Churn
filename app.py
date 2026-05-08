import streamlit as st
import pandas as pd
import joblib
import time

# 1. Page Configuration 
st.set_page_config(
    page_title="Aura Core: Telecom Churn Intelligence", 
    page_icon="💎", 
    layout="wide"
)

# 2.CSS 
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap');
    
    /* Global Overrides */
    * { font-family: 'Plus Jakarta Sans', sans-serif; }
    .stApp { background-color: #FDFDFD; }
    
    /* Hero Banner (Burgundy-Navy Gradient) */
    .hero-container {
        background: linear-gradient(135deg, #1E293B 0%, #800020 100%);
        padding: 3rem 2rem;
        border-radius: 24px;
        color: white;
        margin-bottom: 2.5rem;
        box-shadow: 0 20px 40px rgba(128, 0, 32, 0.15);
    }

    /* Modernizing Number Inputs (Steppers) */
    div[data-testid="stNumberInput"] input {
        text-align: center !important;
        font-weight: 700 !important;
        font-size: 1.2rem !important;
        color: #1E293B !important;
        background-color: #F8FAFC !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 12px !important;
        height: 48px !important;
    }

    div[data-testid="stNumberInput"] button {
        background-color: #F1F5F9 !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 10px !important;
        width: 45px !important;
        height: 45px !important;
        transition: 0.3s;
    }
    
    div[data-testid="stNumberInput"] button:hover {
        background-color: #800020 !important;
        color: white !important;
        border-color: #800020 !important;
    }

    /* Professional Input Cards */
    .input-card {
        background: white;
        padding: 28px;
        border-radius: 22px;
        border: 1px solid #F1F5F9;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05);
        margin-bottom: 15px;
        height: 100%;
    }

    /* Premium Action Button */
    div.stButton > button {
        background: #1E293B !important;
        color: white !important;
        border-radius: 18px !important;
        padding: 25px !important;
        width: 100% !important;
        border: none !important;
        font-weight: 700 !important;
        font-size: 1.2rem !important;
        letter-spacing: 0.8px !important;
        transition: 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
        text-transform: uppercase;
    }
    div.stButton > button:hover {
        background: #800020 !important;
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(128, 0, 32, 0.3) !important;
    }
</style>
""", unsafe_allow_html=True)

# 3.Loader
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('xgb_churn_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception:
        return None, None

model, scaler = load_assets()

# 4. Hero 
st.markdown(f"""
<div class="hero-container">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h1 style="margin: 0; font-size: 3rem; font-weight: 700;">Aura Core Intelligence</h1>
            <p style="margin: 8px 0 0 0; opacity: 0.85; font-size: 1.3rem; font-weight: 300;">Advanced Telecom Churn Diagnostic Dashboard</p>
        </div>
        <div style="background: rgba(255,255,255,0.15); padding: 18px 30px; border-radius: 18px; backdrop-filter: blur(12px); border: 1px solid rgba(255,255,255,0.2);">
            <small style="display: block; opacity: 0.8; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 1px;">Engine Protocol</small>
            <span style="font-weight: 600; font-size: 1.1rem;">● Neural Insights Active</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# 5. Diagnostic Workspace Layout
st.markdown("<h3 style='margin-bottom: 25px; color: #1E293B; font-weight: 700;'>⚙️ Predictive Diagnostic Workspace</h3>", unsafe_allow_html=True)
col_a, col_b, col_c = st.columns(3, gap="large")

with col_a:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.write("#### 👤 Customer Demographics")
    age = st.number_input("Age", 18, 100, 35)
    gender = st.selectbox("Gender Identity", ["Female", "Male"])
    tenure = st.number_input("Tenure (Months)", 0, 120, 24)
    st.markdown('</div>', unsafe_allow_html=True)

with col_b:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.write("#### 📊 Behavioral Metrics")
    usage_freq = st.number_input("Monthly Usage Sessions", 0, 500, 50)
    support_calls = st.number_input("Tech Support Incidents", 0, 50, 2)
    last_int = st.number_input("Last Interaction (Days)", 0, 365, 10)
    st.markdown('</div>', unsafe_allow_html=True)

with col_c:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.write("#### 💳 Financial Health")
    sub_type = st.selectbox("Plan Tier", ["Basic", "Standard", "Premium"])
    contract = st.selectbox("Billing Cycle", ["Annual", "Monthly", "Quarterly"])
    spend = st.number_input("Monthly Spend ($)", 0.0, 50000.0, 1200.0)
    delay = st.number_input("Payment Delay (Days)", 0, 100, 0)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# 6. Prediction
if st.button("Generate Intelligence Report"):
    if model and scaler:
        with st.status("Analyzing Behavioral Patterns...", expanded=False) as status:
            time.sleep(1.5) # UX: Simulates "thinking" time for credibility
            
            # Mapping Categorical Data
            g_map = {"Female": 0, "Male": 1}
            s_map = {"Basic": 0, "Premium": 1, "Standard": 2}
            c_map = {"Annual": 0, "Monthly": 1, "Quarterly": 2}
            
            # Formatting Input Data
            features = pd.DataFrame([[
                age, g_map[gender], tenure, usage_freq, support_calls, 
                delay, s_map[sub_type], c_map[contract], spend, last_int
            ]], columns=[
                'Age', 'Gender', 'Tenure', 'Usage Frequency', 'Support Calls', 
                'Payment Delay', 'Subscription Type', 'Contract Length', 'Total Spend', 'Last Interaction'
            ])
            
            # Prediction
            input_scaled = scaler.transform(features)
            prob = model.predict_proba(input_scaled)[0][1]
            status.update(label="Analysis Successful", state="complete")

        st.markdown("---")
        
        # 7. Strategic Output Layer
        res_col1, res_col2 = st.columns([1, 1.5])
        
        # Intelligence Logic (Telecom Specific Insights)
        advice_points = []
        if contract == "Monthly":
            advice_points.append("• **Critical Contract Risk:** Monthly billing cycles correlate with high volatility in Telecom sectors. Switch to Annual.")
        if support_calls > 3:
            advice_points.append("• **Service Friction Detected:** Customer has frequent tech issues. Priority resolution ticket recommended.")
        if delay > 5:
            advice_points.append("• **Financial Delinquency:** Late payments detected. Offer automated billing or flexible plans.")
        if tenure < 12 and usage_freq < 20:
             advice_points.append("• **Early Lifecycle Warning:** Low engagement for a new user. Send onboarding tutorial.")

        if not advice_points:
            advice_points = ["• Customer exhibits strong retention signals. No immediate action required."]

        advice_html = "<br>".join(advice_points)

        if prob > 0.5:
            res_col1.error(f"### ⚠️ CHURN RISK: {prob:.1%}")
            res_col2.markdown(f"""
            <div style="background: #FEF2F2; padding: 25px; border-radius: 20px; border-left: 6px solid #DC2626;">
                <h4 style="color: #991B1B; margin-top:0;">Retention Strategy</h4>
                <p style="color: #7F1D1D; line-height: 1.6;">{advice_html}</p>
                <hr style="border-color: #FCA5A5;">
                <p style="color: #991B1B; font-weight: 700;">Decision: Trigger High-Value Retention Offer (15% Disc).</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            res_col1.success(f"### ✅ LOYALTY STATUS: {1-prob:.1%}")
            res_col2.markdown(f"""
            <div style="background: #F0FDF4; padding: 25px; border-radius: 20px; border-left: 6px solid #16A34A;">
                <h4 style="color: #166534; margin-top:0;">Growth Insights</h4>
                <p style="color: #14532D; line-height: 1.6;">{advice_html}</p>
                <hr style="border-color: #BBF7D0;">
                <p style="color: #166534; font-weight: 700;">Decision: Eligible for Premium Upsell & Beta Program.</p>
            </div>
            """, unsafe_allow_html=True)
            st.balloons()
    else:
        st.error("Engine failure: Model artifacts missing. Ensure .pkl files are in the root directory.")

# 8. Footer 
st.markdown("""
<br><hr>
<div style="text-align: center; color: #94A3B8; font-size: 0.9rem; padding: 20px;">
    <b>Aura Core: Telecom Churn Intelligence</b> © 2026 • Designed & Developed by <b>Yasmen Wageeh</b>
</div>
""", unsafe_allow_html=True)
