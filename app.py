import streamlit as st
import pandas as pd
import joblib
import time

# 1. Page Configuration (Full Width)
st.set_page_config(page_title="Aura Core | Intelligence", page_icon="💎", layout="wide")

# 2. Advanced CSS for UI/UX Designer Touch
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap');
    
    /* Global Styles */
    * { font-family: 'Plus Jakarta Sans', sans-serif; }
    .stApp { background-color: #FDFDFD; }
    
    /* Hero Banner Gradient (Burgundy to Navy) */
    .hero-container {
        background: linear-gradient(135deg, #1E293B 0%, #800020 100%);
        padding: 3rem 2rem;
        border-radius: 24px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(128, 0, 32, 0.15);
    }

    /* Stepper Styling: Centering text and customizing buttons */
    div[data-testid="stNumberInput"] {
        background-color: transparent !important;
        border: none !important;
    }
    
    /* Centering the Input Value */
    div[data-testid="stNumberInput"] input {
        text-align: center !important;
        font-weight: 700 !important;
        font-size: 1.2rem !important;
        color: #1E293B !important;
        background-color: #F8FAFC !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 12px !important;
        height: 45px !important;
    }

    /* Styling the plus/minus buttons to look modern */
    div[data-testid="stNumberInput"] button {
        background-color: #F1F5F9 !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 10px !important;
        width: 45px !important;
        height: 45px !important;
        transition: all 0.2s ease;
    }
    
    div[data-testid="stNumberInput"] button:hover {
        background-color: #800020 !important;
        color: white !important;
        border-color: #800020 !important;
    }

    /* Cards Styling for grouping */
    .input-card {
        background: white;
        padding: 24px;
        border-radius: 20px;
        border: 1px solid #F1F5F9;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 10px;
    }

    /* The Main Action Button */
    div.stButton > button {
        background: #1E293B;
        color: white;
        border-radius: 16px;
        padding: 25px;
        width: 100%;
        border: none;
        font-weight: 700;
        font-size: 1.1rem;
        letter-spacing: 0.5px;
        transition: 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    div.stButton > button:hover {
        background: #800020;
        transform: translateY(-3px);
        box-shadow: 0 15px 30px rgba(128, 0, 32, 0.25);
    }
</style>
""", unsafe_allow_html=True)

# 3. Model & Assets Loader
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('xgb_churn_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None

model, scaler = load_assets()

# 4. Hero Branding Section
st.markdown(f"""
<div class="hero-container">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h1 style="margin: 0; font-size: 2.8rem; font-weight: 700;">Aura Core Analytics</h1>
            <p style="margin: 5px 0 0 0; opacity: 0.85; font-size: 1.2rem; font-weight: 300;">Strategic Intelligence Dashboard • Version 2.0</p>
        </div>
        <div style="background: rgba(255,255,255,0.15); padding: 15px 25px; border-radius: 15px; backdrop-filter: blur(10px);">
            <small style="display: block; opacity: 0.7;">Engine Status</small>
            <span style="font-weight: 600;">● Neural Network Active</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# 5. Dashboard Grid (The 10 Required Features)
st.markdown("<h3 style='margin-bottom: 20px; color: #1E293B;'>📊 Behavioral Diagnostic Workspace</h3>", unsafe_allow_html=True)
col_a, col_b, col_c = st.columns(3, gap="large")

with col_a:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.write("#### 👤 Profile")
    age = st.number_input("Customer Age", 18, 100, 30)
    gender = st.selectbox("Gender Identity", ["Female", "Male"])
    tenure = st.number_input("Tenure (Total Months)", 0, 120, 12)
    st.markdown('</div>', unsafe_allow_html=True)

with col_b:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.write("#### 📊 Engagement")
    usage_freq = st.number_input("Monthly Usage Sessions", 0, 500, 15)
    support_calls = st.number_input("Technical Support Calls", 0, 50, 1)
    last_int = st.number_input("Days Since Last Login", 0, 365, 5)
    st.markdown('</div>', unsafe_allow_html=True)

with col_c:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.write("#### 💰 Financials")
    sub_type = st.selectbox("Plan Tier", ["Basic", "Standard", "Premium"])
    contract = st.selectbox("Billing Cycle", ["Annual", "Monthly", "Quarterly"])
    spend = st.number_input("Total Spend ($)", 0.0, 50000.0, 500.0)
    delay = st.number_input("Payment Delay (Days)", 0, 100, 2)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# 6. Action & Prediction Logic
if st.button("Generate Diagnostic Report"):
    if model and scaler:
        with st.status("Computing Predictive Analytics...", expanded=False) as status:
            time.sleep(1.2) # To simulate complex calculation for UX
            
            # Feature Encoding Logic
            g_map = {"Female": 0, "Male": 1}
            s_map = {"Basic": 0, "Premium": 1, "Standard": 2}
            c_map = {"Annual": 0, "Monthly": 1, "Quarterly": 2}
            
            # Structure input for model
            features = pd.DataFrame([[
                age, g_map[gender], tenure, usage_freq, support_calls, 
                delay, s_map[sub_type], c_map[contract], spend, last_int
            ]], columns=[
                'Age', 'Gender', 'Tenure', 'Usage Frequency', 'Support Calls', 
                'Payment Delay', 'Subscription Type', 'Contract Length', 'Total Spend', 'Last Interaction'
            ])
            
            # Transformation & Prediction
            input_scaled = scaler.transform(features)
            prob = model.predict_proba(input_scaled)[0][1]
            status.update(label="Analysis Complete", state="complete")

        st.markdown("---")
        
        # Result Presentation Layer
        res_col1, res_col2 = st.columns([1, 1.5])
        
        if prob > 0.5:
            res_col1.error(f"### RISK DETECTED: {prob:.1%}")
            res_col2.markdown(f"""
            <div style="background: #FEF2F2; padding: 20px; border-radius: 15px; border-left: 5px solid #DC2626;">
                <h4 style="color: #991B1B; margin-top:0;">⚠️ High Churn Probability</h4>
                <p style="color: #7F1D1D;">The user behavior matches pattern of departing customers. 
                Immediate retention outreach is recommended. Consider offering a <b>loyalty discount</b>.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            res_col1.success(f"### LOYALTY STATUS: {1-prob:.1%}")
            res_col2.markdown(f"""
            <div style="background: #F0FDF4; padding: 20px; border-radius: 15px; border-left: 5px solid #16A34A;">
                <h4 style="color: #166534; margin-top:0;">✅ Stable Customer Profile</h4>
                <p style="color: #14532D;">Customer exhibits strong retention signals. 
                This user is an excellent candidate for <b>Premium tier upsells</b>.</p>
            </div>
            """, unsafe_allow_html=True)
            st.balloons()
    else:
        st.error("Model files not found in the root directory.")

# 7. Footer (Professional Touch)
st.markdown("<br><hr><p style='text-align: center; color: #94A3B8; font-size: 0.8rem;'>Aura Core Intelligence System © 2026 • Designed for High-Performance Analytics</p>", unsafe_allow_html=True)