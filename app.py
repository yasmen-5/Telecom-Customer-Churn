import streamlit as st
import pandas as pd
import joblib
import time

# 1. Page Configuration
st.set_page_config(page_title="Telecom Customer Churn Intelligence ", page_icon="💎", layout="wide")

# 2. CSS 
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

    /* Centering the Number Input Value */
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
        height: 100%;
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
        return None, None

model, scaler = load_assets()

# 4. Hero Section
st.markdown(f"""
<div class="hero-container">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h1 style="margin: 0; font-size: 2.8rem; font-weight: 700;">Aura Core Analytics</h1>
            <p style="margin: 5px 0 0 0; opacity: 0.85; font-size: 1.2rem; font-weight: 300;">Strategic Intelligence Dashboard • Version 2.1</p>
        </div>
        <div style="background: rgba(255,255,255,0.15); padding: 15px 25px; border-radius: 15px; backdrop-filter: blur(10px);">
            <small style="display: block; opacity: 0.7;">Engine Status</small>
            <span style="font-weight: 600;">● Neural Network Active</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# 5. Dashboard Grid (Inputs)
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
            time.sleep(1.2) # To simulate complex calculation for UX effect
            
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
        
        # 7. Result Presentation Layer 
        res_col1, res_col2 = st.columns([1, 1.5])
        
        if prob > 0.5:
            res_col1.error(f"### ⚠️ HIGH RISK: {prob:.1%}")
            
            # Custom Intelligence Insights
            advice = ""
            if support_calls > 3:
                advice += "• **Tech Issue:** High volume of support calls. Customer needs a technical health check.<br>"
            if delay > 5:
                advice += "• **Billing Issue:** Frequent payment delays. Propose a flexible billing schedule.<br>"
            if usage_freq < 10:
                advice += "• **Low Adoption:** Usage is dropping. Send a personalized 'Feature Discovery' guide.<br>"
            
            if not advice:
                advice = "• Pattern matches departing customers. Immediate proactive outreach is required."

            res_col2.markdown(f"""
            <div style="background: #FEF2F2; padding: 25px; border-radius: 18px; border-left: 6px solid #DC2626;">
                <h4 style="color: #991B1B; margin-top:0;">Actionable Intelligence</h4>
                <p style="color: #7F1D1D; line-height: 1.6;">{advice}</p>
                <hr style="border-color: #FCA5A5;">
                <p style="color: #991B1B; font-weight: 700; font-size: 1.1rem;">Final Decision: Send 15% Retention Discount.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            res_col1.success(f"### ✅ STABLE: {1-prob:.1%}")
            res_col2.markdown(f"""
            <div style="background: #F0FDF4; padding: 25px; border-radius: 18px; border-left: 6px solid #16A34A;">
                <h4 style="color: #166534; margin-top:0;">Growth Potential Detected</h4>
                <p style="color: #14532D; line-height: 1.6;">• Customer exhibits strong long-term loyalty signals.<br>
                • Strategic candidate for <b>Premium Tier</b> migration and cross-selling.</p>
                <hr style="border-color: #BBF7D0;">
                <p style="color: #166534; font-weight: 700; font-size: 1.1rem;">Final Decision: Invite to Exclusive Beta Program.</p>
            </div>
            """, unsafe_allow_html=True)
            st.balloons()
    else:
        st.error("Engine failure: Predictive model or scaler files are missing from the repository.")

# 8. Professional Footer
st.markdown("""
<br><hr>
<div style="text-align: center; color: #94A3B8; font-size: 0.85rem; padding: 10px;">
    Aura Core Intelligence System © 2026 • Strategy & Interface by Yasmen Wageeh
</div>
""", unsafe_allow_html=True)
