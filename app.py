# app.py
import streamlit as st

st.set_page_config(page_title="Heart Health Checker", page_icon="❤️", layout="centered")

st.markdown("""
    <style>
    .centered {
        text-align: center;
    }
    .start-button button {
        background-color: #d63384;
        color: white;
        border-radius: 12px;
        padding: 10px 25px;
        font-size: 18px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <h1 class='centered'>Welcome to the Heart Health Checker</h1>
    <p class='centered'>Created by <a href='https://github.com/CyberGlitchKripto' target='_blank'>CyberGlitchKripto</a></p>
    <hr>
    <h3 class='centered'>🩺 About This App</h3>
    <p class='centered'>This tool uses machine learning to predict the risk of heart disease based on your health data.</p>
    <ul>
        <li>💡 Simple and user-friendly interface</li>
        <li>🧠 Powered by trained AI models</li>
        <li>📄 Generates a downloadable report</li>
        <li>🩺 Gives clear health recommendations</li>
    </ul>
""", unsafe_allow_html=True)

if st.button("👉 Start Health Check", key="start", help="Click to begin", use_container_width=True):
    st.switch_page("pages/health.py")



