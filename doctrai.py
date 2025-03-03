import os
import streamlit as st
from groq import Groq
from typing import Dict, List, Optional
from dotenv import load_dotenv
load_dotenv()

class DoctorAI:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("grok_api_key"))
        self.context = """You are an advanced AI medical assistant with comprehensive knowledge across medical specialties. 
        Provide direct, clear, and actionable medical information and advice based on established medical science and 
        clinical guidelines. Focus on delivering practical insights and evidence-based recommendations."""

    def get_medical_advice(self, symptoms: str) -> str:
        """Get medical advice based on described symptoms"""
        prompt = f"""Based on the following symptoms, provide general medical advice and recommendations:
        Symptoms: {symptoms}
        
        Please include:
        1. Possible causes
        2. General recommendations
        3. When to seek immediate medical attention"""

        response = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": self.context},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.3-70b-specdec",
            temperature=0.5,
            max_tokens=1000
        )
        
        return response.choices[0].message.content

    def get_medication_info(self, medication: str) -> str:
        """Get information about a specific medication"""
        prompt = f"""Provide information about the medication: {medication}
        
        Please include:
        1. Common uses
        2. Typical dosage
        3. Common side effects
        4. Important warnings
        5. Drug interactions to watch for
        """

        response = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": self.context},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.3-70b-specdec",
            temperature=0.5,
            max_tokens=1000
        )
        
        return response.choices[0].message.content

    def get_lifestyle_recommendations(self, condition: str) -> str:
        """Get lifestyle recommendations for managing a specific health condition"""
        prompt = f"""Provide lifestyle recommendations for managing: {condition}
        
        Please include:
        1. Diet recommendations
        2. Exercise suggestions
        3. Stress management techniques
        4. Sleep hygiene tips
        5. Other lifestyle modifications
        """

        response = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": self.context},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.3-70b-specdec",
            temperature=0.5,
            max_tokens=1000
        )
        
        return response.choices[0].message.content

# Streamlit Interface
def main():
    st.set_page_config(page_title="AI Doctor Assistant", layout="wide")
    
    # Custom CSS
    st.markdown("""
        <style>
        .main {
            background-color: #f5f5f5;
        }
        .stButton>button {
            background-color: #ff4b4b;
            color: white;
            border-radius: 10px;
            padding: 10px 25px;
        }
        .stTextInput>div>div>input {
            border-radius: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header
    st.title("🏥 AI Doctor Assistant")
    st.markdown("---")

    # Initialize DoctorAI
    doctor_ai = DoctorAI()

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choose a service:", 
        ["Symptom Checker", "Medication Information", "Lifestyle Recommendations"])

    # Disclaimer
    st.sidebar.markdown("---")
    st.sidebar.warning("""
        ⚠️ **Disclaimer**: 💝 Warning sweetie! Please use this tool responsibly. 🙏 If you have any feedback or issues, feel free to reach out to Shivam 👨‍💻. Stay healthy and take care! 🌟
    """)

    if page == "Symptom Checker":
        st.header("🔍 Symptom Checker")
        symptoms = st.text_area("Describe your symptoms:", 
            height=100,
            placeholder="Example: I have a headache, fever, and sore throat...")
        
        if st.button("Get Medical Advice"):
            if symptoms:
                with st.spinner("Analyzing symptoms..."):
                    advice = doctor_ai.get_medical_advice(symptoms)
                    st.markdown("### Analysis Results")
                    st.write(advice)
            else:
                st.error("Please describe your symptoms first.")

    elif page == "Medication Information":
        st.header("💊 Medication Information")
        medication = st.text_input("Enter medication name:",
            placeholder="Example: ibuprofen")
        
        if st.button("Get Medication Information"):
            if medication:
                with st.spinner("Retrieving information..."):
                    med_info = doctor_ai.get_medication_info(medication)
                    st.markdown("### Medication Details")
                    st.write(med_info)
            else:
                st.error("Please enter a medication name.")

    else:  # Lifestyle Recommendations
        st.header("🌟 Lifestyle Recommendations")
        condition = st.text_input("Enter health condition:",
            placeholder="Example: type 2 diabetes")
        
        if st.button("Get Lifestyle Recommendations"):
            if condition:
                with st.spinner("Generating recommendations..."):
                    recommendations = doctor_ai.get_lifestyle_recommendations(condition)
                    st.markdown("### Recommended Lifestyle Changes")
                    st.write(recommendations)
            else:
                st.error("Please enter a health condition.")

if __name__ == "__main__":
    main()

