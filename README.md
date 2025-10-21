# 🔐 CIRA – Cyber Intelligent Risk Assessment for IIoT using Machine Learning

## 📘 Overview
This project implements a **Cyber Intelligent Risk Assessment (CIRA)** system for the **Industrial Internet of Things (IIoT)** using **machine learning**.  
It predicts cybersecurity risks by analyzing IIoT device data, detecting threats, and assigning risk levels (Low, Medium, High).

---

## ⚙️ Features
- Machine Learning–based threat and risk prediction  
- Flask backend for model inference  
- Streamlit dashboard for risk visualization  
- Risk scoring using likelihood × impact  

---

## 🧠 Tech Stack
**Python**, **Flask**, **Streamlit**, **Scikit-learn**, **XGBoost**, **Pandas**, **NumPy**

---

## 🚀 How to Run
```bash
# Clone the repository
git clone https://github.com/yourusername/CIRA-IIoT.git
cd CIRA-IIoT

# Install dependencies
pip install -r requirements.txt

# Run backend
python backend/app.py

# Run dashboard
streamlit run frontend/dashboard.py
