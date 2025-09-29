# 🧠 Autism Prediction Detection

This project is a **machine learning-based predictive system** that helps detect the likelihood of Autism Spectrum Disorder (ASD) based on user inputs.  
It provides a **Flask web interface** where users can fill out a form, and the trained model predicts whether autism indicators are present.  
If autism is predicted, the app also provides **doctor-like guidance and recommendations**.

---

## 🚀 Features
- Collects responses to **screening questions (A1–A10)** along with demographic data.  
- Uses **pre-trained ML models** (`best_model.pkl`) and **label encoders** (`encoders.pkl`).  
- Predicts whether the user shows signs of autism (`Has Autism` / `No Autism`).  
- If **Has Autism**, provides a **doctor’s prescription section** with guidance and resources.  
- Web interface built with **Flask** (runs on localhost).  
- Clean, modern, and responsive **HTML/CSS templates**.  

---

## 🛠️ Tech Stack
- **Python 3.10+**
- **Flask** – web framework  
- **Pandas / NumPy** – data preprocessing  
- **Scikit-learn** – model training & evaluation  
- **Imbalanced-learn (SMOTE)** – handling imbalanced datasets  
- **XGBoost & Random Forest** – classification models  
- **Matplotlib / Seaborn** – data visualization (in Jupyter Notebook)  

---

## 📂 Project Structure