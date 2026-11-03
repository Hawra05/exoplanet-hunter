# 🪐 Exoplanet Hunter

## 📖 Overview

When a planet passes in front of its host star, it casts a tiny shadow. Detecting this minuscule dip in stellar brightness against a background of cosmic noise and instrumental artifacts is a classic needle-in-a-haystack problem in modern astrophysics. 

**Exoplanet Hunter** is a machine learning exploration built to tackle this exact classification challenge. Using NASA's Kepler Objects of Interest (KOI) data, this project implements a logistic regression model from scratch using only NumPy. It serves as an educational demonstration of how foundational statistical learning can be applied to separate genuine planetary transits from astrophysical false positives.

**🔗 Live Interactive Demo:** https://exoplanet-hunter-a.streamlit.app

---

## 📊 Dataset

The data used in this project originates from the **NASA Exoplanet Archive** and was accessed via Kaggle.
*   **Source:** [Kaggle - NASA Kepler Exoplanet Search Results](https://www.kaggle.com/datasets/nasa/kepler-exoplanet-search-results)
*   **Features:** The model utilizes photometric and stellar parameters recorded by the Kepler space telescope.
*   **Target:** The model classifies objects into binary categories based on their disposition (e.g., Confirmed vs. False Positive).

---

## 🛠️ Methodology

1.  **Data Preprocessing:** Handling missing values and selecting the most predictive numerical features from the Kepler dataset.
2.  **Custom Implementation:** Instead of relying on high-level libraries like Scikit-Learn, the logistic regression algorithm (including gradient descent and the sigmoid function) is written entirely from scratch using **NumPy**. This was done to demonstrate a deep understanding of the underlying mathematics.
3.  **Evaluation:** The model's performance is evaluated using standard classification metrics (Accuracy, Precision, Recall) to ensure it correctly identifies planets without triggering too many false alarms.

---

## ⚙️ Requirements & Installation

To run this project locally, you will need Python installed on your system. Follow the steps below in your terminal:

**1. Clone the repository**
```bash
git clone https://github.com/Hawra05/exoplanet-hunter.git
cd exoplanet-hunter
```
2.Install dependencies
```bash
pip install -r requirements.txt
```
3.Run the application
```bash
streamlit run app.py
```
## ⚠️ Disclaimer

This project is an educational machine learning demonstration and does not constitute scientific confirmation of exoplanets.

## 📄 License

**MIT License**
