# 🪐 Exoplanet Hunter

> An educational machine learning demonstration that uses Kepler Objects of Interest data to distinguish between confirmed exoplanets and false positives.

## 📖 Overview

Can a measured dip in a star's brightness be distinguished from a false alarm? This project uses Kepler Objects of Interest (KOI) data from NASA to build a logistic regression model from scratch with NumPy. The goal is to explore the binary classification problem of identifying genuine exoplanet signals from instrumental noise, eclipsing binaries, and other astrophysical false positives.

**🔗 Live Demo:** [https://exoplanet-hunter-a.streamlit.app](https://exoplanet-hunter-a.streamlit.app)

## 📊 Dataset

The data for this project was obtained from the **NASA Kepler Exoplanet Search Results** dataset via Kaggle. The original source is the **NASA Exoplanet Archive**.

*   **Source:** [Kaggle - NASA Kepler Exoplanet Search Results](https://www.kaggle.com/datasets/nasa/kepler-exoplanet-search-results) | [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)
*   **Target Variable:** The model aims to predict a star's disposition (e.g., `CONFIRMED` vs. `FALSE POSITIVE`).
*   **Features:** The model utilizes photometric and stellar features recorded by NASA before vetting.

## 🛠️ Methodology

1.  **Data Preprocessing:** The dataset was cleaned to handle missing values and select the most relevant features for the classification task.
2.  **Model:** A **logistic regression model** was implemented from scratch using **NumPy**. This was a deliberate choice to demonstrate an understanding of the underlying mathematics, rather than relying on a high-level library.
3.  **Training:** The model was trained on a labeled subset of the data to learn the patterns that distinguish confirmed exoplanets from false positives.
4.  **Evaluation:** The model's performance was evaluated using metrics such as [Accuracy, Precision, Recall, F1-score].

## 🚀 How to Run

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/exoplanet-hunter.git](https://github.com/your-username/exoplanet-hunter.git)
    cd exoplanet-hunter 
