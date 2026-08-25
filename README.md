# 🏆 League of Legends AI Draft Predictor & Assistant

> An advanced Deep Learning web application designed to analyze professional *League of Legends* drafts (LEC, LCK, LPL, LCS), predict match win rates in real-time, and provide data-driven champion recommendations using PyTorch and Streamlit.

---

## 🚀 Live Demo & Overview
This tool bridges the gap between competitive e-sports data and machine learning. Built using professional match data, it parses champion compositions, bans, side selection, and regional meta to evaluate draft advantages dynamically.

* **Live App:** [View Streamlit App](https://esports-draft-predictor.streamlit.app/)

---

## 🛠️ Tech Stack & Libraries
* **Core Language:** Python
* **Deep Learning:** PyTorch (Multi-Layer Perceptron / Neural Network architecture with Dropout regularization)
* **Data Manipulation & Processing:** Pandas, NumPy
* **Machine Learning Pipelines:** Scikit-learn (One-Hot Encoding, Train/Test splitting)
* **Web Interface & UI:** Streamlit

---

## 📊 Methodology & Pipeline
1. **Data Extraction & Cleaning:** Processing professional match data, filtering out team aggregates, and narrowing down observations to major regions (`LEC`, `LCK`, `LPL`, `LCS`).
2. **Feature Engineering:** Applying One-Hot Encoding across sparse categorical features (representing unique champions, bans across 5 phases, side selection, and game roles).
3. **Neural Network Architecture:** A customized Sequential MLP model built with `torch.nn`, featuring hidden layers with ReLU activations, dropout layers to prevent overfitting, and a final Sigmoid activation layer for binary outcome prediction (Blue Side Win Probability).
4. **Inference Engine (Greedy Search):** Simulates available champion pools to compute real-time winrate increments for subsequent draft picks.

---

## ⚙️ Local Installation & Running
If you want to run this project locally on your machine:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Sorixon/esports-draft-predictor.git](https://github.com/Sorixon/esports-draft-predictor.git)
   cd esports-draft-predictor
