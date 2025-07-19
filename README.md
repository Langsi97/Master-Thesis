# ⚽ Master's Thesis: Machine Learning in Football Match Prediction



## 📚 Project Overview

This repository contains the complete research code and analysis for my Master's thesis titled:

**"Machine Learning in Football Match Prediction: A Case Study of the Belgian Jupiler Pro League"**

The goal of this project is to investigate the use of machine learning algorithms in predicting the outcomes of football matches — specifically, Home Win, Draw, or Away Win — and to explore the implications of betting odds, potential market inefficiencies, and odds manipulations. The project aims to raise awareness about the risks associated with sports betting and demonstrate that "the house always wins".

## 🎯 Objectives

- Predict match outcomes using various ML models (Logistic Regression, Random Forest, XGBoost, etc.)
- Integrate bookmaker odds to simulate betting profits and detect potential manipulations
- Educate the public on how AI can be used to highlight the risk and bias in sports betting

## 🧠 Machine Learning Models

The following models were trained and evaluated:

- Logistic Regression
- Random Forest
- Support Vector Machine
- XGBoost
- Naive Bayes

Evaluation metrics include Accuracy, Precision, Recall, F1-score, ROC AUC, and confusion matrices.

## 🏟️ Dataset

- **Source**: Belgian Jupiler Pro League (scrapped from football-data.co.uk, sofifa.com,and transfermkt)
- **Features**: Match statistics, team form, rolling averages, cluster labels
- **Target Variable**: Match outcome (Home Win, Draw, Away Win)

> Note: The dataset has been preprocessed to handle class imbalance (SMOTE), feature scaling, and leakage prevention.

