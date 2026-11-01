# ⚽ Machine Learning in Football Match Prediction
### A Case Study of the Belgian Jupiler Pro League · MSc Thesis, Hasselt University (2025)

**Author:** Langsi Ambe Revelation · MSc Statistics & Data Science
**Stack:** Python · pandas · scikit-learn · XGBoost · imbalanced-learn (SMOTE) · SHAP · Streamlit

---

## 📌 TL;DR

Can machine learning beat the bookmaker? I built an end-to-end pipeline that predicts Home Win / Draw / Away Win for 1,390 Jupiler Pro League matches (2019/20 to 2023/24), turned the model's probabilities into "fair" odds, and compared them with real bookmaker prices in a profit simulation.

**The answer is no, not sustainably.** Even a well-tuned model tops out at roughly 45 to 54% accuracy on a three-way outcome, draws are close to unpredictable, and the bookmaker's built-in margin (the vig) absorbs most of whatever edge the model finds. The data confirms what the industry already prices in: **the house always wins.**

---

## 💼 Business Problem

Sports betting is a high-volume, high-margin market, and the people on each side of it have very different information.

| Stakeholder | The question they care about |
|---|---|
| **Bettors / the public** | Can data or AI give me a reliable edge over the bookmaker? |
| **Bookmakers & trading desks** | Are our odds well calibrated, and where are we exposed to mispricing? |
| **Regulators & responsible-gambling bodies** | How do we evidence the real risk of betting to consumers? |
| **Clubs, media & analytics teams** | Which pre-match factors actually drive results? |

Betting marketing often implies that skill or "insider data" leads to profit. This project tests that claim quantitatively with the same tools a professional trading team would use.

## ✅ Solution

A reproducible prediction and market-analysis pipeline:

1. **Data collection and integration.** Match results and bookmaker odds (football-data.co.uk), player and team ratings (sofifa.com), and squad market values (Transfermarkt), merged into one dataset of 1,390 matches and 51 pre-match features.
2. **Leakage-safe feature engineering.** Only information available before kick-off: rolling form, home/away ratings, squad value, goal-based rolling averages, and K-means team-strength clusters (elbow method).
3. **Modelling.** Five classifiers benchmarked: Logistic Regression, Random Forest, SVM, XGBoost, and Naive Bayes. Class imbalance (draws are under-represented) handled with SMOTE on the training folds only; features scaled where the model requires it.
4. **Evaluation.** Accuracy, precision, recall, weighted F1, ROC AUC, log loss, Brier score, and confusion matrices. Probabilistic metrics matter here, because betting decisions depend on calibrated probabilities, not just the predicted label.
5. **Explainability.** SHAP values identify which features drive each prediction, so results can be read by non-technical stakeholders.
6. **Betting-market layer.** Model probabilities converted into fair odds, compared with Bet365 prices, bookmaker margin (vig/overround) computed, favourite-longshot bias analysed, and a flat-stake profit simulation run on the Random Forest predictions.
7. **Deployment.** An interactive Streamlit app that returns outcome probabilities and implied fair odds for a fixture ([football-match-prediction-system](https://github.com/Langsi97/football-match-prediction-system)).

---

## 📊 Key Results

**Model performance on the held-out test set** (best value per metric in bold)

| Metric | Logistic Regression | **Random Forest** | SVM | XGBoost | Naive Bayes |
|---|---|---|---|---|---|
| Accuracy | 0.515 | **0.540** | 0.452 | 0.485 | 0.510 |
| Balanced accuracy | **0.502** | 0.501 | 0.437 | 0.465 | **0.502** |
| Precision (weighted) | 0.529 | **0.541** | 0.450 | 0.479 | 0.524 |
| Recall (weighted) | 0.515 | **0.540** | 0.452 | 0.485 | 0.510 |
| F1 score (weighted) | 0.497 | 0.493 | 0.448 | 0.476 | **0.500** |
| ROC AUC | 0.691 | **0.696** | 0.615 | 0.666 | 0.662 |
| Log loss ↓ | 1.039 | **0.986** | 1.088 | 1.063 | 1.512 |
| Brier score ↓ | 0.627 | **0.590** | 0.652 | 0.638 | 0.722 |

**Why Random Forest was selected:** it leads on 6 of 8 metrics. More importantly for a betting use case, it has the lowest log loss (0.986) and Brier score (0.590), so its probabilities are the best calibrated. Pricing odds depends on trustworthy probabilities, not just on picking the most likely outcome.

**Benchmark context:** a naive "always predict the home team" baseline typically lands around 45% in top European leagues, and a random guess on three outcomes gets 33%. Random Forest's 54% accuracy and 0.70 ROC AUC show real signal, but only a modest lift above that floor.

**Profit simulation (Random Forest vs Bet365, flat stake):** [N] bets placed · ROI [±x.x%] · net result [±x units]

## 🔍 Most Important Findings

1. **There is a hard ceiling on predictability.** Across all five models, test accuracy ranges from 45.2% (SVM) to 54.0% (Random Forest), and ROC AUC from 0.62 to 0.70. A simple Logistic Regression (51.5%, AUC 0.691) comes within 2.5 points of the best model, while the more complex XGBoost (48.5%) and SVM do worse. Complexity does not unlock a hidden signal. The noise in the sport itself sets the limit.
2. **Draws are the blind spot.** Balanced accuracy stays at about 0.50 for every model, even Random Forest at 54% overall accuracy. Models score well on home and away wins but struggle on draws, even after SMOTE rebalancing. This matters commercially because draws are typically priced at the most attractive odds, which pulls bettors towards the outcome that models handle worst.
3. **Team quality beats short-term form.** SHAP analysis shows that structural strength signals (team ratings, squad market value, strength cluster, home/away quality) carry more weight than recent results. "Hot streaks" are weaker predictors than the betting narrative suggests.
4. **The bookmaker's margin is the real opponent.** Bookmaker odds are not fair odds: the implied probabilities across Home/Draw/Away sum to more than 100%. That overround means a bettor needs to be right more often than the true probability just to break even, and it erases most of the edge the model finds on individual matches.
5. **Pricing is not uniform.** Comparing model-implied fair odds with Bet365 prices reveals favourite-longshot bias: longshots tend to be priced less generously relative to their true chance than favourites. Bettors chasing big payouts are systematically paying the most margin.
6. **Bottom line: the house always wins.** Even when the model identifies matches where it disagrees with the bookmaker, the simulated betting strategy does not produce reliable long-term profit once the vig is applied.

---

## 💡 Business Recommendations

**For bettors and the public**
- Treat sports betting as entertainment spending, not an income strategy. A model that is right half the time still loses money against a priced-in margin.
- Be most cautious with draws and longshots: they look attractive and carry the most hidden margin.

**For regulators and responsible-gambling organisations**
- Use this kind of transparent, explainable analysis in awareness campaigns. Showing that "AI can't beat the odds either" is a concrete, data-backed message.
- Encourage clear disclosure of bookmaker margins so consumers can see the cost of each bet.

**For bookmakers and trading desks**
- Use probabilistic models with calibration metrics (log loss, Brier score) as an independent check on pricing, especially in less-liquid leagues like the Jupiler Pro League where mispricing is more likely.
- Monitor favourite-longshot bias and draw pricing as specific risk areas.

**For clubs and analytics teams**
- Invest in structural quality indicators (squad value, ratings) as the more reliable performance drivers, and treat short-term form as a secondary signal.

---

## ⚠️ Limitations

- **Small sample:** 1,390 matches over five seasons, including the COVID-affected 2019/20 and 2020/21 seasons with empty stadiums that weakened home advantage.
- **Pre-match features only:** no injuries, line-ups, xG or in-play data, which limits the achievable ceiling.
- **Single bookmaker in the simulation:** results use Bet365 prices; line shopping across bookmakers could narrow the margin but is unlikely to remove it.

## 🚀 Next Steps

- Add expected-goals (xG) and line-up/injury data
- Hyperparameter optimisation with Optuna and probability calibration (Platt / isotonic)
- Time-aware (walk-forward) validation across seasons
- Experiment tracking with MLflow, containerisation with Docker, and cloud deployment on Azure
- Extend to other European leagues to test whether the findings generalise

---

## 🗂️ Repository Structure

```
├── Codes.ipynb                                  # Full pipeline: cleaning, features, modelling, evaluation
├── B1920.csv … B2324.csv                        # Season results + bookmaker odds (football-data.co.uk)
├── Belgian_Pro_League_Ratings_2019_2024_Complete.xlsx   # Team/player ratings (sofifa)
├── merged_data_with_seasons.csv                 # Final modelling dataset
├── merged_home_away_ratings_data.csv            # Home/away rating features
├── Model_Predictions_vs_Actual*.xlsx            # Model predictions vs actual outcomes
├── Random_Forest_Odds*.xlsx / RF_Odds_Attached.xlsx     # Fair odds vs bookmaker odds, with vig
├── Profit_Simulation_RF_B365.xlsx               # Betting simulation results
├── saved_models/                                # Trained model artefacts
├── elbow.png                                    # K-means cluster selection
└── Master_Thesis_Langsi_Ambe_Rev (1).pdf        # Full thesis
```

## ▶️ How to Run

```bash
git clone https://github.com/Langsi97/Master-Thesis.git
cd Master-Thesis
pip install pandas numpy scikit-learn xgboost imbalanced-learn shap matplotlib seaborn openpyxl
jupyter notebook Codes.ipynb
```

---

## 📬 Contact

**Langsi Ambe Revelation** · Data Scientist
[LinkedIn](https://www.linkedin.com/in/langsi-ambe97) · [GitHub](https://github.com/Langsi97)

*This project is for research and educational purposes only. It does not encourage or endorse gambling.*
