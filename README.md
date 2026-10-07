# Cancer Mortality Rate Prediction

An experimental machine-learning regression study comparing linear models, deep neural networks, ensemble methods, and feature-engineered CatBoost for predicting age-adjusted cancer mortality rates from demographic, socioeconomic, population, and health-related variables.

## Project overview

This project explores how model choice and feature representation affect regression performance. The workflow progresses from a linear baseline to multiple fully connected neural-network architectures, Random Forest, Gradient Boosting, and CatBoost. It also examines alternative loss functions, regularization, categorical information, domain-informed feature engineering, and cross-validation.

A key outcome is that strong validation and public-leaderboard performance did not translate equally to the larger private leaderboard. The project therefore also serves as a practical case study in generalization and model evaluation.

## What this project demonstrates

- Controlled comparison of linear regression and several DNN architectures
- Target scaling for stable neural-network optimization with SGD
- Comparison of MSE and MAE loss behavior
- Ensemble-tree baselines using Random Forest and Gradient Boosting
- CatBoost modeling with categorical features such as `State` and `binnedInc`
- Domain-informed feature engineering using population-normalized mortality information
- Five-fold cross-validation and comparison with hidden leaderboard performance
- Analysis of local, public-leaderboard, and private-leaderboard generalization

## Methodology

For the controlled baseline experiments, the data are split 70/15/15 into training, validation, and local test sets. Missing numerical values are median-imputed using training-set statistics. Predictors are standardized where appropriate, and the DNN target is standardized during training for numerical stability before predictions are transformed back to the original scale.

The extended CatBoost experiment retains additional categorical information by extracting `State` from `Geography` and keeping `binnedInc`. It also adds the domain-informed feature:

`Deaths_per_100k = avgDeathsPerYear / popEst2015 × 100000`

This feature is related to, but not identical to, the age-adjusted mortality target.

## Results

| Model / experiment | Evaluation | R² |
|---|---|---:|
| Linear Regression | Local test | 0.5964 |
| Random Forest | Local test | 0.6305 |
| Gradient Boosting | Local test | 0.6612 |
| CatBoost + `Deaths_per_100k` | Local test | **0.8529** |
| CatBoost + `Deaths_per_100k` | 5-fold CV mean | **0.8364** |
| Final CatBoost | Kaggle public leaderboard (30%) | **0.83616** |
| Final CatBoost | Kaggle private leaderboard (70%) | **~0.65** |

The CatBoost experiment used a richer feature representation than the controlled baseline models, so its improvement should not be interpreted as a pure architecture-only comparison.

## Generalization: public vs. private leaderboard

The selected model produced a local test R² of 0.8529 and a five-fold cross-validation mean of approximately 0.8364. Its public Kaggle score was also 0.83616. Performance, however, decreased to approximately 0.65 on the larger private leaderboard.

This is an important experimental result rather than something to hide. Agreement between random cross-validation and a public leaderboard did not guarantee equivalent performance on the larger hidden holdout. Both leaderboard results are therefore reported, and the gap indicates that the selected representation was less robust to the full hidden evaluation distribution than the earlier evaluations suggested.

## Project structure

```text
cancer-mortality-regression/
├── README.md
├── LICENSE
├── requirements.txt
├── results.csv
├── .gitignore
├── data/
│   └── README.md
├── figures/
├── models/
└── notebooks/
    └── cancer_mortality_regression.ipynb
```

## Reproduce the experiments

1. Clone the repository.
2. Create a Python environment.
3. Run `pip install -r requirements.txt`.
4. Place `train.csv`, `test.csv`, and `sample_submission.csv` inside `data/`.
5. Open `notebooks/cancer_mortality_regression.ipynb` and run it from top to bottom.

The dataset is intentionally not redistributed here; obtain it from the original authorized source.

## Takeaway

The experiments show that feature representation can produce larger gains than simply increasing neural-network depth. They also illustrate a broader lesson: model selection should not rely on a single validation split or public leaderboard score, and even apparently consistent cross-validation results may not predict performance under a different hidden evaluation distribution.

## Author

**Muhammad Mubinur Rahman**  
PhD Student in Computer Science & Engineering, University of Cincinnati
