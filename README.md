# Football Moneyball

Predicts a FIFA 23 player's overall rating and market value from his attributes, and shows the result as a FIFA-style player card.

## What it does

You enter a player's attributes: age, potential, wage, contract year, height, preferred foot, position, pace, shooting, passing, dribbling, defending, physical, stamina and finishing.

The app then predicts:

- **Overall rating** with XGBoost
- **Market value** (as rated in the game) with Random Forest

The result is displayed as a player card. The card color changes with the rating: bronze, silver, gold and elite.

## Data

[FIFA 23 complete player dataset](https://www.kaggle.com/datasets/stefanoleone992/fifa-23-complete-player-dataset) from Kaggle (`male_players.csv`). Goalkeepers are excluded.

## Pipeline

- Selected the main attributes and removed duplicates and goalkeepers
- Filled missing numeric values with the median and missing club names with the most common one
- Kept the first listed position for players with more than one
- One-hot encoded preferred foot and position
- Scaled the features with `StandardScaler` for the value model
- 80/20 train/test split and 5-fold cross-validation
- Compared Linear Regression and Random Forest for market value, and used XGBoost for overall rating

## Results

| Target | Model | Test R² | 5-fold CV R² |
|---|---|---|---|
| Overall rating | XGBoost | [X] | [X] |
| Market value | Random Forest | [Y] | [Y] |

## Limitations

- The value model learns the values used in the game, not real transfer market prices.
- The value model takes the overall rating and wage as inputs, so a large part of its accuracy comes from them.
- The game calculates the overall rating from the attributes, so a high R² for it is expected.

## Files

- `Football_MoneyBall_.ipynb`: data cleaning, training and evaluation
- `app.py`: Streamlit app that shows the player card
- `rating_model.json`: trained XGBoost model
- `meta.json`: column names used by the models

`value_model.pkl` is not included because of its size. Train the model in the notebook and save it as a dictionary with the keys `model` and `scaler`.

## Run

Models were trained with scikit-learn 1.6.1 and XGBoost 3.4.1. Use the same versions when loading them.

```
pip install -r requirements.txt
streamlit run app.py
```
