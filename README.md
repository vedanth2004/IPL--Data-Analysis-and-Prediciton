# IPL Data Analysis and Prediction

End-to-end analysis of IPL matches from 2008 to 2022, covering exploratory data
analysis, statistical insights, and match winner prediction using machine learning.

---

## Dataset

Two CSV files covering all IPL seasons:

| File | Description |
|---|---|
| `matches.csv` | 1,095 matches — teams, venues, toss, results, player of match |
| `deliveries.csv` | Ball-by-ball delivery data for all matches |

---

## Project Structure

IPL/
├── matches.csv
├── deliveries.csv
└── Ipl_Analysis.ipynb


---

## Analysis Overview

### 1. Loading & Exploring Data
- Loaded both datasets using pandas
- Inspected shape, dtypes, null values, and summary statistics
- `matches.csv` → 1,095 rows × 20 columns
- `deliveries.csv` → ball-by-ball records across all seasons

### 2. Exploratory Data Analysis
Key findings from the analysis:

- **Most matches hosted** → Mumbai (173 matches)
- **Most wins overall** → Mumbai Indians (144 wins)
- **Most Player of the Match awards** → AB de Villiers (25 awards)
- Toss decision trends across seasons
- Venue-wise and city-wise win distributions
- Head-to-head records between teams

### 3. Feature Engineering
Features built for the ML model:

- Team encodings (Label Encoding for team1, team2, winner)
- Venue encoding
- Overall win percentage per team
- Head-to-head win percentage between two teams
- Venue-specific win percentage per team
- Average runs scored and conceded per team
- Net run rate proxy
- Recent form
- Toss winner and toss decision
- Temporal features: year, month, day of week

### 4. Match Winner Prediction

#### Models Trained
| Model | Type | Notes |
|---|---|---|
| Random Forest | Ensemble (decision trees) | Handles non-linear patterns, provides feature importance |
| Logistic Regression | Linear model | Fast, interpretable, good baseline |

- Best model selected automatically based on test set accuracy
- Features scaled for Logistic Regression using StandardScaler
- Both models evaluated on the same train/test split for fair comparison

#### Example Predictions
```python
# Mumbai Indians vs Chennai Super Kings at Wankhede
predict_match_winner('Mumbai Indians', 'Chennai Super Kings',
                     'Wankhede Stadium', 'Mumbai Indians', 'bat')

# RCB vs KKR at Chinnaswamy
predict_match_winner('Royal Challengers Bangalore', 'Kolkata Knight Riders',
                     'M Chinnaswamy Stadium', 'Royal Challengers Bangalore', 'field')
```

Output includes predicted winner and win probabilities for both teams.

---

## Setup

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter

jupyter notebook Ipl_Analysis.ipynb
```

---

## Key Insights

- Mumbai Indians are the most successful IPL franchise by total wins
- Mumbai city has hosted the most IPL matches
- AB de Villiers is the all-time leading Player of the Match award winner
- Head-to-head record and venue-specific win rate are among the strongest
  predictors of match outcome
- Toss decision has moderate influence — teams choosing to field have a slight edge
  in recent seasons

---

## Tech Stack

- **Python 3.13**
- **pandas** — data loading and manipulation
- **numpy** — numerical operations
- **matplotlib / seaborn** — visualizations
- **scikit-learn** — Label Encoding, StandardScaler, Random Forest, Logistic Regression
