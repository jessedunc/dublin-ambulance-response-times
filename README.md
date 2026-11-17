# Dublin Fire Brigade Ambulance Response Times

What drives ambulance response times in Dublin, and can we predict which calls will receive a late response?

This project analyses 407,890 Dublin Fire Brigade ambulance calls from 2021 to 2025, using public data from Smart Dublin. It covers data cleaning, feature engineering, regression and classification.

## Key results

- **Late responses:** the best classifier, logistic regression, flagged **86% of late responses** in 2025 (F1-macro 0.63).
- **Response time:** the best regression model, random forest, predicted response times with a mean absolute error of **32.9 minutes** (R² 0.27), compared with 34.8 minutes for the linear regression baseline.
- **Demand matters most:** city-wide demand, meaning the number of incidents active across Dublin when a call comes in, was the strongest predictor of response time.

## Data cleaning

Much of the work went into making the data trustworthy before any modelling:

- **Combining files:** merged four yearly files and standardised column names that differed between years.
- **Invalid timings:** times are recorded without dates, so gaps that cross midnight were corrected. Calls that took more than 6 hours to dispatch were removed as recording errors.
- **Missing outcomes:** calls with no arrival time were dropped rather than given invented response times.
- **Daylight saving errors:** response times of over 24 hours were traced back to the clock changes and corrected with targeted rules, rather than simply removed as outliers.

After cleaning, 353,396 calls remained.

## Feature engineering

- **Real-time demand:** for every call, the number of incidents active at that moment across the city (`Active City`) and within the same station area (`Active Area`).
- **Calendar features:** hour of day, day of week, month and Irish public holidays.
- **Late response target:** a response counts as late if it is slower than 75% of training-set responses at the same criticality level, so urgent and non-urgent calls are each judged against their own standard.

## Modelling

- **Time-based split:** models were trained on 2021–2024 and tested on 2025 (72,384 calls), so they are always evaluated on future data.
- **Leakage-safe pipelines:** all preprocessing (one-hot encoding and robust scaling) happens inside scikit-learn pipelines, so it is fitted on training data only.
- **Regression models:** linear regression, Ridge, decision tree, random forest and a stacking ensemble.
- **Classification models:** logistic regression, decision tree, K-nearest neighbours, random forest and a stacking ensemble.
- **Model comparison:** 10-fold cross-validation with 95% bootstrap confidence intervals, feature selection and permutation importance.

| Classification model | F1-macro | Late responses caught (recall) |
|---|---|---|
| Logistic regression | 0.63 | 86% |
| Stacking ensemble | 0.63 | 87% |
| Random forest | 0.61 | 90% |
| Decision tree | 0.60 | 87% |
| K-nearest neighbours | 0.60 | 35% |

## Limitations

- **Low R²:** response times are hard to predict from this data alone. Key factors such as traffic, exact incident location and ambulance availability are not included in the public dataset.
- **False alarms:** catching 86% of late responses comes at the cost of false alarms, since only 41% of the calls flagged as late actually were. For a monitoring tool, missing a late response is the more costly mistake, so this trade-off was chosen deliberately.

## Running the notebook

```bash
pip install -r requirements.txt
jupyter notebook dfb_ambulance_response_times.ipynb
```

Running the notebook saves the cleaned data to `preproccesed.csv`. That file is not included in the repository because of its size.

## Data

Dublin Fire Brigade ambulance incident data, published on [Smart Dublin](https://data.smartdublin.ie/dataset/fire-brigade-and-ambulance):

- `da-opendata-2020-to-2022-with-stn-area.csv`
- `2023-open-data-dfb-ambulance.csv`
- `2024-dfb-ambulance-incidents-da.csv`
- `dfb-ambulance-incidents-2025.csv`
