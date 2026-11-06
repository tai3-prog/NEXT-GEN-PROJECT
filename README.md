[Open Walmart Sales Forecasting App](https://next-gen-project-skxuvgteermjdejweepaai.streamlit.app/)

# Walmart Store Sales Forecasting

Weekly sales forecasting for Walmart stores at **Store × Department × Week** level, with a
time-series-safe evaluation, a model comparison, and a Streamlit dashboard.

> **Status:** a validated student/portfolio project. It has **not** been deployed, monitored or tested
> prospectively, and one input-feature question (CPI / Unemployment availability) is **unresolved**
> (see [Limitations](#8-business-insights-and-limitations)).

## Headline results (all numbers come from `reports/results_summary.md`)

| Model | CV MAE | Holdout MAE | Holdout RMSE | Holdout R² |
|---|---:|---:|---:|---:|
| Mean Baseline (DummyRegressor) | 15,276.00 | 15,101.22 | 21,959.53 | -0.0001 |
| Linear Regression | 8,339.38 | 8,037.93 | 12,169.61 | 0.6928 |
| Decision Tree (original, `min_samples_leaf=10`) | 2,915.76 | 1,962.81 | 4,161.92 | 0.9641 |
| Decision Tree (tuned) | 2,888.66 | 1,985.28 | 4,228.36 | 0.9629 |
| **Random Forest (tuned) — selected** | **2,822.41** | **1,883.41** | **3,755.62** | **0.9707** |

* The Random Forest was **selected from cross-validation alone** (lowest mean CV MAE), before any model
  was scored on the holdout. It also happens to have the best holdout scores.
* CV MAE and holdout MAE are **not directly comparable**: CV folds train on only 26–92 weeks and test on
  seasons the model may never have seen (e.g. fold 1 trains on Feb–Jul 2010 and tests on Aug–Dec 2010),
  so CV is more pessimistic than the holdout.
* The tuned Decision Tree improved CV MAE slightly (2,915.76 → 2,888.66) but its holdout MAE was *worse*
  than your original fixed `min_samples_leaf=10` tree (1,985.28 vs 1,962.81). The tuning gain on the Decision Tree
  is within the fold-to-fold noise (CV MAE std ≈ 700), so it should not be presented as a real improvement.

## Quick start

```bash
pip install -r requirements.txt
python Walmart_Sales_analyst.py          # full pipeline: ~16 min on 1 CPU core; writes models/ and reports/
python predict.py                        # predictions for test.csv -> predictions/walmart_test_predictions.csv
python -m streamlit run app.py           # dashboard
```

`WALMART_QUICK_TEST=1 python Walmart_Sales_analyst.py` runs a tiny smoke test (minutes, tiny grids).
Its numbers must **not** be reported. All paths are relative to the script, so the scripts work from any directory.

## Project structure

```
data/                 train.csv, test.csv, features.csv, stores.csv   (unchanged)
models/               walmart_model.pkl (pipeline artifact, version 2) + timestamped backups of earlier artifacts
predictions/          walmart_test_predictions.csv (older file is backed up before being overwritten)
reports/
  results_summary.md  every number of the last run (EDA insights, integrity checks, CV, holdout, errors, audit)
  training_log.txt    full console output of the last run
  model_comparison.csv, cv_results_decision_tree.csv, cv_results_random_forest.csv
  error_by_store.csv, error_by_dept.csv, error_by_month.csv, error_by_week_type.csv
  figures/            12 charts (EDA 01-07, error analysis 08-12)
Walmart_Sales_analyst.py   main script: data checks, EDA, pipelines, CV, tuning, evaluation, saving
predict.py                 batch predictions with the saved pipeline
app.py                     Streamlit dashboard
requirements.txt
```

---

## 1. Problem framing

**Problem.** Estimate the weekly sales of each department in each Walmart store, one week at a time.

**Forecasting objective.** For a given `Store`, `Dept` and sales week, predict `Weekly_Sales`
(the target). The prediction level is Store × Department × Week; the training data has 3,331 store-department
series, 45 stores, 81 departments and 143 weeks (2010-02-05 to 2012-10-26).

**Potential users and decisions** (these are plausible uses, not facts about Walmart's practice):
store or regional planners deciding inventory and replenishment quantities, staff planners estimating
workload by department, and analysts comparing expected with actual sales to spot anomalies.

**Why these metrics.**
* **MAE** is the average miss in the same unit as sales; it is easy to explain and robust to a few huge errors.
  It is the **primary tuning metric** here.
* **RMSE** squares errors, so it punishes large misses; comparing it with MAE shows how heavy-tailed the errors are.
* **R²** says how much of the variation in sales the model explains compared with always predicting the mean.
  It can look high for this data because sales differ enormously between departments and stores; it should
  not be read alone (the mean baseline has R² ≈ 0 and a much larger MAE, which gives context).

The currency is not stated in the data files; the charts label it USD as an assumption.

## 2. Data understanding

**Source.** The four files match the schema and sizes of the public Kaggle competition *Walmart Recruiting –
Store Sales Forecasting* (421,570 training rows, 115,064 test rows, 45 stores). The project folder has no record of
where the files were downloaded, so this is an inference, not a verified provenance.

**Column dictionary** (descriptions follow the public dataset description; "used" refers to this project's model).

| Column | File(s) | Meaning | Used in model |
|---|---|---|---|
| `Store` | all | Store id (1–45) | Yes, one-hot |
| `Dept` | train, test | Department id (81 distinct) | Yes, one-hot |
| `Date` | all | Week date (always a Friday; the **last day** of the week, see below) | Only to derive `year`, `month`, `week` and to split by time |
| `Weekly_Sales` | train | **Target**: sales of that Store/Dept/week | Target |
| `IsHoliday` | all | Special-holiday week flag (Super Bowl, Labor Day, Thanksgiving, Christmas) | Yes |
| `Temperature` | features | Regional average temperature | **No** (not known before the week) |
| `Fuel_Price` | features | Regional fuel cost | **No** (not known before the week) |
| `MarkDown1`–`MarkDown5` | features | Anonymised promotional markdown data | **No** (only 35.7%–49.5% non-null, first value on 2011-11-11) |
| `CPI` | features | Consumer price index | Yes, **availability unaudited** |
| `Unemployment` | features | Unemployment rate | Yes, **availability unaudited** |
| `Type` | stores | Store type A/B/C | Yes, as `Type_A/B/C` |
| `Size` | stores | Store size | Yes |
| `year`, `month`, `week` | derived | From `Date`; `week` is the **ISO week** of the Friday date | Yes |

**Weekly date convention (verified).** Every date in train/test/features is a Friday, exactly 7 days apart.
All ten holidays flagged in the data fall in the 6 days *before* the flagged Friday (e.g. Thanksgiving Thu 2010-11-25 is
flagged on Fri 2010-11-26), so a sales week runs **Saturday → Friday** and the Friday is its closing date.
The dashboard maps any selected day to its week's closing Friday.

**Integrity checks after merging** (run in the script; any failure raises an error):
* No duplicate keys: `features (Store, Date)`, `stores (Store)`, `train` and `test (Store, Dept, Date)` all have 0 duplicates.
* Every `(Store, Date)` pair of train and test exists in `features.csv` (0 missing); every Store exists in `stores.csv`.
* Merges use `validate="many_to_one"`; row counts are unchanged: train 421,570 → 421,570, test 115,064 → 115,064.
* `IsHoliday` is identical in train/test and in `features.csv`.
* Nothing was silently dropped: the MarkDown columns are removed on purpose (reason above), and
  `Temperature`/`Fuel_Price` are removed before modeling (reason below). CPI/Unemployment are missing for 585 weekly
  feature rows (2013-05-03 to 2013-07-26), which affects 38,162 test rows and **no** training rows.

## 3. Exploratory data analysis

The existing analyses (shape/dtypes/missing/duplicates, descriptive statistics, skewness, IQR outliers, sales by
holiday / store type / department / year / month, correlations) are preserved. Charts are in `reports/figures/`:
`01` distribution, `02` weekly sales over time, `03` by month, `04` by holiday, `05` by store type,
`06` top departments, `07` correlation heatmap.

**Insights (all computed from the data):**
1. `Weekly_Sales` is strongly right-skewed (skewness 3.26; mean $15,981 vs median $7,612), so a few very large
   values dominate RMSE.
2. Holiday weeks average $17,036 vs $15,901 for regular weeks (+7.1%), but this rests on only 10 distinct holiday weeks.
3. Strong seasonality: month 12 has the highest average ($19,356) and month 1 the lowest ($14,126).
   This is consistent with seasonal demand, but the data cannot confirm the cause.
4. Store type matters: Type A $20,100, Type B $12,237, Type C $9,520 average weekly sales per record.
5. Sales are concentrated: departments 92, 95, 38, 72, 90 account for 28.5% of all sales, and the one-hot
   department columns carry 67% of the final model's feature importance.

**Outliers.** The IQR rule flags 35,521 rows (8.4%), all above the upper bound (47,395). They are **not removed**:
51.6% of them sit in just five departments (38, 95, 92, 90, 40), 93.8% belong to a Store/Dept/week combination that is also an outlier
in at least two different years (repeating peaks, not one-off glitches), and they are only mildly over-represented in
holiday weeks (7.4% vs 7.0%) and November–December (15.7% vs 12.7%). This points to legitimately large
departments and seasonal peaks, although it cannot prove each individual value is correct. Removing them would train the model to
under-predict the largest sales. 1,285 rows have *negative* sales (likely returns/adjustments); they are real recorded values and are kept.

## 4. Data preprocessing

All preprocessing lives **inside a scikit-learn `Pipeline`** (`ColumnTransformer` + estimator), so it is
fitted only on whichever rows are passed to `fit()`: each CV training fold, the development set, or the full labeled set.

| Columns | Linear models | Tree models |
|---|---|---|
| `CPI`, `Unemployment` | median `SimpleImputer` + `StandardScaler` | median `SimpleImputer` |
| `IsHoliday`, `Size`, `Type_A/B/C`, `year`, `month`, `week` | `StandardScaler` | passed through |
| `Store`, `Dept` | `OneHotEncoder(handle_unknown="ignore")` | same |

* Unknown stores/departments at inference time are encoded as all zeros instead of crashing; the app and
  `predict.py` additionally refuse unseen categories explicitly.
* Verified independently: in CV fold 1 the imputer medians equal the fold-1 *training* medians (CPI 181.677, Unemployment 8.211),
  not the full-development medians (182.163 / 7.962).
* The `Type` → `Type_A/B/C` dummy encoding is a fixed store-level mapping done once before the pipeline; it learns nothing from sales.
* Feature exclusions are preserved: `MarkDown1–5`, `Temperature` and `Fuel_Price` never reach any pipeline.

**CPI and Unemployment — availability audit (unresolved).** The files contain no publication timestamps, so it is **impossible to
verify** that the value attached to a week was known before that week began. Official macroeconomic figures are
typically released after the period they describe. What the data shows: values exist up to 2013-04-26 only; CPI differs
across stores at the same date (median 15 distinct values across stores per week), and within a store it changes
gradually week to week. **Ablation (same CV folds, selected model):** CV MAE is 2,822.41 *with* these two features and **2,662.38
without** them (−5.7%). So they are unverified *and* they did not help on the development folds.
The features were kept because your instructions say to preserve existing decisions unless the evidence is clear; the
ablation is one model on four folds, and it is a strong signal that deserves a follow-up (see Remaining issues).

## 5. Modeling

Four algorithms are compared on the same chronological development split with the same metrics and `random_state=42`:
the mean baseline (`DummyRegressor`), `LinearRegression` (scaled), `DecisionTreeRegressor` and `RandomForestRegressor`.
The table at the top shows the results. The Random Forest was **not assumed** to win; it was simply the best result.
Your original fixed tree (`min_samples_leaf=10`, no depth limit) is kept as a labeled reference row.

## 6. Hyperparameter tuning and cross-validation

* **Date-grouped time-series folds.** `TimeSeriesSplit(n_splits=4)` is applied to the **sorted unique dates** of the
  development period and mapped back to row positions, so all Store/Dept rows of one date stay together.
  Three assertions guard every fold (train strictly earlier than test; no date on both sides; no test date reused).

| Fold | Train weeks | Train ends | Test period | Test weeks |
|---|---:|---|---|---:|
| 1 | 26 | 2010-07-30 | 2010-08-06 → 2010-12-31 | 22 |
| 2 | 48 | 2010-12-31 | 2011-01-07 → 2011-06-03 | 22 |
| 3 | 70 | 2011-06-03 | 2011-06-10 → 2011-11-04 | 22 |
| 4 | 92 | 2011-11-04 | 2011-11-11 → 2012-04-06 | 22 |

* **Search.** Decision Tree: `GridSearchCV` over `min_samples_leaf` ∈ {2, 5, 10, 20, 40} × `max_depth` ∈ {None, 25} (10 candidates).
  Random Forest: `RandomizedSearchCV`, 4 sampled candidates from `min_samples_leaf` ∈ {5, 10, 20} × `max_features` ∈ {0.33, 0.5} × `max_depth` ∈ {None, 25},
  with 50 trees each grown on a 50% bootstrap sample (kept small so it runs on one CPU core). Primary metric: MAE (RMSE and R² are also recorded).
* **Results.** Decision Tree best: `max_depth=None, min_samples_leaf=5` (CV MAE 2,888.66; CV MAE std 692.97). Limiting depth to 25 is far worse (CV MAE ≈ 5,300).
  Random Forest best: `min_samples_leaf=5, max_features=0.33, max_depth=None` (CV MAE 2,822.41). Full tables: `reports/cv_results_*.csv`.
* **Honesty notes on the search.** (1) My first Decision Tree grid had its optimum on the lowest edge (`min_samples_leaf=5`), so I extended it downward
  to 2; the optimum is now interior. (2) The Random Forest search is thin: 3 of the 4 random candidates used `max_depth=25`, which is clearly
  worse, so only one strong candidate was really compared, and its `min_samples_leaf=5` is the lowest value in that space.
  A more thorough search could improve the forest; the current result should be called "lightly tuned".

## 7. Model evaluation

Three different things are reported and must not be mixed up:

| Stage | Data | Used for |
|---|---|---|
| **Cross-validation** | 4 expanding folds inside the **development** weeks (2010-02-05 → 2012-04-06) | Tuning and model *selection* |
| **Holdout evaluation** | Later, untouched weeks (2012-04-13 → 2012-10-26; 29 weeks, 85,809 rows), models fitted on development weeks only | The reported accuracy |
| **Final deployment model** | The selected pipeline **refitted on all** 421,570 labeled rows (→ 2012-10-26) | The saved `walmart_model.pkl` |

Because the deployed model is refitted on all labeled data, **no untouched data remains to score it**; the holdout figures describe
the development-fitted version. `test.csv` has no `Weekly_Sales`, so it is used only to produce predictions, never to compute a metric.
In the script, the holdout is split off (line ~554) but first scored after the selection step; all tuning `.fit` calls use only the development data
(verified by inspecting the script order).

**Selected model:** Random Forest (tuned), holdout **MAE 1,883.41, RMSE 3,755.62, R² 0.9707**; train R² on development data 0.9672
(no large overfitting gap, unlike the Decision Tree's 0.9817).

**Error analysis (holdout; charts `08`–`12`, tables in `reports/error_by_*.csv`):**
* Mean weekly sales per record is $15,781, so MAE is ≈ 11.9% of that; the **median** absolute error is $794, so most predictions are much closer than the mean suggests while a minority of large misses drives RMSE ($3,756).
* Summed to **store-week totals**, MAE is $56,386 = 5.4% of mean store-week sales (individual errors partly cancel).
* Holiday vs regular week: MAE $2,265 (13.9% of mean) vs $1,870 (11.9%). The holdout contains only a single holiday week (2,966 records), so this is weak evidence.
* By month the MAE is fairly even ($1,781–$1,990); April has the largest negative bias (−$857, i.e. over-prediction).
* Hardest stores: 14 (MAE $4,458, 18.0% of its mean), 20, 10, 23, 28. Hardest departments by MAE: 65 (only 29 records, so unreliable), 92, 38, 72, 95 (mostly the large departments).

**Note on reproducibility of your earlier numbers.** Your original fixed tree previously reported validation MAE 1,959.26 / RMSE 4,132.26 / R² 0.9646.
Re-run in this restructured pipeline it gives 1,962.81 / 4,161.92 / 0.9641. The difference is small (0.2% in MAE) and I did not isolate its cause; likely candidates are the new
column order and the now deterministic row order (ties on date), which change tie-breaking inside the tree.

## 8. Business insights and limitations

**What the evidence supports.**
* Among the models compared, the Random Forest had the lowest error on both CV and the holdout, and every tree model beat Linear Regression by a wide margin
  (holdout MAE 1,883–1,985 vs 8,038), so the Store/Dept identity and seasonality are best captured by non-linear models.
* The typical (median) department-week error is ≈ $794 and the average error ≈ $1,883 on weeks around $15.8k, and 5.4% at store-week level. Such an error could give
  a planner a starting estimate for inventory or staffing by department, but this project does not test whether that estimate is *good enough* for any real decision.
* MAE is the average miss in dollars; RMSE ($3,756) being about twice MAE says that a minority of weeks are missed by much more, typically in
  high-volume departments, so a planner should not treat the average error as the worst case.

**Limitations (do not skip these).**
1. **Not shown to be suitable for true pre-week forecasting.** CPI and Unemployment availability before the forecast week is **unverified** (section 4), and removing them lowered CV error.
   Temperature and Fuel_Price were excluded for the same availability reason.
2. **No sales history is used.** The model has no lag or rolling-sales features; it knows store, department, size/type, ISO week/month/year, holiday flag and the two economic indicators.
3. **The `year` feature cannot extrapolate** beyond 2010–2012; later years behave like the latest year seen.
4. **Few holiday events** (10 holiday weeks) support the holiday effect; `IsHoliday` has ≈0.3% feature importance in the final forest, because week-of-year already captures it.
5. **About 2.75 years of data** (143 weeks) cover each season only two or three times; the CV folds with short training windows show how much weaker the model is early on.
6. **Dollar scale and currency** are assumed; **no external business facts** about Walmart were used.
7. **The Random Forest search was small** and the 50-tree forest was chosen for compute reasons.
8. **No production readiness claim:** no monitoring, drift check, prospective test or deployment exists.

## 9. Web application

`app.py` keeps the dashboard's behavior (live forecast without a button, week-ending-Friday logic, read-only CPI/Unemployment taken from `features.csv`,
historical chart and statistics, light theme that stays readable in dark mode) and was updated only where the new artifact required:
* It loads the version-2 artifact (a full pipeline) and builds the exact 12 raw input columns saved in `feature_columns`; it refuses unseen stores/departments.
* New **Model quality** cards show the holdout MAE, RMSE, R², relative error and improvement over the baseline, read from the artifact.
* The **Model details and limitations** panel states the evaluation design, the CPI/Unemployment availability problem, the missing sales history, and that the model is not production-validated.
* It shows a warning if the installed scikit-learn version differs from the one that created the model.

**Verified in this environment:** the app starts with `python -m streamlit run app.py`; the page loads without exceptions; changing store, department or date updates
the forecast; no excluded feature is required; and on 450 sampled test rows (150 with no recorded CPI) the app's predictions equal those of `predict.py`
with 0 mismatches. **Not verified:** deployment to any hosting service, behavior on other operating systems or other library versions.

## Reproducibility

`random_state=42` everywhere, deterministic row ordering (`Date, Store, Dept`), pinned scikit-learn version in `requirements.txt`.
The saved model is a pickle (~52 MB compressed), so load it with the same scikit-learn version that trained it (1.9.1).
