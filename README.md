# Insurance charge estimate

Streamlit page that estimates medical charges with a saved linear regression. Inputs are standardized with `scaler.joblib` (fit on the 80% train split, `random_state` 42) before `linear_regression_model.joblib` predicts.

Sex and smoker use the training codes: female/no = 0, male/yes = 1. The page maps the labels for you.

## Run


```bash
pip install -r requirements.txt
streamlit run app.py
```

## Check

`check.py` reloads the same artifacts and predicts one known training row. It fails if the charge drifts from the number saved when the project was built.

```bash
python check.py
```

The model was saved with scikit-learn 1.6.1. A newer install prints `InconsistentVersionWarning`; `app.py` hides that warning.

## Deploy

The app is [Saavan-Dev/Insurance-charge-estimator](https://github.com/Saavan-Dev/Insurance-charge-estimator) on `main`. `app.py`, `requirements.txt`, both `.joblib` files, and the csv are already in that repo. No secrets file.

1. Open [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Create app. Repository `Saavan-Dev/Insurance-charge-estimator`, branch `main`, main file `app.py`.
3. Deploy. The URL is on the app page when the build finishes.

Push any later change to `main`; Cloud redeploys from that branch. Pin `scikit-learn==1.6.1` in `requirements.txt` before you care that the live number matches `check.py`.
