import warnings  # Step 1: hide a known library warning when the saved model loads.
from pathlib import Path  # Step 2: build file paths that work on any computer.

import joblib  # Step 3: load the saved model and scaler files.
import pandas as pd  # Step 4: hold one input row in a table the scaler understands.
import streamlit as st  # Step 5: draw the web page widgets.
from sklearn.exceptions import InconsistentVersionWarning  # Step 6: name the sklearn version warning.

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)  # Step 7: the model was saved with sklearn 1.6.1; this install is newer, and the warning is expected.

ROOT = Path(__file__).parent  # Step 8: folder that contains this file, the model, the scaler, and the csv.
FEATURES = [  # Step 9: column order the model was trained on. Do not reorder these.
    "sex",  # 0 = female, 1 = male.
    "smoker",  # 0 = no, 1 = yes.
    "children",
    "number_of_past_hospitalizations",
    "claim_amount",
    "past_consultations",
    "hospital_expenditure",
    "annual_salary",
]


def load_artifacts():  # Step 10: read both saved files once per prediction call.
    model = joblib.load(ROOT / "linear_regression_model.joblib")  # Step 11: the linear regression. It expects scaled numbers, not raw ones.
    scaler = joblib.load(ROOT / "scaler.joblib")  # Step 12: mean and spread from the 80% train split (random_state 42).
    return model, scaler  # Step 13: hand both objects back to the caller.


def predict_charges(model, scaler, values):  # Step 14: turn the form answers into one charge estimate.
    row = pd.DataFrame([[values[name] for name in FEATURES]], columns=FEATURES)  # Step 15: one row, columns in FEATURES order.
    scaled = scaler.transform(row)  # Step 16: subtract the training mean and divide by the training spread.
    return float(model.predict(scaled)[0])  # Step 17: model returns a one-item list; take that number.


def main():  # Step 18: build the page. Streamlit reruns this function after every widget change.
    st.set_page_config(page_title="Insurance charges", layout="centered")  # Step 19: browser tab title and a narrow layout.
    st.title("Insurance charge estimate")  # Step 20: page heading.
    st.caption(  # Step 21: short note under the heading about how the inputs are coded.
        "Linear regression trained on the preprocessed insurance table. "
        "Sex and smoker use the training codes: female/no = 0, male/yes = 1. "
        "Inputs are standardized with the scaler fit on the 80% train split (random_state 42) before predict."
    )

    model, scaler = load_artifacts()  # Step 22: load the model and scaler before reading the form.
    sex = st.selectbox("Sex", ["female", "male"])  # Step 23: two labels; mapped to 0 and 1 below.
    smoker = st.selectbox("Smoker", ["no", "yes"])  # Step 24: same coding as training: no = 0, yes = 1.
    children = st.number_input("Children", min_value=0, max_value=5, value=1, step=1)  # Step 25: integer count, default is the training median.
    # shortcut: training rows only have 0 or 1 past hospitalizations; raise the max if you refit
    hospitalizations = st.number_input(  # Step 26: the form allows 0–3, but the fit only saw 0 and 1.
        "Past hospitalizations", min_value=0, max_value=3, value=1, step=1
    )
    claim_amount = st.number_input("Claim amount", min_value=0.0, value=30819.0, step=100.0)  # Step 27: default is the training median claim.
    past_consultations = st.number_input(  # Step 28: visit count. Default 13 is the training median.
        "Past consultations", min_value=0, max_value=40, value=13, step=1
    )
    hospital_expenditure = st.number_input(  # Step 29: money spent. Default is the training median.
        "Hospital expenditure", min_value=0.0, value=6012943.0, step=1000.0
    )
    annual_salary = st.number_input(  # Step 30: this column moves the estimate the most after scaling.
        "Annual salary", min_value=0.0, value=108118300.0, step=1000.0
    )

    values = {  # Step 31: one dict the predictor can read by column name.
        "sex": 1 if sex == "male" else 0,  # Step 32: match LabelEncoder: female 0, male 1.
        "smoker": 1 if smoker == "yes" else 0,  # Step 33: match LabelEncoder: no 0, yes 1.
        "children": children,
        "number_of_past_hospitalizations": hospitalizations,
        "claim_amount": claim_amount,
        "past_consultations": past_consultations,
        "hospital_expenditure": hospital_expenditure,
        "annual_salary": annual_salary,
    }
    estimate = predict_charges(model, scaler, values)  # Step 34: scale the row, then predict.
    st.metric("Estimated charges", f"{estimate:,.2f}")  # Step 35: show the number with thousands separators.

    with st.expander("Dataset"):  # Step 36: extra panel; click to open the source table.
        data = pd.read_csv(ROOT / "new_insurance_data.csv")  # Step 37: raw file. It still has columns the model does not use.
        st.write(f"{len(data)} rows, {len(data.columns)} columns")  # Step 38: size of that file.
        st.dataframe(data.head(20), use_container_width=True)  # Step 39: first 20 rows only, wide enough for the page.


if __name__ == "__main__":  # Step 40: run the page when you start this file, not when check.py imports it.
    main()
