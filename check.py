from app import load_artifacts, predict_charges  # Step 1: reuse the same load and predict functions the page uses.

# First training row of Preprocessed Charges.csv, split test_size=0.2, random_state=42.
ROW = {  # Step 2: one known training row, already label-encoded (sex 0, smoker 1).
    "sex": 0,  # female.
    "smoker": 1,  # yes.
    "children": 0,
    "number_of_past_hospitalizations": 1,
    "claim_amount": 53835.71459,
    "past_consultations": 13,
    "hospital_expenditure": 12510424.33,
    "annual_salary": 299786105.2,
}
EXPECTED = 16272.621685206874  # Step 3: charge the saved model produced for ROW when this project was built.

model, scaler = load_artifacts()  # Step 4: load the same two files app.py loads.
got = predict_charges(model, scaler, ROW)  # Step 5: scale ROW, then predict.
assert abs(got - EXPECTED) < 1e-6, (got, EXPECTED)  # Step 6: stop if the new prediction drifts from the saved one.
print("ok", got)  # Step 7: print the charge when the check passes.
