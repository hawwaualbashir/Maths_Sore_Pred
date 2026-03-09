import streamlit as st 
import pandas as pd
import joblib


model = joblib.load("math_score_prediction_model.pkl")

model_columns = joblib.load("model_columns.pkl")

st.title("STUDENTS MATHS PERFORMANCE PREDICTION")


st.write("Pass and Fail Classification of Students Maths Score Using a Random Forest Model")


form = st.form("prediction_form")


form.subheader("Enter maths score")


gender = form.selectbox(
    "Gender",
    ["male","female"]
)

race = form.selectbox(
    "Race / Ethnicity",
    ["group A","group B","group C","group D","group E"]
)

parent_edu = form.selectbox(
    "Parental Level of Education",
    [
    "some high school",
    "high school",
    "some college",
    "associate's degree",
    "bachelor's degree",
    "master's degree"
    ]
)

lunch = form.selectbox(
    "Lunch Type",
    ["standard","free/reduced"]
)

prep = form.selectbox(
    "Test Preparation Course",
    ["none","completed"]
)



reading_score = form.number_input(
    "Reading Score",
    min_value=0,
    max_value=100,
    value=50
)

writing_score = form.number_input(
    "Writing Score",
    min_value=0,
    max_value=100,
    value=50
)


submit = form.form_submit_button("Predict")



if st.button("Predict"):

    input_data = pd.DataFrame({
        "gender":[gender],
        "race/ethnicity":[race],
        "parental level of education":[parent_edu],
        "lunch":[lunch],
        "test preparation course":[prep],
        "reading score":[reading_score],
        "writing score":[writing_score]
    })


    input_encoded = pd.get_dummies(input_data)

    input_encoded = input_encoded.reindex(columns=model_columns , fill_value=0)



    prediction = model.predict(input_encoded)

    result = "Pass" if prediction[0] == 1 else "Fail"

    st.success(f"Predicted result: {result}")





