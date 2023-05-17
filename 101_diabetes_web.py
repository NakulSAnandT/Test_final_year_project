import numpy as np
import pickle
import streamlit as st
import os
# Set page configuration
st.set_page_config(
    page_title="DIABETES PREDICTION WEB APPLICATION",
    page_icon=":bar_chart:",
    layout="centered",
    initial_sidebar_state="expanded",
)


# Apply custom styles
st.markdown(
    """
    <style>
    body {
        background-color: #000000;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Display image
image = "C:/Users/nakul/Downloads/images.jpeg"  # Replace with the path to your image file
st.image(image, width = 600)

# Load the saved model and scaler

deployment_environment = "render"  # Replace this with the appropriate deployment environment

if deployment_environment == "local":
    file_path = "C:/Users/nakul/OneDrive/Desktop/Hustle!/Diabetes_model.sav"
elif deployment_environment == "render":
    file_path = "/app/Diabetes_model.sav"
else:
    file_path = "default/file/path"  # Set a default file path if none of the environments match

data = pickle.load(open(file_path, 'rb'))

#data = pickle.load(open("C:/Users/nakul/OneDrive/Desktop/Hustle!/Diabetes_model.sav", 'rb'))
loaded_model = data['model']
scaler = data['scaler']

# Creating a function for diabetes prediction
def diab_pred(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Apply scaling to the input data using the loaded scaler
    input_data_scaled = scaler.transform(input_data_reshaped)

    # Make predictions using the loaded model
    prediction = loaded_model.predict(input_data_scaled)
    if prediction[0] == 0:
        return "<span style='color:green;font-weight:bold;'>NOT DIABETIC!</span>"
    else:
        return "<span style='color:red;font-weight:bold;'>DIABETIC!</span> ,Please visit a doctor soon."

def main():
    # Title
    st.title('DIABETES PREDICTION WEB APPLICATION ')

    # User gender selection
    gender = st.radio("Select your gender:", ('Male', 'Female'))

    # Getting the input from the user
    Pregnancies = st.text_input("Enter the Number of Pregnancies you had: ")
    Glucose = st.text_input("Enter Your Glucose Level: ")
    BloodPressure = st.text_input("Enter your Blood Pressure level: ")
    SkinThickness = st.text_input("Enter your skin thickness: ")
    Insulin = st.text_input("Enter your insulin level: ")
    BMI = st.text_input("Enter your BMI Level: ")
    DiabetesPedigreeFunction = st.text_input("Enter Diabetes pedigree fn: ")
    Age = st.text_input("Please enter your age: ")

    diagnosis = ''
    result_shown = False  # Track if result button clicked
    
    # Creating a button for prediction
    if st.button('Show result'):
        result_shown = True
        # Check if any input field is empty
        if '' in [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]:
            st.error("Please fill in all the input fields.")
        else:
            diagnosis = diab_pred([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
            st.markdown(f"You are {diagnosis}.", unsafe_allow_html=True)
    

if __name__ == '__main__':
    main()
