from streamlit_option_menu import option_menu
from joblib import load
import pandas as pd
import numpy as np
import streamlit as st
import utilities as utils
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Setting the page icon and name
st.set_page_config(
    page_title='Fast Health Checkup',
    page_icon='assets/icon.png'
)

# Setting the background cover of the page
utils.add_bg_from_local('assets/resize_cover.png')


# Load the ML model in
with st.spinner('Loading Model...'):
    heart_pred_model = load('heart_pred_model.joblib')

# Load our training data here and calculate the average values of each attributes
data = pd.read_csv('heart.csv')

# Load the saved scaler model in
std_scaler = load('std_scaler.joblib')

healthy_avg = data[data['output'] == 0].mean()
diseased_avg = data[data['output'] == 1].mean()


with st.sidebar:
    st.title('Fast Health Checkup')
    st.markdown('This app uses machine learning to predict heart disease. \n')

    selected = option_menu('Pages',
                          ['Heart Disease Prediction', 'About', 'How To Fill', 'Contact'],
                           menu_icon= ['hospital', 'information', 'eyeglasses', 'envelope'],
                           icons= ['balloon-heart', 'info-square', 'eyeglasses', 'envelope'],
                           default_index = 0)
    st.markdown(
        'Please fill in the information in the main panel.')

if (selected == 'Heart Disease Prediction'):

    st.title("Fast Health Checkup: Heart Disease Prediction")

    # This function will check if the input value is in the right format or not
    def validate_input(input_value, input_type, input_name):
        if input_value:
            try:
                if input_type == int:
                    return int(input_value)
                elif input_type == float:
                    return float(input_value)
            except ValueError:
                st.error(f'{input_name} must be a number.')
        return None

    age = validate_input(st.text_input('Age'), int, 'Age')
    sex = validate_input(st.text_input('Sex (0: Male, 1: Female)'), int, 'Sex')
    cp = validate_input(st.text_input('Chest Pain Type (0: Typical, 1: Atypical, 2: Non-anginal, 3: Asymptomatic)'),
                        int, 'Chest Pain Type')
    trtbps = validate_input(st.text_input('Resting Blood Pressure (mg Hg)'), int, 'Resting Blood Pressure')
    chol = validate_input(st.text_input('Serum Cholesterol (mg/dl)'), int, 'Serum Cholesterol')
    fbs = validate_input(st.text_input('Fasting Blood Sugar >120mg/dl (0: No, 1: Yes)'), int, 'Fasting Blood Sugar')
    restecg = validate_input(st.text_input('Resting ECG (0: Normal, 1: Abnormal)'), int, 'Resting ECG')
    thalachh = validate_input(st.text_input('Maximum Heart Rate'), int, 'Maximum Heart Rate')
    exng = validate_input(st.text_input('Exercised-induced Angina (0: No, 1: Yes)'), int, 'Exercised-induced Angina')
    oldpeak = validate_input(st.text_input('ST Depression Induced'), float, 'ST Depression Induced')
    slp = validate_input(st.text_input('ST Segment Slope (0: Unsloping, 1: Flat, 2: Downsloping)'), int,
                         'ST Segment Slope')
    caa = validate_input(st.text_input('Major Vessels Quantity'), int, 'Major Vessels Quantity')
    thall = validate_input(st.text_input('Thallium Heart Rate (0: Normal, 1: Fixed Defect, 2: Reversible Defect)'), int,
                           'Thallium Heart Rate')

    diagnosis = ' '

    # creating a button for Prediction
    if st.button('Check'):

        # First, I will make sure none of the inputs are empty.
        if all(i is not None and
               str(i).isdigit() for i in [age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, slp, caa, thall]) and '.' in str(oldpeak):
            # Convert strings to appropriate numeric type
            age = int(age)
            sex = int(sex)
            cp = int(cp)
            trtbps = int(trtbps)
            chol = int(chol)
            fbs = int(fbs)
            restecg = int(restecg)
            thalachh = int(thalachh)
            exng = int(exng)
            oldpeak = float(oldpeak)
            slp = int(slp)
            caa = int(caa)
            thall = int(thall)

            # Create a DataFrame from the inputs so that it is represented as the trained data's format
            input_df = pd.DataFrame(
                data=[[age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall]],
                columns=['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh', 'exng', 'oldpeak', 'slp',
                         'caa', 'thall'])

            scaled_input = std_scaler.transform(input_df)

            heart_prediction = heart_pred_model.predict(input_df)

            if (heart_prediction[0] == 1):
                diagnosis = 'You have signs of heart diseases. Please contact your GP as soon as possible.'
            else:
                diagnosis = 'Congratulations! You have a healthy heart.'
            display_chart = True

        else:
            diagnosis = 'Please fill in all fields correctly.'
            display_chart = False

        if display_chart:
            labels = ['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh', 'exng', 'oldpeak', 'slp',
                             'caa', 'thall']
            user_values = [age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall]
            healthy_values = [healthy_avg[col] for col in labels]
            diseased_values = [diseased_avg[col] for col in labels]

            x = np.arange(len(labels))
            width = 0.2

            fig, ax = plt.subplots(figsize=(12, 8))
            rects1 = ax.bar(x - width, user_values, width, label='User')
            rects2 = ax.bar(x, healthy_values, width, label='Healthy Avg')
            rects3 = ax.bar(x + width, diseased_values, width, label='Diseased Avg')

            ax.set_ylabel('Values')
            ax.set_title('User values vs Average values')
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.legend()

            fig.tight_layout()
            st.pyplot(fig)

    st.success(diagnosis)

if (selected == 'About'):
    st.markdown("""<div style="background-color:LightYellow;color:black;padding:20px">
    <h2 style ='color:black'>About this App</h2>
    <p>
    This application is part of a personal project exploring the vast capabilities 
    of machine learning models in the realm of disease prediction by Uyen Ng. The primary objective 
    of this project is to enable early detection of diseases, thereby leading to 
    timely and more effective treatment strategies. 
    </p><p>
    The current model focuses on heart disease prediction. It employs a machine learning 
    model that has been trained on the renowned UCI Heart Disease dataset. The prediction 
    outputs provided by the model are based on a series of health parameters, which 
    have been determined through rigorous data analysis and feature selection techniques.
    </p><p>
    If this model proves to be successful, it will serve as a stepping stone for the 
    development of more advanced and professionally constructed models. Such models could 
    potentially be equipped to predict a wide range of diseases, significantly revolutionising 
    the healthcare sector. The overarching aim is to reduce diagnosis time, lower healthcare 
    costs, and ultimately provide patients with a more streamlined and personalised healthcare 
    experience. 
    </p><p>
    As this project is in its early stages, any feedback or suggestions for improvements 
    are highly welcome.
    </p>
    </div>""", unsafe_allow_html=True)

if (selected == 'How To Fill'):
    st.markdown("""<div style="background-color:LightYellow;color:black;padding:20px">
    <h2 style ='color:black'>All You Need To Know</h2>
    <p>
    Age: Your age in years 
    </p><p>
    Sex: Write '0' if you are biological male, and '1' if you are biological female
    </p><p>
    Chest Pain Type (cp): This refers to the type of chest pain you have experienced. Please input '1' for typical angina (chest pain related to decreased blood supply to the heart), '2' for atypical angina (chest pain not related to the heart), '3' for non-anginal pain (typically sharp pain unrelated to the heart), or '4' for asymptomatic (no pain).
    </p><p>
    Resting Blood Pressure (trtbps): Your usual blood pressure reading. This typically includes two or three numbers.
    </p><p>
    Serum Cholesterol (chol): This is the measurement of certain fats in your blood. Please input your latest cholesterol measurement if you have it. The unit for this measurement is mg/dl.
    </p><p>
    Fasting Blood Sugar (fbs): Please write '1' if your fasting blood sugar level is greater than 120 mg/dl, and '0' if it is less than that.
    </p><p>
    Resting ECG (restecg): This measures your heart's electrical activity. Please input '0' if you have a normal reading, '1' if you have an ST-T wave abnormality, or '2' if you show probable or definite left ventricular hypertrophy if you have the information. 
    </p><p>
    Maximum Heart Rate (thalachh): Please input the maximum rate your heart has achieved during peak exercise (This can easily be obtained from your smartwatches).
    </p><p>
    Exercise Induced Angina (exng): Please write '1' if you have experienced angina (chest pain) during exercise, and '0' if you have not.
    </p><p>
    ST Depression Induced (oldpeak): This is the difference in parts of your electrocardiogram readings before and after you exercise. Please input the difference in millimeters here.
    </p><p>
    ST Segment Slope (slp): This is the slope of the peak exercise ST segment in your electrocardiogram. Please input '0' for an upsloping, '1' for a flat slope, and '2' for a downsloping.
    </p><p>
    Major Vessels Quantity (caa): This is the number of major vessels colored by fluoroscopy, a type of X-ray. Please input a number from '0' to '3'.
    </p><p>
    Thallium Heart Rate (thall): This is a type of blood disorder. Please input '1' if you have a normal form, '2' if you have a fixed defect, or '3' if you have a reversible defect.
    </p><p>
    Please fill in the latest result from your recent health check result, otherwise, you can obtain these information from your GP
    </p>
    </div>""", unsafe_allow_html=True)

if (selected == 'Contact'):
    st.markdown("""
    <style>
    .big-font {
        font-size:30px !important;
        color:black;
        text-align: center;
    }
    .contact-details {
        font-size:20px !important;
        color:black;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background-color:LightYellow;color:black;padding:20px">
    <p class="big-font">Contact Information</p>
    <p class="contact-details">Email: unguye01@student.bbk.ac.uk</p>
    </div>""", unsafe_allow_html=True)
