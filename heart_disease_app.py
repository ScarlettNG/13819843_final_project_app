from streamlit_option_menu import option_menu
import pickle
import pandas as pd
import numpy as np
import streamlit as st
import utilities as utils
import matplotlib.pyplot as plt

# Setting the page icon and name
st.set_page_config(
    page_title='Fast Health Checkup',
    page_icon='assets/icon.png'
)

# Setting the background cover of the page
utils.add_bg_from_local('assets/resize_cover.png')


# Load the ML model in
with st.spinner('Loading Model...'):
    heart_pred_model = pickle.load(open('C:/Users/nguye/OneDrive/Desktop/Heart Disease App Code/heart_pred_model.sav', 'rb'))

# Load our training data here and calculate the average values of each attributes
data = pd.read_csv('C:/Users/nguye/OneDrive/Desktop/Heart Disease App Code/heart.csv')

healthy_avg = data[data['output'] == 0].mean()
diseased_avg = data[data['output'] == 1].mean()


with st.sidebar:
    st.title('Fast Health Checkup')
    st.markdown('This app uses machine learning to predict heart disease. \n')

    selected = option_menu('Pages',
                          ['Heart Disease Prediction', 'About', 'Contact'],
                           menu_icon= ['hospital', 'information', 'envelope'],
                           icons= ['balloon-heart', 'info-square', 'envelope'],
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
