import streamlit as st
import pickle
import pandas as pd

# Load the model
with open('rf_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# You must also load the scaler separately, or skip scaling if not used.
# If you're using a scaler, load it like this:
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Define the prediction function
def predict_churn(input_data):
    input_data_scaled = scaler.transform([input_data])
    prediction = loaded_model.predict(input_data_scaled)
    return prediction[0]

# Streamlit app
st.title('Churn Prediction App')
st.sidebar.header('User Input Features')

def user_input_features():
    montant = st.sidebar.number_input('MONTANT', min_value=0.0, value=1000.0)
    frequence_rech = st.sidebar.number_input('FREQUENCE_RECH', min_value=0.0, value=1.0)
    revenue = st.sidebar.number_input('REVENUE', min_value=0.0, value=1000.0)
    arpu_segment = st.sidebar.number_input('ARPU_SEGMENT', min_value=0.0, value=333.0)
    frequence = st.sidebar.number_input('FREQUENCE', min_value=0.0, value=1.0)
    freq_top_pack = st.sidebar.number_input('FREQ_TOP_PACK', min_value=0.0, value=1.0)

    tenure_options = [
        'TENURE_E 6-9 month', 'TENURE_F 9-12 month', 'TENURE_G 12-15 month',
        'TENURE_H 15-18 month', 'TENURE_I 18-21 month',
        'TENURE_J 21-24 month', 'TENURE_K > 24 month'
    ]
    tenure_selected = st.sidebar.selectbox('Tenure', tenure_options)
    tenure_dict = {col: 0 for col in tenure_options}
    tenure_dict[tenure_selected] = 1

    data = {
        'MONTANT': montant,
        'FREQUENCE_RECH': frequence_rech,
        'REVENUE': revenue,
        'ARPU_SEGMENT': arpu_segment,
        'FREQUENCE': frequence,
        'FREQ_TOP_PACK': freq_top_pack,
    }
    data.update(tenure_dict)
    return pd.DataFrame([data])

input_data = user_input_features()
st.subheader('User Input Features')
st.write(input_data)

prediction = predict_churn(input_data.values[0])
st.subheader('Prediction')
st.write('Churn' if prediction == 1 else 'Not Churn')

st.subheader('Model Coefficients')
if hasattr(loaded_model, 'coef_'):
    coefficients = pd.DataFrame(loaded_model.coef_, input_data.columns, columns=['Coefficient'])
    st.write(coefficients)
st.subheader('Feature Importance')
if hasattr(loaded_model, 'feature_importances_'):
    feature_importance = pd.DataFrame(loaded_model.feature_importances_, input_data.columns, columns=['Importance'])
    st.write(feature_importance.sort_values(by='Importance', ascending=False))


# Add a download button for the model
if st.button('Download Model'):
    with open('rf_model.pkl', 'rb') as model_file:
        model_bytes = model_file.read()
    st.download_button('Download Random Forest Model', model_bytes, file_name='rf_model.pkl')

# Add a download button for the scaler


with open('accuracy.pkl', 'rb') as f:
    accuracy = pickle.load(f)
st.subheader('Model Accuracy')
st.write(f'Accuracy: {accuracy:.2f}')
if st.button('Download Scaler'):
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler_bytes = scaler_file.read()
    st.download_button('Download Scaler', scaler_bytes, file_name='scaler.pkl')



