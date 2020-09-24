import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import catboost
import sklearn

file = open('model.joblib','rb')
model = joblib.load(file)

st.title('Insurance Prediction')
st.subheader('numbers of buildings that were insured')
st.write('Recently, there has been an increase in the number of building collapse in Lagos and major cities in Nigeria. Olusola Insurance Company offers a building insurance policy that protects buildings against damages that could be caused by a fire or vandalism, by a flood or storm.')

html_temp = """
    <div style ='background-color: orange; padding:10px'>
    <h2> Streamlit ML Web App </h2>
    </div>
    """
st.markdown(html_temp, unsafe_allow_html=True)

st.write('Please provide the following details for building insured')
Garden=st.selectbox('Garden',('0','1','2'))
Settlement = st.selectbox('Settlement',('0','1'))
Residential= st.selectbox('Residential',('0','1'))
Building_Type	 = st.selectbox('Building_Type	',('1','2'))
Building_Painted = st.selectbox('Building_Painted',('0','1'))
Insured_Period	= st.slider('Insured_Period	',1.0,1.1)
Building_Dimension= st.slider('Building Dimension',300,1405)
Building_Fenced_N = st.selectbox('Building_Fenced_N	',('0','1'))
Building_Fenced_V= st.selectbox('Building_Fenced_V',('0','1'))
Date_of_Occupancy=st.selectbox('Date_of_Occupancy',('0','1800.0'))	

features = {'Garden':Garden,
'Settlement':Settlement,
'Residential':Residential,
'Building_Painted':Building_Painted,
'Insured_Period':Insured_Period,
'Building_Dimension':Building_Dimension,
'Building_type':Building_Type,
'Building_Fenced_N':Building_Fenced_N,
'Building_Fenced_V':Building_Fenced_V,
'Date_of_Occupancy':Date_of_Occupancy,
}
if st.button('Submit'):
    data = pd.DataFrame(features,index=[0,1])
    st.write(data)

    prediction = model.predict(data)
    proba = model.predict_proba(data)[1]

    if prediction[0] == 0:
        st.error('Building without Claim')
    else:
        st.success('Building with Claim')

    proba_df = pd.DataFrame(proba,columns=['Probability'],index=['Claim','No Claim'])
    proba_df.plot(kind='barh')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()