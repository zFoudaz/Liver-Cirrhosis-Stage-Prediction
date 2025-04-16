import pandas as pd
import pickle
import streamlit as st

model=pickle.load(open('saved_pipeline.sav','rb'))

# Set the layout to wide
st.set_page_config(layout="centered",page_icon="🫀",page_title="Liver prob. detector")

st.title(':lab_coat: Liver Cirrhosis Stage Prediction')
st.caption('This is a web app to predict the stage of liver cirrhosis based on various features.\n')
st.markdown(
    """
    <style>
    .blue-number {
    padding: 0.2em 0.4em;
    overflow-wrap: break-word;
    white-space: pre-wrap;
    margin: 0px;
    border-radius: 0.25rem;
    background: rgb(237, 242, 246);
    color: rgb(0, 112, 224);
    font-family: "Source Code Pro", monospace;
    font-size: 0.75em;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(f"Train Accuracy: <span class='blue-number'>98.42</span>", unsafe_allow_html=True)
st.markdown(f"Validation Accuracy: <span class='blue-number'>95.52</span>", unsafe_allow_html=True)

st.divider()
st.write("""If you need help check --> :grey_question:""")

    
status = st.selectbox('Status', ['D','C','CL'],help='status of the patient C (censored), CL (censored due to liver tx), or D (death)')
drug = st.selectbox('Drug', ['Placebo','D-penicillamine'],help='type of drug D-penicillamine or placebo')
Age = st.number_input('Age',help='age in years',step=1,value=46,min_value=1)

col1, col2, col3 = st.columns(3)
with col1:
    Sex = st.selectbox('Sex', ['M','F'],help='M (male) or F (female)' ,index=1)

with col2:
    ascites = st.selectbox('Ascites', ['Y','N'],help='presence of ascites N (No) or Y (Yes)',index=1)

with col3:
    hepatomegaly = st.selectbox('Hepatomegaly', ['Y','N'],help='presence of hepatomegaly N (No) or Y (Yes)',index=0)
    

col4, col5 =st.columns(2)
with col4:
    spiders = st.selectbox('Spiders', ['Y','N'],help='presence of spiders N (No) or Y (Yes)',index=1) 

with col5:
    edema = st.selectbox('Edema', ['Y','N','S'],help='presence of edema N (no edema and no diuretic therapy for edema), S (edema present without diuretics, or edema resolved by diuretics), or Y (edema despite diuretic therapy)',index=1)
    

col6, col7, col8= st.columns(3)
with col6:
    bilirubin = st.number_input('Bilirubin',help='serum bilirubin in [mg/dl]',value=14.4)

with col7:
    cholesterol = st.number_input('Cholesterol',help='serum cholesterol in [mg/dl]',value=448)

with col8:
    albumin = st.number_input('Albumin',help='albumin in [gm/dl]',value=3.65)
  

col9, col10, col11= st.columns(3)
    
with col9:
   copper = st.number_input('Copper',help='urine copper in [ug/day]',value=34)
   
with col10:
    alk_phos = st.number_input('Alk_Phos',help='alkaline phosphatase in [U/liter]',value=1218)

with col11:
    sgot = st.number_input('SGOT',help='SGOT in [U/ml]',value=60.45)
   

col12, col13 =st.columns(2)
with col12:
    tryglicerides = st.number_input('Tryglicerides',help='triglicerides in [mg/dl]',value=318)
    
with col13:
   platelets = st.number_input('Platelets',help='platelets per cubic [ml/1000]',value=277)


col14, col15 =st.columns(2)
with col14:
   prothrombin = st.number_input('Prothrombin',help='prothrombin time in seconds',value=11)
with col15:
    n_years = st.number_input('N_Years',help='Number of years between registration and the earlier of death, transplantation, or study analysis time in 1986',value=6)


col = st.columns([1, 2, 1])[1]  # center column (wider)

with col:
    st.markdown("""
        <style>
        div.stButton > button {
            width: 100%;
        }
        </style>
    """, unsafe_allow_html=True)
    

if st.button(':nerd_face: Predict'):
    test=pd.DataFrame(
    {
        'Status':[status],
        'Drug':[drug],
        'Age':[Age],
        'Sex':[Sex],
        'Ascites':[ascites],
        'Hepatomegaly':[hepatomegaly],
        'Spiders':[spiders],
        'Edema':[edema],
        'Bilirubin':[bilirubin],
        'Cholesterol':[cholesterol],
        'Albumin':[albumin],
        'Copper':[copper],
        'Alk_Phos':[alk_phos],
        'SGOT':[sgot],
        'Tryglicerides':[tryglicerides],
        'Platelets':[platelets],
        'Prothrombin':[prothrombin],
        'N_Years':[n_years]
    }
    )
    st.markdown(f"The model Predicted that the patient with provided data is in Stage <span class='blue-number'>{model.predict(test)[0]+1}</span>", unsafe_allow_html=True)
    st.write(test)

if st.toggle('Show Test Data :relaxed:'):
    X_test_df = pd.read_csv('test_data.csv')
    st.write(X_test_df)

