import pandas as pd
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report , accuracy_score , confusion_matrix
import streamlit as st



df = pd.read_csv('liver_cirrhosis.csv')
df['Age'] = (df['Age'] / 365).astype('int')
df['N_Years'] = (df['N_Days'] / 365).astype('int')
df.drop('N_Days',axis=1,inplace=True)
df.head()
col_trans = make_column_transformer(
    (OneHotEncoder(drop='first'),['Status','Drug','Age','Sex','Ascites','Hepatomegaly','Spiders','Edema']),
    (StandardScaler(),['Age','Bilirubin','Cholesterol','Albumin','Copper','Alk_Phos','SGOT','Tryglicerides','Platelets','Prothrombin','N_Years']),
    remainder='passthrough'
)


pipeline=make_pipeline(col_trans,XGBClassifier())
X = df.drop('Stage',axis=1)
y= df['Stage']-1
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.05,random_state=42,stratify=y)
pipeline.fit(X_train,y_train)
y_pred=pipeline.predict(X_test)
y_pred_train=pipeline.predict(X_train)




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
Age = st.number_input('Age',help='age in years',step=1,value=1,min_value=1)

col1, col2, col3 = st.columns(3)
with col1:
    Sex = st.selectbox('Sex', ['M','F'],help='M (male) or F (female)')

with col2:
    ascites = st.selectbox('Ascites', ['Y','N'],help='presence of ascites N (No) or Y (Yes)')

with col3:
    hepatomegaly = st.selectbox('Hepatomegaly', ['Y','N'],help='presence of hepatomegaly N (No) or Y (Yes)')
    

col4, col5 =st.columns(2)
with col4:
    spiders = st.selectbox('Spiders', ['Y','N'],help='presence of spiders N (No) or Y (Yes)') 

with col5:
    edema = st.selectbox('Edema', ['Y','N','S'],help='presence of edema N (no edema and no diuretic therapy for edema), S (edema present without diuretics, or edema resolved by diuretics), or Y (edema despite diuretic therapy)')
    

col6, col7, col8= st.columns(3)
with col6:
    bilirubin = st.number_input('Bilirubin',help='serum bilirubin in [mg/dl]')

with col7:
    cholesterol = st.number_input('Cholesterol',help='serum cholesterol in [mg/dl]')

with col8:
    albumin = st.number_input('Albumin',help='albumin in [gm/dl]')
  

col9, col10, col11= st.columns(3)
    
with col9:
   copper = st.number_input('Copper',help='urine copper in [ug/day]')
   
with col10:
    alk_phos = st.number_input('Alk_Phos',help='alkaline phosphatase in [U/liter]')

with col11:
    sgot = st.number_input('SGOT',help='SGOT in [U/ml]')
   

col12, col13 =st.columns(2)
with col12:
    tryglicerides = st.number_input('Tryglicerides',help='triglicerides in [mg/dl]')
    
with col13:
   platelets = st.number_input('Platelets',help='platelets per cubic [ml/1000]')


col14, col15 =st.columns(2)
with col14:
   prothrombin = st.number_input('Prothrombin',help='prothrombin time in seconds')
with col15:
    n_years = st.number_input('N_Years',help='Number of years between registration and the earlier of death, transplantation, or study analysis time in 1986')


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
    st.markdown(f"The model Predicted that the patient with provided data is in Stage <span class='blue-number'>{pipeline.predict(test)[0]+1}</span>", unsafe_allow_html=True)
    st.write(test)

if st.toggle('Show Test Data :relaxed:'):
    X_test_df = pd.read_csv('test_data.csv')
    st.write(X_test_df)
