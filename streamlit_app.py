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



st.title('Liver Cirrhosis Stage Prediction')
st.write('This is a web app to predict the stage of liver cirrhosis based on various features.\n')
st.write('Train Accuracy:',round(accuracy_score(y_train,y_pred_train)*100,2))
st.write('Validation Accuracy:',round(accuracy_score(y_test,y_pred)*100,2))
st.divider()
st.write("""
Status: status of the patient C (censored), CL (censored due to liver tx), or D (death) \n
Drug: type of drug D-penicillamine or placebo \n
Age: age in years \n
Sex: M (male) or F (female) \n
Ascites: presence of ascites N (No) or Y (Yes) \n
Hepatomegaly: presence of hepatomegaly N (No) or Y (Yes) \n
Spiders: presence of spiders N (No) or Y (Yes) \n
Edema: presence of edema N (no edema and no diuretic therapy for edema), S (edema present without diuretics, or edema resolved by diuretics), or Y (edema despite diuretic therapy) \n
Bilirubin: serum bilirubin in [mg/dl]\n
Cholesterol: serum cholesterol in [mg/dl]\n
Albumin: albumin in [gm/dl]\n
Copper: urine copper in [ug/day]\n
Alk_Phos: alkaline phosphatase in [U/liter]\n
SGOT: SGOT in [U/ml]\n
Tryglicerides: triglicerides in [mg/dl]\n
Platelets: platelets per cubic [ml/1000]\n
Prothrombin: prothrombin time in seconds [s]\n
N_Years: Number of years between registration and the earlier of death, transplantation, or study analysis time in 1986\n   
""")

status = st.selectbox('Status', ['D','C','CL'])
drug = st.selectbox('Drug', ['Placebo','D-penicillamine'])
Age = st.number_input('Age')
Sex = st.selectbox('Sex', ['M','F'])
ascites = st.selectbox('Ascites', ['Y','N'])
hepatomegaly = st.selectbox('Hepatomegaly', ['Y','N'])
spiders = st.selectbox('Spiders', ['Y','N'])
edema = st.selectbox('Edema', ['Y','N','S'])
bilirubin = st.number_input('Bilirubin')
cholesterol = st.number_input('Cholesterol')
albumin = st.number_input('Albumin')
copper = st.number_input('Copper')
alk_phos = st.number_input('Alk_Phos')
sgot = st.number_input('SGOT')
tryglicerides = st.number_input('Tryglicerides')
platelets = st.number_input('Platelets')
prothrombin = st.number_input('Prothrombin')
n_years = st.number_input('N_Years')

if st.button('Predict'):
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
    st.write('The model Predicted that the patient with provided data is in Stage',pipeline.predict(test)[0]+1)
    st.write(test)

if st.button('Show Test Data'):
    X_test_df = pd.read_csv('test_data.csv')
    st.write(X_test_df)
#