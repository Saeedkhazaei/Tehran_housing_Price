import streamlit as st

import numpy as np
import pandas as pd
import joblib
from PIL import Image
import time

model = joblib.load('CTBpipe.joblib')
col1,col2 = st.columns([2,1])
col1.header('*The Price of a House in Tehran Will Be Predicted In This Website*') 
col1.write("Learn More >> [link](https://github.com/Saeedkhazaei)")
photo = Image.open("increase-in-home-value.jpg")
photo.resize((800,600))
col2.image(photo)
st.write("---")


Area = st.slider(" Select the Area of House",30,200)
Room = st.slider(" Select the Number of Room",0,5) 
Parking = st.radio("Does the House Have Parking ?", ['Yes','No'])
Warehouse = st.radio("Does the House Have Warehouse ?", ['Yes','No'])
Elevator = st.radio("Does the House Have Elevator ? ", ['Yes','No']) 
Adress = st.selectbox (" Select Wanted Neighbouhood", ['Abazar', 'Abbasabad', 'Absard', 'Abuzar', 'Afsarieh', 'Ahang', 'Air force',
        'Ajudaniye', 'Alborz Complex','Aliabad South', 'Amir Bahador', 'Amirabad', 'Amirieh', 'Andisheh', 'Aqdasieh', 'Araj', 'Argentina',
        'Atabak','Azadshahr', 'Azarbaijan', 'Azari', 'Baghestan', 'Bahar', 'Baqershahr', 'Beryanak', 'Boloorsazi', 'Central Janatabad',
        'Chahardangeh', 'Chardangeh', 'Chardivari', 'Chidz', 'Damavand', 'Darabad', 'Darakeh', 'Darband', 'Daryan No',
        'Dehkade Olampic', 'Dezashib', 'Dolatabad', 'Dorous', 'East Ferdows Boulevard', 'East Pars', 'Ekbatan',
        'Ekhtiarieh', 'Elahieh', 'Elm-o-Sanat', 'Enghelab', 'Eram', 'Eskandari', 'Fallah', 'Farmanieh', 'Fatemi',
        'Feiz Garden', 'Firoozkooh', 'Firoozkooh Kuhsar', 'Gandhi', 'Garden of Saba', 'Gheitarieh', 'Ghiyamdasht',
        'Ghoba', 'Gholhak', 'Gisha', 'Golestan', 'Haft Tir', 'Hakimiyeh', 'Hashemi', 'Hassan Abad', 'Hekmat', 'Heravi',
        'Heshmatieh', 'Hor Square', 'Islamshahr', 'Islamshahr Elahieh', 'Javadiyeh', 'Jeyhoon', 'Jordan', 'Kahrizak',
        'Kamranieh', 'Karimkhan', 'Karoon', 'Kazemabad', 'Keshavarz Boulevard', 'Khademabad Garden', 'Khavaran', 'Komeil',
        'Koohsar', 'Kook', 'Lavasan', 'Lavizan', 'Mahallati', 'Mahmoudieh', 'Majidieh', 'Malard', 'Marzdaran', 'Mehrabad',
        'Mehrabad River River', 'Mehran', 'Mirdamad', 'Mirza Shirazi', 'Moniriyeh', 'Narmak', 'Nasim Shahr', 'Nawab', 'Naziabad',
        'Nezamabad', 'Niavaran', 'North Program Organization', 'Northern Chitgar', 'Northern Janatabad', 'Northern Suhrawardi',
        'Northren Jamalzadeh', 'Ostad Moein', 'Ozgol', 'Pakdasht', 'Pakdasht KhatunAbad', 'Parand', 'Parastar', 'Pardis', 'Pasdaran',
        'Persian Gulf Martyrs Lake', 'Pirouzi', 'Pishva', 'Punak', 'Qalandari', 'Qarchak', 'Qasr-od-Dasht', 'Qazvin Imamzadeh Hassan',
        'Railway', 'Ray', 'Ray - Montazeri', 'Ray - Pilgosh', 'Razi', 'Republic', 'Robat Karim', 'Rudhen', 'Saadat Abad', 'SabaShahr',
        'Sabalan', 'Sadeghieh', 'Safadasht', 'Salehabad', 'Salsabil', 'Sattarkhan', 'Seyed Khandan', 'Shadabad', 'Shahedshahr', 'Shahr-e-Ziba',
        'ShahrAra', 'Shahrake Apadana', 'Shahrake Azadi', 'Shahrake Gharb', 'Shahrake Madaen', 'Shahrake Qods', 'Shahrake Quds',
        'Shahrake Shahid Bagheri', 'Shahrakeh Naft', 'Shahran', 'Shahryar', 'Shams Abad', 'Shoosh', 'Si Metri Ji', 'Sohanak',
        'Southern Chitgar', 'Southern Janatabad', 'Southern Program Organization', 'Southern Suhrawardi', 'Tajrish', 'Tarasht',
        'Taslihat', 'Tehran Now', 'Tehransar', 'Telecommunication', 'Tenant', 'Thirteen November', 'Vahidieh', 'Vahidiyeh',
        'Valiasr', 'Vanak', 'Varamin - Beheshti', 'Velenjak', 'Villa', 'Water Organization', 'Waterfall', 'West Ferdows Boulevard',
        'West Pars', 'Yaftabad', 'Yakhchiabad', 'Yousef Abad', 'Zafar', 'Zaferanieh', 'Zargandeh', 'Zibadasht'] )
def predict(): 
    columns = ['Area', 'Room', 'Parking', 'Warehouse', 'Elevator',
        'Shahran', 'Pardis', 'Shahrake Qods', 'Shahrake Gharb',
       'North Program Organization', 'Andisheh', 'West Ferdows Boulevard',
       'Narmak', 'Saadat Abad', 'Zafar', 'Islamshahr', 'Pirouzi',
       'Shahrake Shahid Bagheri', 'Moniriyeh', 'Velenjak', 'Amirieh',
       'Southern Janatabad', 'Salsabil', 'Zargandeh', 'Feiz Garden',
       'Water Organization', 'ShahrAra', 'Gisha', 'Ray', 'Abbasabad',
       'Ostad Moein', 'Farmanieh', 'Parand', 'Punak', 'Qasr-od-Dasht',
       'Aqdasieh', 'Pakdasht', 'Railway', 'Central Janatabad',
       'East Ferdows Boulevard', 'Pakdasht KhatunAbad', 'Sattarkhan',
       'Baghestan', 'Shahryar', 'Northern Janatabad', 'Daryan No',
       'Southern Program Organization', 'Rudhen', 'West Pars', 'Afsarieh',
       'Marzdaran', 'Dorous', 'Sadeghieh', 'Chahardangeh', 'Baqershahr',
       'Jeyhoon', 'Lavizan', 'Shams Abad', 'Fatemi',
       'Keshavarz Boulevard', 'Kahrizak', 'Qarchak',
       'Northren Jamalzadeh', 'Azarbaijan', 'Bahar',
       'Persian Gulf Martyrs Lake', 'Beryanak', 'Heshmatieh',
       'Elm-o-Sanat', 'Golestan', 'Shahr-e-Ziba', 'Pasdaran',
       'Chardivari', 'Gheitarieh', 'Kamranieh', 'Gholhak', 'Heravi',
       'Hashemi', 'Dehkade Olampic', 'Damavand', 'Republic', 'Zaferanieh',
       'Qazvin Imamzadeh Hassan', 'Niavaran', 'Valiasr', 'Qalandari',
       'Amir Bahador', 'Ekhtiarieh', 'Ekbatan', 'Absard', 'Haft Tir',
       'Mahallati', 'Ozgol', 'Tajrish', 'Abazar', 'Koohsar', 'Hekmat',
       'Parastar', 'Lavasan', 'Majidieh', 'Southern Chitgar', 'Karimkhan',
       'Si Metri Ji', 'Karoon', 'Northern Chitgar', 'East Pars', 'Kook',
       'Air force', 'Sohanak', 'Komeil', 'Azadshahr', 'Zibadasht',
       'Amirabad', 'Dezashib', 'Elahieh', 'Mirdamad', 'Razi', 'Jordan',
       'Mahmoudieh', 'Shahedshahr', 'Yaftabad', 'Mehran', 'Nasim Shahr',
       'Tenant', 'Chardangeh', 'Fallah', 'Eskandari', 'Shahrakeh Naft',
       'Ajudaniye', 'Tehransar', 'Nawab', 'Yousef Abad',
       'Northern Suhrawardi', 'Villa', 'Hakimiyeh', 'Nezamabad',
       'Garden of Saba', 'Tarasht', 'Azari', 'Shahrake Apadana', 'Araj',
       'Vahidieh', 'Malard', 'Shahrake Azadi', 'Darband', 'Vanak',
       'Tehran Now', 'Darabad', 'Eram', 'Atabak', 'Sabalan', 'SabaShahr',
       'Shahrake Madaen', 'Waterfall', 'Ahang', 'Salehabad', 'Pishva',
       'Enghelab', 'Islamshahr Elahieh', 'Ray - Montazeri',
       'Firoozkooh Kuhsar', 'Ghoba', 'Mehrabad', 'Southern Suhrawardi',
       'Abuzar', 'Dolatabad', 'Hor Square', 'Taslihat', 'Kazemabad',
       'Robat Karim', 'Ray - Pilgosh', 'Ghiyamdasht', 'Telecommunication',
       'Mirza Shirazi', 'Gandhi', 'Argentina', 'Seyed Khandan',
       'Shahrake Quds', 'Safadasht', 'Khademabad Garden', 'Hassan Abad',
       'Chidz', 'Khavaran', 'Boloorsazi', 'Mehrabad River River',
       'Varamin - Beheshti', 'Shoosh', 'Thirteen November', 'Darakeh',
       'Aliabad South', 'Alborz Complex', 'Firoozkooh', 'Vahidiyeh',
       'Shadabad', 'Naziabad', 'Javadiyeh', 'Yakhchiabad']
    row = np.zeros(len(columns))
    row[0] = Area
    row[1] = Room
    for i in range(len(columns)):
        if Adress == columns[i]:
            row[i] = 1
    S={"Yes":1,"No":0}
    row[2] =S[Parking]
    row[3] =S[Warehouse]
    row[4] =S[Elevator]
    X = pd.DataFrame([row], columns = columns)
    prediction = model.predict(X)

    progress_bar = col1.progress(0)
    for perdiction_complited in range(100):
        time.sleep(0.05)
        progress_bar.progress(perdiction_complited+1)
    st.write("---")
    st.success("The Price of Wanted House is >>>")
    st.success(prediction)
     
    st.write("---")

    
trigger = st.button('Predict', on_click=predict)


