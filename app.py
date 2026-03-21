import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image


image = Image.open('students-img.jpg')

st.image(image, use_column_width=True)

st.markdown("""
# Analiza sanatatii mintale si a burnout-ului in randul studentilor

"""
)

section = st.sidebar.radio("Navigati catre:",
                     ["Introducere", "Ceva", "EDA"])

if section == "Introducere":
    st.markdown("""
    Viata academica moderna vine adesea la pachet cu un nivel ridicat de stres, presiune,
    performanta si provocari legate de echilibrul dintre viata personala si studii. Sanatatea 
    mintala a studentilor si fenomenul de "burnout" au devenit subiecte critice de discutie
    in mediul educational. Intelegerea factorilor care declanseaza acest sindrom este primul
    pas catre preventie si suport adecvat.

    ***
    
    ### Obiectivul proiectului

    Scopul principal al acestei aplicatii este de a transforma datele brute intr-o poveste usor de
    inteles printr-o Analiza Exploratorie a Datelor (EDA) interactiva. Utilizatorii pot nu doar sa 
    vizualizeze distributia si corelatiile dintre factori, dar si sa interactioneze direct cu setul
    de date.
            
    """)

    uploaded_file = st.file_uploader("Alegeti fisierul")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(df)

        item_count, col_count = df.shape
        st.markdown(f"""
        ### Despre setul de date
        
        Acest proiect are la baza un set de date de mari dimensiuni ({item_count} de inregistrari
        si {col_count} de variabile), creat special pentru a analiza si prezice nivelul de burnout
        al studentilor. Desi este un set de date generat sintetic, el a fost modelat pentru a reflecta
        cu acuratete realitatea, combinand variabile numerice si categorice din trei arii esentiale:
        - Factori academini: volum de studiu, note, prezenta;
        - Factori psihologici: nivelul de stres raportat, anxietate;
        - Factori de stil de viata: calitatea somnului, activitatea fizica, activitati extracuriculare.
    
        """)
elif section == "Ceva":
    st.markdown("Ceva")
elif section == "EDA":
    st.markdown("EDA")