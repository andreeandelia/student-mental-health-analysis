import math
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image


image = Image.open('students-img.jpg')

st.image(image, width="stretch")

st.markdown("""
# Analiza sanatatii mintale si a burnout-ului in randul studentilor

""", text_alignment="justify")

section = st.sidebar.radio("Navigati catre:",
                     ["Introducere", "Preprocesare", "EDA"])

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
            
    """, text_alignment="justify")


    if "my_df" not in st.session_state:
        st.session_state["my_df"] = pd.read_csv("student_mental_health_burnout.csv")

    df = st.session_state["my_df"]
    st.write(df)

    item_count, col_count = df.shape
    st.markdown(f"""
    ### Despre setul de date
        
    Acest proiect are la baza un set de date de mari dimensiuni ({item_count} de inregistrari
    si {col_count} de variabile), creat special pentru a analiza si prezice nivelul de burnout
    al studentilor. Desi este un set de date generat sintetic, el a fost modelat pentru a reflecta
    cu acuratete realitatea, combinand variabile numerice si categorice din trei arii esentiale:
    - Factori academici: volum de studiu, note, prezenta;
    - Factori psihologici: nivelul de stres raportat, anxietate;
    - Factori de stil de viata: calitatea somnului, activitatea fizica, activitati extracuriculare.
        
    #### Prezentarea detaliata a variabilelor
        
    Pentru a oferi o imagine clara asupra informatiilor analizate, am grupat cele 20 de variabile in
    patru categorii logice:
        
    ##### A. Date demografice si academice
        
    Aceasta sectiune descrie profilul de baza al studentului si performanta sa educationala:
    - ***student_id***: Identificator unic atribuit fiecarei inregistrari (variabila nominala);
    - ***age***: Varsta studentului, exprimata in ani (variabila numerica);
    - ***gender***: Genul studentului (variabila categoriala: Male, Female, Other);
    - ***course***: Programul de studiu sau specializarea urmata (variabila categoriala);
    - ***year***: Anul curent de studiu al studentului (variabila categoriala ordinala: 1st, 2nd, 3rd, 4th);
    - ***attendance_percentage***: Rata de prezenta la cursuri, exprimata procentual (variabila numerica continua);
    - ***cgpa***: Media cumulata a notelor, reflectand performanta academica generala (variabila numerica continua).
        
    ##### B. Stil de viata
        
    Variabilele din aceasta categorie cuantifica obiceiurile zilnice care pot influenta starea de bine:
    - ***daily_study_hours***: Timpul mediu, in ore, dedicat studiului individual zilnic (variabile numerica);
    - ***daily_sleep_hours***: Numarul mediu de ore de somn pe noapte (variabila numerica);
    - ***sleep_quality***: Evaluarea subiectiva a calitatii somnului (variabila categorica ordinala: Poor, Average, Good);
    - ***screen_time_hours***: Timpul zilnic estimat petrecut in fata ecranelor, excluzand studiul (variabila numerica);
    - ***physical_activity_hours***: Timpul mediu zilnic alocat exercitiilor fizice sau sportului (variabila numerica);
    - ***internet_quality***: Calitatea conexiunii la internet (variabila categorica: Poor, Average, Good). 
        
    ##### C. Evaluari psihologice si factori de stres
        
    Aceste variabile reprezinta scoruri auto-raportate sau evaluate care masoara diferite dimensiuni ale sanatatii mintale:
    - ***stress_level***: Nivelul general de stres resimtit (variabila categorica ordinala: Low, Medium, High);
    - ***anxiety_score***: Un scor numeric (de la 1 la 10) care indica severitatea simptomelor de anxietate;
    - ***depression_score***: Un scor numeric care masoara prezenta si intensitatea starilor depresive;
    - ***academic_pressure_score***: Scorul care cuantifica presiunea resimtita de student in legatura cu termenele limita, examenele si asteptarile academice;
    - ***financial_stress_score***: Nivelul de ingrijorare cu privire la situatia financiara proprie sau a familiei;
    - ***social_support_score***: Un indicator al spijinului emotional sau practic perceput din partea familiei, prietenilor sau comunitatii.
        
    ##### D. Variabila tinta
        
    - ***burnout_level***: Aceasta este variabila dependenta a setului de date, reprezentand nivelul final de epuizare al studentului (variabila categorica ordinala: Low, Medium, High).
    Toate celelalte caracteristici vor fi analizate din perspectiva impactului pe care il au asupra acestei variabile.
        
    """, text_alignment="justify")

elif section == "Preprocesare":
    st.markdown("""
    ***
    ## Preprocesarea datelor
    """)

    # Initializare dataset prelucrat
    if "processed_df" not in st.session_state:
        st.session_state["processed_df"] = st.session_state["my_df"].copy()

    df = st.session_state["processed_df"]

    # Eliminarea coloanelor
    st.markdown("""
    ### Stergerea coloanelor irelevante
    """)

    col_names = df.columns.tolist()
    default_drop = ["student_id"] if "student_id" in col_names else []

    cols_to_drop = st.multiselect("Alegeti coloana/coloanele pe care doriti sa le eliminati:",
                                  options=col_names,
                                  default=default_drop)

    c1, c2 = st.columns(2)

    with c1:
        if st.button("Aplicati selectia"):
            st.session_state["processed_df"] = df.drop(columns=cols_to_drop)
            st.success("Coloanele selectate au fost eliminate")
            st.rerun()

    with c2:
        if st.button("Resetare dataset"):
            st.session_state["processed_df"] = st.session_state["my_df"].copy()
            st.info("Datasetul a fost resetat la forma initiala")
            st.rerun()

    df = st.session_state["processed_df"]

    st.markdown("### Primele 10 inregistrari")
    st.dataframe(df.head(10), width="stretch")

    st.markdown("""
    ### Analiza detaliata a valorilor lipsa (NaN)
    """)

    nan_df = pd.DataFrame({
        "Variabila": df.columns,
        "Numar NaN": df.isnull().sum().values,
        "Procent NaN": ((df.isnull().sum() / len(df)) * 100).round(2).values
    }).sort_values(by="Procent NaN", ascending=False)

    show_only_nan = st.checkbox("Afiseaza doar variabilele care contin valori lipsa", value=True)

    nan_display = nan_df[nan_df["Numar NaN"] > 0] if show_only_nan else nan_df
    st.dataframe(nan_display, width="stretch")

    if nan_df["Numar NaN"].sum() == 0:
        st.success("Nu exista valori lipsa in dataset")
    else:
        st.warning("Ecita valori lipsa. Acestea trebuie tratate inainte de modelare")

    # Analiza outliers
    st.markdown("""
    ### Analiza outliers
    """)

    numerical_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=["object"]).columns

    outlier_rows = []

    for col in numerical_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        mask = (df[col] < lower) | (df[col] > upper)
        n_outliers = mask.sum()

        outlier_rows.append({
            "Variabila": col,
            "Limita inferioara": round(lower, 2),
            "Limita superioara": round(upper, 2),
            "Numar outliers": int(n_outliers),
            "Procent outliers": round(n_outliers / len(df) * 100, 2)
        })

    outlier_df = pd.DataFrame(outlier_rows).sort_values(
        by="Procent outliers", ascending=False
    )

    show_only_outliers = st.checkbox("Afiseaza doar variabilele care contin outliers", value=True)

    outlier_display = outlier_df[outlier_df["Numar outliers"] > 0] if show_only_outliers else outlier_df
    st.dataframe(outlier_display, width="stretch")

    if (outlier_df["Numar outliers"] == 0).all():
        st.success("Nu au fost identificati outliers numerici semnificativi prin metoda IQR")
    else:
        st.warning("Au fost identificate posibile valori extreme. Acestea trebuie analizate inainte de eliminare")

    st.info(
        "Valorile extreme identificate prin metoda IQR nu sunt eliminate automat, "
        "deoarece unele pot fi observatii valide si relevante pentru analiza"
    )

    # Boxplot individual
    st.markdown("""
        ### Generarea boxplot-urilor pentru variabilele numerice
        """)

    selected_box_col = st.selectbox("Alegeti o variabila numerica:", numerical_cols)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot(df[selected_box_col].dropna())
    ax.set_title(f"Boxplot pentru {selected_box_col}")
    ax.set_ylabel(selected_box_col)
    st.pyplot(fig)

    # Descarcare dataset preprocesat
    st.markdown("### Exportare dataset preprocesat")

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Descarcati datasetul preprocesat",
        data=csv,
        file_name="dataset_preprocesat.csv",
        mime="text/csv"
    )

elif section == "EDA":
    st.markdown("EDA")