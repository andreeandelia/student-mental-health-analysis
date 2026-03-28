import math
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

image = Image.open('ps4-console.png')

st.image(image, width="stretch")

st.markdown("# Analiza vanzarilor de jocuri video", text_alignment="justify")

section = st.sidebar.radio("Navigati catre:",
                           ["Introducere", "Preprocesare", "EDA"])

if section == "Introducere":
    st.markdown("""
    motivatia si obiectivul proiectului
    ***
    """)

    if "uploaded_data" not in st.session_state:
        st.session_state["uploaded_data"] = None
    uploaded_file = st.file_uploader("Incarcati un fisier CSV", type="csv")
    if uploaded_file is not None:
        st.session_state["uploaded_data"] = pd.read_csv(uploaded_file)
        st.session_state["processed_df"] = st.session_state["uploaded_data"].copy()

    df = st.session_state["uploaded_data"]
    if df is None:
        st.stop()

    st.write(df)

    rows, cols = df.shape
    st.markdown(f"Setul de date contine {rows} de inregistrari si {cols} coloane", text_alignment="justify")

elif section == "Preprocesare":
    st.markdown("""
    ***
    ## Preprocesarea datelor
    """)

    # Initializare dataset prelucrat
    if "processed_df" not in st.session_state:
        st.session_state["processed_df"] = st.session_state["uploaded_data"].copy()

    df = st.session_state["processed_df"]

    # Eliminarea coloanelor
    st.markdown("""
    ### Stergerea coloanelor irelevante
    """)

    col_names = df.columns.tolist()
    cols_to_drop = st.multiselect("Alegeti coloana/coloanele pe care doriti sa le eliminati:",
                                  options=col_names,
                                  default=[])

    c1, c2 = st.columns(2)

    with c1:
        if st.button("Aplicati selectia"):
            st.session_state["processed_df"] = df.drop(columns=cols_to_drop)
            st.success("Coloanele selectate au fost eliminate")
            st.rerun()

    with c2:
        if st.button("Resetare dataset"):
            st.session_state["processed_df"] = st.session_state["uploaded_data"].copy()
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
        st.warning("Exita valori lipsa. Acestea trebuie tratate inainte de modelare.")
        st.markdown("***")

        st.markdown("#### Eliminarea randurilor/coloanelor")
        st.info(
            "In cazul acestui set de date, variabilele cu cele mai multe valori lipsa sunt chiar variabilele noastre tinta. "
            "Motiv pentru care recomandam eliminarea ***inregistrarilor*** care nu au variabila ***total_sales***.")

        col_names = df.columns.tolist()
        col_to_drop = st.selectbox("Alegeti coloana dorita:",
                                   options=col_names)
        selected_axis = st.selectbox("Alegeti axa dorita:",
                                     options=["Rand", "Coloana"])
        if st.button("Stergeti"):
            if selected_axis == "Rand":
                st.session_state["processed_df"] = st.session_state["processed_df"].dropna(subset=[col_to_drop])
            else:
                st.session_state["processed_df"] = st.session_state["processed_df"].drop(columns=[col_to_drop], errors="ignore")
            st.success("Datele au fost sterse cu succes!")
            st.rerun()

        df = st.session_state["processed_df"]
        st.markdown("#### Imputarea valorilor lipsa")

        st.markdown("##### Imputare pentru coloanele numerice")
        col_num = df.select_dtypes(include=["float64", "int64"]).columns
        col_na = [col for col in col_num if df[col].isnull().sum() > 0]

        selected_col = st.multiselect("Alegeti coloanele numerice:",
                                      options=col_na)
        selected_method = st.selectbox("Alegeti metoda de impuare:",
                                       options=["Valoarea 0", "Medie", "Mediana"])
        st.info("Pentru a nu distorsiona realitatea, recomandam selectarea optiunii ***Valoarea 0***")
        if st.button("Aplicati imputarea numerica aleasa"):
            for col in selected_col:
                if selected_method == "Valoarea 0":
                    st.session_state["processed_df"][col] = st.session_state["processed_df"][col].fillna(0)
                elif selected_method == "Medie":
                    medie = st.session_state["processed_df"][col].mean()
                    st.session_state["processed_df"][col] = st.session_state["processed_df"][col].fillna(medie)
                elif selected_method == "Mediana":
                    mediana = st.session_state["processed_df"][col].median()
                    st.session_state["processed_df"][col] = st.session_state["processed_df"][col].fillna(mediana)
            st.success("Imputarea numerica a fost realizata cu succes!")
            st.rerun()

        df = st.session_state["processed_df"]
        st.markdown("##### Imputare pentru coloane categoriale")
        col_cat = df.select_dtypes(include=["object"]).columns
        col_na_cat = [col for col in col_cat if df[col].isnull().sum() > 0]

        selected_col_cat = st.multiselect("Alegeti coloanele categoriale:",
                                      options=col_na_cat)
        selected_method_cat = st.selectbox("Alegeti metoda de impuare:",
                                       options=["Necunoscut", "Cea mai frecventa valoare (mod)"])
        st.info("Pentru a nu distorsiona realitatea, recomandam selectarea optiunii ***Necunoscut***")
        if st.button("Aplicati imputarea categoriala aleasa"):
            for col in selected_col_cat:
                if selected_method_cat == "Necunoscut":
                    st.session_state["processed_df"][col] = st.session_state["processed_df"][col].fillna("Necunoscut")
                elif selected_method_cat == "Cea mai frecventa valoare (mod)":
                    modul = st.session_state["processed_df"][col].mode()[0]
                    st.session_state["processed_df"][col] = st.session_state["processed_df"][col].fillna(modul)
            st.success("Imputarea categoriala a fost realizata cu succes!")
            st.rerun()
        st.markdown("***")


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
