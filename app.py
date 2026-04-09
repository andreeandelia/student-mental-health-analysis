import math
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
import seaborn as sb
from sklearn.preprocessing import StandardScaler, LabelEncoder

image = Image.open('ps4-console.png')

st.image(image, width="stretch")

st.markdown("# Analiza vanzarilor de jocuri video", text_alignment="justify")

section = st.sidebar.radio("Navigati catre:",
                           ["Introducere", "Preprocesare", "Analiza exploratorie (EDA)", "Pregatirea datelor pentru ML"])

if section == "Introducere":
    st.markdown("""
    ## Motivația și obiectivul proiectului

    Acest proiect are ca scop analiza vânzărilor de jocuri video pe baza unui set de date real, pentru a evidenția tipare relevante legate de platforme, genuri, regiuni și performanța comercială a titlurilor lansate de-a lungul timpului.

    Motivația proiectului pornește de la interesul pentru industria gaming-ului și de la dorința de a aplica, într-un context practic, concepte de **preprocesare a datelor**, **analiză exploratorie** și **vizualizare interactivă**.

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
        st.warning("Exista valori lipsa. Acestea trebuie tratate inainte de modelare.")
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
                st.session_state["processed_df"] = st.session_state["processed_df"].drop(columns=[col_to_drop],
                                                                                         errors="ignore")
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

    df = st.session_state["processed_df"]
    # Analiza outliers
    st.markdown("""
    ### Analiza outliers
    """)

    selected_outliers_method = st.selectbox("Alegeti metoda de analiza",
                                            options=["IQR (Interquartile Range)", "Boxplot"])

    numerical_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=["object"]).columns

    if selected_outliers_method == "IQR (Interquartile Range)":
        st.markdown("#### Detalii Outliers prin metoda IQR")
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
        st.dataframe(outlier_display, use_container_width=True)

        if (outlier_df["Numar outliers"] == 0).all():
            st.success("Nu au fost identificati outliers numerici semnificativi prin metoda IQR")
        else:
            st.warning("Au fost identificate posibile valori extreme. Acestea trebuie analizate inainte de eliminare")

        st.info(
            "Valorile extreme identificate prin metoda IQR nu sunt eliminate automat, "
            "deoarece unele pot fi observatii valide si relevante pentru analiza"
        )
    elif selected_outliers_method == "Boxplot":
        # Boxplot individual
        st.markdown("#### Generarea boxplot-urilor pentru variabilele numerice")

        selected_box_col = st.selectbox("Alegeti o variabila numerica:", numerical_cols)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.boxplot(df[selected_box_col].dropna(), vert=False)
        ax.set_title(f"Boxplot pentru distributia {selected_box_col}")
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

elif section == "Analiza exploratorie (EDA)":
    st.markdown("""
    ***
    ## Analiza Exploratorie a Datelor""")

    if "eda_df" not in st.session_state:
        st.session_state["eda_df"] = st.session_state["processed_df"].copy()
    df = st.session_state["eda_df"]

    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_year"] = df["release_date"].dt.year

    st.subheader("1. Care sunt cele mai bine vandute titluri la nivel global?")
    top_n = st.slider("Selectati numarul de jocuri:", min_value=5, max_value=20, value=10)

    top_games = df.sort_values(by="total_sales", ascending=False).head(top_n)

    fig1 = px.bar(
        top_games,
        x="title",
        y="total_sales",
        color="console",
        title=f"Top {top_n} jocuri dupa vanzarile globale (mil. copii)",
        labels={"title": "Titlul jocului", "total_sales": "Vanzari totale (mil.)", "console": "Consola"},
        hover_data=["publisher", "genre"]
    )
    fig1.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("***")
    st.subheader("2. Ce an a avut cele mai multe vanzari? Este industria in crestere?")

    sales_by_year = df[df["release_year"] <= 2024].groupby("release_year")["total_sales"].sum().reset_index()

    fig2 = px.line(
        sales_by_year,
        x="release_year",
        y="total_sales",
        markers=True,
        title="Vanzarile totale globale pe ani",
        labels={"release_year": "Anul lansarii", "total_sales": "Vanzarile globale (mil.)"}
    )
    fig2.update_traces(line_color="#2ca02c")
    st.plotly_chart(fig2, use_container_width=True)

    best_year = sales_by_year.loc[sales_by_year["total_sales"].idxmax()]
    st.info(
        f"Anul cu cele mai mari vanzari a fost **{int(best_year["release_year"])}**, generand aproximativ **{best_year["total_sales"]:.2f} milioane** de copii vandute.")

    st.markdown("***")
    st.subheader("3. Exista console care se specializeaza pe anumite genuri?")

    top_consoles = df["console"].value_counts().head(10).index.tolist()
    selected_consoles = st.multiselect("Selectati consolele pentru a le compara",
                                       options=top_consoles,
                                       default=top_consoles[:5])
    df_filtered_consoles = df[df["console"].isin(selected_consoles)]
    console_genre_matrix = pd.crosstab(
        index=df_filtered_consoles["console"],
        columns=df_filtered_consoles["genre"],
        values=df_filtered_consoles["total_sales"],
        aggfunc="sum"
    ).fillna(0)

    fig3, ax3 = plt.subplots(figsize=(12, 6))

    sb.heatmap(console_genre_matrix, annot=True, fmt=".1f", cmap="YlGnBu", linewidths=.5, ax=ax3)
    ax3.set_title("Heatmap: Vanzari totale (mil.) pe console si genuri")
    ax3.set_xlabel("Genul jocului")
    ax3.set_ylabel("Consola")
    st.pyplot(fig3)

    st.markdown("***")
    st.subheader("4. Ce titluri sunt populare intr-o regiune, dar nu si in alta?")

    regiuni_dict = {
        "na_sales": "America de Nord",
        "pal_sales": "Europa & Africa",
        "jp_sales": "Japonia",
        "other_sales": "Restul lumii"
    }

    col1, col2 = st.columns(2)
    with col1:
        regiunea_x = st.selectbox("Alegeti prima regiune (axa X):",
                                  options=list(regiuni_dict.keys()),
                                  format_func=lambda x: regiuni_dict[x])
    with col2:
        regiunea_y = st.selectbox("Alegeti a doua regiune (axa Y):",
                                  options=list(regiuni_dict.keys()),
                                  format_func=lambda x: regiuni_dict[x])

    prag_vanzari = st.slider("Afisati doar jocurile cu vanzari de peste (mil. copii):", min_value=0.1, max_value=9.8, value=1.0, step=0.1)

    df_regional = df[(df[regiunea_x] >= prag_vanzari) | (df[regiunea_y] >= prag_vanzari)]

    if df_regional.empty:
        st.warning("Nu exista jocuri care sa indeplineasca acest prag de vanzari pentru regiunile selectate.")
    else:
        fig4 = px.scatter(
            df_regional,
            x=regiunea_x,
            y=regiunea_y,
            color="genre",
            hover_name="title",
            hover_data=["console", "total_sales"],
            title=f"Comparatie: {regiuni_dict[regiunea_x]} vs. {regiuni_dict[regiunea_y]}",
            labels={regiunea_x: f"Vanzari {regiuni_dict[regiunea_x]}", regiunea_y: f"Vanzari {regiuni_dict[regiunea_y]}"}
        )

        max_val = max(df_regional[regiunea_x].max(), df_regional[regiunea_y].max())
        fig4.add_shape(type="line", line=dict(dash="dash", color="gray"), x0=0, y0=0, x1=max_val, y1=max_val)

        st.plotly_chart(fig4, use_container_width=True)

elif section == "Analiza exploratorie (EDA)":
    st.markdown("""
    ***
    ## Analiza Exploratorie a Datelor""")

    if "processed_df" not in st.session_state or st.session_state["processed_df"] is None:
        st.warning("Incarcati un fisier inainte de a accesa analiza exploratorie a datelor!")
        st.stop()

    if "eda_df" not in st.session_state:
        st.session_state["eda_df"] = st.session_state["processed_df"].copy()
    df = st.session_state["eda_df"]

    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_year"] = df["release_date"].dt.year

    st.subheader("1. Care sunt cele mai bine vandute titluri la nivel global?")
    top_n = st.slider("Selectati numarul de jocuri:", min_value=5, max_value=20, value=10)

    top_games = df.sort_values(by="total_sales", ascending=False).head(top_n)

    fig1 = px.bar(
        top_games,
        x="title",
        y="total_sales",
        color="console",
        title=f"Top {top_n} jocuri dupa vanzarile globale (mil. copii)",
        labels={"title": "Titlul jocului", "total_sales": "Vanzari totale (mil.)", "console": "Consola"},
        hover_data=["publisher", "genre"]
    )
    fig1.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("***")
    st.subheader("2. Ce an a avut cele mai multe vanzari? Este industria in crestere?")

    sales_by_year = df[df["release_year"] <= 2024].groupby("release_year")["total_sales"].sum().reset_index()

    fig2 = px.line(
        sales_by_year,
        x="release_year",
        y="total_sales",
        markers=True,
        title="Vanzarile totale globale pe ani",
        labels={"release_year": "Anul lansarii", "total_sales": "Vanzarile globale (mil.)"}
    )
    fig2.update_traces(line_color="#2ca02c")
    st.plotly_chart(fig2, use_container_width=True)

    best_year = sales_by_year.loc[sales_by_year["total_sales"].idxmax()]
    st.info(
        f"Anul cu cele mai mari vanzari a fost **{int(best_year["release_year"])}**, generand aproximativ **{best_year["total_sales"]:.2f} milioane** de copii vandute.")

    st.markdown("***")
    st.subheader("3. Exista console care se specializeaza pe anumite genuri?")

    top_consoles = df["console"].value_counts().head(10).index.tolist()
    selected_consoles = st.multiselect("Selectati consolele pentru a le compara",
                                       options=top_consoles,
                                       default=top_consoles[:5])
    df_filtered_consoles = df[df["console"].isin(selected_consoles)]
    console_genre_matrix = pd.crosstab(
        index=df_filtered_consoles["console"],
        columns=df_filtered_consoles["genre"],
        values=df_filtered_consoles["total_sales"],
        aggfunc="sum"
    ).fillna(0)

    fig3, ax3 = plt.subplots(figsize=(12, 6))

    sb.heatmap(console_genre_matrix, annot=True, fmt=".1f", cmap="YlGnBu", linewidths=.5, ax=ax3)
    ax3.set_title("Heatmap: Vanzari totale (mil.) pe console si genuri")
    ax3.set_xlabel("Genul jocului")
    ax3.set_ylabel("Consola")
    st.pyplot(fig3)

    st.markdown("***")
    st.subheader("4. Ce titluri sunt populare intr-o regiune, dar nu si in alta?")

    regiuni_dict = {
        "na_sales": "America de Nord",
        "pal_sales": "Europa & Africa",
        "jp_sales": "Japonia",
        "other_sales": "Restul lumii"
    }

    col1, col2 = st.columns(2)
    with col1:
        regiunea_x = st.selectbox("Alegeti prima regiune (axa X):",
                                  options=list(regiuni_dict.keys()),
                                  format_func=lambda x: regiuni_dict[x])
    with col2:
        regiunea_y = st.selectbox("Alegeti a doua regiune (axa Y):",
                                  options=list(regiuni_dict.keys()),
                                  format_func=lambda x: regiuni_dict[x])

    prag_vanzari = st.slider("Afisati doar jocurile cu vanzari de peste (mil. copii):", min_value=0.1, max_value=9.8, value=1.0, step=0.1)

    df_regional = df[(df[regiunea_x] >= prag_vanzari) | (df[regiunea_y] >= prag_vanzari)]

    if df_regional.empty:
        st.warning("Nu exista jocuri care sa indeplineasca acest prag de vanzari pentru regiunile selectate.")
    else:
        fig4 = px.scatter(
            df_regional,
            x=regiunea_x,
            y=regiunea_y,
            color="genre",
            hover_name="title",
            hover_data=["console", "total_sales"],
            title=f"Comparatie: {regiuni_dict[regiunea_x]} vs. {regiuni_dict[regiunea_y]}",
            labels={regiunea_x: f"Vanzari {regiuni_dict[regiunea_x]}", regiunea_y: f"Vanzari {regiuni_dict[regiunea_y]}"}
        )

        max_val = max(df_regional[regiunea_x].max(), df_regional[regiunea_y].max())
        fig4.add_shape(type="line", line=dict(dash="dash", color="gray"), x0=0, y0=0, x1=max_val, y1=max_val)

        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("***")
    st.subheader("5. Matricea de corelatie: Cum se influenteaza variabilele intre ele?")
    st.write(
        "Acest grafic ne arata corelatia matematica dintre variabilele numerice. O valoare apropiata de 1 indica o legatura directa puternica.")

    df_numeric = df.select_dtypes(include=[np.number])

    corr_matrix = df_numeric.corr()

    fig5, ax5 = plt.subplots(figsize=(10, 8))
    sb.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, linewidths=0.5, ax=ax5)
    ax5.set_title("Matricea de corelatie a variabilelor numerice")

    st.pyplot(fig5)

elif section == "Pregatirea datelor pentru ML":
    st.markdown("""
        ***
        ## Pregatirea datelor pentru Machine Learning""")

    if "eda_df" not in st.session_state or st.session_state["eda_df"] is None:
        st.warning("Finalizati preprocesarea si vizitati EDA inainte de a ajunge aici!")
        st.stop()

    if "ml_df" not in st.session_state:
        st.session_state["ml_df"] = st.session_state["eda_df"].copy()

    cols_to_drop = ["title", "release_date"]

    st.session_state["ml_df"] = st.session_state["ml_df"].drop(columns=cols_to_drop, errors="ignore")

    df_ml = st.session_state["ml_df"]

    # codificarea datelor
    st.markdown("### Codificarea variabilelor categorice")
    st.write("Alegeti coloanele de tip text si metoda matematica prin care doriti sa le transformati in numere.")

    col_cat = df_ml.select_dtypes(include=["object"]).columns.tolist()

    col_num = df_ml.select_dtypes(include=["float64", "int64"]).columns.tolist()

    cols_to_encode = st.multiselect("Alegeti coloanele pe care doriti sa le codificati (ex: genre, console):",
                                    options=col_cat,
                                    default=["genre", "console"] if "genre" in col_cat else [])

    encoding_method = st.selectbox("Alegeti metoda de codificare:",
                                   ["One-Hot Encoding (genereaza coloane separate de 0/1)", "Label Encoding (inlocuieste textul cu cifre: 0, 1, 2, ...)", "Frequency Encoding (inlocuieste cu numarul de aparitii)", "Target Encoding (inlocuieste cu media variabilei tinta)"])

    target_col = None
    if encoding_method == "Target Encoding (inlocuieste cu media variabilei tinta)":
        target_col = st.selectbox("Alegeti variabila tinta", options=col_num)

    if st.button("Aplicati codificarea"):
        if cols_to_encode:
            if encoding_method == "One-Hot Encoding (genereaza coloane separate de 0/1)":
                st.session_state["ml_df"] = pd.get_dummies(st.session_state["ml_df"], columns=cols_to_encode,
                                                           drop_first=True)
                st.success("One-Hot Encoding aplicat cu succes!")
                st.rerun()
            elif encoding_method == "Label Encoding (inlocuieste textul cu cifre: 0, 1, 2, ...)":
                le = LabelEncoder()
                for col in cols_to_encode:
                    st.session_state["ml_df"][col] = le.fit_transform(st.session_state["ml_df"][col].astype(str))
                st.success("Label Encoding aplicat cu succes!")
                st.rerun()
            elif encoding_method == "Frequency Encoding (inlocuieste cu numarul de aparitii)":
                for col in cols_to_encode:
                    # Calculam de cate ori apare fiecare categorie
                    freq_map = st.session_state["ml_df"][col].value_counts().to_dict()
                    # Cream o coloana noua cu frecventa si o stergem pe cea veche text
                    st.session_state["ml_df"][f"{col}_freq"] = st.session_state["ml_df"][col].map(freq_map)
                    st.session_state["ml_df"] = st.session_state["ml_df"].drop(columns=[col])
                st.success(f"Frequency Encoding aplicat cu succes!")
                st.rerun()
            elif encoding_method == "Target Encoding (inlocuieste cu media variabilei tinta)":
                if target_col:
                    for col in cols_to_encode:
                        # Calculam media variabilei tinta pentru fiecare categorie
                        target_mean_map = st.session_state["ml_df"].groupby(col)[target_col].mean().to_dict()
                        # Mapam valorile
                        st.session_state["ml_df"][f"{col}_target_enc"] = st.session_state["ml_df"][col].map(
                            target_mean_map)
                        st.session_state["ml_df"] = st.session_state["ml_df"].drop(columns=[col])
                    st.success(f"Target Encoding (bazat pe {target_col}) aplicat cu succes!")
                    st.rerun()
                else:
                    st.error("Selectați o variabila tinta pentru a putea aplica Target Encoding!")
        else:
            st.error("Selectati cel putin o coloana!")

    # scalarea datelor
    st.markdown("### Scalarea variabilelor numerice")
    st.write("Vom folosi `StandardScaler` pentru a aduce valorile mari (ex: vanzarile) la aceeasi scara cu cele mici (ex: scorurile)")

    df_ml = st.session_state["ml_df"]

    col_num = df_ml.select_dtypes(include=["float64", "int64"]).columns.tolist()

    default_scale_cols = [col for col in col_num if col != "total_sales"]

    cols_to_scale = st.multiselect("Alegeti coloanele pe care doriti sa le scalati:",
                                   options=col_num,
                                   default=default_scale_cols)

    if st.button("Aplicati scalarea standard"):
        if cols_to_scale:
            scaler = StandardScaler()

            st.session_state["ml_df"][cols_to_scale] = scaler.fit_transform(st.session_state["ml_df"][cols_to_scale])
            st.success("Scalarea a fost realizata cu succes!")
            st.rerun()
        else:
            st.error("Selectati cel putin o coloana!")

    st.markdown("***")
    st.markdown("### Previzualizarea datelor pregatite pentru ML")
    st.dataframe(st.session_state["ml_df"].head(10), width="stretch")

    st.markdown("### Exportare dataset ML")
    csv_ml = st.session_state["ml_df"].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Descarcati datasetul pregatit pentru ML",
        data=csv_ml,
        file_name="dataset_ml_ready.csv",
        mime="text/csv"
    )
