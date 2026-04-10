import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
import seaborn as sb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import statsmodels.api as sm

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score


image = Image.open('ps4-console.png')

st.image(image, width="stretch")

st.markdown("# Analiza vanzarilor de jocuri video", text_alignment="justify")

section = st.sidebar.radio("Navigati catre:",
                           ["Introducere", "Preprocesare", "Analiza exploratorie (EDA)", "Pregatirea datelor pentru ML", "Modelare ML - Clusterizare KMeans", "Modelare ML - Regresie liniara multipla", "Modelare ML - Random Forest Regressor", "Modelare ML - Regresie logistica"])

if section == "Introducere":
    st.markdown("""
    ## Motivația și obiectivul proiectului

    Acest proiect are ca scop analiza vânzărilor de jocuri video pe baza unui set de date real, pentru a evidenția tipare relevante legate de platforme, genuri, regiuni și performanța comercială a titlurilor lansate de-a lungul timpului.

    Motivația proiectului pornește de la interesul pentru industria gaming-ului și de la dorința de a aplica, într-un context practic, concepte de **preprocesare a datelor**, **analiză exploratorie** și **vizualizare interactivă**.

    """, text_alignment="justify")

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
        st.markdown("##### Imputare pentru coloane categorice")
        col_cat = df.select_dtypes(include=["object"]).columns
        col_na_cat = [col for col in col_cat if df[col].isnull().sum() > 0]

        selected_col_cat = st.multiselect("Alegeti coloanele categorice:",
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
        f"Anul cu cele mai mari vanzari a fost **{int(best_year['release_year'])}**, generand aproximativ **{best_year['total_sales']:.2f} milioane** de copii vandute.")

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

elif section == "Modelare ML - Clusterizare KMeans":
    st.markdown("""
        ***
        ## Modelare ML - Clusterizare KMeans
    """)

    if "ml_df" not in st.session_state or st.session_state["ml_df"] is None:
        st.warning("Pregatiti mai intai datele in sectiunea anterioara!")
        st.stop()

    if "kmeans_df" not in st.session_state:
        st.session_state["kmeans_df"] = None
    if "kmeans_features" not in st.session_state:
        st.session_state["kmeans_features"] = None
    if "kmeans_k" not in st.session_state:
        st.session_state["kmeans_k"] = None
    if "kmeans_score" not in st.session_state:
        st.session_state["kmeans_score"] = None
    if "kmeans_profile" not in st.session_state:
        st.session_state["kmeans_profile"] = None
    if "kmeans_pca_used" not in st.session_state:
        st.session_state["kmeans_pca_used"] = None
    if "kmeans_pca_variance" not in st.session_state:
        st.session_state["kmeans_pca_variance"] = None
    if "kmeans_plot_df" not in st.session_state:
        st.session_state["kmeans_plot_df"] = None

    df_model = st.session_state["ml_df"].copy()

    st.markdown("### Selectarea variabilelor pentru clusterizare")
    st.write("KMeans poate fi aplicat doar pe variabile numerice. Alegeti cel putin doua variabile.")

    numerical_cols = df_model.select_dtypes(include=[np.number]).columns.tolist()

    if len(numerical_cols) < 2:
        st.error("Nu exista suficiente variabile numerice pentru a realiza clusterizarea.")
        st.stop()

    default_features = numerical_cols[:4] if len(numerical_cols) >= 4 else numerical_cols

    selected_features = st.multiselect(
        "Alegeti variabilele numerice:",
        options=numerical_cols,
        default=default_features
    )

    use_pca = st.selectbox(
        "Doriti sa aplicati PCA inainte de KMeans?",
        options=["Nu", "Da"]
    )

    n_components = None
    if use_pca == "Da":
        max_components = len(selected_features) if len(selected_features) >= 2 else 2
        n_components = st.slider(
            "Alegeti numarul de componente principale:",
            min_value=2,
            max_value=max_components,
            value=2
        )
        st.info(
            "PCA se aplica dupa codificare si scalare. Aceasta metoda reduce dimensionalitatea datelor "
            "si poate imbunatati interpretarea clusterelor."
        )

    k = st.slider("Alegeti numarul de clustere (K):", min_value=2, max_value=8, value=3)

    if st.button("Aplicati clusterizarea KMeans"):
        if len(selected_features) < 2:
            st.error("Selectati cel putin doua variabile pentru clusterizare!")
        else:
            df_kmeans_raw = df_model[selected_features].dropna().copy()

            if len(df_kmeans_raw) < k:
                st.error("Numarul de observatii este prea mic pentru numarul de clustere selectat.")
            else:
                if use_pca == "Da":
                    pca = PCA(n_components=n_components)
                    X_transformed = pca.fit_transform(df_kmeans_raw)

                    pca_columns = [f"PC{i+1}" for i in range(n_components)]
                    df_kmeans_model = pd.DataFrame(X_transformed, columns=pca_columns, index=df_kmeans_raw.index)

                    model = KMeans(n_clusters=k, random_state=42, n_init=10)
                    clusters = model.fit_predict(df_kmeans_model)

                    df_kmeans_result = df_kmeans_raw.copy()
                    for col in pca_columns:
                        df_kmeans_result[col] = df_kmeans_model[col]
                    df_kmeans_result["Cluster"] = clusters

                    cluster_profile = df_kmeans_result.groupby("Cluster")[selected_features].mean().round(2)
                    silhouette = silhouette_score(df_kmeans_model, clusters)
                    explained_variance = pca.explained_variance_ratio_.sum()

                    st.session_state["kmeans_df"] = df_kmeans_result.copy()
                    st.session_state["kmeans_features"] = selected_features
                    st.session_state["kmeans_k"] = k
                    st.session_state["kmeans_score"] = silhouette
                    st.session_state["kmeans_profile"] = cluster_profile
                    st.session_state["kmeans_pca_used"] = "Da"
                    st.session_state["kmeans_pca_variance"] = explained_variance
                    st.session_state["kmeans_plot_df"] = df_kmeans_result[pca_columns + ["Cluster"]].copy()

                    st.success("Clusterizarea KMeans cu PCA a fost realizata cu succes!")

                else:
                    model = KMeans(n_clusters=k, random_state=42, n_init=10)
                    clusters = model.fit_predict(df_kmeans_raw)

                    df_kmeans_result = df_kmeans_raw.copy()
                    df_kmeans_result["Cluster"] = clusters

                    cluster_profile = df_kmeans_result.groupby("Cluster")[selected_features].mean().round(2)
                    silhouette = silhouette_score(df_kmeans_raw, clusters)

                    st.session_state["kmeans_df"] = df_kmeans_result.copy()
                    st.session_state["kmeans_features"] = selected_features
                    st.session_state["kmeans_k"] = k
                    st.session_state["kmeans_score"] = silhouette
                    st.session_state["kmeans_profile"] = cluster_profile
                    st.session_state["kmeans_pca_used"] = "Nu"
                    st.session_state["kmeans_pca_variance"] = None
                    st.session_state["kmeans_plot_df"] = df_kmeans_result.copy()

                    st.success("Clusterizarea KMeans a fost realizata cu succes!")

    if (
        st.session_state["kmeans_df"] is not None and
        st.session_state["kmeans_features"] is not None and
        st.session_state["kmeans_score"] is not None and
        st.session_state["kmeans_profile"] is not None and
        st.session_state["kmeans_pca_used"] is not None
    ):
        st.markdown("***")
        st.markdown("### Rezultatele clusterizarii")

        st.info(
            f"Scorul de separare intre clustere (Silhouette Score) este: **{st.session_state['kmeans_score']:.3f}**"
        )

        if st.session_state["kmeans_pca_used"] == "Da" and st.session_state["kmeans_pca_variance"] is not None:
            st.write(
                f"Proportia din variatia totala explicata de componentele PCA selectate este: "
                f"**{st.session_state['kmeans_pca_variance']:.2%}**"
            )

        st.markdown("### Datasetul clusterizat")
        st.dataframe(st.session_state["kmeans_df"], width="stretch")

        st.markdown("### Dimensiunea fiecarui cluster")
        cluster_count = st.session_state["kmeans_df"]["Cluster"].value_counts().sort_index().reset_index()
        cluster_count.columns = ["Cluster", "Numar observatii"]

        fig_count = px.bar(
            cluster_count,
            x="Cluster",
            y="Numar observatii",
            title="Numarul de observatii din fiecare cluster",
            labels={"Cluster": "Cluster", "Numar observatii": "Numar observatii"}
        )
        st.plotly_chart(fig_count, use_container_width=True)

        st.markdown("### Profilul mediu al clusterelor")
        st.dataframe(st.session_state["kmeans_profile"], width="stretch")

        st.markdown("### Reprezentare grafica a clusterelor")
        if st.session_state["kmeans_pca_used"] == "Da":
            fig_cluster = px.scatter(
                st.session_state["kmeans_plot_df"],
                x="PC1",
                y="PC2",
                color=st.session_state["kmeans_plot_df"]["Cluster"].astype(str),
                title="Clusterizarea jocurilor dupa primele doua componente principale",
                labels={
                    "PC1": "Componenta principala 1",
                    "PC2": "Componenta principala 2",
                    "color": "Cluster"
                }
            )
            st.plotly_chart(fig_cluster, use_container_width=True)
        else:
            if len(st.session_state["kmeans_features"]) >= 2:
                fig_cluster = px.scatter(
                    st.session_state["kmeans_plot_df"],
                    x=st.session_state["kmeans_features"][0],
                    y=st.session_state["kmeans_features"][1],
                    color=st.session_state["kmeans_plot_df"]["Cluster"].astype(str),
                    title=f"Clusterizarea jocurilor dupa {st.session_state['kmeans_features'][0]} si {st.session_state['kmeans_features'][1]}",
                    labels={
                        st.session_state["kmeans_features"][0]: st.session_state["kmeans_features"][0],
                        st.session_state["kmeans_features"][1]: st.session_state["kmeans_features"][1],
                        "color": "Cluster"
                    },
                    hover_data=st.session_state["kmeans_features"]
                )
                st.plotly_chart(fig_cluster, use_container_width=True)

        st.markdown("### Interpretare economica")
        if st.session_state["kmeans_pca_used"] == "Da":
            st.write(
                "Clusterizarea KMeans a fost aplicata dupa reducerea dimensionalitatii prin PCA. "
                "Aceasta abordare comprima informatia din mai multe variabile numerice in cateva componente principale, "
                "permitand identificarea unor segmente de jocuri video cu profiluri comerciale similare."
            )
        else:
            st.write(
                "Clusterizarea KMeans grupeaza jocurile video in categorii omogene, pe baza caracteristicilor numerice selectate. "
                "Astfel, putem identifica segmente de jocuri cu performanta comerciala similara, jocuri cu succes regional ridicat "
                "sau titluri care se diferentiaza clar fata de restul pietei."
            )

        st.markdown("### Exportare rezultate clusterizare")
        csv_cluster = st.session_state["kmeans_df"].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Descarcati datasetul clusterizat",
            data=csv_cluster,
            file_name="dataset_clusterizat_kmeans.csv",
            mime="text/csv"
        )


elif section == "Modelare ML - Regresie liniara multipla":
    st.markdown("""
        ***
        ## Modelare ML - Regresie liniara multipla
    """)

    if "ml_df" not in st.session_state or st.session_state["ml_df"] is None:
        st.warning("Pregatiti mai intai datele in sectiunea anterioara!")
        st.stop()

    if "reg_metrics" not in st.session_state:
        st.session_state["reg_metrics"] = None
    if "reg_coef_df" not in st.session_state:
        st.session_state["reg_coef_df"] = None
    if "reg_pred_df" not in st.session_state:
        st.session_state["reg_pred_df"] = None
    if "reg_summary_df" not in st.session_state:
        st.session_state["reg_summary_df"] = None
    if "reg_target" not in st.session_state:
        st.session_state["reg_target"] = None
    if "reg_features" not in st.session_state:
        st.session_state["reg_features"] = None

    df_model = st.session_state["ml_df"].copy()

    st.markdown("### Selectarea variabilei tinta si a predictorilor")
    st.write("Regresia liniara multipla explica influenta mai multor variabile independente asupra unei variabile tinta numerice.")

    numerical_cols = df_model.select_dtypes(include=[np.number]).columns.tolist()

    if len(numerical_cols) < 2:
        st.error("Nu exista suficiente variabile numerice pentru a realiza regresia.")
        st.stop()

    default_target = "total_sales" if "total_sales" in numerical_cols else numerical_cols[0]

    target_col = st.selectbox("Alegeti variabila tinta:", options=numerical_cols,
                              index=numerical_cols.index(default_target))

    predictor_options = [col for col in numerical_cols if col != target_col]

    default_predictors = predictor_options[:4] if len(predictor_options) >= 4 else predictor_options

    selected_features = st.multiselect(
        "Alegeti variabilele independente:",
        options=predictor_options,
        default=default_predictors
    )

    test_size_percent = st.slider("Alegeti procentul de date pentru testare (%):",
                                  min_value=10, max_value=40, value=20, step=5)

    if st.button("Aplicati regresia liniara multipla"):
        if len(selected_features) < 1:
            st.error("Selectati cel putin o variabila independenta!")
        else:
            df_reg = df_model[[target_col] + selected_features].dropna().copy()

            if len(df_reg) < 10:
                st.error("Nu exista suficiente observatii pentru a estima modelul de regresie.")
            else:
                X = df_reg[selected_features]
                y = df_reg[target_col]

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=test_size_percent / 100,
                    random_state=42
                )

                model = LinearRegression()
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)

                coef_df = pd.DataFrame({
                    "Variabila": selected_features,
                    "Coeficient": model.coef_
                }).sort_values(by="Coeficient", ascending=False)

                pred_df = pd.DataFrame({
                    "Valori reale": y_test.values,
                    "Valori prezise": y_pred
                })

                X_sm = sm.add_constant(X)
                ols_model = sm.OLS(y, X_sm).fit()

                summary_df = pd.DataFrame({
                    "Variabila": ols_model.params.index,
                    "Coeficient": ols_model.params.values,
                    "P-value": ols_model.pvalues.values
                })

                st.session_state["reg_metrics"] = {
                    "MAE": mae,
                    "RMSE": rmse,
                    "R2": r2,
                    "Intercept": model.intercept_
                }
                st.session_state["reg_coef_df"] = coef_df
                st.session_state["reg_pred_df"] = pred_df
                st.session_state["reg_summary_df"] = summary_df
                st.session_state["reg_target"] = target_col
                st.session_state["reg_features"] = selected_features

                st.success("Regresia liniara multipla a fost estimata cu succes!")

    if (
        st.session_state["reg_metrics"] is not None and
        st.session_state["reg_coef_df"] is not None and
        st.session_state["reg_pred_df"] is not None and
        st.session_state["reg_summary_df"] is not None
    ):
        st.markdown("***")
        st.markdown("### Rezultatele regresiei")

        st.markdown("### Indicatori de performanta ai modelului")
        st.write(f"**MAE:** {st.session_state['reg_metrics']['MAE']:.4f}")
        st.write(f"**RMSE:** {st.session_state['reg_metrics']['RMSE']:.4f}")
        st.write(f"**R²:** {st.session_state['reg_metrics']['R2']:.4f}")
        st.write(f"**Intercept:** {st.session_state['reg_metrics']['Intercept']:.4f}")

        st.markdown("### Coeficientii modelului (scikit-learn)")
        st.dataframe(st.session_state["reg_coef_df"], width="stretch")

        st.markdown("### Coeficienti si semnificatie statistica (statsmodels)")
        st.dataframe(st.session_state["reg_summary_df"], width="stretch")

        st.markdown("### Comparatie intre valorile reale si cele prezise")
        st.dataframe(st.session_state["reg_pred_df"].head(20), width="stretch")

        fig_pred = px.scatter(
            st.session_state["reg_pred_df"],
            x="Valori reale",
            y="Valori prezise",
            title=f"Valori reale vs valori prezise pentru {st.session_state['reg_target']}",
            labels={"Valori reale": "Valori reale", "Valori prezise": "Valori prezise"}
        )
        fig_pred.add_shape(
            type="line",
            x0=st.session_state["reg_pred_df"]["Valori reale"].min(),
            y0=st.session_state["reg_pred_df"]["Valori reale"].min(),
            x1=st.session_state["reg_pred_df"]["Valori reale"].max(),
            y1=st.session_state["reg_pred_df"]["Valori reale"].max(),
            line=dict(dash="dash", color="gray")
        )
        st.plotly_chart(fig_pred, use_container_width=True)

        st.markdown("### Interpretare economica")
        st.write(
            "Regresia liniara multipla estimeaza influenta simultana a mai multor factori asupra variabilei tinta. "
            "Coeficientii pozitivi sugereaza o relatie directa, iar coeficientii negativi sugereaza o relatie inversa. "
            "Valoarea R² arata in ce masura modelul explica variatia variabilei tinta, iar p-value ne ajuta sa identificam "
            "variabilele cu influenta statistica relevanta."
        )

        st.markdown("### Exportare rezultate regresie")
        csv_reg = st.session_state["reg_pred_df"].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Descarcati tabelul cu valori reale si prezise",
            data=csv_reg,
            file_name="rezultate_regresie_liniara.csv",
            mime="text/csv"
        )

elif section == "Modelare ML - Random Forest Regressor":
    st.markdown("""
        ***
        ## Modelare ML - Random Forest Regressor
    """)

    if "ml_df" not in st.session_state or st.session_state["ml_df"] is None:
        st.warning("Pregatiti mai intai datele in sectiunea anterioara!")
        st.stop()

    if "rf_metrics" not in st.session_state:
        st.session_state["rf_metrics"] = None
    if "rf_importance_df" not in st.session_state:
        st.session_state["rf_importance_df"] = None
    if "rf_pred_df" not in st.session_state:
        st.session_state["rf_pred_df"] = None
    if "rf_target" not in st.session_state:
        st.session_state["rf_target"] = None
    if "rf_features" not in st.session_state:
        st.session_state["rf_features"] = None

    df_model = st.session_state["ml_df"].copy()

    st.markdown("### Selectarea variabilei tinta si a predictorilor")
    st.write("Random Forest Regressor este un algoritm de predictie care poate surprinde relatii complexe si neliniare intre variabile.")

    numerical_cols = df_model.select_dtypes(include=[np.number]).columns.tolist()

    if len(numerical_cols) < 2:
        st.error("Nu exista suficiente variabile numerice pentru a realiza modelul Random Forest.")
        st.stop()

    default_target = "total_sales" if "total_sales" in numerical_cols else numerical_cols[0]

    target_col = st.selectbox(
        "Alegeti variabila tinta:",
        options=numerical_cols,
        index=numerical_cols.index(default_target)
    )

    predictor_options = [col for col in numerical_cols if col != target_col]

    if target_col == "total_sales":
        st.info(
            "Recomandare: pentru o analiza mai corecta, evitati folosirea directa a variabilelor regionale "
            "(na_sales, pal_sales, jp_sales, other_sales) ca predictori pentru total_sales."
        )

    preferred_predictors = [col for col in predictor_options if col not in ["na_sales", "pal_sales", "jp_sales", "other_sales"]]
    default_predictors = preferred_predictors[:4] if len(preferred_predictors) >= 4 else predictor_options[:4]

    selected_features = st.multiselect(
        "Alegeti variabilele independente:",
        options=predictor_options,
        default=default_predictors
    )

    test_size_percent = st.slider(
        "Alegeti procentul de date pentru testare (%):",
        min_value=10, max_value=40, value=20, step=5
    )

    n_estimators = st.slider(
        "Alegeti numarul de arbori (n_estimators):",
        min_value=50, max_value=300, value=100, step=50
    )

    max_depth_option = st.selectbox(
        "Alegeti adancimea maxima a arborilor:",
        options=["Fara limita", 5, 10, 15, 20]
    )

    if st.button("Aplicati modelul Random Forest Regressor"):
        if len(selected_features) < 1:
            st.error("Selectati cel putin o variabila independenta!")
        else:
            df_rf = df_model[[target_col] + selected_features].dropna().copy()

            if len(df_rf) < 10:
                st.error("Nu exista suficiente observatii pentru a estima modelul Random Forest.")
            else:
                X = df_rf[selected_features]
                y = df_rf[target_col]

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=test_size_percent / 100,
                    random_state=42
                )

                max_depth_value = None if max_depth_option == "Fara limita" else max_depth_option

                model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth_value,
                    random_state=42
                )

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)

                importance_df = pd.DataFrame({
                    "Variabila": selected_features,
                    "Importanta": model.feature_importances_
                }).sort_values(by="Importanta", ascending=False)

                pred_df = pd.DataFrame({
                    "Valori reale": y_test.values,
                    "Valori prezise": y_pred
                })

                st.session_state["rf_metrics"] = {
                    "MAE": mae,
                    "RMSE": rmse,
                    "R2": r2,
                    "n_estimators": n_estimators,
                    "max_depth": max_depth_option
                }
                st.session_state["rf_importance_df"] = importance_df
                st.session_state["rf_pred_df"] = pred_df
                st.session_state["rf_target"] = target_col
                st.session_state["rf_features"] = selected_features

                st.success("Modelul Random Forest Regressor a fost estimat cu succes!")

    if (
        st.session_state["rf_metrics"] is not None and
        st.session_state["rf_importance_df"] is not None and
        st.session_state["rf_pred_df"] is not None
    ):
        st.markdown("***")
        st.markdown("### Rezultatele modelului Random Forest")

        st.markdown("### Indicatori de performanta ai modelului")
        st.write(f"**MAE:** {st.session_state['rf_metrics']['MAE']:.4f}")
        st.write(f"**RMSE:** {st.session_state['rf_metrics']['RMSE']:.4f}")
        st.write(f"**R²:** {st.session_state['rf_metrics']['R2']:.4f}")
        st.write(f"**Numar arbori:** {st.session_state['rf_metrics']['n_estimators']}")
        st.write(f"**Adancime maxima:** {st.session_state['rf_metrics']['max_depth']}")

        st.markdown("### Importanta variabilelor")
        st.dataframe(st.session_state["rf_importance_df"], width="stretch")

        fig_importance = px.bar(
            st.session_state["rf_importance_df"],
            x="Variabila",
            y="Importanta",
            title="Importanta variabilelor in modelul Random Forest",
            labels={"Variabila": "Variabila", "Importanta": "Importanta"}
        )
        st.plotly_chart(fig_importance, use_container_width=True)

        st.markdown("### Comparatie intre valorile reale si cele prezise")
        st.dataframe(st.session_state["rf_pred_df"].head(20), width="stretch")

        fig_pred = px.scatter(
            st.session_state["rf_pred_df"],
            x="Valori reale",
            y="Valori prezise",
            title=f"Valori reale vs valori prezise pentru {st.session_state['rf_target']}",
            labels={"Valori reale": "Valori reale", "Valori prezise": "Valori prezise"}
        )
        fig_pred.add_shape(
            type="line",
            x0=st.session_state["rf_pred_df"]["Valori reale"].min(),
            y0=st.session_state["rf_pred_df"]["Valori reale"].min(),
            x1=st.session_state["rf_pred_df"]["Valori reale"].max(),
            y1=st.session_state["rf_pred_df"]["Valori reale"].max(),
            line=dict(dash="dash", color="gray")
        )
        st.plotly_chart(fig_pred, use_container_width=True)

        st.markdown("### Interpretare economica")
        st.write(
            "Modelul Random Forest Regressor estimeaza variabila tinta prin combinarea mai multor arbori de decizie. "
            "Acesta poate surprinde relatii complexe dintre variabile si permite identificarea predictorilor cu cea mai mare influenta "
            "asupra performantelor comerciale ale jocurilor video."
        )

        st.markdown("### Exportare rezultate Random Forest")
        csv_rf = st.session_state["rf_pred_df"].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Descarcati tabelul cu valori reale si prezise",
            data=csv_rf,
            file_name="rezultate_random_forest.csv",
            mime="text/csv"
        )

elif section == "Modelare ML - Regresie logistica":
    st.markdown("""
        ***
        ## Modelare ML - Regresie logistica
    """)

    if "ml_df" not in st.session_state or st.session_state["ml_df"] is None:
        st.warning("Pregatiti mai intai datele in sectiunea anterioara!")
        st.stop()

    if "log_metrics" not in st.session_state:
        st.session_state["log_metrics"] = None
    if "log_coef_df" not in st.session_state:
        st.session_state["log_coef_df"] = None
    if "log_pred_df" not in st.session_state:
        st.session_state["log_pred_df"] = None
    if "log_conf_matrix_df" not in st.session_state:
        st.session_state["log_conf_matrix_df"] = None
    if "log_target_base" not in st.session_state:
        st.session_state["log_target_base"] = None
    if "log_features" not in st.session_state:
        st.session_state["log_features"] = None
    if "log_class_distribution" not in st.session_state:
        st.session_state["log_class_distribution"] = None

    df_model = st.session_state["ml_df"].copy()

    st.markdown("### Definirea variabilei tinta binare")
    st.write(
        "Regresia logistica este un model de clasificare. In acest caz, vom transforma o variabila numerica "
        "intr-o variabila binara care indica daca un joc poate fi considerat succes comercial sau nu."
    )

    numerical_cols = df_model.select_dtypes(include=[np.number]).columns.tolist()

    if len(numerical_cols) < 2:
        st.error("Nu exista suficiente variabile numerice pentru a realiza modelul de regresie logistica.")
        st.stop()

    default_target_base = "total_sales" if "total_sales" in numerical_cols else numerical_cols[0]

    target_base_col = st.selectbox(
        "Alegeti variabila numerica pe baza careia se construieste tinta binara:",
        options=numerical_cols,
        index=numerical_cols.index(default_target_base)
    )

    threshold_method = st.selectbox(
        "Alegeti metoda de definire a pragului:",
        options=["Mediana", "Prag personalizat"]
    )

    if threshold_method == "Mediana":
        threshold_value = float(df_model[target_base_col].median())
        st.info(f"Pragul ales automat (mediana) este: **{threshold_value:.4f}**")
    else:
        min_value = float(df_model[target_base_col].min())
        max_value = float(df_model[target_base_col].max())
        default_value = float(df_model[target_base_col].median())

        threshold_value = st.slider(
            "Alegeti pragul personalizat:",
            min_value=min_value,
            max_value=max_value,
            value=default_value
        )

    st.markdown("### Selectarea predictorilor")
    predictor_options = [col for col in numerical_cols if col != target_base_col]

    if target_base_col == "total_sales":
        st.info(
            "Recomandare: pentru o analiza mai corecta, evitati folosirea directa a variabilelor regionale "
            "(na_sales, pal_sales, jp_sales, other_sales) ca predictori pentru total_sales."
        )

    preferred_predictors = [col for col in predictor_options if col not in ["na_sales", "pal_sales", "jp_sales", "other_sales"]]
    default_predictors = preferred_predictors[:4] if len(preferred_predictors) >= 4 else predictor_options[:4]

    selected_features = st.multiselect(
        "Alegeti variabilele independente:",
        options=predictor_options,
        default=default_predictors
    )

    test_size_percent = st.slider(
        "Alegeti procentul de date pentru testare (%):",
        min_value=10, max_value=40, value=20, step=5
    )

    if st.button("Aplicati modelul de regresie logistica"):
        if len(selected_features) < 1:
            st.error("Selectati cel putin o variabila independenta!")
        else:
            df_log = df_model[[target_base_col] + selected_features].dropna().copy()

            if len(df_log) < 10:
                st.error("Nu exista suficiente observatii pentru a estima modelul de regresie logistica.")
            else:
                df_log["target_binar"] = (df_log[target_base_col] > threshold_value).astype(int)

                class_distribution = df_log["target_binar"].value_counts().sort_index().reset_index()
                class_distribution.columns = ["Clasa", "Numar observatii"]

                if df_log["target_binar"].nunique() < 2:
                    st.error("Variabila tinta binara are o singura clasa. Modificati pragul pentru a obtine doua clase.")
                else:
                    X = df_log[selected_features]
                    y = df_log["target_binar"]

                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y,
                        test_size=test_size_percent / 100,
                        random_state=42,
                        stratify=y
                    )

                    model = LogisticRegression(max_iter=1000)
                    model.fit(X_train, y_train)

                    y_pred = model.predict(X_test)
                    y_prob = model.predict_proba(X_test)[:, 1]

                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, zero_division=0)
                    recall = recall_score(y_test, y_pred, zero_division=0)
                    f1 = f1_score(y_test, y_pred, zero_division=0)

                    conf_matrix = confusion_matrix(y_test, y_pred)
                    conf_matrix_df = pd.DataFrame(
                        conf_matrix,
                        index=["Real 0", "Real 1"],
                        columns=["Prezis 0", "Prezis 1"]
                    )

                    coef_df = pd.DataFrame({
                        "Variabila": selected_features,
                        "Coeficient": model.coef_[0]
                    }).sort_values(by="Coeficient", ascending=False)

                    pred_df = pd.DataFrame({
                        "Valoare reala": y_test.values,
                        "Valoare prezisa": y_pred,
                        "Probabilitate clasa 1": y_prob
                    })

                    st.session_state["log_metrics"] = {
                        "Accuracy": accuracy,
                        "Precision": precision,
                        "Recall": recall,
                        "F1": f1,
                        "Threshold": threshold_value
                    }
                    st.session_state["log_coef_df"] = coef_df
                    st.session_state["log_pred_df"] = pred_df
                    st.session_state["log_conf_matrix_df"] = conf_matrix_df
                    st.session_state["log_target_base"] = target_base_col
                    st.session_state["log_features"] = selected_features
                    st.session_state["log_class_distribution"] = class_distribution

                    st.success("Modelul de regresie logistica a fost estimat cu succes!")

    if (
        st.session_state["log_metrics"] is not None and
        st.session_state["log_coef_df"] is not None and
        st.session_state["log_pred_df"] is not None and
        st.session_state["log_conf_matrix_df"] is not None and
        st.session_state["log_class_distribution"] is not None
    ):
        st.markdown("***")
        st.markdown("### Rezultatele modelului de regresie logistica")

        st.markdown("### Distributia claselor")
        st.dataframe(st.session_state["log_class_distribution"], width="stretch")

        fig_class = px.bar(
            st.session_state["log_class_distribution"],
            x="Clasa",
            y="Numar observatii",
            title="Distributia observatiilor pe clase",
            labels={"Clasa": "Clasa", "Numar observatii": "Numar observatii"}
        )
        st.plotly_chart(fig_class, use_container_width=True)

        st.markdown("### Indicatori de performanta ai modelului")
        st.write(f"**Accuracy:** {st.session_state['log_metrics']['Accuracy']:.4f}")
        st.write(f"**Precision:** {st.session_state['log_metrics']['Precision']:.4f}")
        st.write(f"**Recall:** {st.session_state['log_metrics']['Recall']:.4f}")
        st.write(f"**F1-score:** {st.session_state['log_metrics']['F1']:.4f}")
        st.write(f"**Pragul utilizat:** {st.session_state['log_metrics']['Threshold']:.4f}")

        st.markdown("### Coeficientii modelului")
        st.dataframe(st.session_state["log_coef_df"], width="stretch")

        fig_coef = px.bar(
            st.session_state["log_coef_df"],
            x="Variabila",
            y="Coeficient",
            title="Coeficientii modelului de regresie logistica",
            labels={"Variabila": "Variabila", "Coeficient": "Coeficient"}
        )
        st.plotly_chart(fig_coef, use_container_width=True)

        st.markdown("### Matricea de confuzie")
        st.dataframe(st.session_state["log_conf_matrix_df"], width="stretch")

        fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
        sb.heatmap(st.session_state["log_conf_matrix_df"], annot=True, fmt="d", cmap="Blues", ax=ax_cm)
        ax_cm.set_title("Matricea de confuzie")
        st.pyplot(fig_cm)

        st.markdown("### Comparatie intre valorile reale si cele prezise")
        st.dataframe(st.session_state["log_pred_df"].head(20), width="stretch")

        fig_prob = px.scatter(
            st.session_state["log_pred_df"],
            x="Valoare reala",
            y="Probabilitate clasa 1",
            color=st.session_state["log_pred_df"]["Valoare prezisa"].astype(str),
            title="Probabilitatea estimata pentru clasa 1",
            labels={
                "Valoare reala": "Valoare reala",
                "Probabilitate clasa 1": "Probabilitate clasa 1",
                "color": "Clasa prezisa"
            }
        )
        st.plotly_chart(fig_prob, use_container_width=True)

        st.markdown("### Interpretare economica")
        st.write(
            "Modelul de regresie logistica estimeaza probabilitatea ca un joc sa apartina clasei de succes comercial, "
            "in functie de predictorii selectati. Coeficientii pozitivi sugereaza ca o crestere a variabilei respective "
            "mareste probabilitatea de apartenenta la clasa 1, iar coeficientii negativi reduc aceasta probabilitate."
        )

        st.markdown("### Exportare rezultate regresie logistica")
        csv_log = st.session_state["log_pred_df"].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Descarcati tabelul cu rezultate clasificare",
            data=csv_log,
            file_name="rezultate_regresie_logistica.csv",
            mime="text/csv"
        )