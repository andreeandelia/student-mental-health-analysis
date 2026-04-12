# Analiza Vânzărilor de Jocuri Video

## Descriere

Acest proiect este o aplicație web interactivă dezvoltată în Python cu Streamlit, pentru analiza setului de date VGChartz (`video-games-sales.csv`).

## Setul de date

Fișierul folosit în proiect: `video-games-sales.csv`.

Exemple de coloane relevante:

- Vânzări globale și regionale: `total_sales`, `na_sales`, `pal_sales`, `jp_sales`, `other_sales`
- Informații despre joc: `title`, `genre`, `console`, `publisher`, `developer`, `release_date`
- Scoruri: `critic_score`, `user_score`

## Funcționalități implementate

### 1. Introducere și încărcare date

- Upload CSV din interfața Streamlit.
- Afișare date brute și dimensiunea dataset-ului.
- Persistența datelor în sesiune prin `st.session_state`.

### 2. Preprocesare

- Eliminare coloane irelevante.
- Reset dataset la forma inițială.
- Analiza valorilor lipsă (număr + procent NaN).
- Ștergere rânduri/coloane cu valori lipsă după selecție.
- Imputare interactivă:
  - numeric: `0`, medie, mediana
  - categoric: `Necunoscut`, mod
- Analiza outliers:
  - metoda IQR (raport tabelar)
  - boxplot pe variabila selectată
- Export dataset preprocesat (`dataset_preprocesat.csv`).

### 3. Analiza exploratorie (EDA)

- Top jocuri după vânzări globale.
- Evoluția vânzărilor în timp (pe ani).
- Heatmap vânzări pe console și genuri.
- Comparație regională între două piețe (scatter).
- Matrice de corelație pentru variabile numerice.

### 4. Pregătirea datelor pentru ML

- Eliminare automată a coloanelor text greu modelabile direct (`title`, `release_date`).
- Codificare variabile categorice:
  - One-Hot Encoding
  - Label Encoding
  - Frequency Encoding
  - Target Encoding
- Scalare variabile numerice cu `StandardScaler`.
- Export dataset gata pentru ML (`dataset_ml_ready.csv`).

### 5. Modelare ML

#### 5.1 Clusterizare KMeans

- Selecție variabile numerice.
- Opțiune PCA înainte de KMeans.
- Selecție `K` (numărul de clustere).
- Evaluare cu Silhouette Score.
- Profilare clustere + vizualizare + export (`dataset_clusterizat_kmeans.csv`).

#### 5.2 Regresie liniară multiplă

- Selecție variabilă țintă numerică și predictori.
- Split train/test configurabil.
- Metrici: MAE, RMSE, R2.
- Coeficienți model (scikit-learn) + semnificație statistică (statsmodels).
- Export predicții (`rezultate_regresie_liniara.csv`).

#### 5.3 Random Forest Regressor

- Selecție țintă/predictori.
- Hiperparametri configurabili (`n_estimators`, `max_depth`).
- Metrici: MAE, RMSE, R2.
- Importanța variabilelor + comparație real/predicție.
- Export predicții (`rezultate_random_forest.csv`).

#### 5.4 Regresie logistică (clasificare binară)

- Transformarea unei variabile numerice într-o țintă binară pe baza unui prag (mediana sau prag personalizat).
- Split train/test stratificat.
- Metrici: Accuracy, Precision, Recall, F1.
- Matrice de confuzie + coeficienți model.

## Structura proiectului

- `app.py` - aplicația principală Streamlit
- `video-games-sales.csv` - datasetul de intrare
- `ps4-console.png` - imagine folosită în interfață
- `README.md` - documentația proiectului

## Cerințe

- Python 3.10+ (recomandat)

Biblioteci Python utilizate:

- streamlit
- pandas
- numpy
- matplotlib
- seaborn
- plotly
- pillow
- scikit-learn
- statsmodels

## Instalare și rulare

1. (Opțional) Creează și activează un mediu virtual.
2. Instalează dependențele:

```bash
pip install streamlit pandas numpy matplotlib seaborn plotly pillow scikit-learn statsmodels
```

3. Rulează aplicația:

```bash
streamlit run app.py
```

4. În browser:

- se deschide automat pagina Streamlit (de obicei `http://localhost:8501`).

## Flux recomandat în aplicație

1. Introducere: încarcă CSV-ul.
2. Preprocesare: tratează NaN/outliers și exportă dacă este nevoie.
3. EDA: analizează pattern-urile principale.
4. Pregătire ML: codifică + scalează datele.
5. Rulează unul sau mai multe modele ML în secțiunile dedicate.

## Observații

- Aplicația folosește `st.session_state`, deci datele procesate sunt păstrate pe parcursul navigării între secțiuni.
- Pentru rezultate ML relevante, este recomandat să finalizezi preprocesarea înainte de modelare.
