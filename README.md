# 🎮 Analiza Exploratorie a Jocurilor Video (VGChartz 2024)

## 📌 Descrierea și Obiectivul Proiectului
Acest proiect este o aplicație web interactivă construită în **Python**, folosind bibliotecile **Streamlit** și **Pandas**. Obiectivul principal este realizarea unei Analize Exploratorii de Date (EDA) asupra pieței jocurilor video, oferind utilizatorului control total asupra modului în care datele sunt procesate și vizualizate.

## 📊 Setul de Date
Analiza se bazează pe setul de date `video-games-sales.csv`, care oferă o imagine de ansamblu asupra performanței jocurilor video la nivel global. Variabilele principale includ:
* **Vânzări (în milioane):** Globale (`total_sales`), America de Nord (`na_sales`), Japonia (`jp_sales`), Europa și Africa (`pal_sales`), etc.
* **Recepție:** Scoruri acordate de critici (`critic_score`).
* **Metadate:** Platformă (`console`), Gen (`genre`), Publisher, Developer și Data lansării.

## 🛠️ Stadiul Proiectului și Funcționalități

Proiectul este structurat momentan în 3 etape majore:

- [x] **Etapa 1: Încărcarea și Inspectarea Datelor**
  - Încărcarea dinamică a fișierului CSV direct din interfață.
  - Utilizarea `st.session_state` pentru a memora setul de date și a preveni resetarea aplicației la fiecare interacțiune.

- [x] **Etapa 2: Curățarea și Preprocesarea Datelor (Data Cleaning)**
  - Eliminarea coloanelor fără valoare predictivă/analitică (ex. URL-urile imaginilor `img` și `last_update`).
  - Filtrarea rândurilor esențiale: eliminarea jocurilor care nu au înregistrate vânzări totale (`total_sales`).
  - **Imputare interactivă:** Utilizatorul poate alege din interfață metoda de completare a valorilor lipsă:
    - *Pentru date numerice:* Înlocuire cu 0 (ideal pentru vânzările regionale lipsă), Medie sau Mediană.
    - *Pentru date text:* Înlocuire cu "Necunoscut" sau cu cea mai frecventă valoare (Modulul).

- [ ] **Etapa 3: Analiza Exploratorie (EDA) și Vizualizări**
  - *În dezvoltare* - Urmează implementarea graficelor interactive pentru analiza vânzărilor și a scorurilor.

## 🚀 Rularea Aplicației
Pentru a porni aplicația pe plan local, asigurați-vă că aveți instalate pachetele necesare și rulați în terminal comanda:
`streamlit run [nume_fisier].py`