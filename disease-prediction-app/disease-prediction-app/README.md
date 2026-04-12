# 🩺 MediPredict — AI Disease Prediction Web App

An end-to-end machine-learning web application that predicts diseases from
selected symptoms and returns descriptions, diet plans, medications,
precautions, and workout recommendations — all in one clean, dark-themed UI.

---

## 📸 Screenshots

> After login, select symptoms from the multi-select dropdown and click
> **Analyse Symptoms** to see the AI prediction with supporting medical info.

---

## 🗂 Project Structure

```
disease-prediction-app/
│
├── app.py                        ← Main Streamlit app (router + pages)
├── auth.py                       ← User registration & login (SQLite)
│
├── model/
│   ├── train_model.py            ← Standalone training script
│   ├── model.pkl                 ← Trained RandomForest (compressed, ~15 MB)
│   └── feature_columns.csv       ← Ordered symptom column names
│
├── data/
│   ├── symptoms_disease.csv      ← Training dataset (symptoms → disease)
│   ├── descriptions.csv          ← Disease descriptions
│   ├── diets.csv                 ← Dietary recommendations per disease
│   ├── medications.csv           ← Common medications per disease
│   ├── precautions.csv           ← Precautions per disease
│   └── workouts.csv              ← Exercise recommendations per disease
│
├── utils/
│   ├── __init__.py
│   ├── preprocess.py             ← Symptom → feature vector conversion
│   └── predictor.py              ← Model inference + CSV data lookup
│
├── .streamlit/
│   └── config.toml               ← Streamlit theme & server config
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit + custom CSS |
| ML Model | scikit-learn RandomForestClassifier |
| Data | pandas + NumPy |
| Auth | SQLite via Python `sqlite3` |
| Serialisation | joblib (compressed pickle) |

---

## 🚀 Local Setup

### 1 — Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/disease-prediction-app.git
cd disease-prediction-app
```

### 2 — Create and activate a virtual environment

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### 4 — (Optional) Retrain the model

If you want to retrain from scratch with the full dataset:

```bash
python model/train_model.py
```

The pre-trained `model/model.pkl` is already included so this step is
**not required** to run the app.

### 5 — Launch the app

```bash
streamlit run app.py
```

The app will open automatically at **http://localhost:8501**.

---

## 🌐 Streamlit Cloud Deployment

1. **Push your repo to GitHub** (the `.gitignore` already excludes `users.db`
   and the large training CSV).

2. Go to [share.streamlit.io](https://share.streamlit.io) and click
   **New app**.

3. Connect your GitHub account and select the repo / branch / `app.py`.

4. Click **Deploy** — Streamlit Cloud will install `requirements.txt`
   automatically.

> **Note on `users.db`:** The SQLite file is ephemeral on Streamlit Cloud
> (it resets on each cold start). For production persistence, replace the
> SQLite backend in `auth.py` with a hosted DB (e.g. Supabase or PlanetScale).

---

## 📊 Dataset Format

### `data/symptoms_disease.csv`

Binary feature matrix — one row per patient record:

```
diseases,symptom_a,symptom_b,symptom_c,...
acne,0,1,0,...
pneumonia,1,0,1,...
```

### `data/descriptions.csv`

```
Disease,Description
acne,"Acne is a skin condition that occurs when..."
pneumonia,"Pneumonia is an infection that inflames..."
```

### `data/diets.csv`

```
Disease,Diet_1,Diet_2,Diet_3,Diet_4,Diet_5
acne,Reduce dairy products,Eat more fruits,...
```

### `data/medications.csv`

```
Disease,Medication_1,Medication_2,...,Medication_5
acne,Benzoyl peroxide cream,Salicylic acid wash,...
```

### `data/precautions.csv`

```
Disease,Precaution_1,...,Precaution_5
acne,Wash face twice daily,...
```

### `data/workouts.csv`

```
Disease,Workout_1,...,Workout_5
acne,Regular moderate exercise,...
```

---

## 🤖 ML Model Details

| Attribute | Value |
|---|---|
| Algorithm | `RandomForestClassifier` |
| Estimators | 150 trees |
| Max depth | 25 |
| Features | 377 binary symptom flags |
| Classes | 713 diseases |
| Training rows | ~25,000 (stratified sample — 40 per disease) |
| Test accuracy | **~76.5%** |
| Model size | ~15 MB (compressed) |

The model receives a binary vector (1 = symptom present, 0 = absent) and
returns the predicted disease plus class-probability scores used for the
top-5 confidence bar chart.

---

## 🔐 Authentication

- Passwords are hashed with **SHA-256** before storage — plain text is
  never persisted.
- User data lives in `users.db` (auto-created on first run, excluded from
  git via `.gitignore`).
- Session state is managed by Streamlit's `st.session_state`.

---

## ⚠️ Medical Disclaimer

This application is intended for **educational and informational purposes
only**. It is **not** a substitute for professional medical advice,
diagnosis, or treatment. Always seek the guidance of a qualified healthcare
provider with any questions you have regarding a medical condition.

---

## 🙏 Acknowledgements

- Dataset inspired by publicly available symptom–disease corpora.
- UI inspired by GitHub's dark design system.
- Built with [Streamlit](https://streamlit.io) and
  [scikit-learn](https://scikit-learn.org).
