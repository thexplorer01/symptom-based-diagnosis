# symptom-based-diagnosis

**Disease Detection from Symptoms** — 3rd year project

## Team
- Ayush Patil
- Chinmay Raut
- Pushpak Patil

**College:** St. John college of engineering and management

## Overview
A machine learning project that predicts likely disease(s) from a set of symptoms.
Provides a Streamlit demo app for interactive testing.

## Folder structure
```
symptom-based-diagnosis/
├─ data/
│  └─ dataset.csv
├─ src/
│  ├─ data_generator.py
│  ├─ train_model.py
│  └─ app_streamlit.py
├─ models/
│  └─ model.joblib
├─ report/
│  └─ Project_Report.docx
├─ ppt/
│  └─ Presentation.pptx
├─ requirements.txt
└─ README.md
```

## Setup
1. Create virtual env:
   ```
   python -m venv venv
   source venv/bin/activate   # Linux/macOS
   venv\Scripts\activate    # Windows
   ```

2. Install requirements:
   ```
   pip install -r requirements.txt
   ```

3. Run (if you want to regenerate the dataset and retrain):
   ```
   python src/data_generator.py
   python src/train_model.py
   streamlit run src/app_streamlit.py
   ```

## Notes
- This is a prototype using a synthetic dataset. Not for clinical use.
