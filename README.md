# XAI-PAL Streamlit Research Demonstrator

## Run locally
1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Install dependencies:
   `pip install -r requirements.txt`
4. Start the app:
   `streamlit run app.py`

The two Excel workbooks must remain beside `app.py`:
- `Teachers form (Responses) (1).xlsx`
- `Parent_ Guardian form (Responses) (1).xlsx`

## Deploy on Streamlit Community Cloud
1. Create a GitHub repository and upload all four files (app.py, requirements.txt, README.md and the two Excel workbooks).
2. In Streamlit Community Cloud, choose **New app**, select the repository and `app.py`, then deploy.

## Research integrity
The survey dashboard uses the uploaded teacher and parent/guardian questionnaire data. The model sandbox is trained on synthetic data generated inside the app. It is a demonstration only, not a validated model or a result of the study. Do not use it for decisions about real students.
