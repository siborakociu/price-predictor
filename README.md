# Car price-predictor system - hybrid AI (XGBoost + Rule-based reasoning)

This project presents a hybrid AI system for car price prediction that combines:
-	Statistical learning – that uses an XGBoost regression model
- Symbolic reasoning- that uses a rule based system
The system is ran as an interactive Streamlit web application, allowing users to input car details and get an interpretable price prediction with explanations.

### Files / Code Description
vehicle-price-prediction-xgboost.ipynb- Jupyter notebook, which contains the full development workflow for the car price prediction project. It includes data preprocessing, feature engineering, model training using XGBoost, and the integration of rule-based reasoning for hybrid predictions, as well as visualizations and performance evaluation of the models.

streamlit_app.py- This script runs the interactive web application for car price prediction. Users can input vehicle details and receive predicted prices along with explanations, combining the XGBoost model predictions with rule-based adjustments for improved interpretability.

pkl files- used to save and load Python objects.

