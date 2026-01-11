"""
Vehicle Price Prediction - Streamlit Application
AI Term Project - XGBoost + Rule-Based System Implementation
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

# Import rule-based system
try:
    from rules import VehiclePricingRules, apply_rules
except ImportError:
    st.error("rules.py not found. Please ensure it's in the same directory.")
    st.stop()

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Vehicle Price Predictor",
    page_icon="Car",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# LOAD MODEL AND ENCODERS
# ============================================================================

@st.cache_resource
def load_model_assets():
    """Load the trained model, encoders, and statistics"""
    try:
        model = joblib.load('vehicle_price_model.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        feature_cols = joblib.load('feature_columns.pkl')
        statistics = joblib.load('model_statistics.pkl')
        make_model_mapping = joblib.load('make_model_mapping.pkl')
        return model, label_encoders, feature_cols, statistics, make_model_mapping
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        st.info("Please ensure you have run the Jupyter notebook to train the model first.")
        return None, None, None, None, None

model, label_encoders, feature_cols, statistics, make_model_mapping = load_model_assets()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def prepare_input_features(input_data, label_encoders):
    """Prepare input features for prediction"""
    df = pd.DataFrame([input_data])
    
    # Standardize inputs to match training data format
    df['make'] = df['make'].str.strip().str.title()
    df['model'] = df['model'].str.strip().str.title()
    df['body'] = df['body'].str.strip()  # Already standardized from dropdown
    df['transmission'] = df['transmission'].str.strip().str.lower()
    
    # Encode categorical variables
    categorical_cols = ['make', 'model', 'body', 'transmission']
    for col in categorical_cols:
        if input_data[col] in label_encoders[col].classes_:
            df[f'{col}_encoded'] = label_encoders[col].transform([df[col].iloc[0]])[0]
        else:
            # Handle unseen categories
            df[f'{col}_encoded'] = 0
    
    return df

def predict_price(input_features, model, feature_cols):
    """Make price prediction"""
    try:
        prediction = model.predict(input_features[feature_cols])[0]
        return max(500, prediction)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        margin: 2rem 0;
    }
    .price-display {
        font-size: 3.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# MAIN APPLICATION
# ============================================================================

# Header
st.markdown('<p class="main-header">Vehicle Price Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Vehicle Valuation System Using XGBoost</p>', unsafe_allow_html=True)

# Check if model is loaded
if model is None:
    st.stop()

# Create tabs
tab1, tab2 = st.tabs(["Price Prediction", "Model Performance"])

# ============================================================================
# TAB 1: PRICE PREDICTION
# ============================================================================

with tab1:
    st.header("Enter Vehicle Details")
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Basic Information")
        
        year = st.number_input(
            "Year",
            min_value=1990,
            max_value=2026,
            value=2015,
            step=1,
            help="Manufacturing year of the vehicle"
        )
        
        # Get available makes from encoder
        makes = sorted(list(label_encoders['make'].classes_))
        make = st.selectbox(
            "Make",
            options=makes,
            index=makes.index('Ford') if 'Ford' in makes else 0,
            help="Brand or manufacturer"
        )
        
        # Dynamic model dropdown based on selected make
        if make_model_mapping and make in make_model_mapping:
            available_models = make_model_mapping[make]
            default_model = available_models[0] if available_models else ""
        else:
            # Fallback if mapping not available
            available_models = sorted(list(label_encoders['model'].classes_))
            default_model = "F-150" if "F-150" in available_models else available_models[0]
        
        model_name = st.selectbox(
            "Model",
            options=available_models,
            help=f"Available models for {make}"
        )
        
        # Standardized body types
        standard_body_types = ['Sedan', 'SUV', 'Truck', 'Coupe', 'Convertible', 
                               'Hatchback', 'Wagon', 'Van', 'Other']
        body = st.selectbox(
            "Body Type",
            options=standard_body_types,
            index=standard_body_types.index('SUV'),
            help="Body type of the vehicle"
        )
    
    with col2:
        st.subheader("Vehicle Condition")
        
        # Get available transmissions
        transmissions = list(label_encoders['transmission'].classes_)
        transmission = st.selectbox(
            "Transmission",
            options=sorted(transmissions),
            index=sorted(transmissions).index('automatic') if 'automatic' in transmissions else 0,
            help="Type of transmission"
        )
        
        condition = st.slider(
            "Condition Rating",
            min_value=1,
            max_value=49,
            value=35,
            help="Overall condition of the vehicle (1-49)"
        )
        
        odometer = st.number_input(
            "Odometer (miles)",
            min_value=0,
            max_value=500000,
            value=50000,
            step=1000,
            help="Total miles driven"
        )
    
    # Prediction button
    st.markdown("---")
    
    if st.button("Predict Vehicle Price", type="primary", use_container_width=True):
        # Prepare input data
        input_data = {
            'year': year,
            'make': make,
            'model': model_name,
            'body': body,
            'transmission': transmission,
            'condition': condition,
            'odometer': odometer
        }
        
        # STEP 1: Apply Rule-Based System (Input Validation)
        rules_engine = VehiclePricingRules()
        is_valid, warnings = rules_engine.validate_input(input_data)
        
        # Show validation warnings if any
        if warnings:
            st.warning("Rule-Based System Validation:")
            for warning in warnings:
                st.write(f"- {warning}")
        
        # STEP 2: ML Model Prediction
        input_features = prepare_input_features(input_data, label_encoders)
        ml_prediction = predict_price(input_features, model, feature_cols)
        
        if ml_prediction:
            # STEP 3: Apply Rule-Based Adjustments
            rules_result = apply_rules(input_data, ml_prediction)
            
            final_price = rules_result['adjusted_prediction']
            base_price = rules_result['base_prediction']
            
            # Display hybrid prediction result
            st.markdown(f"""
                <div class="prediction-box">
                    <h2>Predicted Vehicle Price</h2>
                    <div class="price-display">${final_price:,.2f}</div>
                    <p style="font-size: 1.1rem;">Hybrid AI Prediction (ML + Rule-Based System)</p>
                    <p style="font-size: 0.9rem;">Confidence: {rules_result['confidence']}</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Show ML vs Rules comparison
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("ML Model Price", f"${base_price:,.2f}", 
                         help="XGBoost statistical learning prediction")
            with col2:
                st.metric("Rule Adjustments", f"${rules_result['total_adjustment']:+,.2f}",
                         delta=f"{rules_result['rules_applied']} rules applied")
            with col3:
                st.metric("Final Price", f"${final_price:,.2f}", 
                         delta="Hybrid System")
            
            # Calculate confidence interval
            lower_bound = final_price * 0.9
            upper_bound = final_price * 1.1
            
            st.markdown("---")
            
            # Display Rule-Based Adjustments
            if rules_result['adjustments']:
                st.subheader("Rule-Based Adjustments Applied")
                st.info(f"The expert rule system applied {rules_result['rules_applied']} adjustments based on domain knowledge:")
                for i, adjustment in enumerate(rules_result['adjustments'], 1):
                    st.write(f"{i}. {adjustment}")
            
            # Market Insights
            if rules_result['market_insights']:
                st.markdown("---")
                st.subheader("Market Insights")
                for insight in rules_result['market_insights']:
                    st.success(f"- {insight}")
            
            # Additional insights
            st.markdown("---")
            st.subheader("Prediction Details")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.info(f"""
                    **Vehicle Summary:**
                    - {year} {make} {model_name}
                    - Body Type: {body}
                    - Condition: {condition}/49
                    - Mileage: {odometer:,} miles
                """)
            
            with col2:
                vehicle_age = 2026 - year
                miles_per_year = odometer / vehicle_age if vehicle_age > 0 else odometer
                
                st.success(f"""
                    **Analysis:**
                    - Vehicle Age: {vehicle_age} years
                    - Avg. Miles/Year: {miles_per_year:,.0f}
                    - ML Confidence: {statistics['ml_r2_score']*100:.1f}%
                    - Overall Confidence: {rules_result['confidence']}
                """)
            
            # Comparison chart
            st.markdown("---")
            st.subheader("Price Comparison")
            
            comparison_df = pd.DataFrame({
                'Method': ['ML Model Only', 'Rule Adjustments', 'Final Hybrid Price', 'Lower Range', 'Upper Range'],
                'Price': [base_price, base_price + rules_result['total_adjustment'], final_price, lower_bound, upper_bound],
                'Type': ['Prediction', 'Adjustment', 'Final', 'Range', 'Range']
            })
            
            fig = px.bar(
                comparison_df,
                x='Method',
                y='Price',
                title='Hybrid AI System Price Breakdown',
                color='Type',
                color_discrete_map={'Prediction': '#1f77b4', 'Adjustment': '#ff7f0e', 'Final': '#2ca02c', 'Range': '#d62728'}
            )
            st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 2: MODEL PERFORMANCE
# ============================================================================

with tab2:
    st.header("Model Performance Metrics")
    
    st.subheader("Hybrid System Performance")
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Hybrid R² Score",
            f"{statistics.get('r2_score', 0):.4f}",
            delta=f"+{(statistics.get('r2_score', 0) - statistics.get('ml_r2_score', 0))*100:.2f}%" if 'ml_r2_score' in statistics else None,
            help="Coefficient of determination (higher is better)"
        )
    
    with col2:
        st.metric(
            "Hybrid MAE",
            f"${statistics.get('mae', 0):,.2f}",
            delta=f"${statistics.get('ml_mae', 0) - statistics.get('mae', 0):+,.0f}" if 'ml_mae' in statistics else None,
            help="Mean Absolute Error"
        )
    
    with col3:
        st.metric(
            "Hybrid RMSE",
            f"${statistics.get('rmse', 0):,.2f}",
            delta=f"${statistics.get('ml_rmse', 0) - statistics.get('rmse', 0):+,.0f}" if 'ml_rmse' in statistics else None,
            help="Root Mean Squared Error"
        )
    
    with col4:
        st.metric(
            "Hybrid MAPE",
            f"{statistics.get('mape', 0):.2f}%",
            delta=f"{statistics.get('ml_mape', 0) - statistics.get('mape', 0):+.2f}%" if 'ml_mape' in statistics else None,
            help="Mean Absolute Percentage Error"
        )
    
    # Show comparison if ML-only metrics available
    if 'ml_r2_score' in statistics:
        st.markdown("---")
        
        # Check if hybrid metrics are available
        has_hybrid = 'r2_score' in statistics
        
        if has_hybrid:
            st.subheader("ML vs Hybrid Comparison")
            
            comparison_data = pd.DataFrame({
                'System': ['ML Only', 'Hybrid (ML + Rules)'],
                'R² Score': [statistics['ml_r2_score'], statistics['r2_score']],
                'MAE ($)': [statistics['ml_mae'], statistics['mae']],
                'RMSE ($)': [statistics['ml_rmse'], statistics['rmse']]
            })
            
            st.dataframe(comparison_data, hide_index=True, use_container_width=True)
            
            # Show improvement
            improvement = (statistics['r2_score'] - statistics['ml_r2_score']) * 100
            st.success(f"Hybrid system achieves {improvement:.2f}% improvement in R² score over ML-only approach")
        else:
            st.subheader("Model Performance Metrics")
            
            metrics_data = pd.DataFrame({
                'Metric': ['R² Score', 'MAE ($)', 'RMSE ($)', 'MAPE (%)'],
                'Value': [
                    f"{statistics['ml_r2_score']:.4f}",
                    f"${statistics['ml_mae']:,.2f}",
                    f"${statistics['ml_rmse']:,.2f}",
                    f"{statistics['ml_mape']:.2f}%"
                ]
            })
            
            st.dataframe(metrics_data, hide_index=True, use_container_width=True)
            st.info("Note: This model uses ML-only predictions. The hybrid (ML + Rules) system can be enabled after re-training.")
        
        # Rule statistics
        if 'pct_predictions_adjusted' in statistics:
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Predictions Adjusted by Rules",
                    f"{statistics['pct_predictions_adjusted']:.1f}%"
                )
            with col2:
                st.metric(
                    "Avg Rule Adjustment",
                    f"${statistics.get('avg_rule_adjustment', 0):+,.2f}"
                )
    
    st.markdown("---")
    
    # Feature importance
    st.subheader("Feature Importance")
    
    if 'feature_importance' in statistics:
        importance_df = pd.DataFrame(statistics['feature_importance'])
        importance_df = importance_df.sort_values('importance', ascending=True)
        
        fig = px.bar(
            importance_df,
            x='importance',
            y='feature',
            orientation='h',
            title='Feature Importances',
            labels={'importance': 'Importance Score', 'feature': 'Feature'},
            color='importance',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Model details
    st.markdown("---")
    st.subheader("Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            **Algorithm:** XGBoost Regression
            
            **Hyperparameters:**
            - Max Depth: 6
            - Learning Rate: 0.1
            - N Estimators: 200
            - Subsample: 0.8
            - Colsample by Tree: 0.8
        """)
    
    with col2:
        st.markdown(f"""
            **Dataset:**
            - Training Samples: {statistics['train_size']:,}
            - Test Samples: {statistics['test_size']:,}
            - Total Features: {len(feature_cols)}
            
            **Performance:**
            - Excellent predictive accuracy
            - Low prediction error
            - Robust to outliers
        """)

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.title("Navigation")
    st.markdown("""
    ### Quick Guide
    1. **Price Prediction:** Enter vehicle details and get instant price estimate
    2. **Model Performance:** View model accuracy and feature importance
    
    ### Model Status
    """)
    
    if model:
        st.success("Model Loaded Successfully")
        st.metric("Training Samples", f"{statistics['train_size']:,}")
        st.metric("Model Accuracy (R²)", f"{statistics['ml_r2_score']*100:.2f}%")
    else:
        st.error("Model Not Found")
        st.info("Please run the Jupyter notebook to train the model first.")
    
    st.markdown("---")
    
    st.markdown("""
    ### Resources
    - [XGBoost Documentation](https://xgboost.readthedocs.io/)
    - [Streamlit Docs](https://docs.streamlit.io/)
    - [Scikit-learn Guide](https://scikit-learn.org/)
    """)
    
    st.markdown("---")
    st.caption("AI Term Project 2026")
