"""
Shared utility functions for Vehicle Price Prediction
Can be used in both Jupyter Notebook and Streamlit app
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

class VehiclePricePredictor:
    """
    Wrapper class for vehicle price prediction
    Handles model loading, feature preparation, and prediction
    """
    
    def __init__(self, model_path='vehicle_price_model.pkl',
                 encoders_path='label_encoders.pkl',
                 features_path='feature_columns.pkl'):
        """
        Initialize the predictor by loading model and encoders
        
        Parameters:
        -----------
        model_path : str
            Path to the saved XGBoost model
        encoders_path : str
            Path to the saved label encoders
        features_path : str
            Path to the saved feature column names
        """
        self.model = joblib.load(model_path)
        self.label_encoders = joblib.load(encoders_path)
        self.feature_cols = joblib.load(features_path)
        
    def prepare_features(self, input_data):
        """
        Prepare input features for prediction
        
        Parameters:
        -----------
        input_data : dict
            Dictionary containing vehicle information
            Required keys: year, make, model, body, transmission, condition, odometer
        
        Returns:
        --------
        pd.DataFrame
            Prepared features ready for prediction
        """
        # Convert to DataFrame
        df = pd.DataFrame([input_data])
        
        # Standardize inputs to match training data format
        df['make'] = df['make'].str.strip().str.title()
        df['model'] = df['model'].str.strip().str.title()
        df['body'] = df['body'].str.strip()
        df['transmission'] = df['transmission'].str.strip().str.lower()
        
        # Encode categorical variables
        categorical_cols = ['make', 'model', 'body', 'transmission']
        for col in categorical_cols:
            try:
                if df[col].iloc[0] in self.label_encoders[col].classes_:
                    df[f'{col}_encoded'] = self.label_encoders[col].transform([df[col].iloc[0]])[0]
                else:
                    # Handle unseen categories
                    df[f'{col}_encoded'] = 0
            except KeyError:
                df[f'{col}_encoded'] = 0
        
        return df
    
    def predict(self, input_data):
        """
        Predict vehicle price
        
        Parameters:
        -----------
        input_data : dict
            Dictionary containing vehicle information
        
        Returns:
        --------
        float
            Predicted price
        """
        features = self.prepare_features(input_data)
        prediction = self.model.predict(features[self.feature_cols])[0]
        return max(500, prediction)
    
    def predict_with_confidence(self, input_data, confidence_level=0.1):
        """
        Predict vehicle price with confidence interval
        
        Parameters:
        -----------
        input_data : dict
            Dictionary containing vehicle information
        confidence_level : float
            Confidence interval width (default: 0.1 for ±10%)
        
        Returns:
        --------
        dict
            Dictionary with 'prediction', 'lower_bound', 'upper_bound'
        """
        prediction = self.predict(input_data)
        lower = prediction * (1 - confidence_level)
        upper = prediction * (1 + confidence_level)
        
        return {
            'prediction': prediction,
            'lower_bound': lower,
            'upper_bound': upper,
            'confidence_level': confidence_level * 100
        }
    
    def get_feature_importance(self):
        """
        Get feature importance from the model
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with feature names and importance scores
        """
        importance_df = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df


def load_and_validate_data(filepath):
    """
    Load and validate the vehicle dataset
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
    
    Returns:
    --------
    pd.DataFrame
        Loaded and validated dataframe
    """
    required_columns = [
        'year', 'make', 'model', 'body', 'transmission',
        'condition', 'odometer', 'sellingprice'
    ]
    
    df = pd.read_csv(filepath)
    
    # Check for required columns
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Basic validation
    assert df.shape[0] > 0, "Dataset is empty"
    assert df['year'].min() >= 1900, "Invalid year values"
    assert df['sellingprice'].min() > 0, "Invalid price values"
    
    return df


def filter_outliers(df, year_min=1990, year_max=2026,
                    odometer_max=500000, price_min=500,
                    price_max=150000):
    """
    Filter outliers from the dataset
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    year_min, year_max : int
        Valid year range
    odometer_max : int
        Maximum valid odometer reading
    price_min, price_max : int
        Valid price range
    
    Returns:
    --------
    pd.DataFrame
        Filtered dataframe
    """
    # Standardize categorical data before filtering
    df['make'] = df['make'].str.strip().str.title()
    df['model'] = df['model'].str.strip().str.title()
    df['transmission'] = df['transmission'].fillna('unknown').str.strip().str.lower()
    
    df_filtered = df[
        (df['year'] >= year_min) &
        (df['year'] <= year_max) &
        (df['odometer'] > 0) &
        (df['odometer'] <= odometer_max) &
        (df['sellingprice'] >= price_min) &
        (df['sellingprice'] <= price_max) &
        (df['condition'] >= 1) &
        (df['condition'] <= 49)
    ].copy()
    
    # IQR-based outlier removal for price
    Q1 = df_filtered['sellingprice'].quantile(0.01)
    Q3 = df_filtered['sellingprice'].quantile(0.99)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    df_filtered = df_filtered[
        (df_filtered['sellingprice'] >= lower_bound) &
        (df_filtered['sellingprice'] <= upper_bound)
    ]
    
    return df_filtered


def stratified_sample(df, target_size=100000, random_state=42):
    """
    Perform stratified sampling on the dataset
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_size : int
        Target sample size
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    pd.DataFrame
        Sampled dataframe
    """
    # Create stratification bins
    df = df.copy()
    df['year_bin'] = pd.cut(df['year'], bins=10, labels=False)
    df['price_bin'] = pd.qcut(df['sellingprice'], q=10, labels=False, duplicates='drop')
    
    # Perform stratified sampling
    df_sample = df.groupby(['year_bin', 'price_bin'], group_keys=False).apply(
        lambda x: x.sample(
            min(len(x), max(1, int(len(x) * target_size / len(df)))),
            random_state=random_state
        )
    ).reset_index(drop=True)
    
    # Remove temporary binning columns
    df_sample = df_sample.drop(['year_bin', 'price_bin'], axis=1)
    
    return df_sample


def standardize_body_type(body):
    """
    Standardize body type classifications
    
    Parameters:
    -----------
    body : str
        Raw body type string
    
    Returns:
    --------
    str
        Standardized body type
    """
    import pandas as pd
    
    if pd.isna(body):
        return 'Unknown'
    
    body = str(body).lower().strip()
    
    # Coupe variations
    if 'coupe' in body or 'cpe' in body:
        return 'Coupe'
    
    # Sedan variations
    if 'sedan' in body or 'sdn' in body:
        return 'Sedan'
    
    # SUV variations
    if 'suv' in body or 'sport utility' in body:
        return 'SUV'
    
    # Truck variations
    if 'truck' in body or 'pickup' in body:
        return 'Truck'
    
    # Van variations
    if 'van' in body or 'minivan' in body:
        return 'Van'
    
    # Wagon variations
    if 'wagon' in body or 'wgn' in body:
        return 'Wagon'
    
    # Convertible variations
    if 'convertible' in body or 'conv' in body or 'cabriolet' in body:
        return 'Convertible'
    
    # Hatchback variations
    if 'hatchback' in body or 'hatch' in body:
        return 'Hatchback'
    
    # Default
    return 'Other'


def calculate_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    
    Returns:
    --------
    dict
        Dictionary of metrics
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    metrics = {
        'r2_score': r2_score(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    }
    
    return metrics


def format_price(price):
    """
    Format price for display
    
    Parameters:
    -----------
    price : float
        Price value
    
    Returns:
    --------
    str
        Formatted price string
    """
    return f"${price:,.2f}"


def get_vehicle_summary(input_data):
    """
    Generate a human-readable vehicle summary
    
    Parameters:
    -----------
    input_data : dict
        Vehicle information dictionary
    
    Returns:
    --------
    str
        Formatted summary string
    """
    summary = f"{input_data['year']} {input_data['make']} {input_data['model']}\n"
    summary += f"Body: {input_data['body']} | Transmission: {input_data['transmission']}\n"
    summary += f"Condition: {input_data['condition']}/49 | Odometer: {input_data['odometer']:,} miles"
    
    return summary


# Example usage and testing
if __name__ == "__main__":
    print("Vehicle Price Prediction Utilities")
    print("=" * 50)
    
    # Example: Create a predictor (requires trained model files)
    try:
        predictor = VehiclePricePredictor()
        print("Model loaded successfully")
        
        # Example prediction
        example_vehicle = {
            'year': 2015,
            'make': 'Ford',
            'model': 'F-150',
            'body': 'Truck',
            'transmission': 'automatic',
            'condition': 35,
            'odometer': 50000
        }
        
        result = predictor.predict_with_confidence(example_vehicle)
        print(f"\nExample Prediction:")
        print(f"Vehicle: {get_vehicle_summary(example_vehicle)}")
        print(f"Predicted Price: {format_price(result['prediction'])}")
        print(f"Range: {format_price(result['lower_bound'])} - {format_price(result['upper_bound'])}")
        
        # Show feature importance
        print(f"\nTop 5 Important Features:")
        importance = predictor.get_feature_importance()
        for idx, row in importance.head(5).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
            
    except FileNotFoundError:
        print("Model files not found. Please train the model first.")
        print("Run the Jupyter notebook to generate model files.")
