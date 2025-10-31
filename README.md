# Overview

This is a machine learning web application built with Streamlit that predicts house prices based on property characteristics. The application uses a Random Forest Regressor model trained on housing data to provide price predictions based on seven key features: lot area, year built, floor square footage, number of bathrooms, bedrooms, and total rooms.

**Recent Updates (October 2025)**:
- Added interactive data exploration with feature distributions, correlation heatmap, and price analysis
- Implemented train/validation metrics comparison with overfitting detection
- Added prediction history tracking to compare multiple house configurations
- Enhanced visualizations using Plotly for interactivity
- Improved model performance reporting with MAE, R², and RMSE metrics

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture

**Framework**: Streamlit web framework
- **Decision**: Use Streamlit for rapid prototyping and deployment of ML applications
- **Rationale**: Streamlit provides a simple Python-first approach to building interactive web apps without requiring frontend development expertise
- **Key Features**: 
  - Interactive sidebar sliders for user input
  - Real-time predictions with confidence intervals
  - Wide layout configuration for better visualization space
  - Custom page title and icon (🏠)
  - Session-based prediction history tracking
  - Interactive data exploration tabs:
    - Feature distributions with statistics
    - Correlation heatmap
    - Price vs feature scatter plots with trendlines
  - Comprehensive model performance metrics (train/validation comparison)

## Backend Architecture

**Machine Learning Pipeline**:
- **Model**: Random Forest Regressor from scikit-learn
- **Decision**: Use Random Forest for house price prediction
- **Rationale**: Random Forest models handle non-linear relationships well, are robust to outliers, and provide good out-of-box performance for regression tasks
- **Features Used**: 7 numerical features
  - LotArea: Property lot size
  - YearBuilt: Construction year
  - FirstFlrSF: First floor square footage
  - SecondFlrSF: Second floor square footage  
  - FullBath: Number of full bathrooms
  - BedroomAbvGr: Bedrooms above grade
  - TotRmsAbvGrd: Total rooms above grade

**Caching Strategy**:
- `@st.cache_data` decorator for data loading
- `@st.cache_resource` decorator for model training (changed from cache_data for better semantic fit)
- **Decision**: Implement caching to prevent redundant data loading and model retraining
- **Rationale**: Improves application performance and user experience by avoiding expensive operations on each interaction
- **Safety**: Model preparation function defensively copies input data to prevent mutation

**Session State Management**:
- `prediction_history`: Stores user-saved predictions for comparison
- **Decision**: Use Streamlit session state for prediction tracking
- **Rationale**: Enables users to compare multiple house configurations within a session without database complexity

**Data Processing**:
- Feature selection approach using specific columns from dataset
- Train-test split (80/20) for model validation
- Evaluation metrics: MAE, R² Score, RMSE (calculated for both train and validation sets)
- Overfitting detection: Automatic warning when train/validation performance diverges significantly

## Error Handling

- Try-catch blocks for data loading with user-friendly error messages
- FileNotFoundError handling for missing dataset
- Generic exception handling for unexpected errors

# External Dependencies

## Python Libraries

**Core Dependencies**:
- `streamlit`: Web application framework
- `pandas`: Data manipulation and analysis
- `scikit-learn`: Machine learning library (RandomForestRegressor, train_test_split, metrics)
- `matplotlib`: Static plotting and visualization
- `seaborn`: Statistical data visualization (v0.13.2)
- `plotly`: Interactive plotting (express and graph_objects modules, v6.3.1)
- `numpy`: Numerical computing
- `statsmodels`: Statistical models (v0.14.5) - required for Plotly trendline functionality

## Data Source

**Dataset**: `train_houses.csv`
- **Type**: CSV file containing housing data
- **Location**: Expected in the root directory
- **Required Columns**: LotArea, YearBuilt, FirstFlrSF, SecondFlrSF, FullBath, BedroomAbvGr, TotRmsAbvGrd, SalePrice
- **Purpose**: Training data for the Random Forest model and source for prediction features

## Visualization Libraries

**Decision**: Multi-library visualization approach
- matplotlib for feature importance bar chart
- Plotly for interactive visualizations (histograms, heatmaps, scatter plots)
- **Rationale**: Provides flexibility for different visualization needs and user interaction patterns
- **Interactive Features**: Hover tooltips, zoom, pan, and download options on all Plotly charts

## Known Limitations

**SHAP Explainability**:
- SHAP library could not be installed due to environment dependency conflicts with Linux/Python 3.12 in the uv package manager
- Feature importance from Random Forest is provided as an alternative
- Future consideration: Explore alternative explainability libraries or environment configurations
