import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# Configure page
st.set_page_config(
    page_title="House Price Prediction App",
    page_icon="🏠",
    layout="wide"
)

st.write("""
# House Price Prediction App 🏠

This app predicts house prices using machine learning based on key property characteristics!
""")
st.write('---')

@st.cache_data
def load_data():
    """Load and prepare the housing data"""
    try:
        data = pd.read_csv('train_houses.csv')
        return data
    except FileNotFoundError:
        st.error("Error: 'train_houses.csv' file not found. Please ensure the dataset is available.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_resource
def prepare_model(_data):
    """Prepare and train the Random Forest model"""
    if _data is None:
        return None, None, None, None, None, None, None
    
    # Copy data to avoid mutation
    data = _data.copy()
    
    # Define features
    features = ['LotArea', 'YearBuilt', 'FirstFlrSF', 'SecondFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
    
    # Check if all features exist in the dataset
    missing_features = [f for f in features if f not in data.columns]
    if missing_features:
        st.error(f"Missing features in dataset: {missing_features}")
        return None, None, None, None, None, None, None
    
    # Check if target exists
    if 'SalePrice' not in data.columns:
        st.error("Target variable 'SalePrice' not found in dataset")
        return None, None, None, None, None, None, None
    
    # Prepare features and target
    X = data[features].fillna(data[features].mean())
    y = data['SalePrice'].fillna(data['SalePrice'].mean())
    
    # Split the data
    train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=1)
    
    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=1)
    model.fit(train_X, train_y)
    
    # Calculate training and validation performance
    train_predictions = model.predict(train_X)
    val_predictions = model.predict(val_X)
    
    train_mae = mean_absolute_error(train_y, train_predictions)
    val_mae = mean_absolute_error(val_y, val_predictions)
    train_r2 = r2_score(train_y, train_predictions)
    val_r2 = r2_score(val_y, val_predictions)
    train_rmse = np.sqrt(mean_squared_error(train_y, train_predictions))
    val_rmse = np.sqrt(mean_squared_error(val_y, val_predictions))
    
    metrics = {
        'train_mae': train_mae,
        'val_mae': val_mae,
        'train_r2': train_r2,
        'val_r2': val_r2,
        'train_rmse': train_rmse,
        'val_rmse': val_rmse,
        'val_predictions': val_predictions,
        'val_actual': val_y
    }
    
    return model, val_mae, features, metrics, train_X, val_X, train_y

# Initialize session state for prediction history
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Load data and train model
data = load_data()
if data is not None:
    model, mae, features, metrics, train_X, val_X, train_y = prepare_model(data)
    
    if model is not None:
        # Sidebar for input parameters
        st.sidebar.header('🏠 Specify House Characteristics')
        st.sidebar.write("Use the sliders below to input the characteristics of the house:")
        
        def user_input_features():
            """Create input widgets for user parameters"""
            # Get min/max values from the dataset for realistic ranges
            LotArea = st.sidebar.slider(
                'Lot Area (sq ft)', 
                int(data['LotArea'].min()), 
                int(data['LotArea'].max()), 
                int(data['LotArea'].median()),
                help="Size of the lot in square feet"
            )
            
            YearBuilt = st.sidebar.slider(
                'Year Built', 
                int(data['YearBuilt'].min()), 
                int(data['YearBuilt'].max()), 
                int(data['YearBuilt'].median()),
                help="Year the house was originally constructed"
            )
            
            FirstFlrSF = st.sidebar.slider(
                'First Floor Area (sq ft)', 
                int(data['FirstFlrSF'].min()), 
                int(data['FirstFlrSF'].max()), 
                int(data['FirstFlrSF'].median()),
                help="First floor square footage"
            )
            
            SecondFlrSF = st.sidebar.slider(
                'Second Floor Area (sq ft)', 
                int(data['SecondFlrSF'].min()), 
                int(data['SecondFlrSF'].max()), 
                int(data['SecondFlrSF'].median()),
                help="Second floor square footage"
            )
            
            FullBath = st.sidebar.slider(
                'Full Bathrooms', 
                int(data['FullBath'].min()), 
                int(data['FullBath'].max()), 
                int(data['FullBath'].median()),
                help="Number of full bathrooms"
            )
            
            BedroomAbvGr = st.sidebar.slider(
                'Bedrooms Above Ground', 
                int(data['BedroomAbvGr'].min()), 
                int(data['BedroomAbvGr'].max()), 
                int(data['BedroomAbvGr'].median()),
                help="Number of bedrooms above ground level"
            )
            
            TotRmsAbvGrd = st.sidebar.slider(
                'Total Rooms Above Ground', 
                int(data['TotRmsAbvGrd'].min()), 
                int(data['TotRmsAbvGrd'].max()), 
                int(data['TotRmsAbvGrd'].median()),
                help="Total number of rooms above ground (excludes bathrooms)"
            )
            
            # Create dataframe with user inputs
            user_data = {
                'LotArea': LotArea,
                'YearBuilt': YearBuilt,
                'FirstFlrSF': FirstFlrSF,
                'SecondFlrSF': SecondFlrSF,
                'FullBath': FullBath,
                'BedroomAbvGr': BedroomAbvGr,
                'TotRmsAbvGrd': TotRmsAbvGrd
            }
            
            return pd.DataFrame(user_data, index=[0])
        
        # Get user input
        input_df = user_input_features()
        
        # Main panel layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Display input parameters
            st.header('📋 Specified Input Parameters')
            st.dataframe(input_df, width='stretch')
            
            # Model performance metrics
            st.header('📊 Model Performance')
            
            # Display metrics in columns
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            with metric_col1:
                st.metric(
                    label="Validation MAE",
                    value=f"${metrics['val_mae']:,.0f}",
                    delta=f"{metrics['train_mae'] - metrics['val_mae']:,.0f}" if metrics['train_mae'] < metrics['val_mae'] else f"-${metrics['val_mae'] - metrics['train_mae']:,.0f}",
                    help="Average absolute difference between predicted and actual prices on validation data"
                )
            with metric_col2:
                st.metric(
                    label="Validation R² Score",
                    value=f"{metrics['val_r2']:.3f}",
                    help="Proportion of variance explained by the model (closer to 1.0 is better)"
                )
            with metric_col3:
                st.metric(
                    label="Validation RMSE",
                    value=f"${metrics['val_rmse']:,.0f}",
                    help="Root Mean Squared Error - penalizes larger errors more heavily"
                )
            
            # Train vs Validation comparison
            with st.expander("📈 View Train vs Validation Metrics"):
                comp_col1, comp_col2 = st.columns(2)
                
                with comp_col1:
                    st.subheader("Training Performance")
                    st.write(f"**MAE:** ${metrics['train_mae']:,.0f}")
                    st.write(f"**R² Score:** {metrics['train_r2']:.4f}")
                    st.write(f"**RMSE:** ${metrics['train_rmse']:,.0f}")
                
                with comp_col2:
                    st.subheader("Validation Performance")
                    st.write(f"**MAE:** ${metrics['val_mae']:,.0f}")
                    st.write(f"**R² Score:** {metrics['val_r2']:.4f}")
                    st.write(f"**RMSE:** ${metrics['val_rmse']:,.0f}")
                
                # Overfitting check
                if metrics['train_r2'] - metrics['val_r2'] > 0.1:
                    st.warning("⚠️ Model may be overfitting - training performance significantly better than validation")
                elif abs(metrics['train_r2'] - metrics['val_r2']) < 0.05:
                    st.success("✅ Good balance between training and validation performance")
                
                # Predictions vs Actual scatter plot
                st.subheader("Validation Set: Predicted vs Actual Prices")
                fig_scatter = px.scatter(
                    x=metrics['val_actual'], 
                    y=metrics['val_predictions'],
                    labels={'x': 'Actual Price ($)', 'y': 'Predicted Price ($)'},
                    title="Validation Predictions vs Actual Values"
                )
                # Add diagonal line
                fig_scatter.add_trace(
                    go.Scatter(
                        x=[metrics['val_actual'].min(), metrics['val_actual'].max()],
                        y=[metrics['val_actual'].min(), metrics['val_actual'].max()],
                        mode='lines',
                        name='Perfect Prediction',
                        line=dict(color='red', dash='dash')
                    )
                )
                st.plotly_chart(fig_scatter, width='stretch')
            
            # Dataset info
            st.header('📈 Dataset Information')
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Total Houses", f"{len(data):,}")
            with col_b:
                st.metric("Average Price", f"${data['SalePrice'].mean():,.0f}")
            with col_c:
                st.metric("Price Range", f"${data['SalePrice'].min():,.0f} - ${data['SalePrice'].max():,.0f}")
        
        with col2:
            # Make prediction
            prediction = model.predict(input_df)[0]
            
            st.header('🎯 Predicted Sale Price')
            
            # Display prediction with formatting
            st.success(f"## ${prediction:,.0f}")
            
            # Confidence interval (rough estimate based on MAE)
            lower_bound = max(0, prediction - mae)
            upper_bound = prediction + mae
            
            st.info(f"""
            **Estimated Price Range:** ${lower_bound:,.0f} - ${upper_bound:,.0f}
            
            *This range is based on the model's Mean Absolute Error and represents typical prediction accuracy.*
            """)
            
            # Save prediction to history
            if st.button("💾 Save This Prediction", key="save_prediction"):
                prediction_entry = {
                    'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'price': prediction,
                    **input_df.iloc[0].to_dict()
                }
                st.session_state.prediction_history.append(prediction_entry)
                st.success("Prediction saved to history!")
            
            # Feature importance visualization
            st.header('🔍 Feature Importance')
            feature_importance = pd.DataFrame({
                'feature': features,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=True)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(feature_importance['feature'], feature_importance['importance'])
            ax.set_xlabel('Importance Score')
            ax.set_title('Random Forest Feature Importance')
            ax.grid(axis='x', alpha=0.3)
            
            st.pyplot(fig)
        
        st.write('---')
        
        # Prediction History
        if len(st.session_state.prediction_history) > 0:
            st.header('📜 Prediction History')
            st.write(f"You have saved {len(st.session_state.prediction_history)} prediction(s)")
            
            history_df = pd.DataFrame(st.session_state.prediction_history)
            
            # Display table
            st.dataframe(
                history_df.style.format({
                    'price': '${:,.0f}',
                    'LotArea': '{:,.0f}',
                    'FirstFlrSF': '{:,.0f}',
                    'SecondFlrSF': '{:,.0f}'
                }),
                width='stretch'
            )
            
            # Price comparison chart
            if len(st.session_state.prediction_history) > 1:
                fig_history = px.bar(
                    history_df,
                    x='timestamp',
                    y='price',
                    title='Price Comparison Across Predictions',
                    labels={'timestamp': 'Time', 'price': 'Predicted Price ($)'}
                )
                st.plotly_chart(fig_history, width='stretch')
            
            # Clear history button
            if st.button("🗑️ Clear Prediction History", key="clear_history"):
                st.session_state.prediction_history = []
                st.rerun()
        
        st.write('---')
        
        # Data Exploration Section
        st.header('📊 Data Exploration & Feature Analysis')
        
        tab1, tab2, tab3 = st.tabs(["Feature Distributions", "Feature Correlations", "Price Analysis"])
        
        with tab1:
            st.subheader("Distribution of Key Features")
            
            # Select feature to visualize
            selected_feature = st.selectbox(
                "Select a feature to visualize:",
                features,
                key="dist_feature"
            )
            
            # Create distribution plot
            fig_dist = px.histogram(
                data,
                x=selected_feature,
                nbins=50,
                title=f'Distribution of {selected_feature}',
                labels={selected_feature: selected_feature},
                marginal="box"
            )
            st.plotly_chart(fig_dist, width='stretch')
            
            # Show statistics
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
            with col_stat1:
                st.metric("Mean", f"{data[selected_feature].mean():,.2f}")
            with col_stat2:
                st.metric("Median", f"{data[selected_feature].median():,.2f}")
            with col_stat3:
                st.metric("Min", f"{data[selected_feature].min():,.2f}")
            with col_stat4:
                st.metric("Max", f"{data[selected_feature].max():,.2f}")
        
        with tab2:
            st.subheader("Feature Correlation Heatmap")
            
            # Calculate correlation matrix
            correlation_data = data[features + ['SalePrice']].corr()
            
            # Create heatmap using plotly
            fig_corr = px.imshow(
                correlation_data,
                labels=dict(color="Correlation"),
                x=correlation_data.columns,
                y=correlation_data.columns,
                color_continuous_scale='RdBu_r',
                aspect="auto",
                title="Feature Correlation Matrix"
            )
            fig_corr.update_xaxes(side="bottom")
            st.plotly_chart(fig_corr, width='stretch')
            
            # Show strongest correlations with price
            st.subheader("Strongest Correlations with Sale Price")
            price_corr = correlation_data['SalePrice'].drop('SalePrice').sort_values(ascending=False)
            
            col_corr1, col_corr2 = st.columns(2)
            with col_corr1:
                st.write("**Positive Correlations:**")
                for feat, corr in price_corr.head(3).items():
                    st.write(f"• {feat}: {corr:.3f}")
            with col_corr2:
                st.write("**Negative Correlations:**")
                for feat, corr in price_corr.tail(3).items():
                    st.write(f"• {feat}: {corr:.3f}")
        
        with tab3:
            st.subheader("Sale Price vs Features")
            
            # Select feature to compare with price
            price_feature = st.selectbox(
                "Select a feature to compare with Sale Price:",
                features,
                key="price_feature"
            )
            
            # Create scatter plot
            fig_scatter_price = px.scatter(
                data,
                x=price_feature,
                y='SalePrice',
                trendline="ols",
                title=f'Sale Price vs {price_feature}',
                labels={price_feature: price_feature, 'SalePrice': 'Sale Price ($)'},
                opacity=0.5
            )
            st.plotly_chart(fig_scatter_price, width='stretch')
        
        st.write('---')
        
        # Additional insights
        st.header('💡 Model Insights')
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Most Important Features")
            top_features = feature_importance.nlargest(3, 'importance')['feature'].tolist()
            for i, feature in enumerate(top_features, 1):
                st.write(f"{i}. {feature}")
        
        with col2:
            st.subheader("Your House vs Average")
            total_sqft = input_df['FirstFlrSF'].iloc[0] + input_df['SecondFlrSF'].iloc[0]
            avg_sqft = data['FirstFlrSF'].mean() + data['SecondFlrSF'].mean()
            
            if total_sqft > avg_sqft:
                st.write(f"✅ Larger than average ({total_sqft:,.0f} vs {avg_sqft:,.0f} sq ft)")
            else:
                st.write(f"📊 Smaller than average ({total_sqft:,.0f} vs {avg_sqft:,.0f} sq ft)")
        
        with col3:
            st.subheader("Price Comparison")
            avg_price = data['SalePrice'].mean()
            if prediction > avg_price:
                st.write(f"💰 Above average price")
                st.write(f"Premium: ${prediction - avg_price:,.0f}")
            else:
                st.write(f"💵 Below average price")
                st.write(f"Savings: ${avg_price - prediction:,.0f}")
        
        # Methodology
        with st.expander("📖 About This Model"):
            st.write("""
            **Model Type:** Random Forest Regressor
            
            **Features Used:**
            - **LotArea**: Size of the property lot in square feet
            - **YearBuilt**: Original construction year
            - **FirstFlrSF**: First floor square footage
            - **SecondFlrSF**: Second floor square footage  
            - **FullBath**: Number of full bathrooms
            - **BedroomAbvGr**: Number of bedrooms above ground
            - **TotRmsAbvGrd**: Total rooms above ground level
            
            **How it works:**
            The Random Forest algorithm creates multiple decision trees and averages their predictions to provide more accurate and stable estimates. The model was trained on historical house sale data and uses the seven most predictive features to estimate house prices.
            
            **Limitations:**
            - Predictions are based on historical data and market conditions may change
            - Model accuracy varies with unusual property characteristics
            - Local market factors not captured in these features may affect actual prices
            """)
    else:
        st.error("Unable to train the model due to data issues. Please check the dataset format and try again.")
else:
    st.error("Unable to load the dataset. Please ensure 'train_houses.csv' is available and properly formatted.")
    
    # Show expected format
    st.subheader("Expected Dataset Format")
    st.write("""
    The CSV file should contain the following columns:
    - LotArea, YearBuilt, FirstFlrSF, SecondFlrSF, FullBath, BedroomAbvGr, TotRmsAbvGrd, SalePrice
    """)
