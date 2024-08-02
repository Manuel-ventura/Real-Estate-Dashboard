import dash
import os
import json
from dash import dcc, html, Dash
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score

# Load data
df = pd.read_csv('./data/upgrated_data_final.csv')

# Load models
lin_reg_model = joblib.load('models/lin_reg_model.pkl')
tree_model = joblib.load('models/tree_reg_model.pkl')
forest_model = joblib.load('models/forest_reg_model.pkl')
tuned_rf_model = joblib.load('models/tuned_rf_model.pkl')

# Separate features and target variable
X = df[['square_meters', 'bedrooms', 'parking', 'bathroom',
        'cluster', 'latitude', 'longitude', 'location_num']]
y = df['price']

# Make predictions
df['lin_reg_pred'] = lin_reg_model.predict(X)
df['tree_pred'] = tree_model.predict(X)
df['forest_pred'] = forest_model.predict(X)
df['tuned_rf_pred'] = tuned_rf_model.predict(X)

# Calculate residuals
df['lin_reg_resid'] = y - df['lin_reg_pred']
df['tree_resid'] = y - df['tree_pred']
df['forest_resid'] = y - df['forest_pred']
df['tuned_rf_resid'] = y - df['tuned_rf_pred']

# Add Model Performance Metrics
metrics = {
    'Linear Regression': {
        'MSE': mean_squared_error(y, df['lin_reg_pred']),
        'R2': r2_score(y, df['lin_reg_pred'])
    },
    'Decision Tree': {
        'MSE': mean_squared_error(y, df['tree_pred']),
        'R2': r2_score(y, df['tree_pred'])
    },
    'Random Forest': {
        'MSE': mean_squared_error(y, df['forest_pred']),
        'R2': r2_score(y, df['forest_pred'])
    },
    'Tuned Random Forest': {
        'MSE': mean_squared_error(y, df['tuned_rf_pred']),
        'R2': r2_score(y, df['tuned_rf_pred'])
    }
}

# Summary Statistics
summary_stats = df.describe().T

# Filter Options
filter_options = dcc.Dropdown(
    id='location-filter',
    options=[{'label': loc, 'value': loc} for loc in df['location'].unique()],
    multi=True,
    placeholder="Select locations"
)

# Color palette and styles
colors = {
    'background': '#f5f5f5',
    'text': '#222222',
    'palette': ['#a54657', '#edae49', '#94b33d', '#dd2c2f', '#af5500', '#fdc700', '#222222', '#4361ee', '#4895ef', '#4cc9f0', '#582630', '#222222']
}

# Graphs
fig_lin_reg = px.scatter(df, x='price', y='lin_reg_pred', trendline='ols',
                         title='Linear Regression: Predictions vs. Actual Values', color_discrete_sequence=[colors['palette'][0]])
# Regression line color
fig_lin_reg.data[1].line.color = colors['palette'][10]

fig_tree = px.scatter(df, x='price', y='tree_pred', trendline='ols',
                      title='Decision Tree: Predictions vs. Actual Values', color_discrete_sequence=[colors['palette'][1]])
# Regression line color
fig_tree.data[1].line.color = colors['palette'][10]

fig_forest = px.scatter(df, x='price', y='forest_pred', trendline='ols',
                        title='Random Forest: Predictions vs. Actual Values', color_discrete_sequence=[colors['palette'][2]])
# Regression line color
fig_forest.data[1].line.color = colors['palette'][10]

fig_tuned_rf = px.scatter(df, x='price', y='tuned_rf_pred', trendline='ols',
                          title='Tuned Random Forest: Predictions vs. Actual Values', color_discrete_sequence=[colors['palette'][3]])
# Regression line color
fig_tuned_rf.data[1].line.color = colors['palette'][10]

# Residuals Plot for Tuned Random Forest with trendline
fig_residuals = px.scatter(df, x='price', y='tuned_rf_resid',
                           title='Tuned Random Forest: Residuals',
                           color_discrete_sequence=[colors['palette'][4]],
                           trendline='ols')
# Customize the trendline color
fig_residuals.data[1].line.color = colors['palette'][10]

# Feature Importance
feature_importance = pd.Series(
    tuned_rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
fig_feature_importance = px.bar(feature_importance, x=feature_importance.index, y=feature_importance.values,
                                title='Feature Importance', color_discrete_sequence=[colors['palette'][5]])

# Interactive map
with open('geo-map/property_heat_map.html', 'r', encoding='utf-8') as f:
    map_html = f.read()

# Define the external stylesheets
external_stylesheets = [dbc.themes.BOOTSTRAP]

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[
                dbc.themes.BOOTSTRAP, "/styles/styles.css"])

# Helper function to convert metrics dictionary to a DataFrame


def metrics_to_dataframe(metrics):
    data = []
    for model, metric in metrics.items():
        data.append([model, metric['MSE'], metric['R2']])
    df_metrics = pd.DataFrame(data, columns=['Model', 'MSE', 'R2'])
    return df_metrics


# Convert metrics to DataFrame
df_metrics = metrics_to_dataframe(metrics)

app.layout = dbc.Container(
    [
        # Title Section
        dbc.Row(
            dbc.Col(
                html.H1(
                    "House Price Analysis Dashboard in El Salvador",
                    style={'textAlign': 'center',
                           'color': colors['text'], 'fontWeight': 'bold', 'fontFamily': 'Urbanist', 'marginBottom': '20px'}
                )
            )
        ),

        # Description Section
        dbc.Row(
            dbc.Col(
                html.P(
                    "This dashboard presents an exploratory analysis and predictive models of house prices in El Salvador.",
                    style={'textAlign': 'center',
                           'color': colors['text'], 'fontFamily': 'Urbanist', 'marginBottom': '40px'}
                )
            )
        ),

        # Exploratory Analysis Section
        dbc.Row(
            dbc.Col(
                html.Div(
                    children=[
                        html.H2("Exploratory Analysis", style={
                                'textAlign': 'center', 'fontFamily': 'Urbanist', 'fontWeight': 'bold', 'marginBottom': '20px'}),
                        dcc.Markdown(
                            """
                            ### 1. **Summary Statistics**
                            - The average property price is **$374,913.44** with a standard deviation of **$342,244.45**.
                            - The average property size is **238.53 square meters**.
                            - Properties typically have around **3 bedrooms**, **2 bathrooms**, and **2 parking spaces**.
                            - The data includes properties from various locations with different characteristics.
                            """,
                            style={'textAlign': 'left',
                                   'fontFamily': 'Urbanist', 'marginBottom': '20px'}
                        ),
                        html.Img(src='assets/average_price_by_location.png',
                                 style={'width': '100%', 'marginBottom': '20px'}),
                        dcc.Markdown(
                            """
                            ### 2. **Price per Square Meter Distribution**
                            - Prices vary significantly, reflecting the diversity of properties and locations.
                            """,
                            style={'textAlign': 'left',
                                   'fontFamily': 'Urbanist', 'marginBottom': '20px'}
                        ),
                        html.Img(src='assets/boxplot_price_per_square_meter.png',
                                 style={'width': '100%', 'marginBottom': '20px'}),
                        dcc.Markdown(
                            """
                            ### 3. **Average Price by Location**
                            - San Salvador and its surroundings have higher average prices, indicating higher demand and better infrastructure.
                            """,
                            style={'textAlign': 'left',
                                   'fontFamily': 'Urbanist', 'marginBottom': '20px'}
                        ),
                        html.Img(src='assets/distribution_of_square_meters_by_location.png',
                                 style={'width': '100%', 'marginBottom': '20px'}),
                        dcc.Markdown(
                            """
                            ### 4. **Square Meters Distribution by Location**
                            - Urban properties are smaller compared to rural ones.
                            """,
                            style={'textAlign': 'left',
                                   'fontFamily': 'Urbanist', 'marginBottom': '20px'}
                        ),
                        html.Img(src='assets/feature_importance_random_forest.png',
                                 style={'width': '100%', 'marginBottom': '20px'}),
                        dcc.Markdown(
                            """
                            ### 5. **Feature Importance**
                            - The Random Forest model identifies **square meters**, **location**, and **number of bedrooms** as the most influential features in predicting property prices.
                            """,
                            style={'textAlign': 'left',
                                   'fontFamily': 'Urbanist', 'marginBottom': '20px'}
                        ),
                        html.Img(src='assets/histogram_price_per_square_meter.png',
                                 style={'width': '100%', 'marginBottom': '20px'}),
                        dcc.Markdown(
                            """
                            ### 6. **Residual Analysis**
                            - The Tuned Random Forest model shows the best performance with the least residual errors.
                            """,
                            style={'textAlign': 'left',
                                   'fontFamily': 'Urbanist', 'marginBottom': '20px'}
                        ),
                        html.Img(src='assets/tuned_random_forest_residuals.png',
                                 style={'width': '100%', 'marginBottom': '20px'})
                    ],
                    style={
                        'padding': '20px', 'backgroundColor': colors['background'], 'borderRadius': '10px', 'textAlign': 'center'}
                ),
                width=12
            )
        ),
        dbc.Row(
            dbc.Col(
                html.Div(
                    children=[
                        html.H2("Predictive Model Results", style={
                                'textAlign': 'center', 'fontFamily': 'Urbanist', 'fontWeight': 'bold'}),
                        dcc.Markdown(
                            """
                            - **Linear Regression**: Positive relationship between actual and predicted prices, with some dispersion.
                            - **Decision Tree**: Captures non-linearities but with more dispersion.
                            - **Random Forest**: Better alignment and accuracy due to the combination of multiple trees.
                            - **Tuned Random Forest**: Hyperparameter optimization significantly improves model performance.
                            """,
                            style={'textAlign': 'left',
                                   'fontFamily': 'Urbanist', 'marginBottom': '20px'}
                        ),
                        html.Img(src='assets/histograms_of_residuals.png',
                                 style={'width': '100%', 'marginBottom': '20px'}),
                        html.Img(src='assets/scatter_plot_of_predictions_vs_actual_values.png',
                                 style={'width': '100%', 'marginBottom': '20px'}),
                    ],
                    style={
                        'padding': '20px', 'backgroundColor': colors['background'], 'borderRadius': '10px', 'textAlign': 'center'}
                ),
                width=12
            )
        ),
        dbc.Row(
            dbc.Col(
                html.Div(
                    children=[
                        html.H2("Conclusions", style={
                                'textAlign': 'center', 'fontFamily': 'Urbanist', 'fontWeight': 'bold'}),
                        dcc.Markdown(
                            """
                            - **Predictive models**, especially the Tuned Random Forest, provide accurate predictions.
                            - **Exploratory analysis** reveals important trends in the price distribution and property features.
                            - **Data visualization and analysis** offer a powerful tool to understand the real estate market in El Salvador.
                            - **Future work** should focus on collecting more data and exploring advanced machine learning techniques for improved accuracy.
                            """,
                            style={'textAlign': 'left',
                                   'fontFamily': 'Urbanist', 'marginBottom': '20px'}
                        )
                    ],
                    style={
                        'padding': '20px', 'backgroundColor': colors['background'], 'borderRadius': '10px', 'textAlign': 'center'}
                ),
                width=12
            )
        ),



        # Summary Statistics Section
        dcc.Markdown(
            """
            ## **Summary Statistics**
            """,
            style={'textAlign': 'center',
                   'fontFamily': 'Urbanist', 'marginBottom': '20px'}
        ),
        html.Div(
            dbc.Table.from_dataframe(summary_stats, striped=True, bordered=True, hover=True, style={
                'fontFamily': 'Urbanist'}),
            style={'display': 'flex',
                   'justifyContent': 'center'}
        ),

        # Model Performance Metrics Section
        dbc.Row(
            dbc.Col(
                html.Div(
                    children=[
                        html.H3("Model Performance Metrics", style={
                                'textAlign': 'center', 'fontFamily': 'Urbanist', 'fontWeight': 'bold'}),
                        dbc.Table.from_dataframe(df_metrics, striped=True, bordered=True, hover=True, style={
                                                 'fontFamily': 'Urbanist'})
                    ],
                    style={
                        'padding': '20px', 'backgroundColor': colors['background'], 'borderRadius': '10px', 'textAlign': 'center'}
                ),
                width=12,
                style={'textAlign': 'center'}
            )
        ),

        # Graphs Section
        dbc.Row(
            [
                dbc.Col(dcc.Graph(figure=fig_lin_reg), md=6),
                dbc.Col(dcc.Graph(figure=fig_tree), md=6),
            ],
            style={'marginBottom': '20px'}
        ),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(figure=fig_forest), md=6),
                dbc.Col(dcc.Graph(figure=fig_tuned_rf), md=6),
            ],
            style={'marginBottom': '20px'}
        ),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(figure=fig_residuals), md=6),
                dbc.Col(dcc.Graph(figure=fig_feature_importance), md=6),
            ],
            style={'marginBottom': '20px'}
        ),

        # Heat Map Information Section
        dbc.Row(
            dbc.Col(
                html.Div(
                    children=[
                        html.H3("Heat Map Information", style={
                            'textAlign': 'center', 'fontFamily': 'Urbanist', 'fontWeight': 'bold'}),
                        dcc.Markdown(
                            f"""
                    <span style='color: {colors['text']}'>
                    The heat map shows the density of properties in El Salvador.
                    Areas with more properties are represented by more intense colors,
                    indicating higher density.
                    It is represented this way because the exact address of the property is not available.
                    </span>
                    """,
                            style={'textAlign': 'center',
                                   'fontFamily': 'Urbanist'},
                            dangerously_allow_html=True
                        )
                    ],
                    style={
                        'padding': '20px', 'backgroundColor': colors['background'], 'borderRadius': '10px', 'textAlign': 'center'}
                ),
                width=12,
                style={'textAlign': 'center'}
            )
        ),
        # Map Section
        dbc.Row(
            dbc.Col(
                html.Div(
                    children=[
                        html.Iframe(
                            id='map',
                            srcDoc=map_html,
                            width='100%',
                            height='600'
                        )
                    ],
                    className='graph-container'
                ),
                width=12
            )
        )
    ],
    className='container',
    style={'backgroundColor': colors['background'], 'padding': '20px'}
)

if __name__ == "__main__":
    app.run_server(debug=True)
