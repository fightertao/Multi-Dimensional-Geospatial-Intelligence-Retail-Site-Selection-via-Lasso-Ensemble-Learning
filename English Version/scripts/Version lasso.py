import pandas as pd
import numpy as np
import folium
import os
import requests
import matplotlib.pyplot as plt
from folium.plugins import MousePosition

# 1. Configuration
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
#  Define 8 Gravity Centers (Centers of Influence)
CENTERS = [
    # Primary Centers (Weight: 1.5) - Core Urban Dual-Hubs
    {"name": "Wenchangge (Old City)", "lat": 32.3944, "lon": 119.4335, "weight": 1.5},
    {"name": "Jinghuacheng (New Town)", "lat": 32.3853, "lon": 119.3747, "weight": 1.5},

    # Secondary Centers (Weight: 1.0) - Sub-centers & Core Districts
    {"name": "Guangling New Town (East)", "lat": 32.3935, "lon": 119.4950, "weight": 1.0},
    {"name": "Wanda/Erligiao (South)", "lat": 32.3685, "lon": 119.3993, "weight": 1.0},
    {"name": "Jiangdu Center (Golden Eagle)", "lat": 32.4052, "lon": 119.5789, "weight": 1.0},

    # Tertiary Centers (Weight: 0.7) - Outlying County Centers
    {"name": "Gaoyou Center (Shimao)", "lat": 32.7896, "lon": 119.4486, "weight": 0.7},
    {"name": "Baoying Center (Wuyue)", "lat": 33.2118, "lon": 119.3500, "weight": 0.7},
    {"name": "Yizheng Center (Gulou)", "lat": 32.2681, "lon": 119.1832, "weight": 0.7}
]

 # 2. Load logic
def smart_load(file_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, file_name)
    if not os.path.exists(path): return pd.DataFrame()
    df = pd.read_csv(path, encoding='utf-8-sig')
    coord_col = [c for c in df.columns if c in ['坐标', 'location', 'Coordinates']][0]
    df[['lon', 'lat']] = df[coord_col].str.split(',', expand=True).astype(float)
    return df

df_shops = smart_load("../data/allteashops.csv")
df_k12   = smart_load("../data/primaryandmiddleschool.csv")
df_all_s = smart_load("../data/schools.csv")
df_ind   = smart_load("../data/corporateparksandenterprises.csv")
df_hosp  = smart_load("../data/hospitals.csv")
df_mall  = smart_load("../data/malls.csv")
df_off   = smart_load("../data/enterprises.csv")
df_bus   = smart_load("../data/bus stations.csv")

# 3. Grid-based Feature Extraction
STEP = 0.01 # The step size is set to a grid with a side length of 1 km.
grid_data = []
LON_MIN, LON_MAX = df_shops['lon'].min(), df_shops['lon'].max()
LAT_MIN, LAT_MAX = df_shops['lat'].min(), df_shops['lat'].max()
def count_in_grid(df_target, ln, lt):
    if df_target.empty: return 0
    return len(df_target[(df_target['lon'] >= ln) & (df_target['lon'] < ln + STEP) &
                         (df_target['lat'] >= lt) & (df_target['lat'] < lt + STEP)])
print("Extracting 36-dimensional spatial features...")
for ln in np.arange(LON_MIN, LON_MAX, STEP):
    for lt in np.arange(LAT_MIN, LAT_MAX, STEP):
        #Multi-center Weighted Gravity Calculation
        total_gravity = 0
        for c in CENTERS:
            # Calculate the Euclidean distance between the current grid cell and the gravity center
            d = np.sqrt((ln - c['lon']) ** 2 + (lt - c['lat']) ** 2)
            # Gravity Formula: Weight / (Distance + Smoothing Term)
            # The smoothing term (0.01) prevents division by zero and moderates extreme values at the center
            total_gravity += c['weight'] / (d + 0.01)
        # Assign the cumulative gravity score as the 'prox'
        prox = total_gravity
        # Append extracted features to grid_data list
        grid_data.append({
            'ln': ln, 'lt': lt,
            'k12': count_in_grid(df_k12, ln, lt),       # Education POI density
            'all_s': count_in_grid(df_all_s, ln, lt),   # General service POI density
            'ind': count_in_grid(df_ind, ln, lt),       # Industrial POI density
            'hosp': count_in_grid(df_hosp, ln, lt),     # Healthcare POI density
            'mall': count_in_grid(df_mall, ln, lt),     # Commercial mall POI density
            'off': count_in_grid(df_off, ln, lt),      # Office/Business POI density
            'bus': count_in_grid(df_bus, ln, lt),      # Transportation (Bus stop) density
            'prox': prox,                               # Accessibility score from 8 gravity centers
            'y_actual': count_in_grid(df_shops, ln, lt) # Target variable: Actual shop count per cell
        })
df_grid = pd.DataFrame(grid_data)


# 4. Machine Learning Implementation (Lasso Regression)
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

# 4.1. Define Base Features
features = ['k12', 'all_s', 'ind', 'hosp', 'mall', 'off', 'bus', 'prox']
X = df_grid[features]
y = df_grid['y_actual']

# 4.2. Feature Engineering: Interaction Terms
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X)
feature_names = poly.get_feature_names_out(features)

# 4.3.Feature Scaling (Essential for Regularized Models)
scaler = StandardScaler()
X_poly_scaled = scaler.fit_transform(X_poly)

# 4.4.Model Training with Cross-Validation
print("\n" + "="*60)
print("Running Lasso Regularization & Hyperparameter Optimization...")
model = LassoCV(cv=5, random_state=42, max_iter=10000)
model.fit(X_poly_scaled, y)

# 4.5. Prediction & Performance Metrics
df_grid['y_pred'] = model.predict(X_poly_scaled)
df_grid['gap'] = df_grid['y_pred'] - df_grid['y_actual']

print(f"Lasso Optimization Complete! Features Retained: {sum(model.coef_ != 0)} / {len(feature_names)}")
print("="*60)

# 4.6.Display Key Driving Factors
print("\nTop Driving Factors Identified by Lasso:")
important_feats = sorted(zip(feature_names, model.coef_), key=lambda x: abs(x[1]), reverse=True)
for name, val in important_feats:
    if val != 0:
        status = "[Positive Impact]" if val > 0 else "[Negative Impact]"
        print(f"Feature: {name:25} | Coefficient: {val:8.4f} | {status}")

# 5. Rigorous Model Validation (Lasso)
print("\n" + "="*60)
print("Generalization Capability Validation (Using Scaled Data)")
print("="*60)

# 5.1. Dataset Splitting using X_poly_scaled
X_train, X_test, y_train, y_test = train_test_split(
    X_poly_scaled, y, test_size=0.3, random_state=42
)

# 5.2. Consistency Check using LassoCV
model_valid = LassoCV(cv=5, random_state=42)
model_valid.fit(X_train, y_train)

# 5.3. Prediction & Evaluation
y_train_pred = model_valid.predict(X_train)
y_test_pred  = model_valid.predict(X_test)

r2_train = r2_score(y_train, y_train_pred)
r2_test  = r2_score(y_test, y_test_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)

print(f"Train R²: {r2_train:.4f}")
print(f"Test  R²: {r2_test:.4f} (Lasso typically enhances test set stability)")
print(f"Test MAE: {mae_test:.2f}")
    # Assessment of Generalization
if r2_train - r2_test > 0.15:
    print("Overfitting detected: Potentially due to high spatial autocorrelation among features.")
else:
    print("Strong Generalization: Regularization (L1) is effectively controlling complexity.")

# 6. Top-K Spatial Coverage Verification
print("\n" + "="*60)
print("Site Selection Capability Analysis (Ranked by Lasso Predictions)")
print("="*60)
df_sorted = df_grid.sort_values(by='y_pred', ascending=False)

top_ratio = 0.2
top_k = int(len(df_sorted) * top_ratio)
df_top = df_sorted.head(top_k)

total_shops = df_grid['y_actual'].sum()
covered_shops = df_top['y_actual'].sum()
coverage_rate = covered_shops / total_shops if total_shops > 0 else 0

print(f"Top {int(top_ratio*100)}% of grids covered {coverage_rate*100:.2f}% of existing shops.")

if coverage_rate >= 0.6:
    print("Predictive Strength: Exceptional (Lasso effectively pinpointed core business districts).")
elif coverage_rate >= 0.4:
    print("Predictive Strength: Satisfactory")
else:
    print("Predictive Strength: Weak")
print("="*60)
# 6. 1.Ranking Grids by Predicted Potential）
df_sorted = df_grid.sort_values(by='y_pred', ascending=False)

# 6. 2.Selecting Top 20% High-Potential Grids
top_ratio = 0.2
top_k = int(len(df_sorted) * top_ratio)
df_top = df_sorted.head(top_k)

# 6. 3.Coverage Rate Calculation (Spatial Recall)
total_shops = df_grid['y_actual'].sum()
covered_shops = df_top['y_actual'].sum()
coverage_rate = covered_shops / total_shops if total_shops > 0 else 0
print(f" Top {int(top_ratio*100)}% of grids covered {coverage_rate*100:.2f}% of existing shops.")

#  6. 4. Performance Benchmark
if coverage_rate >= 0.6:
    print("Predictive Strength: Exceptional ")
elif coverage_rate >= 0.5:
    print("Predictive Strength: Excellent ")
elif coverage_rate >= 0.4:
    print("Predictive Strength: Satisfactory")
else:
    print("Predictive Strength: Low (Requires feature engineering refinement)")

print("="*60)

# 7. Build  base maps and visualizations
amap_url = 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png'
attr = '© OpenStreetMap contributors, © CARTO'
MAP_LAT = CENTERS[0]['lat']
MAP_LON = CENTERS[0]['lon']
m = folium.Map(
    location=[MAP_LAT, MAP_LON],
    zoom_start=11,
    tiles=amap_url,
    attr='AutoNavi'
)
geojson_url = "https://geo.datav.aliyun.com/areas_v3/bound/321000_full.json"
try:
    folium.GeoJson(
        requests.get(geojson_url).json(),
        name='Yangzhou Administrative Boundary',
        interactive=False,
        style_function=lambda x: {'fillColor': 'transparent', 'color': '#1E90FF', 'weight': 2, 'opacity': 0.4}
    ).add_to(m)
except: pass
potential_layer = folium.FeatureGroup(name='Potential site selection Thermal (Green)', show=True)
for _, row in df_grid.iterrows():
    if row['gap'] > 0.5:
        color = '#00441b' if row['gap'] > 3 else '#238b45' if row['gap'] > 1 else '#a1d99b'
        folium.Rectangle(
            [[row['lt'], row['ln']], [row['lt'] + STEP, row['ln'] + STEP]],
            fill=True, color='white', weight=0.5, fill_color=color, fill_opacity=0.6,
            popup=f"Potential value:{round(row['gap'], 2)}",
            tooltip=f"Potential: {round(row['gap'], 2)}"
        ).add_to(potential_layer)
potential_layer.add_to(m)
bus_layer = folium.FeatureGroup(name='Bus station(Blue points)', show=False)
for _, row in df_bus.iterrows():
    folium.CircleMarker(location=[row['lat'], row['lon']], radius=1, color='#1E90FF', fill=True).add_to(bus_layer)
bus_layer.add_to(m)

centers_layer = folium.FeatureGroup(name='Gravitational center (Five-pointed star)')
for c in CENTERS:
    star_radius = 15 * c['weight']
    folium.RegularPolygonMarker(
        location=[c['lat'], c['lon']], number_of_sides=5, radius=star_radius,
        rotation=35, color='#FFD700', fill_color='#FFD700', fill_opacity=0.7,
        popup=f"{c['name']}: {c['weight']}"
    ).add_to(centers_layer)
centers_layer.add_to(m)
shop_layer = folium.FeatureGroup(name='The current distribution of shops(Red points)', show=True)
for _, row in df_shops.iterrows():
    folium.CircleMarker(location=[row['lat'], row['lon']], radius=2, color='#FF0000', fill=True).add_to(shop_layer)
shop_layer.add_to(m)
top_opportunities = df_grid[df_grid['y_actual'] <= 1].sort_values(by='gap', ascending=False).head(5)
print("\n" + "★" * 60)
print(f" Yangzhou Milk Tea Site Selection: Top 5 Golden Points Recommended by 36-D Model")
print("★" * 60)
gold_layer = folium.FeatureGroup(name='Recommended gold points', show=True)

for i, (idx, row) in enumerate(top_opportunities.iterrows()):
    print(f"Recommended Site {i + 1}: Coordinates ({row['lt']:.4f}, {row['ln']:.4f})")
    print(f"   - Predicted Potential Score: {row['gap']:.2f}")
    print(f"   - Current Competition (Existing Shops): {int(row['y_actual'])}")
    print(f"   - Primary Drivers: Verify 'University + Shopping Mall' synergy at this location.")
    folium.Marker(
        location=[row['lt'], row['ln']],
        popup=folium.Popup(f"<b>gold point No.{i + 1}</b><br>Potential value: {round(row['gap'], 2)}", max_width=200),
        icon=folium.Icon(color='orange', icon='leaf', prefix='fa')
    ).add_to(gold_layer)

gold_layer.add_to(m)
print("\n Golden sites have been plotted.refresh the HTML file to view!")
MousePosition(lng_first=True, position='bottomright').add_to(m)
folium.LayerControl(collapsed=False).add_to(m)
m.save("Version Lasso.html")