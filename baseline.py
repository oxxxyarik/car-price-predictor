import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

CSV_FILE_PATH = "C:/Users/Ярослав/Desktop/project/autoru-used-cars.csv" # твой путь к CSV

print("Загрузка данных для Baseline...")
df_raw = pd.read_csv(CSV_FILE_PATH)
df = df_raw.dropna(subset=['year', 'price']).copy()
df = df[(df['price'] >= 30000) & (df['price'] <= 25000000)]
df = df[(df['year'] >= 1970) & (df['year'] <= 2026)]

df['mileage'] = df['mileage'].fillna(df['mileage'].median())
df['power'] = df['power'].fillna(df['power'].median())

features = ['brand', 'bodyType', 'transmission', 'year', 'mileage', 'power']
cat_cols = ['brand', 'bodyType', 'transmission']
num_cols = ['year', 'mileage', 'power']

for col in cat_cols:
    df[col] = df[col].fillna("Unknown").astype(str)

X = df[features]
y = df['price']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# DUMMY BASELINE
median_price = y_train.median()
dummy_preds = np.full_like(y_val, fill_value=median_price)

print("\n" + "=" * 40)
print("  DUMMY BASELINE")
print("=" * 40)
print(f"MAE  : {mean_absolute_error(y_val, dummy_preds):,.0f} руб.")
print(f"MAPE : {mean_absolute_percentage_error(y_val, dummy_preds)*100:.2f}%")
print(f"R²   : {r2_score(y_val, dummy_preds):.4f}")

# LINEAR MODEL BASELINE
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ]
)

linear_model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', Ridge())
])

print("\nОбучение Линейной регрессии...")
linear_model.fit(X_train, y_train)
linear_preds = linear_model.predict(X_val)

print("\n" + "=" * 40)
print("  LINEAR REGRESSION BASELINE")
print("=" * 40)
print(f"MAE  : {mean_absolute_error(y_val, linear_preds):,.0f} руб.")
print(f"MAPE : {mean_absolute_percentage_error(y_val, linear_preds)*100:.2f}%")
print(f"R²   : {r2_score(y_val, linear_preds):.4f}")
print("=" * 40)