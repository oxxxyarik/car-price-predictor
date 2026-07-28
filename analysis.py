import os
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split

CSV_FILE_PATH = "C:/Users/Ярослав/Desktop/project/autoru-used-cars.csv"

print("Загрузка данных...")
df_raw = pd.read_csv(CSV_FILE_PATH)
print(f"Загружено: {df_raw.shape[0]} строк, {df_raw.shape[1]} колонок")

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df[(df['price'] >= 30000) & (df['price'] <= 25000000)]
    df = df.dropna(subset=['year', 'price'])
    df = df[(df['year'] >= 1970) & (df['year'] <= 2026)]

    df['mileage'] = df['mileage'].fillna(df.groupby('year')['mileage'].transform('median'))
    df['mileage'] = df['mileage'].fillna(df['mileage'].median())

    df['engine_vol'] = df['engineDisplacement'].str.extract(r'(\d+\.\d+)').astype(float)
    df['engine_vol'] = df['engine_vol'].fillna(df['engine_vol'].median())

    df['desc_len'] = df['description'].fillna('').apply(len)

    CURRENT_YEAR = 2026
    df['car_age'] = CURRENT_YEAR - df['year']
    df['km_per_year'] = df['mileage'] / (df['car_age'] + 1)

    features = [
        'brand', 'name', 'bodyType', 'color', 'fuelType', 
        'transmission', 'power', 'engine_vol', 'year', 'mileage', 
        'car_age', 'km_per_year', 'desc_len', 'location', 'price'
    ]
    df = df[features]

    cat_cols = ['brand', 'name', 'bodyType', 'color', 'fuelType', 'transmission', 'location']
    for col in cat_cols:
        df[col] = df[col].fillna("Unknown").astype(str)

    df['power'] = df['power'].fillna(df['power'].median())

    return df

print("Очистка и подготовка признаков...")
df_clean = preprocess_data(df_raw)
print(f"Строк после очистки: {df_clean.shape[0]}")

TARGET_COL = 'price'
X = df_clean.drop(columns=[TARGET_COL])

y_log = np.log1p(df_clean[TARGET_COL])

cat_features = ['brand', 'name', 'bodyType', 'color', 'fuelType', 'transmission', 'location']

# Train / Validation (80 / 20)
X_train, X_val, y_train, y_val = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)
model = CatBoostRegressor(
    iterations=3200,
    learning_rate=0.04,
    depth=7,
    loss_function='RMSE',
    eval_metric='RMSE',
    cat_features=['brand', 'name', 'bodyType', 'color', 'fuelType', 'transmission', 'location'],
    random_seed=42,
    verbose=100
)

print("\nСтарт обучения CatBoost...")
model.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    early_stopping_rounds=100,
    use_best_model=True
)

preds_val_log = model.predict(X_val)

preds_val_rub = np.expm1(preds_val_log)
y_val_rub = np.expm1(y_val)

mae = mean_absolute_error(y_val_rub, preds_val_rub)
mape = mean_absolute_percentage_error(y_val_rub, preds_val_rub) * 100
r2 = r2_score(y_val_rub, preds_val_rub)

importance = model.get_feature_importance()

feature_importance = pd.DataFrame({
    "Feature": model.feature_names_,
    "Importance": importance
})

print(feature_importance.sort_values("Importance", ascending=False).head(10))

print("\n" + "=" * 40)
print("     ИТОГОВЫЕ МЕТРИКИ МОДЕЛИ")
print("=" * 40)
print(f"Средняя ошибка (MAE)      : {mae:,.0f} руб.")
print(f"Относительная ошибка (MAPE): {mape:.2f}%")
print(f"Точность модели (R²)      : {r2:.4f}")
print("=" * 40)

model.save_model("car_price_model.cbm")
print("\nМодель сохранена в 'car_price_model.cbm'!")