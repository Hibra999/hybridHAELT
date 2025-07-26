from utils.data import scrape_candles_to_dataframe
from utils.features import create_robust_features
from utils.model import ForecasterDualModel
from utils.plots import plot_results
import warnings
import numpy as np
warnings.filterwarnings("ignore")

# Configuración
OPTIMIZE_HYPERPARAMS = False  
SYMBOL = 'SOL/USDT'  # o 'SOL/USDT'

# 1. Obtener datos
print("1. Fetching data...")
data = scrape_candles_to_dataframe('binance', 3, SYMBOL, '1h', '2025-03-01T00:00:00Z', 1000)
print(f"Total data: {len(data)}")

# 2. Crear features
print("\n2. Creating features...")
df = create_robust_features(data)

# 3. Preparar datos
forecaster = ForecasterDualModel(sequence_length=168, forecast_horizon=72)
X, y = forecaster.prepare_data(df)

print("\n4. Splitting data...")
months_test = 1  # Número de meses para test
months_val = 1   # Número de meses para validación

# Calcular índices
test_size = 730 * months_test  # 730 horas por mes
val_size = 730 * months_val

X_train = X[:-test_size-val_size]
y_train = y[:-test_size-val_size]
X_val = X[-test_size-val_size:-test_size]
y_val = y[-test_size-val_size:-test_size]
X_test = X[-test_size:]
y_test = y[-test_size:]

print(f"Train: {len(X_train)} samples")
print(f"Val: {len(X_val)} samples")
print(f"Test: {len(X_test)} samples")
# 5. Entrenar modelos
print("\n4. Training models...")
if OPTIMIZE_HYPERPARAMS:
    print("Optimization enabled....")

forecaster.train_models(X_train, y_train, X_val, y_val, optimize_params=OPTIMIZE_HYPERPARAMS)

# 6. Simulación en tiempo real
print("\n6. Running real-time simulation...")
n_simulation_steps = len(X_test) - 1 

simulation_result = forecaster.simulate_real_time_forecast(
            X_test, y_test, n_steps=n_simulation_steps, verbose=True
        )

# 7. Visualizar resultados
print("\n7. Plotting results...")
plot_results(simulation_result)