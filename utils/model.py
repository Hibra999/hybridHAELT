import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, LSTM, Dense, Dropout, LayerNormalization, 
                                   MultiHeadAttention, Conv1D, GlobalAveragePooling1D,
                                   BatchNormalization, Concatenate, Embedding, Add, Layer)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# Crear position embeddings usando Lambda layer
from tensorflow.keras.layers import Lambda
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
from collections import deque
import optuna
from optuna.samplers import TPESampler
from datetime import datetime, timedelta
import ccxt
import re


class HoltWintersDecompositionLayer(Layer):
    """
    Implementación exacta del bloque de descomposición Holt-Winters del paper Helformer.
    """
    
    def __init__(self, season_length=24, **kwargs):
        super(HoltWintersDecompositionLayer, self).__init__(**kwargs)
        self.season_length = season_length
        
    def build(self, input_shape):
        super(HoltWintersDecompositionLayer, self).build(input_shape)
        
        # Parámetros locales aprendibles α y γ (entre 0 y 1)
        self.alpha = self.add_weight(
            name='alpha',
            shape=(1,),
            initializer=tf.keras.initializers.Constant(0.3),
            trainable=True,
            constraint=tf.keras.constraints.MinMaxNorm(min_value=0.01, max_value=0.99)
        )
        
        self.gamma = self.add_weight(
            name='gamma',
            shape=(1,),
            initializer=tf.keras.initializers.Constant(0.3),
            trainable=True,
            constraint=tf.keras.constraints.MinMaxNorm(min_value=0.01, max_value=0.99)
        )
    
    def compute_output_shape(self, input_shape):
        """Calcula la forma de salida."""
        # Input shape: (batch_size, seq_len, n_features)
        # Output shape: (batch_size, seq_len, n_features + 3)
        return (input_shape[0], input_shape[1], input_shape[2] + 3)
        
    @tf.function
    def call(self, inputs, training=None):
        """
        inputs: tensor de shape (batch_size, sequence_length, n_features)
        Usa solo la primera columna (Close price) para la descomposición HW
        """
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        
        # Extraer serie de precios (primera columna)
        prices = inputs[:, :, 0]  # shape: (batch_size, seq_len)
        
        # Extraer valores escalares de alpha y gamma
        alpha_scalar = self.alpha[0]  # Acceder directamente al elemento
        gamma_scalar = self.gamma[0]
        
        # Implementación vectorizada de Holt-Winters
        def apply_holt_winters_vectorized(prices_batch):
            # prices_batch shape: (batch_size, seq_len)
            
            # Inicializar arrays
            level = tf.zeros_like(prices_batch)
            seasonal = tf.zeros_like(prices_batch)
            deseasonalized = tf.zeros_like(prices_batch)
            
            # Nivel inicial: promedio de la primera temporada
            initial_level = tf.reduce_mean(prices_batch[:, :self.season_length], axis=1, keepdims=True)
            
            # Inicializar primera temporada
            level = tf.concat([
                tf.tile(initial_level, [1, self.season_length]),
                level[:, self.season_length:]
            ], axis=1)
            
            seasonal = tf.concat([
                tf.zeros([batch_size, self.season_length]),
                seasonal[:, self.season_length:]
            ], axis=1)
            
            deseasonalized = tf.concat([
                prices_batch[:, :self.season_length],
                deseasonalized[:, self.season_length:]
            ], axis=1)
            
            # Aplicar Holt-Winters iterativamente
            for t in tf.range(self.season_length, seq_len):
                # Índices para acceso vectorizado
                t_prev = t - 1
                t_season = t - self.season_length
                
                # Obtener valores previos
                s_prev = seasonal[:, t_season]
                l_prev = level[:, t_prev]
                price_t = prices_batch[:, t]
                
                # Ecuaciones HW
                l_t = alpha_scalar * (price_t - s_prev) + (1 - alpha_scalar) * l_prev
                s_t = gamma_scalar * (price_t - l_t) + (1 - gamma_scalar) * s_prev
                y_t = price_t - s_t
                
                # Actualizar usando tf.tensor_scatter_nd_update
                indices = tf.stack([tf.range(batch_size), tf.fill([batch_size], t)], axis=1)
                
                level = tf.tensor_scatter_nd_update(level, indices, l_t)
                seasonal = tf.tensor_scatter_nd_update(seasonal, indices, s_t)
                deseasonalized = tf.tensor_scatter_nd_update(deseasonalized, indices, y_t)
            
            return level, seasonal, deseasonalized
        
        # Aplicar HW vectorizado
        level_batch, seasonal_batch, deseasonalized_batch = apply_holt_winters_vectorized(prices)
        
        # Expandir dimensiones para concatenar
        level_expanded = tf.expand_dims(level_batch, -1)  # (batch, seq_len, 1)
        seasonal_expanded = tf.expand_dims(seasonal_batch, -1)
        deseasonalized_expanded = tf.expand_dims(deseasonalized_batch, -1)
        
        # Concatenar: [datos desestacionalizados, features originales, nivel, estacionalidad]
        output = tf.concat([
            deseasonalized_expanded,  # Componente principal desestacionalizado
            inputs,                   # Todas las features originales (n_features)
            level_expanded,           # Componente de nivel
            seasonal_expanded         # Componente estacional
        ], axis=-1)
        
        # Output shape: (batch_size, seq_len, n_features + 3)
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'season_length': self.season_length
        })
        return config


class ForecasterDualModel:
    
    def __init__(self, sequence_length=168, forecast_horizon=72):
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        
        # Modelos
        self.models = {}
        self.model_scores = {}
        self.model_weights = {'lightgbm': 0.5, 'helformer': 0.5}
        
        # Meta-learner
        self.meta_learner = None
        
        # Scalers
        self.scalers = {
            'features': StandardScaler(),
            'target': StandardScaler(),
            'robust': RobustScaler(),
            'minmax': MinMaxScaler()
        }
        
        # Feature names
        self.feature_names = None
        
        # Performance tracking
        self.performance_history = {
            'lightgbm': deque(maxlen=100),
            'helformer': deque(maxlen=100)
        }
        self.weight_history = []
        
        # Online learning parameters
        self.online_learning_rate = 0.01
        self.weight_momentum = 0.9
        self.min_weight = 0.1
        
        # LightGBM optimized parameters (estado del arte)
        self.lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 255,
            'learning_rate': 0.02,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.7,
            'bagging_freq': 5,
            'verbose': -1,
            'n_estimators': 2000,
            'max_depth': -1,
            'min_child_samples': 20,
            'subsample_for_bin': 200000,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'min_split_gain': 0.01,
            'min_child_weight': 0.001,
            'random_state': 42,
            'n_jobs': -1,
            'importance_type': 'gain',
            'enable_bundle': True,
            'max_conflict_rate': 0.999,
            'extra_trees': True
        }
        
        # Helformer default parameters
        self.helformer_best_params = None

    def build_helformer(self, n_features, sequence_length):
        """
        Construir modelo Helformer exactamente según el paper.
        """
        # Parámetros por defecto o los mejores encontrados
        if self.helformer_best_params is not None:
            params = self.helformer_best_params
        else:
            params = {
                'season_length': 24,
                'd_model': 256,
                'n_heads': 8,
                'n_blocks': 4,
                'lstm_units': 128,
                'dropout_rate': 0.1,
                'learning_rate': 0.001
            }
        
        # Input layer
        inputs = Input(shape=(sequence_length, n_features), name='input_features')
        
        # 1. Bloque de descomposición Holt-Winters (componente clave del Helformer)
        hw_decomposed = HoltWintersDecompositionLayer(
            season_length=params.get('season_length', 24),
            name='holt_winters_decomposition'
        )(inputs)
        
        # hw_decomposed shape: (batch, seq_len, n_features + 3)
        
        # 2. Proyección a dimensión del modelo
        x = Dense(params['d_model'], name='input_projection')(hw_decomposed)
        x = LayerNormalization(name='input_norm')(x)
        
        # 3. Bloques de Multi-Head Attention + LSTM
        for block_idx in range(params['n_blocks']):
            # Multi-head attention (procesa patrones globales)
            attn_output = MultiHeadAttention(
                num_heads=params['n_heads'],
                key_dim=params['d_model'] // params['n_heads'],
                dropout=params['dropout_rate'],
                name=f'multi_head_attention_{block_idx}'
            )(x, x)  # Self-attention
            
            # Add & Norm
            x = Add(name=f'add_attention_{block_idx}')([x, attn_output])
            x = LayerNormalization(name=f'norm_attention_{block_idx}')(x)
            
            # LSTM layer (según el paper, reemplaza al FFN)
            lstm_output = LSTM(
                units=params['lstm_units'],
                return_sequences=True,
                dropout=params['dropout_rate'],
                recurrent_dropout=params['dropout_rate'],
                name=f'lstm_{block_idx}'
            )(x)
            
            # Proyectar de vuelta a d_model
            lstm_output = Dense(params['d_model'], name=f'lstm_projection_{block_idx}')(lstm_output)
            
            # Add & Norm
            x = Add(name=f'add_lstm_{block_idx}')([x, lstm_output])
            x = LayerNormalization(name=f'norm_lstm_{block_idx}')(x)
        
        # 4. Agregación para predicción (un paso adelante)
        # Usar atención para ponderar la importancia de cada paso temporal
        attention_scores = Dense(1, activation='softmax', name='temporal_attention')(x)
        
        # Weighted sum usando los scores de atención
        x = Lambda(
            lambda inputs: tf.reduce_sum(inputs[0] * inputs[1], axis=1),
            name='weighted_aggregation'
        )([x, attention_scores])
        
        # 5. Capas de salida
        x = Dense(128, activation='relu', name='output_dense_1')(x)
        x = Dropout(params['dropout_rate'], name='output_dropout_1')(x)
        x = Dense(64, activation='relu', name='output_dense_2')(x)
        x = Dropout(params['dropout_rate']/2, name='output_dropout_2')(x)
        
        # Salida final: predicción del precio
        outputs = Dense(1, name='price_prediction')(x)
        
        # Crear modelo
        model = Model(inputs=inputs, outputs=outputs, name='Helformer')
        
        # Compilar con optimizador y métricas
        model.compile(
            optimizer=Adam(learning_rate=params.get('learning_rate', 0.001)),
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        return model

    def optimize_helformer_hyperparameters(self, X_train_seq, y_train_seq, X_val_seq, y_val_seq,
                                         n_features, sequence_length, n_trials=30):
        """
        Optimizar hiperparámetros del Helformer usando Optuna.
        """
        def objective(trial):
            # Hiperparámetros a optimizar
            params = {
                'season_length': trial.suggest_int('season_length', 12, 48, step=6),
                'd_model': trial.suggest_categorical('d_model', [128, 256, 384, 512]),
                'n_heads': trial.suggest_categorical('n_heads', [4, 8, 12, 16]),
                'n_blocks': trial.suggest_int('n_blocks', 2, 6),
                'lstm_units': trial.suggest_int('lstm_units', 64, 256, step=32),
                'dropout_rate': trial.suggest_float('dropout_rate', 0.0, 0.3, step=0.05),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128])
            }
            
            # Validar que n_heads divide a d_model
            if params['d_model'] % params['n_heads'] != 0:
                return float('inf')
            
            try:
                # Guardar temporalmente los parámetros
                self.helformer_best_params = params
                
                # Construir modelo
                model = self.build_helformer(n_features, sequence_length)
                
                # Callbacks
                callbacks = [
                    EarlyStopping(
                        monitor='val_loss',
                        patience=10,
                        restore_best_weights=True,
                        verbose=0
                    ),
                    ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.5,
                        patience=5,
                        min_lr=1e-6,
                        verbose=0
                    )
                ]
                
                # Entrenar
                history = model.fit(
                    X_train_seq, y_train_seq,
                    validation_data=(X_val_seq, y_val_seq),
                    epochs=30,
                    batch_size=params['batch_size'],
                    callbacks=callbacks,
                    verbose=0
                )
                
                # Limpiar memoria
                tf.keras.backend.clear_session()
                
                return min(history.history['val_loss'])
                
            except Exception as e:
                print(f"Error en trial: {e}")
                return float('inf')
        
        # Crear estudio Optuna
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=42)
        )
        
        # Optimizar
        study.optimize(
            objective,
            n_trials=n_trials,
            show_progress_bar=True
        )
        
        print(f"\nMejores hiperparámetros encontrados para Helformer:")
        print(f"Mejor pérdida de validación: {study.best_value:.6f}")
        for param, value in study.best_params.items():
            print(f"  {param}: {value}")
        
        # Guardar mejores parámetros
        self.helformer_best_params = study.best_params
        
        return study.best_params

    def prepare_data(self, data, target_col='target'):
        """Preparar datos para entrenamiento."""
        
        # Columnas a excluir (incluir datetime y timestamp)
        exclude_cols = ['target', 'datetime', 'timestamp']
        
        # Obtener solo columnas numéricas
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Filtrar columnas válidas (numéricas y no en exclude_cols)
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Eliminar características constantes o con muchos NaN
        valid_features = []
        for col in feature_cols:
            if data[col].nunique() > 1 and data[col].notna().sum() > len(data) * 0.5:
                valid_features.append(col)
        
        # IMPORTANTE: Asegurar que 'Close' sea la primera columna para Holt-Winters
        if 'Close' in valid_features:
            valid_features.remove('Close')
            valid_features = ['Close'] + valid_features
        
        X = data[valid_features].values
        y = data[target_col].values
        
        self.feature_names = valid_features
        print(f"Prepared data shape: X={X.shape}, y={y.shape}")
        print(f"Number of features: {len(valid_features)}")
        print(f"First feature (for HW decomposition): {valid_features[0]}")
        
        return X, y
    
    def prepare_sequences(self, X, y=None):
        """Preparar secuencias para Helformer."""
        X_seq = []
        y_seq = []
        
        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:(i + self.sequence_length)])
            if y is not None:
                y_seq.append(y[i + self.sequence_length])
        
        return np.array(X_seq), np.array(y_seq) if y is not None else None
    
    def optimize_lightgbm_params(self, X_train, y_train, X_val, y_val):
        """Optimizar hiperparámetros de LightGBM con Optuna."""
        print("Optimizing LightGBM hyperparameters...")
        
        def objective(trial):
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('num_leaves', 31, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
                'min_split_gain': trial.suggest_float('min_split_gain', 1e-3, 0.1, log=True),
                'subsample_for_bin': trial.suggest_int('subsample_for_bin', 20000, 300000),
                'n_estimators': trial.suggest_int('n_estimators', 100, 3000),
                'random_state': 42,
                'verbose': -1,
                'n_jobs': -1
            }
            
            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
            
            pred = model.predict(X_val)
            return mean_squared_error(y_val, pred)
        
        study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=50, show_progress_bar=True)
        
        print(f"Best MSE: {study.best_value:.6f}")
        return study.best_params
    
    def train_models(self, X_train, y_train, X_val, y_val, optimize_params=False):
        """Entrenar LightGBM y Helformer."""
        print("\n" + "="*60)
        print("TRAINING MODELS")
        print("="*60)
        
        # Escalar datos
        X_train_scaled = self.scalers['features'].fit_transform(X_train)
        y_train_scaled = self.scalers['target'].fit_transform(y_train.reshape(-1, 1)).ravel()
        X_val_scaled = self.scalers['features'].transform(X_val)
        y_val_scaled = self.scalers['target'].transform(y_val.reshape(-1, 1)).ravel()
        
        # 1. Entrenar LightGBM
        print("\n1. Training LightGBM...")
        
        if optimize_params:
            # Optimizar parámetros
            best_params = self.optimize_lightgbm_params(
                X_train_scaled, y_train_scaled, 
                X_val_scaled, y_val_scaled
            )
            self.lgb_params.update(best_params)
        
        self.models['lightgbm'] = lgb.LGBMRegressor(**self.lgb_params)
        self.models['lightgbm'].fit(
            X_train_scaled, y_train_scaled,
            eval_set=[(X_val_scaled, y_val_scaled)],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)]
        )
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.models['lightgbm'].feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 15 Most Important Features (LightGBM):")
        print(feature_importance.head(30).to_string(index=False))
        
        # 2. Entrenar Helformer
        print("\n2. Training Helformer...")
        
        # Preparar secuencias
        X_train_seq, y_train_seq = self.prepare_sequences(X_train_scaled, y_train_scaled)
        X_val_seq, y_val_seq = self.prepare_sequences(X_val_scaled, y_val_scaled)
        
        if len(X_train_seq) > 0:
            # Optimización con Optuna si se solicita
            if optimize_params:
                print("Optimizing Helformer hyperparameters with Optuna...")
                self.optimize_helformer_hyperparameters(
                    X_train_seq, y_train_seq, X_val_seq, y_val_seq,
                    n_features=X_train.shape[1], 
                    sequence_length=self.sequence_length,
                    n_trials=20  # Menos trials para ser más rápido
                )
            
            # Construir modelo
            self.models['helformer'] = self.build_helformer(
                n_features=X_train.shape[1],
                sequence_length=self.sequence_length
            )
            
            # Callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=20,
                    restore_best_weights=True,
                    mode='min'
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=10,
                    min_lr=1e-6,
                    verbose=1
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    'best_helformer.keras',
                    monitor='val_loss',
                    save_best_only=True,
                    mode='min',
                    verbose=0
                )
            ]
            
            # Entrenar
            epochs = 5
            batch_size = self.helformer_best_params.get('batch_size', 32) if self.helformer_best_params else 32
            
            history = self.models['helformer'].fit(
                X_train_seq, y_train_seq,
                validation_data=(X_val_seq, y_val_seq),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
            
            # Cargar mejor modelo
            self.models['helformer'].load_weights('best_helformer.keras')
            
            # Mostrar parámetros aprendidos de Holt-Winters
            try:
                hw_layer = self.models['helformer'].get_layer('holt_winters_decomposition')
                alpha = hw_layer.alpha.numpy()[0]
                gamma = hw_layer.gamma.numpy()[0]
                print(f"\nLearned Holt-Winters parameters:")
                print(f"  Alpha (α): {alpha:.4f}")
                print(f"  Gamma (γ): {gamma:.4f}")
            except Exception as e:
                print(f"Could not retrieve HW parameters: {e}")
            
            # Imprimir resumen del modelo
            print("\nHelformer Model Summary:")
            print(f"Total parameters: {self.models['helformer'].count_params():,}")
                    
        # 3. Entrenar meta-learner
        print("\n3. Training Meta-Learner...")
        self._train_meta_learner(X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled)
        
        print("\nAll models trained successfully!")
    
    def _train_meta_learner(self, X_train, y_train, X_val, y_val):
        """Entrenar meta-learner que combina predicciones."""
        
        # Obtener predicciones de modelos base
        train_preds = []
        val_preds = []
        
        # LightGBM predictions
        train_preds.append(self.models['lightgbm'].predict(X_train))
        val_preds.append(self.models['lightgbm'].predict(X_val))
        
        # Helformer predictions
        X_train_seq, y_train_seq = self.prepare_sequences(X_train, y_train)
        X_val_seq, y_val_seq = self.prepare_sequences(X_val, y_val)
        
        if len(X_train_seq) > 0:
            # Predicciones para secuencias
            helformer_train_pred = self.models['helformer'].predict(X_train_seq, verbose=0).ravel()
            helformer_val_pred = self.models['helformer'].predict(X_val_seq, verbose=0).ravel()
            
            # Alinear predicciones
            full_train_pred = np.zeros(len(y_train))
            full_val_pred = np.zeros(len(y_val))
            
            full_train_pred[self.sequence_length:self.sequence_length+len(helformer_train_pred)] = helformer_train_pred
            full_val_pred[self.sequence_length:self.sequence_length+len(helformer_val_pred)] = helformer_val_pred
            
            # Forward fill
            for i in range(len(full_train_pred)):
                if full_train_pred[i] == 0 and i > 0:
                    full_train_pred[i] = full_train_pred[i-1]
            for i in range(len(full_val_pred)):
                if full_val_pred[i] == 0 and i > 0:
                    full_val_pred[i] = full_val_pred[i-1]
            
            train_preds.append(full_train_pred)
            val_preds.append(full_val_pred)
        
        # Stack predictions
        train_meta_features = np.column_stack(train_preds)
        val_meta_features = np.column_stack(val_preds)
        
        # MODIFICACIÓN: Usar TODOS los features en lugar de solo los top
        train_meta_features = np.hstack([train_meta_features, X_train])
        val_meta_features = np.hstack([val_meta_features, X_val])
                
        # Construir meta-learner
        input_dim = train_meta_features.shape[1]
        
        inputs = Input(shape=(input_dim,))
        x = Dense(64, activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(32, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Dense(16, activation='relu')(x)
        outputs = Dense(1)(x)
        
        self.meta_learner = Model(inputs=inputs, outputs=outputs)
        self.meta_learner.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
        
        # Entrenar
        self.meta_learner.fit(
            train_meta_features, y_train,
            validation_data=(val_meta_features, y_val),
            epochs=100,
            batch_size=32,
            callbacks=[
                EarlyStopping(patience=15, restore_best_weights=True),
                ReduceLROnPlateau(patience=5, factor=0.5)
            ],
            verbose=0
        )
        
        # Actualizar pesos basados en rendimiento
        lgb_mse = mean_squared_error(y_val, val_preds[0])
        helformer_mse = mean_squared_error(y_val[self.sequence_length:], 
                                         val_preds[1][self.sequence_length:])
        
        # Pesos inversamente proporcionales al error
        total_inv_mse = 1/lgb_mse + 1/helformer_mse
        self.model_weights['lightgbm'] = (1/lgb_mse) / total_inv_mse
        self.model_weights['helformer'] = (1/helformer_mse) / total_inv_mse
        
        print(f"Initial model weights - LightGBM: {self.model_weights['lightgbm']:.3f}, "
              f"Helformer: {self.model_weights['helformer']:.3f}")
    
    def predict(self, X_test, use_meta_learner=True):
        """Hacer predicciones con el ensemble."""
        X_test_scaled = self.scalers['features'].transform(X_test)
        
        predictions = {}
        
        # LightGBM predictions
        predictions['lightgbm'] = self.models['lightgbm'].predict(X_test_scaled)
        
        # Helformer predictions
        X_test_seq, _ = self.prepare_sequences(X_test_scaled)
        if len(X_test_seq) > 0:
            helformer_pred = self.models['helformer'].predict(X_test_seq, verbose=0).ravel()
            
            # Alinear predicciones
            full_pred = np.zeros(len(X_test))
            full_pred[self.sequence_length:self.sequence_length+len(helformer_pred)] = helformer_pred
            
            # Forward fill
            for i in range(len(full_pred)):
                if full_pred[i] == 0 and i > 0:
                    full_pred[i] = full_pred[i-1]
            
            predictions['helformer'] = full_pred
        else:
            predictions['helformer'] = np.zeros(len(X_test))
        
        # Ensemble prediction
        if use_meta_learner and self.meta_learner is not None:
            # Preparar features para meta-learner
            pred_matrix = np.column_stack([predictions['lightgbm'], predictions['helformer']])
            
            # MODIFICACIÓN: Usar TODOS los features
            meta_features = np.hstack([pred_matrix, X_test_scaled])
            
            ensemble_pred = self.meta_learner.predict(meta_features, verbose=0).ravel()
        else:   
            # Weighted average
            weights = np.array([self.model_weights['lightgbm'], self.model_weights['helformer']])
            pred_matrix = np.column_stack([predictions['lightgbm'], predictions['helformer']])
            ensemble_pred = np.average(pred_matrix, weights=weights, axis=1)
        
        # Inverse transform
        ensemble_pred = self.scalers['target'].inverse_transform(ensemble_pred.reshape(-1, 1)).ravel()
        
        for name in predictions:
            predictions[name] = self.scalers['target'].inverse_transform(
                predictions[name].reshape(-1, 1)
            ).ravel()
        
        return ensemble_pred, predictions
    
    def update_model_weights_online(self, predictions, actual_price):
        """Actualizar pesos del modelo en línea."""
        errors = {}
        
        for name, pred in predictions.items():
            error = abs(pred - actual_price)
            errors[name] = error
            self.performance_history[name].append(error)
        
        # Calcular rendimiento reciente
        recent_performance = {}
        for name in ['lightgbm', 'helformer']:
            if len(self.performance_history[name]) > 0:
                recent_errors = list(self.performance_history[name])
                weights = np.exp(-0.1 * np.arange(len(recent_errors)))
                weights = weights / weights.sum()
                recent_performance[name] = np.average(recent_errors, weights=weights)
            else:
                recent_performance[name] = float('inf')
        
        # Actualizar pesos
        performance_scores = {}
        for name in ['lightgbm', 'helformer']:
            if recent_performance[name] != float('inf') and recent_performance[name] > 0:
                performance_scores[name] = 1.0 / recent_performance[name]
            else:
                performance_scores[name] = 1.0
        
        # Aplicar momentum
        new_weights = {}
        total_score = sum(performance_scores.values())
        
        for name in ['lightgbm', 'helformer']:
            target_weight = performance_scores[name] / total_score if total_score > 0 else 0.5
            current_weight = self.model_weights.get(name, 0.5)
            new_weight = (self.weight_momentum * current_weight + 
                         (1 - self.weight_momentum) * target_weight)
            new_weight = max(new_weight, self.min_weight)
            new_weights[name] = new_weight
        
        # Normalizar
        total_weight = sum(new_weights.values())
        self.model_weights = {name: w/total_weight for name, w in new_weights.items()}
        
        self.weight_history.append(self.model_weights.copy())
        
        return self.model_weights
    
    def simulate_real_time_forecast(self, X_test, y_test, n_steps=None, verbose=True):
        """Simular predicción en tiempo real con adaptación de pesos."""
        if n_steps is None:
            n_steps = min(len(X_test) - 1, 168)  # Default 1 semana
        else:
            n_steps = min(n_steps, len(X_test) - 1)
        
        if verbose:
            print(f"\nSimulating real-time forecast for {n_steps} steps...")
            print("="*60)
        
        predictions = []
        actual_values = []
        all_model_predictions = {'lightgbm': [], 'helformer': []}
        weight_evolution = []
        errors = []
        
        for step in range(n_steps):
            # Features actuales
            current_features = X_test[step].reshape(1, -1)
            
            # Predicción
            pred, model_preds = self.predict(current_features, use_meta_learner=True)
            
            # Guardar predicciones
            predictions.append(pred[0])
            actual_values.append(y_test[step])
            
            for name in ['lightgbm', 'helformer']:
                all_model_predictions[name].append(model_preds[name][0])
            
            # Calcular error
            error = abs(pred[0] - y_test[step])
            errors.append(error)
            
            # Actualizar pesos
            if step > 0:
                pred_dict = {name: model_preds[name][0] for name in ['lightgbm', 'helformer']}
                self.update_model_weights_online(pred_dict, y_test[step])
            
            weight_evolution.append(self.model_weights.copy())
            
            # Progress
            if verbose and (step + 1) % 24 == 0:
                recent_mae = np.mean(errors[-24:])
                print(f"Step {step + 1}/{n_steps}: Recent 24h MAE = ${recent_mae:.2f}")
                print(f"Weights - LightGBM: {self.model_weights['lightgbm']:.3f}, "
                      f"Helformer: {self.model_weights['helformer']:.3f}")
        
        # Métricas finales
        predictions = np.array(predictions)
        actual_values = np.array(actual_values)
        
        mae = mean_absolute_error(actual_values, predictions)
        rmse = np.sqrt(mean_squared_error(actual_values, predictions))
        mape = np.mean(np.abs((actual_values - predictions) / actual_values)) * 100
        
        if verbose:
            print("\n" + "="*60)
            print("REAL-TIME SIMULATION RESULTS")
            print("="*60)
            print(f"Total steps: {len(predictions)}")
            print(f"MAE: ${mae:.2f}")
            print(f"RMSE: ${rmse:.2f}")
            print(f"MAPE: {mape:.2f}%")
            print("\nFinal Model Weights:")
            print(f"  LightGBM: {self.model_weights['lightgbm']:.3f}")
            print(f"  Helformer: {self.model_weights['helformer']:.3f}")
        
        return {
            'predictions': predictions,
            'actual_values': actual_values,
            'all_model_predictions': all_model_predictions,
            'weight_evolution': weight_evolution,
            'errors': errors,
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        }