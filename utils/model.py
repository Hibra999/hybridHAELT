import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, QuantileTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, roc_auc_score
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.linear_model import Ridge, Lasso, ElasticNet
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from datetime import time
# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings

import tensorflow as tf
# Additional warning suppression
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
tf.get_logger().setLevel('ERROR')

# For older TF versions compatibility
try:
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except AttributeError:
    pass

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (Input, LSTM, Dense, Dropout, LayerNormalization,
                                   MultiHeadAttention, Conv1D, GlobalAveragePooling1D,
                                   BatchNormalization, Concatenate, Embedding, Add, Layer,
                                   Flatten, Reshape, GlobalMaxPooling1D, Lambda, GRU,
                                   Bidirectional, TimeDistributed, Permute, RepeatVector,
                                   Activation, Multiply)
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
from collections import deque
import optuna
from optuna.samplers import TPESampler
import ccxt
import re

# Try to import advanced libraries
try:
    from pytorch_tabnet.tab_model import TabNetRegressor, TabNetClassifier
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False
    print("TabNet not available. Install with: pip install pytorch-tabnet")

try:
    from flaml import AutoML
    FLAML_AVAILABLE = True
except ImportError:
    FLAML_AVAILABLE = False
    print("FLAML not available. Install with: pip install flaml")


class TemporalConvNet(Layer):
    """Temporal Convolutional Network"""

    def __init__(self, num_channels, kernel_size=3, dropout=0.2, **kwargs):
        super(TemporalConvNet, self).__init__(**kwargs)
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.layers = []

    def build(self, input_shape):
        num_levels = len(self.num_channels)

        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_shape[-1] if i == 0 else self.num_channels[i-1]
            out_channels = self.num_channels[i]

            # Dilated convolution
            conv = Conv1D(
                out_channels, self.kernel_size,
                padding='causal', dilation_rate=dilation_size,
                activation='relu'
            )

            # Batch norm
            batch_norm = BatchNormalization()

            # Dropout
            dropout = Dropout(self.dropout)

            # Residual connection
            if in_channels != out_channels:
                residual = Conv1D(out_channels, 1, padding='same')
            else:
                residual = None

            self.layers.append({
                'conv': conv,
                'batch_norm': batch_norm,
                'dropout': dropout,
                'residual': residual
            })

    def call(self, inputs, training=None):
        x = inputs

        for layer in self.layers:
            residual = x

            # Apply convolution
            x = layer['conv'](x)
            x = layer['batch_norm'](x, training=training)
            x = tf.keras.layers.Activation('relu')(x)
            x = layer['dropout'](x, training=training)

            # Apply residual connection
            if layer['residual'] is not None:
                residual = layer['residual'](residual)

            x = x + residual

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_channels': self.num_channels,
            'kernel_size': self.kernel_size,
            'dropout': self.dropout
        })
        return config


class EnhancedHierarchicalMetaLearner:
    """Enhanced Hierarchical Meta-Learning System with Only 2 Models"""

    def __init__(self, sequence_length=168, forecast_horizon=1):
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon

        # Base models - ONLY 2 MODELS
        self.base_models = {
            'regression': {},
            'classification': {}
        }

        # Model scores and weights
        self.model_scores = {
            'regression': {},
            'classification': {}
        }
        self.model_weights = {
            'regression': {},
            'classification': {}
        }

        # Hierarchical meta-learners
        self.meta_learners = {
            'level1_regression': None,      # First level: combine regression predictions
            'level1_classification': None,   # First level: combine classification predictions
            'level2_final': None            # Second level: combine level1 outputs
        }

        # Scalers
        self.scalers = {
            'features': StandardScaler(),
            'features_robust': RobustScaler(),
            'features_quantile': QuantileTransformer(n_quantiles=1000, output_distribution='normal'),
            'target': StandardScaler(),
            'meta_features': StandardScaler()
        }

        # Feature names - IMPORTANT for consistency
        self.feature_names = None

        # Performance tracking
        self.performance_history = {}

        # Stacking parameters
        self.n_folds = 5
        self.use_time_series_cv = True

        # Expert weights for ensemble models
        self.expert_weights = {}

        # Model parameters
        self._initialize_model_parameters()

    def _initialize_model_parameters(self):
        """Initialize parameters for only 2 models"""

        # LightGBM parameters (best tree-based model)
        self.lgb_params_base = {
            'boosting_type': 'gbdt',
            'num_leaves': 255,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.7,
            'bagging_freq': 5,
            'verbose': -1,
            'max_depth': -1,
            'min_child_samples': 20,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'n_jobs': -1
        }

        self.lgb_params_regression = {
            **self.lgb_params_base,
            'objective': 'regression',
            'metric': 'rmse',
            'learning_rate': 0.02,
            'n_estimators': 2000
        }

        self.lgb_params_classification = {
            **self.lgb_params_base,
            'objective': 'binary',
            'metric': 'binary_logloss',
            'learning_rate': 0.03,
            'n_estimators': 1500
        }

    def create_binary_labels(self, y):
        """Convert continuous price targets to binary up/down labels"""
        returns = np.diff(y, prepend=y[0])
        binary_labels = (returns > 0).astype(int)
        return binary_labels

    def build_tcn_gru_attention(self, n_features, sequence_length):
        """
        Builds a state-of-the-art time-series model using TCN, Bi-GRU, and Attention.
        """
        inputs = Input(shape=(sequence_length, n_features))

        # 1. Initial Conv layer to create a richer feature representation
        x = Conv1D(filters=64, kernel_size=3, padding='causal', activation='relu')(inputs)
        x = LayerNormalization()(x)
        x = Dropout(0.2)(x)

        # 2. Temporal Convolutional Network (TCN) to extract hierarchical temporal features
        tcn_channels = [64, 128, 64] # Number of filters for each TCN layer
        x = TemporalConvNet(num_channels=tcn_channels, kernel_size=3, dropout=0.2)(x)

        # 3. Bidirectional GRU to capture long-term sequential dependencies in both directions
        x = Bidirectional(GRU(64, return_sequences=True))(x)

        # 4. Attention mechanism to focus on the most relevant time steps
        attention = Dense(1, activation='tanh')(x)
        attention = Flatten()(attention)
        attention = Activation('softmax')(attention)
        attention = RepeatVector(128)(attention) # 64 units * 2 for bidirectional GRU
        attention = Permute([2, 1])(attention)

        # Apply attention weights
        weighted_representation = Multiply()([x, attention])
        pooled_representation = Lambda(lambda xin: K.sum(xin, axis=1))(weighted_representation)

        # 5. Final MLP for prediction
        x = Dense(128, activation='relu')(pooled_representation)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        # 6. Dual outputs for regression and classification
        regression_output = Dense(1, name='regression')(x)
        classification_output = Dense(1, activation='sigmoid', name='classification')(x)

        model = Model(inputs=inputs, outputs=[regression_output, classification_output])

        model.compile(
            optimizer=AdamW(learning_rate=0.001, weight_decay=0.01),
            loss={'regression': 'mse', 'classification': 'binary_crossentropy'},
            metrics={'regression': 'mae', 'classification': 'accuracy'}
        )
        return model

    def build_hierarchical_meta_learner_level1(self, n_models, task_type='regression'):
        """Build first level meta-learner for specific task"""
        input_dim = n_models

        inputs = Input(shape=(input_dim,))

        # Feature expansion
        x = Dense(128, activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        # Self-attention on model predictions
        x_reshaped = Reshape((n_models, 1))(inputs)
        x_expanded = Dense(64)(x_reshaped)
        attention = MultiHeadAttention(num_heads=4, key_dim=16)(x_expanded, x_expanded)
        attention_flat = Flatten()(attention)

        # Combine with dense features
        x = Concatenate()([x, attention_flat])

        # Deep layers
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)

        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        # Task-specific output
        if task_type == 'regression':
            output = Dense(1)(x)
            loss = 'mse'
            metrics = ['mae']
        else:
            output = Dense(1, activation='sigmoid')(x)
            loss = 'binary_crossentropy'
            metrics = ['accuracy', tf.keras.metrics.AUC(name='auc')]

        model = Model(inputs=inputs, outputs=output)
        model.compile(optimizer=AdamW(learning_rate=0.001, weight_decay=0.01),
                     loss=loss, metrics=metrics)

        return model

    def build_hierarchical_meta_learner_level2(self, n_features_from_data):
        """Build second level meta-learner that combines level1 outputs with original features"""
        # Input: 2 level1 outputs (regression, classification) + original features
        input_dim = 2 + n_features_from_data

        inputs = Input(shape=(input_dim,))

        # Separate level1 predictions and original features
        level1_preds = Lambda(lambda x: x[:, :2])(inputs)
        original_features = Lambda(lambda x: x[:, 2:])(inputs)

        # Process level1 predictions
        level1_processed = Dense(32, activation='relu')(level1_preds)
        level1_processed = BatchNormalization()(level1_processed)

        # Process original features with dimension reduction
        features_processed = Dense(256, activation='relu')(original_features)
        features_processed = BatchNormalization()(features_processed)
        features_processed = Dropout(0.3)(features_processed)
        features_processed = Dense(128, activation='relu')(features_processed)
        features_processed = BatchNormalization()(features_processed)
        features_processed = Dropout(0.2)(features_processed)

        # Combine
        combined = Concatenate()([level1_processed, features_processed])

        # Deep processing
        x = Dense(256, activation='relu')(combined)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)

        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        # Final regression output
        output = Dense(1)(x)

        model = Model(inputs=inputs, outputs=output)
        model.compile(optimizer=AdamW(learning_rate=0.001, weight_decay=0.01),
                     loss='mse', metrics=['mae', 'mape'])

        return model

    def prepare_data(self, data, target_col='target'):
        """Prepare data for training"""
        exclude_cols = ['target', 'datetime', 'timestamp']
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]

        # If feature_names already set (during prediction), use those exact features
        if self.feature_names is not None:
            # Use only the features that were used during training
            missing_features = [f for f in self.feature_names if f not in data.columns]
            if missing_features:
                print(f"Warning: Missing features: {missing_features[:5]}...")  # Show first 5
                # Add missing features with zeros
                for feat in missing_features:
                    data[feat] = 0

            X = data[self.feature_names].values
            if target_col in data.columns:
                y = data[target_col].values
            else:
                # For prediction, use Close as target
                y = data['Close'].values
        else:
            # First time - determine valid features
            valid_features = []
            for col in feature_cols:
                if data[col].nunique() > 1 and data[col].notna().sum() > len(data) * 0.5:
                    valid_features.append(col)

            # Ensure 'Close' is the first column for consistency
            if 'Close' in valid_features:
                valid_features.remove('Close')
                valid_features = ['Close'] + valid_features

            X = data[valid_features].values
            y = data[target_col].values

            self.feature_names = valid_features
            print(f"Prepared data shape: X={X.shape}, y={y.shape}")
            print(f"Number of features: {len(valid_features)}")

        return X, y

    def prepare_sequences(self, X, y=None):
        """Prepare sequences for neural network models"""
        X_seq = []
        y_seq = []

        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:(i + self.sequence_length)])
            if y is not None:
                y_seq.append(y[i + self.sequence_length])

        return np.array(X_seq), np.array(y_seq) if y is not None else None

    def train_base_models(self, X_train, y_train, X_val, y_val):
        """Train only 2 base models"""
        print("\n" + "="*60)
        print("TRAINING BASE MODELS (2 MODELS ONLY)")
        print("="*60)

        # Create binary labels
        y_train_binary = self.create_binary_labels(y_train)
        y_val_binary = self.create_binary_labels(y_val)

        # Scale features
        X_train_scaled = self.scalers['features'].fit_transform(X_train)
        X_val_scaled = self.scalers['features'].transform(X_val)

        y_train_scaled = self.scalers['target'].fit_transform(y_train.reshape(-1, 1)).ravel()
        y_val_scaled = self.scalers['target'].transform(y_val.reshape(-1, 1)).ravel()

        # 1. Train LightGBM (Best tree-based model)
        print("\n1. Training LightGBM...")
        self.base_models['regression']['lightgbm'] = lgb.LGBMRegressor(**self.lgb_params_regression)
        self.base_models['regression']['lightgbm'].fit(
            X_train_scaled, y_train_scaled,
            eval_set=[(X_val_scaled, y_val_scaled)],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)]
        )

        self.base_models['classification']['lightgbm'] = lgb.LGBMClassifier(**self.lgb_params_classification)
        self.base_models['classification']['lightgbm'].fit(
            X_train_scaled, y_train_binary,
            eval_set=[(X_val_scaled, y_val_binary)],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)]
        )

        # 2. Train TCN-GRU-Attention Model (New advanced model)
        print("\n2. Training TCN-GRU-Attention Model...")

        # Prepare sequences
        X_train_seq, y_train_seq = self.prepare_sequences(X_train_scaled, y_train_scaled)
        X_val_seq, y_val_seq = self.prepare_sequences(X_val_scaled, y_val_scaled)

        # Binary labels for sequences
        y_train_seq_binary = self.create_binary_labels(
            self.scalers['target'].inverse_transform(y_train_seq.reshape(-1, 1)).ravel()
        )
        y_val_seq_binary = self.create_binary_labels(
            self.scalers['target'].inverse_transform(y_val_seq.reshape(-1, 1)).ravel()
        )

        if len(X_train_seq) > 0:
            # Common callbacks
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
            ]

            # Build and train
            self.base_models['regression']['tcn_gru_attention'] = self.build_tcn_gru_attention(
                n_features=X_train.shape[1],
                sequence_length=self.sequence_length
            )

            self.base_models['regression']['tcn_gru_attention'].fit(
                X_train_seq,
                {'regression': y_train_seq, 'classification': y_train_seq_binary},
                validation_data=(X_val_seq, {'regression': y_val_seq, 'classification': y_val_seq_binary}),
                epochs=100,
                batch_size=32,
                callbacks=callbacks,
                verbose=0
            )

            # Also add classification output
            self.base_models['classification']['tcn_gru_attention'] = self.base_models['regression']['tcn_gru_attention']

        print("\nAll base models trained successfully!")

        # Print model count
        n_regression_models = len(self.base_models['regression'])
        n_classification_models = len(self.base_models['classification'])
        print(f"\nTotal models trained:")
        print(f"  Regression models: {n_regression_models}")
        print(f"  Classification models: {n_classification_models}")

    def get_stacking_predictions(self, X_train, y_train_reg, y_train_clf, X_val):
        """Get out-of-fold predictions using stacking"""
        n_train = len(X_train)

        # Initialize prediction arrays
        train_preds_reg = {}
        train_preds_clf = {}
        val_preds_reg = {}
        val_preds_clf = {}

        # Get scaled version
        X_train_scaled = self.scalers['features'].transform(X_train)
        X_val_scaled = self.scalers['features'].transform(X_val)

        # Choose cross-validation strategy
        if self.use_time_series_cv:
            cv = TimeSeriesSplit(n_splits=self.n_folds)
        else:
            cv = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        # LightGBM stacking
        print(f"\nGenerating stacking predictions for lightgbm...")

        # Initialize arrays
        train_preds_reg['lightgbm'] = np.zeros(n_train)
        train_preds_clf['lightgbm'] = np.zeros(n_train)
        val_preds_reg['lightgbm'] = []
        val_preds_clf['lightgbm'] = []

        # Cross-validation
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_scaled)):
            X_fold_train = X_train_scaled[train_idx]
            y_fold_train_reg = y_train_reg[train_idx]
            y_fold_train_clf = y_train_clf[train_idx]

            X_fold_val = X_train_scaled[val_idx]

            # Train fold models
            fold_reg_model = lgb.LGBMRegressor(**self.lgb_params_regression)
            fold_clf_model = lgb.LGBMClassifier(**self.lgb_params_classification)

            # Fit models
            fold_reg_model.fit(X_fold_train, y_fold_train_reg)
            fold_clf_model.fit(X_fold_train, y_fold_train_clf)

            # Get OOF predictions
            train_preds_reg['lightgbm'][val_idx] = fold_reg_model.predict(X_fold_val)
            train_preds_clf['lightgbm'][val_idx] = fold_clf_model.predict_proba(X_fold_val)[:, 1]

            # Predict on validation set
            val_preds_reg['lightgbm'].append(fold_reg_model.predict(X_val_scaled))
            val_preds_clf['lightgbm'].append(fold_clf_model.predict_proba(X_val_scaled)[:, 1])

        # Average validation predictions
        val_preds_reg['lightgbm'] = np.mean(val_preds_reg['lightgbm'], axis=0)
        val_preds_clf['lightgbm'] = np.mean(val_preds_clf['lightgbm'], axis=0)

        # Neural network predictions (no CV, use trained models)
        X_train_seq, y_train_seq = self.prepare_sequences(self.scalers['features'].transform(X_train))
        X_val_seq, _ = self.prepare_sequences(X_val_scaled)

        if 'tcn_gru_attention' in self.base_models['regression']:
            # Get predictions
            nn_pred_train = self.base_models['regression']['tcn_gru_attention'].predict(X_train_seq, verbose=0)
            nn_pred_val = self.base_models['regression']['tcn_gru_attention'].predict(X_val_seq, verbose=0)

            # Extract regression and classification predictions
            if isinstance(nn_pred_train, list):
                train_reg = nn_pred_train[0].ravel()
                train_clf = nn_pred_train[1].ravel()
                val_reg = nn_pred_val[0].ravel()
                val_clf = nn_pred_val[1].ravel()
            else: # Fallback for dictionary output
                train_reg = nn_pred_train['regression'].ravel()
                train_clf = nn_pred_train['classification'].ravel()
                val_reg = nn_pred_val['regression'].ravel()
                val_clf = nn_pred_val['classification'].ravel()

            # Align predictions
            train_preds_reg['tcn_gru_attention'] = self._align_nn_predictions(train_reg, n_train)
            train_preds_clf['tcn_gru_attention'] = self._align_nn_predictions(train_clf, n_train)
            val_preds_reg['tcn_gru_attention'] = self._align_nn_predictions(val_reg, len(X_val))
            val_preds_clf['tcn_gru_attention'] = self._align_nn_predictions(val_clf, len(X_val))

        return train_preds_reg, train_preds_clf, val_preds_reg, val_preds_clf

    def _align_nn_predictions(self, predictions, target_length):
        """Align neural network predictions to target length by padding and forward-filling."""
        aligned = np.zeros(target_length)
        # Predictions start after the first sequence is complete
        start_idx = self.sequence_length
        pred_len = len(predictions)

        # Ensure we don't write past the end of the array
        end_idx = min(start_idx + pred_len, target_length)
        
        # Place predictions into the aligned array
        if end_idx > start_idx:
            aligned[start_idx:end_idx] = predictions[:end_idx-start_idx]

        # Forward-fill the initial part (before first prediction)
        if pred_len > 0:
            aligned[:start_idx] = aligned[start_idx]

        # Forward-fill any gaps at the end if predictions were shorter than the target
        if end_idx < target_length and pred_len > 0:
             aligned[end_idx:] = aligned[end_idx - 1]

        return aligned

    def train_hierarchical_meta_learners(self, X_train, y_train, X_val, y_val,
                                       train_preds_reg, train_preds_clf,
                                       val_preds_reg, val_preds_clf):
        """Train hierarchical meta-learners"""
        print("\n" + "="*60)
        print("TRAINING HIERARCHICAL META-LEARNERS")
        print("="*60)

        # Create binary labels
        y_train_binary = self.create_binary_labels(y_train)
        y_val_binary = self.create_binary_labels(y_val)

        # Scale targets
        y_train_scaled = self.scalers['target'].transform(y_train.reshape(-1, 1)).ravel()
        y_val_scaled = self.scalers['target'].transform(y_val.reshape(-1, 1)).ravel()

        # 1. Train Level 1 Regression Meta-Learner
        print("\n1. Training Level 1 Regression Meta-Learner...")

        # Stack regression predictions
        train_meta_reg = np.column_stack([train_preds_reg[model] for model in sorted(train_preds_reg.keys())])
        val_meta_reg = np.column_stack([val_preds_reg[model] for model in sorted(val_preds_reg.keys())])

        # Scale meta features
        train_meta_reg_scaled = self.scalers['meta_features'].fit_transform(train_meta_reg)
        val_meta_reg_scaled = self.scalers['meta_features'].transform(val_meta_reg)

        # Build and train
        self.meta_learners['level1_regression'] = self.build_hierarchical_meta_learner_level1(
            n_models=train_meta_reg.shape[1],
            task_type='regression'
        )

        self.meta_learners['level1_regression'].fit(
            train_meta_reg_scaled, y_train_scaled,
            validation_data=(val_meta_reg_scaled, y_val_scaled),
            epochs=200,
            batch_size=64,
            callbacks=[
                EarlyStopping(patience=20, restore_best_weights=True),
                ReduceLROnPlateau(patience=10, factor=0.5, min_lr=1e-6)
            ],
            verbose=0
        )

        # Get level 1 regression predictions
        level1_reg_train = self.meta_learners['level1_regression'].predict(train_meta_reg_scaled, verbose=0).ravel()
        level1_reg_val = self.meta_learners['level1_regression'].predict(val_meta_reg_scaled, verbose=0).ravel()

        # 2. Train Level 1 Classification Meta-Learner
        print("\n2. Training Level 1 Classification Meta-Learner...")

        # Stack classification predictions
        train_meta_clf = np.column_stack([train_preds_clf[model] for model in sorted(train_preds_clf.keys())])
        val_meta_clf = np.column_stack([val_preds_clf[model] for model in sorted(val_preds_clf.keys())])

        # Build and train
        self.meta_learners['level1_classification'] = self.build_hierarchical_meta_learner_level1(
            n_models=train_meta_clf.shape[1],
            task_type='classification'
        )

        self.meta_learners['level1_classification'].fit(
            train_meta_clf, y_train_binary,
            validation_data=(val_meta_clf, y_val_binary),
            epochs=200,
            batch_size=64,
            callbacks=[
                EarlyStopping(patience=20, restore_best_weights=True),
                ReduceLROnPlateau(patience=10, factor=0.5, min_lr=1e-6)
            ],
            verbose=0
        )

        # Get level 1 classification predictions
        level1_clf_train = self.meta_learners['level1_classification'].predict(train_meta_clf, verbose=0).ravel()
        level1_clf_val = self.meta_learners['level1_classification'].predict(val_meta_clf, verbose=0).ravel()

        # 3. Train Level 2 Final Meta-Learner
        print("\n3. Training Level 2 Final Meta-Learner...")

        # Combine level 1 predictions with original features
        X_train_scaled = self.scalers['features'].transform(X_train)
        X_val_scaled = self.scalers['features'].transform(X_val)

        train_level2_input = np.column_stack([level1_reg_train, level1_clf_train, X_train_scaled])
        val_level2_input = np.column_stack([level1_reg_val, level1_clf_val, X_val_scaled])

        # Build and train
        self.meta_learners['level2_final'] = self.build_hierarchical_meta_learner_level2(
            n_features_from_data=X_train.shape[1]
        )

        self.meta_learners['level2_final'].fit(
            train_level2_input, y_train_scaled,
            validation_data=(val_level2_input, y_val_scaled),
            epochs=200,
            batch_size=64,
            callbacks=[
                EarlyStopping(patience=25, restore_best_weights=True),
                ReduceLROnPlateau(patience=12, factor=0.5, min_lr=1e-6)
            ],
            verbose=0
        )

        # Evaluate meta-learners
        level1_reg_mae = mean_absolute_error(y_val_scaled, level1_reg_val)
        level1_clf_auc = roc_auc_score(y_val_binary, level1_clf_val)

        final_pred = self.meta_learners['level2_final'].predict(val_level2_input, verbose=0).ravel()
        final_mae = mean_absolute_error(y_val_scaled, final_pred)

        print(f"\nMeta-learner Performance:")
        print(f"  Level 1 Regression MAE: {level1_reg_mae:.4f}")
        print(f"  Level 1 Classification AUC: {level1_clf_auc:.3f}")
        print(f"  Level 2 Final MAE: {final_mae:.4f}")

        # Calculate model weights based on performance
        self._calculate_model_weights(val_preds_reg, val_preds_clf, y_val_scaled, y_val_binary)

    def _calculate_model_weights(self, val_preds_reg, val_preds_clf, y_val_scaled, y_val_binary):
        """Calculate initial model weights based on validation performance"""

        # Regression weights
        for model_name, preds in val_preds_reg.items():
            mse = mean_squared_error(y_val_scaled, preds)
            self.model_scores['regression'][model_name] = 1.0 / (mse + 1e-6)

        # Normalize regression weights
        total_score = sum(self.model_scores['regression'].values())
        if total_score > 0:
            self.model_weights['regression'] = {
                name: score/total_score
                for name, score in self.model_scores['regression'].items()
            }
        else:
             self.model_weights['regression'] = {name: 1/len(val_preds_reg) for name in val_preds_reg}


        # Classification weights
        for model_name, preds in val_preds_clf.items():
            try:
                auc = roc_auc_score(y_val_binary, preds)
                self.model_scores['classification'][model_name] = auc
            except:
                self.model_scores['classification'][model_name] = 0.5

        # Normalize classification weights
        total_score = sum(self.model_scores['classification'].values())
        if total_score > 0:
            self.model_weights['classification'] = {
                name: score/total_score
                for name, score in self.model_scores['classification'].items()
            }
        else:
            self.model_weights['classification'] = {name: 1/len(val_preds_clf) for name in val_preds_clf}


        print("\nInitial Model Weights:")
        print("\nRegression:")
        for name, weight in sorted(self.model_weights['regression'].items()):
            print(f"  {name}: {weight:.3f}")
        print("\nClassification:")
        for name, weight in sorted(self.model_weights['classification'].items()):
            print(f"  {name}: {weight:.3f}")

    def train(self, X_train, y_train, X_val, y_val):
        """Main training method"""

        # Train base models
        self.train_base_models(X_train, y_train, X_val, y_val)

        # Get stacking predictions
        y_train_binary = self.create_binary_labels(y_train)

        y_train_scaled = self.scalers['target'].transform(y_train.reshape(-1, 1)).ravel()

        train_preds_reg, train_preds_clf, val_preds_reg, val_preds_clf = self.get_stacking_predictions(
            X_train, y_train_scaled, y_train_binary, X_val
        )

        # Train hierarchical meta-learners
        self.train_hierarchical_meta_learners(
            X_train, y_train, X_val, y_val,
            train_preds_reg, train_preds_clf,
            val_preds_reg, val_preds_clf
        )

        print("\nTraining completed successfully!")

    def predict(self, X_test, return_all_predictions=False):
        """Make predictions using the hierarchical ensemble"""

        # Get predictions from all base models
        base_preds_reg = {}
        base_preds_clf = {}

        # Scaled version
        X_test_scaled = self.scalers['features'].transform(X_test)

        # LightGBM
        base_preds_reg['lightgbm'] = self.base_models['regression']['lightgbm'].predict(X_test_scaled)
        base_preds_clf['lightgbm'] = self.base_models['classification']['lightgbm'].predict_proba(X_test_scaled)[:, 1]

        # Neural network model
        if 'tcn_gru_attention' in self.base_models['regression']:
            # FIX: Smartly handle sequence creation for single vs. batch prediction
            if len(X_test_scaled) < self.sequence_length:
                # Not enough data for a single sequence, predict zeros
                nn_reg_preds = np.zeros(len(X_test_scaled))
                nn_clf_preds = np.zeros(len(X_test_scaled))
            else:
                # Prepare sequences from the input data
                X_test_seq, _ = self.prepare_sequences(X_test_scaled)

                if X_test_seq.shape[0] > 0:
                    # We have at least one full sequence to predict from
                    nn_pred = self.base_models['regression']['tcn_gru_attention'].predict(X_test_seq, verbose=0)
                    
                    if isinstance(nn_pred, list):
                        reg_pred_seq = nn_pred[0].ravel()
                        clf_pred_seq = nn_pred[1].ravel()
                    else: # Fallback for dict
                        reg_pred_seq = nn_pred['regression'].ravel()
                        clf_pred_seq = nn_pred['classification'].ravel()
                    
                    # Align predictions back to the original X_test length
                    nn_reg_preds = self._align_nn_predictions(reg_pred_seq, len(X_test_scaled))
                    nn_clf_preds = self._align_nn_predictions(clf_pred_seq, len(X_test_scaled))
                else:
                    # This case handles len(X_test_scaled) == self.sequence_length
                    # Create a single batch for prediction
                    X_test_seq_single = np.expand_dims(X_test_scaled, axis=0)
                    nn_pred_single = self.base_models['regression']['tcn_gru_attention'].predict(X_test_seq_single, verbose=0)

                    if isinstance(nn_pred_single, list):
                        reg_pred_val = nn_pred_single[0].ravel()[0]
                        clf_pred_val = nn_pred_single[1].ravel()[0]
                    else: # Fallback for dict
                        reg_pred_val = nn_pred_single['regression'].ravel()[0]
                        clf_pred_val = nn_pred_single['classification'].ravel()[0]
                    
                    # The prediction is for the next step, so we fill the whole array with it
                    nn_reg_preds = np.full(len(X_test_scaled), reg_pred_val)
                    nn_clf_preds = np.full(len(X_test_scaled), clf_pred_val)

            base_preds_reg['tcn_gru_attention'] = nn_reg_preds
            base_preds_clf['tcn_gru_attention'] = nn_clf_preds

        # Level 1 predictions
        # Regression
        meta_reg = np.column_stack([base_preds_reg[model] for model in sorted(base_preds_reg.keys())])
        meta_reg_scaled = self.scalers['meta_features'].transform(meta_reg)
        level1_reg = self.meta_learners['level1_regression'].predict(meta_reg_scaled, verbose=0).ravel()

        # Classification
        meta_clf = np.column_stack([base_preds_clf[model] for model in sorted(base_preds_clf.keys())])
        level1_clf = self.meta_learners['level1_classification'].predict(meta_clf, verbose=0).ravel()

        # Level 2 final prediction
        level2_input = np.column_stack([level1_reg, level1_clf, X_test_scaled])
        final_pred_scaled = self.meta_learners['level2_final'].predict(level2_input, verbose=0).ravel()

        # Inverse transform
        final_pred = self.scalers['target'].inverse_transform(final_pred_scaled.reshape(-1, 1)).ravel()

        # Also inverse transform base predictions
        for model_name in base_preds_reg:
            base_preds_reg[model_name] = self.scalers['target'].inverse_transform(
                base_preds_reg[model_name].reshape(-1, 1)
            ).ravel()

        if return_all_predictions:
            return {
                'final_prediction': final_pred,
                'level1_regression': self.scalers['target'].inverse_transform(level1_reg.reshape(-1, 1)).ravel(),
                'level1_classification': level1_clf,
                'base_regression': base_preds_reg,
                'base_classification': base_preds_clf,
                'ensemble_confidence': level1_clf,
                'ensemble_direction': (level1_clf > 0.5).astype(int)
            }
        else:
            return final_pred

# Register custom layers with Keras for proper serialization
tf.keras.utils.get_custom_objects().update({
    'TemporalConvNet': TemporalConvNet,
})

def simulate_enhanced_real_time_forecast(model, test_data, scaler_X, scaler_y, forecast_horizon=168, update_interval=10):
    """
    Simulate real-time forecasting with a rolling window.
    """
    print(f"\nSimulating real-time forecast for {forecast_horizon} steps...")

    predictions = []
    actual_values = []
    all_predictions = []
    ensemble_confidence = []
    ensemble_direction = []

    # Need to maintain a rolling window of historical data for sequence-based models
    sequence_length = model.sequence_length

    # Initialize with historical data from the test set
    if len(test_data) < sequence_length:
        raise ValueError("Test data must be at least as long as the model's sequence length.")
    
    # The simulation starts after the first full sequence_length of data is available
    sim_start_index = sequence_length
    historical_window_df = test_data.iloc[:sim_start_index].copy()
    
    for step in range(forecast_horizon):
        
        current_step_index = sim_start_index + step
        if current_step_index >= len(test_data):
            print(f"Stopping simulation at step {step} as we have run out of test data.")
            break
            
        # Prepare the current window for prediction
        X_window, _ = model.prepare_data(historical_window_df, target_col='Close')

        # Make a prediction for the next time step
        # The 'predict' method is now robust enough to handle this
        all_preds = model.predict(X_window, return_all_predictions=True)

        # The prediction for the next step is the last value in the returned array
        current_pred = all_preds['final_prediction'][-1]
        current_confidence = all_preds['ensemble_confidence'][-1]
        current_direction = all_preds['ensemble_direction'][-1]

        # Store predictions and actuals
        predictions.append(current_pred)
        ensemble_confidence.append(current_confidence)
        ensemble_direction.append(current_direction)
        
        # Store all model predictions for detailed plotting
        all_predictions.append({
            'step': step,
            'final': current_pred,
            'level1_reg': all_preds['level1_regression'][-1],
            'level1_clf': current_confidence,
            'base_reg': {k: v[-1] for k, v in all_preds['base_regression'].items()},
            'base_clf': {k: v[-1] for k, v in all_preds['base_classification'].items()}
        })

        # Get actual value for this step
        actual = test_data.iloc[current_step_index]['Close']
        actual_values.append(actual)

        # Update the historical window for the next iteration
        # Drop the oldest data point and append the new actual one
        next_data_point = test_data.iloc[current_step_index:current_step_index+1]
        historical_window_df = pd.concat([historical_window_df.iloc[1:], next_data_point], ignore_index=True)

        # Print progress
        if (step + 1) % update_interval == 0:
            error = abs(current_pred - actual)
            accuracy = 100 * (1 - error / actual) if actual != 0 else 0
            print(f"Step {step+1}/{forecast_horizon}: "
                  f"Predicted={current_pred:.2f}, "
                  f"Actual={actual:.2f}, "
                  f"Accuracy={accuracy:.1f}%, "
                  f"Confidence={current_confidence:.3f}")

    # Calculate final metrics
    if not actual_values:
        print("\nSimulation Complete! No actual values were processed.")
        return {}

    mae = mean_absolute_error(actual_values, predictions)
    rmse = np.sqrt(mean_squared_error(actual_values, predictions))
    mape = np.mean(np.abs((np.array(actual_values) - np.array(predictions)) / np.array(actual_values))) * 100

    # Direction accuracy
    actual_directions = (np.diff(actual_values, prepend=actual_values[0]) > 0).astype(int)
    pred_directions = np.array(ensemble_direction)
    # Align lengths for comparison
    min_len = min(len(actual_directions), len(pred_directions))
    direction_accuracy = accuracy_score(actual_directions[:min_len], pred_directions[:min_len])

    avg_confidence = np.mean(ensemble_confidence)

    print(f"\nSimulation Complete!")
    print(f"MAE: ${mae:.2f}")
    print(f"RMSE: ${rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"Direction Accuracy: {direction_accuracy:.2%}")
    print(f"Average Confidence: {avg_confidence:.3f}")
    
    # Prepare complete results for plotting
    base_reg_preds = {model: [] for model in all_predictions[0]['base_reg'].keys()}
    for step_preds in all_predictions:
        for model_name, pred_val in step_preds['base_reg'].items():
            base_reg_preds[model_name].append(pred_val)

    return {
        'predictions': predictions,
        'actual_values': actual_values,
        'all_model_predictions': {
            'base_regression': base_reg_preds,
            'level1_regression': [p['level1_reg'] for p in all_predictions],
            'level1_classification': [p['level1_clf'] for p in all_predictions]
        },
        'confidence_scores': ensemble_confidence,
        'direction_predictions': ensemble_direction,
        'direction_accuracies': [(predictions[i] > predictions[i-1]) == (actual_values[i] > actual_values[i-1])
                                if i > 0 else 0.5 for i in range(len(predictions))],
        'errors': [abs(predictions[i] - actual_values[i]) for i in range(len(predictions))],
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'direction_accuracy': direction_accuracy,
        'avg_confidence': avg_confidence
    }