import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, QuantileTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, roc_auc_score
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.linear_model import Ridge, Lasso, ElasticNet
import lightgbm as lgb
import catboost as cb
from datetime import time
from tqdm import tqdm

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
tf.get_logger().setLevel('ERROR')

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (Input, LSTM, Dense, Dropout, LayerNormalization,
                                   MultiHeadAttention, Conv1D, GlobalAveragePooling1D,
                                   BatchNormalization, Concatenate, Embedding, Add, Layer,
                                   Flatten, Reshape, GlobalMaxPooling1D, Lambda, GRU,
                                   Bidirectional, TimeDistributed, Permute, RepeatVector,
                                   Activation, Multiply, Conv2D, MaxPooling1D, AveragePooling1D)
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy, Huber
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
from collections import deque
import optuna
from optuna.samplers import TPESampler


class TqdmCallback(tf.keras.callbacks.Callback):
    """Custom Keras callback for tqdm progress bars"""
    def __init__(self, epochs, desc="Training"):
        self.epochs = epochs
        self.desc = desc
        self.pbar = None
        
    def on_train_begin(self, logs=None):
        self.pbar = tqdm(total=self.epochs, desc=self.desc, position=1, leave=False)
        
    def on_epoch_end(self, epoch, logs=None):
        self.pbar.update(1)
        if logs:
            loss = logs.get('loss', 0)
            val_loss = logs.get('val_loss', 0)
            self.pbar.set_postfix({'loss': f'{loss:.4f}', 'val_loss': f'{val_loss:.4f}'})
            
    def on_train_end(self, logs=None):
        if self.pbar:
            self.pbar.close()


class TemporalConvNet(Layer):
    """Enhanced Temporal Convolutional Network with residual connections"""
    
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


class TransformerBlock(Layer):
    """Transformer block for time series"""
    
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout)
        self.ffn = Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(d_model),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        
    def call(self, inputs, training = None):
        attn_output = self.attention(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class EnhancedHierarchicalMetaLearner:
    """Enhanced Hierarchical Meta-Learning System with 3 Models (No XGBoost) and Multiple Hierarchy Levels"""
    
    def __init__(self, sequence_length=168, forecast_horizon=1):
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        
        # Base models - EXACTLY 3 MODELS (removed XGBoost)
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
        
        # Hierarchical meta-learners - EXPANDED TO 5 LEVELS
        self.meta_learners = {
            # Level 1: Task-specific combiners
            'level1_regression': None,      
            'level1_classification': None,
            
            # Level 2: Advanced task fusion
            'level2_regression_enhanced': None,
            'level2_classification_enhanced': None,
            
            # Level 3: Cross-task integration
            'level3_cross_task': None,
            
            # Level 4: Uncertainty-aware ensemble
            'level4_uncertainty': None,
            
            # Level 5: Final attention-based predictor
            'level5_final': None
        }
        
        # Scalers
        self.scalers = {
            'features': StandardScaler(),
            'features_robust': RobustScaler(),
            'features_quantile': QuantileTransformer(n_quantiles=1000, output_distribution='normal'),
            'target': StandardScaler(),
            'meta_features': StandardScaler(),
            'meta_features_l2': StandardScaler(),
            'meta_features_l3': StandardScaler()
        }
        
        # Feature names
        self.feature_names = None
        
        # Performance tracking
        self.performance_history = {}
        
        # Stacking parameters
        self.n_folds = 5
        self.use_time_series_cv = True
        
        # Expert weights
        self.expert_weights = {}
        
        # Model parameters
        self._initialize_model_parameters()
        
    def _initialize_model_parameters(self):
        """Initialize parameters for 3 models (removed XGBoost)"""
        
        # LightGBM parameters
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
        """Enhanced TCN-GRU model with Transformer components"""
        inputs = Input(shape=(sequence_length, n_features))
        
        # 1. Initial feature transformation
        x = Conv1D(filters=64, kernel_size=3, padding='causal', activation='relu')(inputs)
        x = LayerNormalization()(x)
        x = Dropout(0.2)(x)
        
        # 2. Enhanced Temporal Convolutional Network
        tcn_channels = [64, 128, 256, 128, 64]
        x_tcn = TemporalConvNet(num_channels=tcn_channels, kernel_size=3, dropout=0.2)(x)
        
        # 3. Bidirectional GRU with more units
        x_gru = Bidirectional(GRU(128, return_sequences=True, dropout=0.2))(x_tcn)
        
        # 4. Add Transformer block
        x_trans = TransformerBlock(d_model=256, num_heads=8, ff_dim=512, dropout=0.1)(x_gru)
        
        # 5. Enhanced attention mechanism
        attention = Dense(1, activation='tanh')(x_trans)
        attention = Flatten()(attention)
        attention = Activation('softmax')(attention)
        attention = RepeatVector(256)(attention)
        attention = Permute([2, 1])(attention)
        
        # Apply attention
        weighted = Multiply()([x_trans, attention])
        pooled = Lambda(lambda xin: K.sum(xin, axis=1))(weighted)
        
        # 6. Deep MLP with skip connections
        x = Dense(512, activation='relu')(pooled)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        x_skip = Dense(256)(x)
        
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        
        x = Add()([x, x_skip])
        
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        # 7. Multiple outputs
        regression_output = Dense(1, name='regression')(x)
        classification_output = Dense(1, activation='sigmoid', name='classification')(x)
        
        # Uncertainty output
        uncertainty_output = Dense(1, activation='softplus', name='uncertainty')(x)
        
        model = Model(inputs=inputs, outputs=[regression_output, classification_output, uncertainty_output])
        
        model.compile(
            optimizer=AdamW(learning_rate=0.001, weight_decay=0.01),
            loss={
                'regression': Huber(delta=1.0),
                'classification': 'binary_crossentropy',
                'uncertainty': 'mse'
            },
            metrics={
                'regression': ['mae', 'mape'],
                'classification': ['accuracy', tf.keras.metrics.AUC()],
                'uncertainty': 'mae'
            }
        )
        return model
    
    def build_transformer_lstm_model(self, n_features, sequence_length):
        """Pure Transformer + LSTM hybrid model"""
        inputs = Input(shape=(sequence_length, n_features))
        
        # Initial projection
        x = Dense(128)(inputs)
        
        # Stack of transformer blocks
        for _ in range(3):
            x = TransformerBlock(d_model=128, num_heads=8, ff_dim=256, dropout=0.1)(x)
        
        # LSTM processing
        x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))(x)
        x = Bidirectional(LSTM(64, return_sequences=False, dropout=0.2))(x)
        
        # Dense layers
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        # Outputs
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
        """Enhanced Level 1 meta-learner with deeper architecture"""
        input_dim = n_models
        
        inputs = Input(shape=(input_dim,))
        
        # Feature expansion with gating
        x = Dense(256, activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        # Gating mechanism
        gate = Dense(n_models, activation='sigmoid')(x)
        gated_input = Multiply()([inputs, gate])
        
        # Self-attention on model predictions
        x_reshaped = Reshape((n_models, 1))(gated_input)
        x_expanded = Dense(64)(x_reshaped)
        attention = MultiHeadAttention(num_heads=4, key_dim=16)(x_expanded, x_expanded)
        attention_flat = Flatten()(attention)
        
        # Combine features
        x = Concatenate()([x, attention_flat, gated_input])
        
        # Deeper processing
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        # Task-specific output
        if task_type == 'regression':
            output = Dense(1, kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(x)
            loss = Huber(delta=1.0)
            metrics = ['mae', 'mape']
        else:
            output = Dense(1, activation='sigmoid')(x)
            loss = 'binary_crossentropy'
            metrics = ['accuracy', tf.keras.metrics.AUC(name='auc')]
        
        model = Model(inputs=inputs, outputs=output)
        model.compile(optimizer=AdamW(learning_rate=0.001, weight_decay=0.01),
                     loss=loss, metrics=metrics)
        
        return model
    
    def build_hierarchical_meta_learner_level2(self, n_features_from_level1, task_type='regression'):
        """Level 2: Advanced task-specific ensemble"""
        inputs = Input(shape=(n_features_from_level1,))
        
        # Multi-path processing
        # Path 1: Direct transformation
        path1 = Dense(128, activation='relu')(inputs)
        path1 = BatchNormalization()(path1)
        path1 = Dropout(0.3)(path1)
        
        # Path 2: Attention-based
        path2 = Reshape((n_features_from_level1, 1))(inputs)
        path2 = MultiHeadAttention(num_heads=2, key_dim=32)(path2, path2)
        path2 = Flatten()(path2)
        path2 = Dense(128, activation='relu')(path2)
        path2 = BatchNormalization()(path2)
        
        # Combine paths
        combined = Concatenate()([path1, path2])
        
        # Deep processing
        x = Dense(256, activation='relu')(combined)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        # Output
        if task_type == 'regression':
            output = Dense(1)(x)
            loss = 'mse'
            metrics = ['mae']
        else:
            output = Dense(1, activation='sigmoid')(x)
            loss = 'binary_crossentropy'
            metrics = ['accuracy']
        
        model = Model(inputs=inputs, outputs=output)
        model.compile(optimizer=AdamW(learning_rate=0.001, weight_decay=0.01),
                     loss=loss, metrics=metrics)
        
        return model
    
    def build_hierarchical_meta_learner_level3(self, n_inputs):
        """Level 3: Cross-task integration"""
        inputs = Input(shape=(n_inputs,))
        
        # Separate processing for regression and classification inputs
        reg_features = Lambda(lambda x: x[:, :n_inputs//2])(inputs)
        clf_features = Lambda(lambda x: x[:, n_inputs//2:])(inputs)
        
        # Cross-attention between tasks
        reg_reshaped = Reshape((n_inputs//2, 1))(reg_features)
        clf_reshaped = Reshape((n_inputs//2, 1))(clf_features)
        
        # Cross-attention
        cross_attention_reg = MultiHeadAttention(num_heads=4, key_dim=16)(reg_reshaped, clf_reshaped)
        cross_attention_clf = MultiHeadAttention(num_heads=4, key_dim=16)(clf_reshaped, reg_reshaped)
        
        # Flatten
        cross_reg_flat = Flatten()(cross_attention_reg)
        cross_clf_flat = Flatten()(cross_attention_clf)
        
        # Combine all features
        combined = Concatenate()([inputs, cross_reg_flat, cross_clf_flat])
        
        # Deep processing
        x = Dense(512, activation='relu')(combined)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        # Output
        output = Dense(1)(x)
        
        model = Model(inputs=inputs, outputs=output)
        model.compile(optimizer=AdamW(learning_rate=0.001, weight_decay=0.01),
                     loss=Huber(delta=1.0), metrics=['mae', 'mape'])
        
        return model
    
    def build_hierarchical_meta_learner_level4(self, n_inputs):
        """Level 4: Uncertainty-aware ensemble"""
        inputs = Input(shape=(n_inputs,))
        
        # Feature processing with uncertainty modeling
        x = Dense(256, activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        # Split into mean and variance paths
        mean_path = Dense(128, activation='relu')(x)
        mean_path = BatchNormalization()(mean_path)
        mean_path = Dropout(0.2)(mean_path)
        
        var_path = Dense(128, activation='relu')(x)
        var_path = BatchNormalization()(var_path)
        var_path = Dropout(0.2)(var_path)
        
        # Combine with uncertainty weighting
        combined = Concatenate()([mean_path, var_path])
        
        x = Dense(256, activation='relu')(combined)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        # Dual outputs
        mean_output = Dense(1, name='mean')(x)
        variance_output = Dense(1, activation='softplus', name='variance')(x)
        
        model = Model(inputs=inputs, outputs=[mean_output, variance_output])
        
        def custom_loss(y_true, y_pred_mean, y_pred_var):
            """Negative log-likelihood for Gaussian"""
            return 0.5 * tf.math.log(y_pred_var) + 0.5 * tf.square(y_true - y_pred_mean) / y_pred_var
        
        model.compile(
            optimizer=AdamW(learning_rate=0.001, weight_decay=0.01),
            loss={'mean': 'mse', 'variance': 'mse'},
            metrics={'mean': ['mae', 'mape']}
        )
        
        return model
    
    def build_hierarchical_meta_learner_level5(self, n_inputs, n_original_features):
        """Level 5: Final attention-based predictor with original features"""
        # Inputs: predictions + original features
        input_dim = n_inputs + n_original_features
        
        inputs = Input(shape=(input_dim,))
        
        # Separate predictions and features
        predictions = Lambda(lambda x: x[:, :n_inputs])(inputs)
        features = Lambda(lambda x: x[:, n_inputs:])(inputs)
        
        # Process predictions with self-attention
        pred_reshaped = Reshape((n_inputs, 1))(predictions)
        pred_expanded = Dense(64)(pred_reshaped)
        
        # Multi-head self-attention
        pred_attention = MultiHeadAttention(
            num_heads=8,
            key_dim=32,
            dropout=0.1
        )(pred_expanded, pred_expanded)
        pred_attention_flat = Flatten()(pred_attention)
        
        # Process original features
        feat_processed = Dense(256, activation='relu')(features)
        feat_processed = BatchNormalization()(feat_processed)
        feat_processed = Dropout(0.3)(feat_processed)
        feat_processed = Dense(128, activation='relu')(feat_processed)
        feat_processed = BatchNormalization()(feat_processed)
        
        # Combine all information
        combined = Concatenate()([predictions, pred_attention_flat, feat_processed])
        
        # Final deep processing
        x = Dense(512, activation='relu')(combined)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        
        # Residual block
        x_res = Dense(256)(x)
        
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        
        x = Add()([x, x_res])
        
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        # Final output with confidence
        price_output = Dense(1, name='price')(x)
        confidence_output = Dense(1, activation='sigmoid', name='confidence')(x)
        
        model = Model(inputs=inputs, outputs=[price_output, confidence_output])
        
        model.compile(
            optimizer=AdamW(learning_rate=0.001, weight_decay=0.01),
            loss={'price': Huber(delta=1.0), 'confidence': 'binary_crossentropy'},
            metrics={'price': ['mae', 'mape'], 'confidence': 'accuracy'}
        )
        
        return model
    
    def prepare_data(self, data, target_col='target'):
        """Prepare data for training"""
        exclude_cols = ['target', 'datetime', 'timestamp']
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # If feature_names already set, use those exact features
        if self.feature_names is not None:
            missing_features = [f for f in self.feature_names if f not in data.columns]
            if missing_features:
                print(f"Warning: Missing features: {missing_features[:5]}...")
                # Add progress bar if many missing features
                if len(missing_features) > 100:
                    for feat in tqdm(missing_features, desc="Adding missing features", leave=False):
                        data[feat] = 0
                else:
                    for feat in missing_features:
                        data[feat] = 0
                    
            X = data[self.feature_names].values
            if target_col in data.columns:
                y = data[target_col].values
            else:
                y = data['Close'].values
        else:
            # First time - determine valid features
            print("Determining valid features...")
            valid_features = []
            
            # Add progress bar for feature validation if many features
            if len(feature_cols) > 100:
                feature_iterator = tqdm(feature_cols, desc="Validating features", leave=False)
            else:
                feature_iterator = feature_cols
                
            for col in feature_iterator:
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
        
        # Use tqdm only if there are many sequences to prepare
        n_sequences = len(X) - self.sequence_length
        if n_sequences > 1000:
            iterator = tqdm(range(n_sequences), desc="Preparing sequences", leave=False)
        else:
            iterator = range(n_sequences)
        
        for i in iterator:
            X_seq.append(X[i:(i + self.sequence_length)])
            if y is not None:
                y_seq.append(y[i + self.sequence_length])
        
        return np.array(X_seq), np.array(y_seq) if y is not None else None
    
    def train_base_models(self, X_train, y_train, X_val, y_val):
        """Train exactly 3 base models (removed XGBoost)"""
        print("\n" + "="*60)
        print("TRAINING BASE MODELS (3 MODELS)")
        print("="*60)
        
        # Create binary labels
        y_train_binary = self.create_binary_labels(y_train)
        y_val_binary = self.create_binary_labels(y_val)
        
        # Scale features
        print("\nScaling features...")
        X_train_scaled = self.scalers['features'].fit_transform(X_train)
        X_val_scaled = self.scalers['features'].transform(X_val)
        
        y_train_scaled = self.scalers['target'].fit_transform(y_train.reshape(-1, 1)).ravel()
        y_val_scaled = self.scalers['target'].transform(y_val.reshape(-1, 1)).ravel()
        
        # Progress bar for base models
        base_models_pbar = tqdm(total=3, desc="Training Base Models", position=0)
        
        # 1. Train LightGBM
        base_models_pbar.set_description("Training LightGBM")
        self.base_models['regression']['lightgbm'] = lgb.LGBMRegressor(**self.lgb_params_regression)
        self.base_models['regression']['lightgbm'].fit(
            X_train_scaled, y_train_scaled,
            eval_set=[(X_val_scaled, y_val_scaled)],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
        )
        
        self.base_models['classification']['lightgbm'] = lgb.LGBMClassifier(**self.lgb_params_classification)
        self.base_models['classification']['lightgbm'].fit(
            X_train_scaled, y_train_binary,
            eval_set=[(X_val_scaled, y_val_binary)],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
        )
        base_models_pbar.update(1)
        
        # 2. Train Enhanced TCN-GRU-Attention Model
        base_models_pbar.set_description("Training TCN-GRU-Attention")
        
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
        
        # Prepare uncertainty targets (using rolling std as proxy)
        y_train_uncertainty = pd.Series(y_train_scaled).rolling(20).std().fillna(0).values[self.sequence_length:]
        y_val_uncertainty = pd.Series(y_val_scaled).rolling(20).std().fillna(0).values[self.sequence_length:]
        epo = 100
        if len(X_train_seq) > 0:
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6),
                TqdmCallback(epo, "TCN-GRU-Attention")
            ]
            
            self.base_models['regression']['tcn_gru_attention'] = self.build_tcn_gru_attention(
                n_features=X_train.shape[1],
                sequence_length=self.sequence_length
            )
            
            self.base_models['regression']['tcn_gru_attention'].fit(
                X_train_seq,
                {
                    'regression': y_train_seq,
                    'classification': y_train_seq_binary,
                    'uncertainty': y_train_uncertainty
                },
                validation_data=(
                    X_val_seq,
                    {
                        'regression': y_val_seq,
                        'classification': y_val_seq_binary,
                        'uncertainty': y_val_uncertainty
                    }
                ),
                epochs=epo,
                batch_size=32,
                callbacks=callbacks,
                verbose=0
            )
            
            self.base_models['classification']['tcn_gru_attention'] = self.base_models['regression']['tcn_gru_attention']
        base_models_pbar.update(1)
        
        # 3. Train Transformer-LSTM Model
        base_models_pbar.set_description("Training Transformer-LSTM")
        
        if len(X_train_seq) > 0:
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6),
                TqdmCallback(epo, "Transformer-LSTM")
            ]
            
            self.base_models['regression']['transformer_lstm'] = self.build_transformer_lstm_model(
                n_features=X_train.shape[1],
                sequence_length=self.sequence_length
            )
            
            self.base_models['regression']['transformer_lstm'].fit(
                X_train_seq,
                {'regression': y_train_seq, 'classification': y_train_seq_binary},
                validation_data=(X_val_seq, {'regression': y_val_seq, 'classification': y_val_seq_binary}),
                epochs=epo,
                batch_size=32,
                callbacks=callbacks,
                verbose=0
            )
            
            self.base_models['classification']['transformer_lstm'] = self.base_models['regression']['transformer_lstm']
        base_models_pbar.update(1)
        base_models_pbar.close()
        
        print("\nAll 3 base models trained successfully!")
        
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
        
        # Tree-based models stacking (only LightGBM now)
        for model_name in ['lightgbm']:
            print(f"\nGenerating stacking predictions for {model_name}...")
            
            # Initialize arrays
            train_preds_reg[model_name] = np.zeros(n_train)
            train_preds_clf[model_name] = np.zeros(n_train)
            val_preds_reg[model_name] = []
            val_preds_clf[model_name] = []
            
            # Cross-validation with progress bar
            fold_pbar = tqdm(enumerate(cv.split(X_train_scaled)), total=self.n_folds, 
                            desc=f"CV Folds for {model_name}", position=0)
            
            for fold, (train_idx, val_idx) in fold_pbar:
                X_fold_train = X_train_scaled[train_idx]
                y_fold_train_reg = y_train_reg[train_idx]
                y_fold_train_clf = y_train_clf[train_idx]
                
                X_fold_val = X_train_scaled[val_idx]
                
                # Train fold models
                fold_reg_model = lgb.LGBMRegressor(**self.lgb_params_regression)
                fold_clf_model = lgb.LGBMClassifier(**self.lgb_params_classification)
                
                # Fit models silently
                fold_reg_model.fit(X_fold_train, y_fold_train_reg)
                fold_clf_model.fit(X_fold_train, y_fold_train_clf)
                
                # Get OOF predictions
                train_preds_reg[model_name][val_idx] = fold_reg_model.predict(X_fold_val)
                if hasattr(fold_clf_model, 'predict_proba'):
                    train_preds_clf[model_name][val_idx] = fold_clf_model.predict_proba(X_fold_val)[:, 1]
                else:
                    train_preds_clf[model_name][val_idx] = fold_clf_model.predict(X_fold_val)
                
                # Predict on validation set
                val_preds_reg[model_name].append(fold_reg_model.predict(X_val_scaled))
                if hasattr(fold_clf_model, 'predict_proba'):
                    val_preds_clf[model_name].append(fold_clf_model.predict_proba(X_val_scaled)[:, 1])
                else:
                    val_preds_clf[model_name].append(fold_clf_model.predict(X_val_scaled))
                
                fold_pbar.set_postfix({'fold': fold+1})
            
            # Average validation predictions
            val_preds_reg[model_name] = np.mean(val_preds_reg[model_name], axis=0)
            val_preds_clf[model_name] = np.mean(val_preds_clf[model_name], axis=0)
        
        # Neural network predictions (no CV, use trained models)
        print("\nGenerating neural network predictions...")
        nn_pbar = tqdm(['tcn_gru_attention', 'transformer_lstm'], desc="NN Predictions")
        
        X_train_seq, y_train_seq = self.prepare_sequences(self.scalers['features'].transform(X_train))
        X_val_seq, _ = self.prepare_sequences(X_val_scaled)
        
        for nn_model_name in nn_pbar:
            nn_pbar.set_description(f"Predicting with {nn_model_name}")
            
            if nn_model_name in self.base_models['regression']:
                nn_pred_train = self.base_models['regression'][nn_model_name].predict(X_train_seq, verbose=0)
                nn_pred_val = self.base_models['regression'][nn_model_name].predict(X_val_seq, verbose=0)
                
                # Extract predictions
                if isinstance(nn_pred_train, list):
                    train_reg = nn_pred_train[0].ravel()
                    train_clf = nn_pred_train[1].ravel()
                    val_reg = nn_pred_val[0].ravel()
                    val_clf = nn_pred_val[1].ravel()
                else:
                    train_reg = nn_pred_train['regression'].ravel()
                    train_clf = nn_pred_train['classification'].ravel()
                    val_reg = nn_pred_val['regression'].ravel()
                    val_clf = nn_pred_val['classification'].ravel()
                
                # Align predictions
                train_preds_reg[nn_model_name] = self._align_nn_predictions(train_reg, n_train)
                train_preds_clf[nn_model_name] = self._align_nn_predictions(train_clf, n_train)
                val_preds_reg[nn_model_name] = self._align_nn_predictions(val_reg, len(X_val))
                val_preds_clf[nn_model_name] = self._align_nn_predictions(val_clf, len(X_val))
        
        return train_preds_reg, train_preds_clf, val_preds_reg, val_preds_clf
    
    def _align_nn_predictions(self, predictions, target_length):
        """Align neural network predictions to target length"""
        aligned = np.zeros(target_length)
        start_idx = self.sequence_length
        pred_len = len(predictions)
        
        end_idx = min(start_idx + pred_len, target_length)
        
        if end_idx > start_idx:
            aligned[start_idx:end_idx] = predictions[:end_idx-start_idx]
        
        if pred_len > 0:
            aligned[:start_idx] = aligned[start_idx]
        
        if end_idx < target_length and pred_len > 0:
            aligned[end_idx:] = aligned[end_idx - 1]
        
        return aligned
    
    def train_hierarchical_meta_learners(self, X_train, y_train, X_val, y_val,
                                       train_preds_reg, train_preds_clf,
                                       val_preds_reg, val_preds_clf):
        """Train 5-level hierarchical meta-learners"""
        print("\n" + "="*60)
        print("TRAINING 5-LEVEL HIERARCHICAL META-LEARNERS")
        print("="*60)
        
        # Create binary labels
        y_train_binary = self.create_binary_labels(y_train)
        y_val_binary = self.create_binary_labels(y_val)
        
        # Scale targets
        y_train_scaled = self.scalers['target'].transform(y_train.reshape(-1, 1)).ravel()
        y_val_scaled = self.scalers['target'].transform(y_val.reshape(-1, 1)).ravel()
        
        # Progress bar for hierarchy levels
        hierarchy_pbar = tqdm(total=5, desc="Training Hierarchy Levels", position=0)
        
        # Level 1: Task-specific meta-learners
        hierarchy_pbar.set_description("Training Level 1 Meta-Learners")
        
        # Regression
        train_meta_reg = np.column_stack([train_preds_reg[model] for model in sorted(train_preds_reg.keys())])
        val_meta_reg = np.column_stack([val_preds_reg[model] for model in sorted(val_preds_reg.keys())])
        
        train_meta_reg_scaled = self.scalers['meta_features'].fit_transform(train_meta_reg)
        val_meta_reg_scaled = self.scalers['meta_features'].transform(val_meta_reg)
        
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
                EarlyStopping(patience=25, restore_best_weights=True),
                ReduceLROnPlateau(patience=12, factor=0.5, min_lr=1e-6),
                TqdmCallback(200, "Level 1 Regression")
            ],
            verbose=0
        )
        
        # Classification
        train_meta_clf = np.column_stack([train_preds_clf[model] for model in sorted(train_preds_clf.keys())])
        val_meta_clf = np.column_stack([val_preds_clf[model] for model in sorted(val_preds_clf.keys())])
        
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
                EarlyStopping(patience=25, restore_best_weights=True),
                ReduceLROnPlateau(patience=12, factor=0.5, min_lr=1e-6),
                TqdmCallback(200, "Level 1 Classification")
            ],
            verbose=0
        )
        
        # Get Level 1 predictions
        level1_reg_train = self.meta_learners['level1_regression'].predict(train_meta_reg_scaled, verbose=0).ravel()
        level1_reg_val = self.meta_learners['level1_regression'].predict(val_meta_reg_scaled, verbose=0).ravel()
        
        level1_clf_train = self.meta_learners['level1_classification'].predict(train_meta_clf, verbose=0).ravel()
        level1_clf_val = self.meta_learners['level1_classification'].predict(val_meta_clf, verbose=0).ravel()
        hierarchy_pbar.update(1)
        
        # Level 2: Enhanced task-specific ensemble
        hierarchy_pbar.set_description("Training Level 2 Enhanced Meta-Learners")
        
        # Combine Level 1 predictions with base predictions for enhanced ensemble
        train_l2_reg_input = np.column_stack([level1_reg_train] + [train_preds_reg[m] for m in sorted(train_preds_reg.keys())])
        val_l2_reg_input = np.column_stack([level1_reg_val] + [val_preds_reg[m] for m in sorted(val_preds_reg.keys())])
        
        train_l2_reg_scaled = self.scalers['meta_features_l2'].fit_transform(train_l2_reg_input)
        val_l2_reg_scaled = self.scalers['meta_features_l2'].transform(val_l2_reg_input)
        
        self.meta_learners['level2_regression_enhanced'] = self.build_hierarchical_meta_learner_level2(
            n_features_from_level1=train_l2_reg_input.shape[1],
            task_type='regression'
        )
        
        self.meta_learners['level2_regression_enhanced'].fit(
            train_l2_reg_scaled, y_train_scaled,
            validation_data=(val_l2_reg_scaled, y_val_scaled),
            epochs=150,
            batch_size=64,
            callbacks=[
                EarlyStopping(patience=20, restore_best_weights=True),
                ReduceLROnPlateau(patience=10, factor=0.5, min_lr=1e-6),
                TqdmCallback(150, "Level 2 Regression")
            ],
            verbose=0
        )
        
        # Get Level 2 predictions
        level2_reg_train = self.meta_learners['level2_regression_enhanced'].predict(train_l2_reg_scaled, verbose=0).ravel()
        level2_reg_val = self.meta_learners['level2_regression_enhanced'].predict(val_l2_reg_scaled, verbose=0).ravel()
        hierarchy_pbar.update(1)
        
        # Level 3: Cross-task integration
        hierarchy_pbar.set_description("Training Level 3 Cross-Task Integration")
        
        # Combine Level 2 regression and Level 1 classification
        train_l3_input = np.column_stack([level2_reg_train, level1_clf_train])
        val_l3_input = np.column_stack([level2_reg_val, level1_clf_val])
        
        self.meta_learners['level3_cross_task'] = self.build_hierarchical_meta_learner_level3(
            n_inputs=train_l3_input.shape[1]
        )
        
        self.meta_learners['level3_cross_task'].fit(
            train_l3_input, y_train_scaled,
            validation_data=(val_l3_input, y_val_scaled),
            epochs=150,
            batch_size=64,
            callbacks=[
                EarlyStopping(patience=20, restore_best_weights=True),
                ReduceLROnPlateau(patience=10, factor=0.5, min_lr=1e-6),
                TqdmCallback(150, "Level 3 Cross-Task")
            ],
            verbose=0
        )
        
        # Get Level 3 predictions
        level3_train = self.meta_learners['level3_cross_task'].predict(train_l3_input, verbose=0).ravel()
        level3_val = self.meta_learners['level3_cross_task'].predict(val_l3_input, verbose=0).ravel()
        hierarchy_pbar.update(1)
        
        # Level 4: Uncertainty-aware ensemble
        hierarchy_pbar.set_description("Training Level 4 Uncertainty-Aware Ensemble")
        
        # Combine all previous levels
        train_l4_input = np.column_stack([level1_reg_train, level2_reg_train, level3_train, level1_clf_train])
        val_l4_input = np.column_stack([level1_reg_val, level2_reg_val, level3_val, level1_clf_val])
        
        train_l4_scaled = self.scalers['meta_features_l3'].fit_transform(train_l4_input)
        val_l4_scaled = self.scalers['meta_features_l3'].transform(val_l4_input)
        
        self.meta_learners['level4_uncertainty'] = self.build_hierarchical_meta_learner_level4(
            n_inputs=train_l4_scaled.shape[1]
        )
        
        self.meta_learners['level4_uncertainty'].fit(
            train_l4_scaled,
            {'mean': y_train_scaled, 'variance': np.abs(y_train_scaled - level3_train)},
            validation_data=(val_l4_scaled, {'mean': y_val_scaled, 'variance': np.abs(y_val_scaled - level3_val)}),
            epochs=150,
            batch_size=64,
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.5, min_lr=1e-6),
                TqdmCallback(150, "Level 4 Uncertainty")
            ],
            verbose=0
        )
        
        # Get Level 4 predictions
        level4_pred = self.meta_learners['level4_uncertainty'].predict(train_l4_scaled, verbose=0)
        if isinstance(level4_pred, list):
            level4_train = level4_pred[0].ravel()
        else:
            level4_train = level4_pred['mean'].ravel()
        
        level4_pred_val = self.meta_learners['level4_uncertainty'].predict(val_l4_scaled, verbose=0)
        if isinstance(level4_pred_val, list):
            level4_val = level4_pred_val[0].ravel()
        else:
            level4_val = level4_pred_val['mean'].ravel()
        hierarchy_pbar.update(1)
        
        # Level 5: Final attention-based predictor
        hierarchy_pbar.set_description("Training Level 5 Final Attention Ensemble")
        
        # Combine all levels with original features
        X_train_scaled = self.scalers['features'].transform(X_train)
        X_val_scaled = self.scalers['features'].transform(X_val)
        
        train_final_input = np.column_stack([
            level1_reg_train, level2_reg_train, level3_train, level4_train,
            level1_clf_train, X_train_scaled
        ])
        val_final_input = np.column_stack([
            level1_reg_val, level2_reg_val, level3_val, level4_val,
            level1_clf_val, X_val_scaled
        ])
        
        # Create confidence labels
        train_confidence = 1 - np.abs(y_train_scaled - level4_train) / (np.abs(y_train_scaled) + 1e-6)
        val_confidence = 1 - np.abs(y_val_scaled - level4_val) / (np.abs(y_val_scaled) + 1e-6)
        train_confidence = np.clip(train_confidence, 0, 1)
        val_confidence = np.clip(val_confidence, 0, 1)
        
        self.meta_learners['level5_final'] = self.build_hierarchical_meta_learner_level5(
            n_inputs=5,  # 5 prediction inputs
            n_original_features=X_train.shape[1]
        )
        
        self.meta_learners['level5_final'].fit(
            train_final_input,
            {'price': y_train_scaled, 'confidence': train_confidence},
            validation_data=(val_final_input, {'price': y_val_scaled, 'confidence': val_confidence}),
            epochs=200,
            batch_size=64,
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', patience=12, factor=0.5, min_lr=1e-6),
                TqdmCallback(200, "Level 5 Final")
            ],
            verbose=0
        )
        hierarchy_pbar.update(1)
        hierarchy_pbar.close()
        
        # Evaluate performance
        print("\nEvaluating hierarchical performance...")
        final_pred = self.meta_learners['level5_final'].predict(val_final_input, verbose=0)
        if isinstance(final_pred, list):
            final_price = final_pred[0].ravel()
        else:
            final_price = final_pred['price'].ravel()
        
        final_mae = mean_absolute_error(y_val_scaled, final_price)
        
        print(f"\nHierarchical Meta-Learner Performance:")
        print(f"  Level 1 Regression MAE: {mean_absolute_error(y_val_scaled, level1_reg_val):.4f}")
        print(f"  Level 2 Enhanced MAE: {mean_absolute_error(y_val_scaled, level2_reg_val):.4f}")
        print(f"  Level 3 Cross-Task MAE: {mean_absolute_error(y_val_scaled, level3_val):.4f}")
        print(f"  Level 4 Uncertainty MAE: {mean_absolute_error(y_val_scaled, level4_val):.4f}")
        print(f"  Level 5 Final MAE: {final_mae:.4f}")
        
        # Calculate model weights
        self._calculate_model_weights(val_preds_reg, val_preds_clf, y_val_scaled, y_val_binary)
    
    def _calculate_model_weights(self, val_preds_reg, val_preds_clf, y_val_scaled, y_val_binary):
        """Calculate model weights based on performance"""
        
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
        
        print("\nModel Weights:")
        print("\nRegression:")
        for name, weight in sorted(self.model_weights['regression'].items()):
            print(f"  {name}: {weight:.3f}")
        print("\nClassification:")
        for name, weight in sorted(self.model_weights['classification'].items()):
            print(f"  {name}: {weight:.3f}")
    
    def train(self, X_train, y_train, X_val, y_val):
        """Main training method"""
        print("\n" + "="*60)
        print("STARTING ENHANCED HIERARCHICAL META-LEARNING TRAINING")
        print("="*60)
        
        # Overall training progress
        main_pbar = tqdm(total=3, desc="Overall Training Progress", position=0)
        
        # Train base models
        main_pbar.set_description("Phase 1: Training Base Models")
        self.train_base_models(X_train, y_train, X_val, y_val)
        main_pbar.update(1)
        
        # Get stacking predictions
        main_pbar.set_description("Phase 2: Generating Stacking Predictions")
        y_train_binary = self.create_binary_labels(y_train)
        y_train_scaled = self.scalers['target'].transform(y_train.reshape(-1, 1)).ravel()
        
        train_preds_reg, train_preds_clf, val_preds_reg, val_preds_clf = self.get_stacking_predictions(
            X_train, y_train_scaled, y_train_binary, X_val
        )
        main_pbar.update(1)
        
        # Train hierarchical meta-learners
        main_pbar.set_description("Phase 3: Training Hierarchical Meta-Learners")
        self.train_hierarchical_meta_learners(
            X_train, y_train, X_val, y_val,
            train_preds_reg, train_preds_clf,
            val_preds_reg, val_preds_clf
        )
        main_pbar.update(1)
        main_pbar.close()
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
    
    def predict(self, X_test, return_all_predictions=False):
        """Make predictions using the 5-level hierarchical ensemble"""
        
        # Get predictions from all base models
        base_preds_reg = {}
        base_preds_clf = {}
        
        # Scaled version
        X_test_scaled = self.scalers['features'].transform(X_test)
        
        # Tree-based models (only LightGBM now)
        for model_name in ['lightgbm']:
            if model_name in self.base_models['regression']:
                base_preds_reg[model_name] = self.base_models['regression'][model_name].predict(X_test_scaled)
                base_preds_clf[model_name] = self.base_models['classification'][model_name].predict_proba(X_test_scaled)[:, 1]
        
        # Neural network models
        X_test_seq = None
        is_simulation_step = (X_test_scaled.shape[0] == self.sequence_length)

        if X_test_scaled.shape[0] >= self.sequence_length:
            if is_simulation_step:
                # Handle single window for one prediction (simulation case)
                X_test_seq = X_test_scaled.reshape(1, self.sequence_length, -1)
            else:
                # Handle batch of data for multiple predictions (validation/test case)
                X_test_seq, _ = self.prepare_sequences(X_test_scaled)

        for nn_model_name in ['tcn_gru_attention', 'transformer_lstm']:
            if nn_model_name in self.base_models['regression']:
                if X_test_seq is not None and X_test_seq.shape[0] > 0:
                    nn_pred = self.base_models['regression'][nn_model_name].predict(X_test_seq, verbose=0)
                    
                    if isinstance(nn_pred, list):
                        reg_pred_seq = nn_pred[0].ravel()
                        clf_pred_seq = nn_pred[1].ravel()
                    else:
                        reg_pred_seq = nn_pred['regression'].ravel()
                        clf_pred_seq = nn_pred['classification'].ravel()

                    if is_simulation_step:
                        # For simulation, we have one prediction. Pad it to match X_test length.
                        # The simulation loop only uses the last element, but meta-learners need a full array.
                        # We fill the array with the single valid prediction.
                        base_preds_reg[nn_model_name] = np.full(len(X_test_scaled), reg_pred_seq[0])
                        base_preds_clf[nn_model_name] = np.full(len(X_test_scaled), clf_pred_seq[0])
                    else:
                        # For batch prediction, align the sequences
                        base_preds_reg[nn_model_name] = self._align_nn_predictions(reg_pred_seq, len(X_test_scaled))
                        base_preds_clf[nn_model_name] = self._align_nn_predictions(clf_pred_seq, len(X_test_scaled))
                else:
                    # If no sequences could be formed, fill with zeros as a fallback.
                    base_preds_reg[nn_model_name] = np.zeros(len(X_test_scaled))
                    base_preds_clf[nn_model_name] = np.zeros(len(X_test_scaled))

        # Verify we have all 3 models
        expected_models = ['lightgbm', 'tcn_gru_attention', 'transformer_lstm']
        missing_models = [m for m in expected_models if m not in base_preds_reg]
        if missing_models:
            print(f"Warning: Missing predictions from models: {missing_models}")
            # Fill with zeros for missing models
            for model in missing_models:
                base_preds_reg[model] = np.zeros(len(X_test_scaled))
                base_preds_clf[model] = np.zeros(len(X_test_scaled))
        
        # Level 1 predictions
        meta_reg = np.column_stack([base_preds_reg[model] for model in sorted(base_preds_reg.keys())])
        meta_reg_scaled = self.scalers['meta_features'].transform(meta_reg)
        level1_reg = self.meta_learners['level1_regression'].predict(meta_reg_scaled, verbose=0).ravel()
        
        meta_clf = np.column_stack([base_preds_clf[model] for model in sorted(base_preds_clf.keys())])
        level1_clf = self.meta_learners['level1_classification'].predict(meta_clf, verbose=0).ravel()
        
        # Level 2 predictions
        l2_reg_input = np.column_stack([level1_reg] + [base_preds_reg[m] for m in sorted(base_preds_reg.keys())])
        l2_reg_scaled = self.scalers['meta_features_l2'].transform(l2_reg_input)
        level2_reg = self.meta_learners['level2_regression_enhanced'].predict(l2_reg_scaled, verbose=0).ravel()
        
        # Level 3 predictions
        l3_input = np.column_stack([level2_reg, level1_clf])
        level3 = self.meta_learners['level3_cross_task'].predict(l3_input, verbose=0).ravel()
        
        # Level 4 predictions
        l4_input = np.column_stack([level1_reg, level2_reg, level3, level1_clf])
        l4_scaled = self.scalers['meta_features_l3'].transform(l4_input)
        level4_pred = self.meta_learners['level4_uncertainty'].predict(l4_scaled, verbose=0)
        
        if isinstance(level4_pred, list):
            level4 = level4_pred[0].ravel()
            level4_uncertainty = level4_pred[1].ravel()
        else:
            level4 = level4_pred['mean'].ravel()
            level4_uncertainty = level4_pred['variance'].ravel()
        
        # Level 5 final prediction
        final_input = np.column_stack([
            level1_reg, level2_reg, level3, level4,
            level1_clf, X_test_scaled
        ])
        
        final_pred = self.meta_learners['level5_final'].predict(final_input, verbose=0)
        
        if isinstance(final_pred, list):
            final_price_scaled = final_pred[0].ravel()
            final_confidence = final_pred[1].ravel()
        else:
            final_price_scaled = final_pred['price'].ravel()
            final_confidence = final_pred['confidence'].ravel()
        
        # Inverse transform
        final_price = self.scalers['target'].inverse_transform(final_price_scaled.reshape(-1, 1)).ravel()
        
        # Also inverse transform other predictions
        for model_name in base_preds_reg:
            base_preds_reg[model_name] = self.scalers['target'].inverse_transform(
                base_preds_reg[model_name].reshape(-1, 1)
            ).ravel()
        
        level1_reg = self.scalers['target'].inverse_transform(level1_reg.reshape(-1, 1)).ravel()
        level2_reg = self.scalers['target'].inverse_transform(level2_reg.reshape(-1, 1)).ravel()
        level3 = self.scalers['target'].inverse_transform(level3.reshape(-1, 1)).ravel()
        level4 = self.scalers['target'].inverse_transform(level4.reshape(-1, 1)).ravel()
        
        if return_all_predictions:
            return {
                'final_prediction': final_price,
                'confidence': final_confidence,
                'uncertainty': level4_uncertainty,
                'level1_regression': level1_reg,
                'level1_classification': level1_clf,
                'level2_regression': level2_reg,
                'level3_cross_task': level3,
                'level4_uncertainty': level4,
                'base_regression': base_preds_reg,
                'base_classification': base_preds_clf,
                'ensemble_confidence': final_confidence,
                'ensemble_direction': (level1_clf > 0.5).astype(int)
            }
        else:
            return final_price


# Keep the original simulate function name
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
    hierarchy_predictions = {
        'level1': [],
        'level2': [],
        'level3': [],
        'level4': [],
        'level5': []
    }
    
    sequence_length = model.sequence_length
    
    if len(test_data) < sequence_length:
        raise ValueError("Test data must be at least as long as the model's sequence length.")
    
    sim_start_index = sequence_length
    historical_window_df = test_data.iloc[:sim_start_index].copy()
    
    # Progress bar for simulation
    sim_pbar = tqdm(range(forecast_horizon), desc="Simulating forecast", position=0)
    
    for step in sim_pbar:
        current_step_index = sim_start_index + step
        if current_step_index >= len(test_data):
            print(f"\nStopping simulation at step {step} as we have run out of test data.")
            break
        
        # Prepare the current window for prediction
        X_window, _ = model.prepare_data(historical_window_df, target_col='Close')
        
        # Make a prediction for the next time step
        all_preds = model.predict(X_window, return_all_predictions=True)
        
        # The prediction for the next step is the last value in the returned array
        current_pred = all_preds['final_prediction'][-1]
        current_confidence = all_preds['ensemble_confidence'][-1]
        current_direction = all_preds['ensemble_direction'][-1]
        
        # Store predictions
        predictions.append(current_pred)
        ensemble_confidence.append(current_confidence)
        ensemble_direction.append(current_direction)
        
        # Store hierarchy predictions
        hierarchy_predictions['level1'].append(all_preds['level1_regression'][-1])
        hierarchy_predictions['level2'].append(all_preds['level2_regression'][-1])
        hierarchy_predictions['level3'].append(all_preds['level3_cross_task'][-1])
        hierarchy_predictions['level4'].append(all_preds['level4_uncertainty'][-1])
        hierarchy_predictions['level5'].append(current_pred)
        
        # Store all model predictions
        all_predictions.append({
            'step': step,
            'final': current_pred,
            'confidence': current_confidence,
            'uncertainty': all_preds['uncertainty'][-1],
            'level1_reg': all_preds['level1_regression'][-1],
            'level1_clf': all_preds['level1_classification'][-1],
            'base_reg': {k: v[-1] for k, v in all_preds['base_regression'].items()},
            'base_clf': {k: v[-1] for k, v in all_preds['base_classification'].items()}
        })
        
        # Get actual value for this step
        actual = test_data.iloc[current_step_index]['Close']
        actual_values.append(actual)
        
        # Update the historical window
        next_data_point = test_data.iloc[current_step_index:current_step_index+1]
        historical_window_df = pd.concat([historical_window_df.iloc[1:], next_data_point], ignore_index=True)
        
        # Update progress bar
        error = abs(current_pred - actual)
        accuracy = 100 * (1 - error / actual) if actual != 0 else 0
        sim_pbar.set_postfix({
            'Pred': f'{current_pred:.2f}',
            'Actual': f'{actual:.2f}',
            'Acc': f'{accuracy:.1f}%',
            'Conf': f'{current_confidence:.3f}'
        })
        
        # Print detailed progress at intervals
        if (step + 1) % update_interval == 0:
            tqdm.write(f"Step {step+1}/{forecast_horizon}: "
                      f"Predicted={current_pred:.2f}, "
                      f"Actual={actual:.2f}, "
                      f"Accuracy={accuracy:.1f}%, "
                      f"Confidence={current_confidence:.3f}")
    
    sim_pbar.close()
    
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
    min_len = min(len(actual_directions), len(pred_directions))
    direction_accuracy = accuracy_score(actual_directions[:min_len], pred_directions[:min_len])
    
    avg_confidence = np.mean(ensemble_confidence)
    
    print(f"\nSimulation Complete!")
    print(f"MAE: ${mae:.2f}")
    print(f"RMSE: ${rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"Direction Accuracy: {direction_accuracy:.2%}")
    print(f"Average Confidence: {avg_confidence:.3f}")
    
    # Prepare base model predictions
    base_reg_preds = {model: [] for model in all_predictions[0]['base_reg'].keys()}
    for step_preds in all_predictions:
        for model_name, pred_val in step_preds['base_reg'].items():
            base_reg_preds[model_name].append(pred_val)
    
    return {
        'predictions': predictions,
        'actual_values': actual_values,
        'all_model_predictions': {
            'base_regression': base_reg_preds,
            'level1_regression': hierarchy_predictions['level1'],
            'level2_regression': hierarchy_predictions['level2'],
            'level3_cross_task': hierarchy_predictions['level3'],
            'level4_uncertainty': hierarchy_predictions['level4'],
            'level5_final': hierarchy_predictions['level5']
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


# Register custom layers
tf.keras.utils.get_custom_objects().update({
    'TemporalConvNet': TemporalConvNet,
    'TransformerBlock': TransformerBlock
})