import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, roc_auc_score
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, LSTM, Dense, Dropout, LayerNormalization, 
                                   MultiHeadAttention, Conv1D, GlobalAveragePooling1D,
                                   BatchNormalization, Concatenate, Embedding, Add, Layer,
                                   Flatten, Reshape, GlobalMaxPooling1D, Lambda)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
from collections import deque
import optuna
from optuna.samplers import TPESampler
import ccxt
import re


class ResNetBlock1D(Layer):
    """1D ResNet block for feature extraction"""
    
    def __init__(self, filters, kernel_size=3, stride=1, **kwargs):
        super(ResNetBlock1D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        
    def build(self, input_shape):
        # Main path
        self.conv1 = Conv1D(self.filters, self.kernel_size, 
                           strides=self.stride, padding='same', activation=None)
        self.bn1 = BatchNormalization()
        self.conv2 = Conv1D(self.filters, self.kernel_size, 
                           strides=1, padding='same', activation=None)
        self.bn2 = BatchNormalization()
        
        # Shortcut path
        if self.stride != 1 or input_shape[-1] != self.filters:
            self.shortcut = Conv1D(self.filters, 1, strides=self.stride, padding='same')
            self.shortcut_bn = BatchNormalization()
        else:
            self.shortcut = None
            
    def call(self, inputs, training=None):
        # Main path
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        
        # Shortcut
        if self.shortcut is not None:
            shortcut = self.shortcut(inputs)
            shortcut = self.shortcut_bn(shortcut, training=training)
        else:
            shortcut = inputs
            
        # Add and activate
        x = Add()([x, shortcut])
        x = tf.nn.relu(x)
        
        return x


class TemporalSelfAttention(Layer):
    """Temporal self-attention layer"""
    
    def __init__(self, d_model, num_heads=4, **kwargs):
        super(TemporalSelfAttention, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
    def build(self, input_shape):
        # Q, K, V projections
        self.wq = Dense(self.d_model, use_bias=False)
        self.wk = Dense(self.d_model, use_bias=False)
        self.wv = Dense(self.d_model, use_bias=False)
        
        # Output projection
        self.dense = Dense(self.d_model)
        
        # Layer norm
        self.layernorm = LayerNormalization(epsilon=1e-6)
        
    def call(self, inputs, training=None):
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        
        # Linear projections in batch from d_model => h x d_k
        Q = self.wq(inputs)  # [batch_size, seq_len, d_model]
        K = self.wk(inputs)
        V = self.wv(inputs)
        
        # Reshape for multi-head attention
        Q = tf.reshape(Q, (batch_size, seq_len, self.num_heads, self.d_k))
        K = tf.reshape(K, (batch_size, seq_len, self.num_heads, self.d_k))
        V = tf.reshape(V, (batch_size, seq_len, self.num_heads, self.d_k))
        
        # Transpose for attention computation
        Q = tf.transpose(Q, perm=[0, 2, 1, 3])  # [batch_size, num_heads, seq_len, d_k]
        K = tf.transpose(K, perm=[0, 2, 1, 3])
        V = tf.transpose(V, perm=[0, 2, 1, 3])
        
        # Attention
        matmul_qk = tf.matmul(Q, K, transpose_b=True)  # [batch_size, num_heads, seq_len, seq_len]
        
        # Scale
        dk = tf.cast(self.d_k, tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        # Softmax
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        
        # Apply attention to values
        attention_output = tf.matmul(attention_weights, V)  # [batch_size, num_heads, seq_len, d_k]
        
        # Transpose back
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        
        # Concatenate heads
        concat_attention = tf.reshape(attention_output, 
                                    (batch_size, seq_len, self.d_model))
        
        # Final linear layer
        output = self.dense(concat_attention)
        
        # Add & Norm
        output = self.layernorm(inputs + output)
        
        return output


class TransformerBlock(Layer):
    """Transformer encoder block"""
    
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        
    def build(self, input_shape):
        # Multi-head attention
        self.att = MultiHeadAttention(
            num_heads=self.num_heads, 
            key_dim=self.embed_dim // self.num_heads,
            dropout=self.dropout_rate
        )
        
        # Feed forward network
        self.ffn = tf.keras.Sequential([
            Dense(self.ff_dim, activation="relu"),
            Dense(self.embed_dim),
        ])
        
        # Layer normalization
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        
        # Dropout
        self.dropout1 = Dropout(self.dropout_rate)
        self.dropout2 = Dropout(self.dropout_rate)
        
    def call(self, inputs, training=None):
        # Multi-head attention
        attn_output = self.att(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # Feed forward
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2


class ForecasterDualModel:
    """Forecaster with LightGBM and HAELT for both regression and classification"""
    
    def __init__(self, sequence_length=168, forecast_horizon=72):
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        
        # Models
        self.models = {}
        self.model_scores = {}
        self.model_weights = {'lightgbm': 0.5, 'haelt': 0.5}
        
        # Expert weights for HAELT internal ensemble
        self.expert_weights = {'lstm_path': 0.5, 'transformer_path': 0.5}
        self.expert_losses = {'lstm_path': deque(maxlen=100), 
                            'transformer_path': deque(maxlen=100)}
        
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
            'haelt': deque(maxlen=100)
        }
        self.weight_history = []
        
        # Online learning parameters
        self.online_learning_rate = 0.01
        self.weight_momentum = 0.9
        self.min_weight = 0.1
        
        # HAELT hyperparameters
        self.haelt_params = {
            'resnet_filters': [64, 128, 256],
            'd_model': 128,
            'attention_heads': 4,
            'lstm_units': [128, 64, 32],
            'transformer_embed_dim': 64,
            'transformer_heads': 4,
            'transformer_ff_dim': 128,
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'temperature': 1.0,  # For softmax weight computation
            'window_k': 20  # Window size for rolling validation loss
        }
        
        # LightGBM parameters for regression
        self.lgb_params_regression = {
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
        
        # LightGBM parameters for classification
        self.lgb_params_classification = self.lgb_params_regression.copy()
        self.lgb_params_classification.update({
            'objective': 'binary',
            'metric': 'binary_logloss'
        })

    def build_haelt(self, n_features, sequence_length):
        """
        Build HAELT model with dual outputs for both regression and classification
        """
        inputs = Input(shape=(sequence_length, n_features), name='input_features')
        
        # 1. ResNet-based Feature Extraction
        x = inputs
        for i, filters in enumerate(self.haelt_params['resnet_filters']):
            x = ResNetBlock1D(filters, kernel_size=3, stride=1 if i == 0 else 2, 
                            name=f'resnet_block_{i}')(x)
            x = Dropout(self.haelt_params['dropout_rate'])(x)
        
        # Project to d_model dimensions
        x = Conv1D(self.haelt_params['d_model'], 1, padding='same', name='projection')(x)
        x = LayerNormalization(name='projection_norm')(x)
        
        # 2. Temporal Self-Attention
        attention_output = TemporalSelfAttention(
            d_model=self.haelt_params['d_model'],
            num_heads=self.haelt_params['attention_heads'],
            name='temporal_attention'
        )(x)
        
        # 3. Parallel Branches
        # LSTM Branch
        lstm_branch = attention_output
        for i, units in enumerate(self.haelt_params['lstm_units']):
            return_sequences = i < len(self.haelt_params['lstm_units']) - 1
            lstm_branch = LSTM(
                units=units,
                return_sequences=return_sequences,
                dropout=self.haelt_params['dropout_rate'],
                recurrent_dropout=self.haelt_params['dropout_rate'],
                name=f'lstm_{i}'
            )(lstm_branch)
            if return_sequences:
                lstm_branch = LayerNormalization(name=f'lstm_norm_{i}')(lstm_branch)
        
        # LSTM expert outputs (both regression and classification)
        lstm_expert_regression = Dense(1, name='lstm_expert_regression')(lstm_branch)
        lstm_expert_classification = Dense(1, activation='sigmoid', name='lstm_expert_classification')(lstm_branch)
        
        # Transformer Branch
        transformer_branch = attention_output
        
        # Project to transformer dimensions if needed
        if self.haelt_params['d_model'] != self.haelt_params['transformer_embed_dim']:
            transformer_branch = Dense(self.haelt_params['transformer_embed_dim'], 
                                     name='transformer_projection')(transformer_branch)
        
        # Apply transformer blocks
        transformer_branch = TransformerBlock(
            embed_dim=self.haelt_params['transformer_embed_dim'],
            num_heads=self.haelt_params['transformer_heads'],
            ff_dim=self.haelt_params['transformer_ff_dim'],
            dropout_rate=self.haelt_params['dropout_rate'],
            name='transformer_block'
        )(transformer_branch)
        
        # Global pooling for transformer output
        transformer_branch = GlobalAveragePooling1D(name='transformer_pooling')(transformer_branch)
        
        # Transformer expert outputs (both regression and classification)
        transformer_expert_regression = Dense(1, name='transformer_expert_regression')(transformer_branch)
        transformer_expert_classification = Dense(1, activation='sigmoid', 
                                                name='transformer_expert_classification')(transformer_branch)
        
        # 4. Concatenate both branches for joint processing
        combined = Concatenate(name='branch_concat')([lstm_branch, transformer_branch])
        
        # 5. Shared representation for both tasks
        shared = Dense(128, activation='relu', name='shared_1')(combined)
        shared = BatchNormalization(name='shared_bn_1')(shared)
        shared = Dropout(self.haelt_params['dropout_rate'])(shared)
        
        shared = Dense(64, activation='relu', name='shared_2')(shared)
        shared = BatchNormalization(name='shared_bn_2')(shared)
        shared = Dropout(self.haelt_params['dropout_rate'] / 2)(shared)
        
        # 6. Task-specific heads
        # Regression head
        regression_head = Dense(32, activation='relu', name='regression_head_1')(shared)
        regression_head = Dense(16, activation='relu', name='regression_head_2')(regression_head)
        final_regression = Dense(1, name='final_regression')(regression_head)
        
        # Classification head
        classification_head = Dense(32, activation='relu', name='classification_head_1')(shared)
        classification_head = Dense(16, activation='relu', name='classification_head_2')(classification_head)
        final_classification = Dense(1, activation='sigmoid', name='final_classification')(classification_head)
        
        # Create model with multiple outputs
        model = Model(
            inputs=inputs,
            outputs={
                # Main outputs
                'final_regression': final_regression,
                'final_classification': final_classification,
                # Expert outputs for regression
                'lstm_expert_regression': lstm_expert_regression,
                'transformer_expert_regression': transformer_expert_regression,
                # Expert outputs for classification
                'lstm_expert_classification': lstm_expert_classification,
                'transformer_expert_classification': transformer_expert_classification
            },
            name='HAELT'
        )
        
        # Compile with both losses
        model.compile(
            optimizer=Adam(learning_rate=self.haelt_params['learning_rate']),
            loss={
                # Regression losses
                'final_regression': 'mse',
                'lstm_expert_regression': 'mse',
                'transformer_expert_regression': 'mse',
                # Classification losses
                'final_classification': 'binary_crossentropy',
                'lstm_expert_classification': 'binary_crossentropy',
                'transformer_expert_classification': 'binary_crossentropy'
            },
            loss_weights={
                # Main task weights
                'final_regression': 1.0,
                'final_classification': 1.0,
                # Expert auxiliary losses
                'lstm_expert_regression': 0.2,
                'transformer_expert_regression': 0.2,
                'lstm_expert_classification': 0.2,
                'transformer_expert_classification': 0.2
            },
            metrics={
                'final_regression': ['mae', 'mape'],
                'final_classification': ['accuracy', tf.keras.metrics.AUC(name='auc')],
                'lstm_expert_regression': ['mae'],
                'transformer_expert_regression': ['mae'],
                'lstm_expert_classification': ['accuracy'],
                'transformer_expert_classification': ['accuracy']
            }
        )
        
        return model

    def create_binary_labels(self, y):
        """Convert continuous price targets to binary up/down labels"""
        # Calculate returns
        returns = np.diff(y, prepend=y[0])
        # Binary labels: 1 if price goes up, 0 if down
        binary_labels = (returns > 0).astype(int)
        return binary_labels

    def update_expert_weights(self, lstm_loss_reg, transformer_loss_reg, lstm_loss_clf, transformer_loss_clf):
        """Update expert weights based on combined performance"""
        # Combine regression and classification losses
        lstm_combined_loss = 0.5 * lstm_loss_reg + 0.5 * lstm_loss_clf
        transformer_combined_loss = 0.5 * transformer_loss_reg + 0.5 * transformer_loss_clf
        
        # Add to history
        self.expert_losses['lstm_path'].append(lstm_combined_loss)
        self.expert_losses['transformer_path'].append(transformer_combined_loss)
        
        # Calculate average losses over window
        if len(self.expert_losses['lstm_path']) >= self.haelt_params['window_k']:
            lstm_avg_loss = np.mean(list(self.expert_losses['lstm_path'])[-self.haelt_params['window_k']:])
            transformer_avg_loss = np.mean(list(self.expert_losses['transformer_path'])[-self.haelt_params['window_k']:])
            
            # Compute softmax weights
            tau = self.haelt_params['temperature']
            exp_lstm = np.exp(-lstm_avg_loss / tau)
            exp_transformer = np.exp(-transformer_avg_loss / tau)
            
            total_exp = exp_lstm + exp_transformer
            
            self.expert_weights['lstm_path'] = exp_lstm / total_exp
            self.expert_weights['transformer_path'] = exp_transformer / total_exp

    def prepare_data(self, data, target_col='target'):
        """Prepare data for training."""
        exclude_cols = ['target', 'datetime', 'timestamp']
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        valid_features = []
        for col in feature_cols:
            if data[col].nunique() > 1 and data[col].notna().sum() > len(data) * 0.5:
                valid_features.append(col)
        
        # IMPORTANT: Ensure 'Close' is the first column for consistency
        if 'Close' in valid_features:
            valid_features.remove('Close')
            valid_features = ['Close'] + valid_features
        
        X = data[valid_features].values
        y = data[target_col].values
        
        self.feature_names = valid_features
        print(f"Prepared data shape: X={X.shape}, y={y.shape}")
        print(f"Number of features: {len(valid_features)}")
        print(f"First feature: {valid_features[0]}")
        
        return X, y

    def prepare_sequences(self, X, y=None):
        """Prepare sequences for HAELT."""
        X_seq = []
        y_seq = []
        
        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:(i + self.sequence_length)])
            if y is not None:
                y_seq.append(y[i + self.sequence_length])
        
        return np.array(X_seq), np.array(y_seq) if y is not None else None

    def train_models(self, X_train, y_train, X_val, y_val, optimize_params=False):
        """Train LightGBM (both regression and classification) and HAELT models."""
        print("\n" + "="*60)
        print("TRAINING MODELS (Dual Task: Regression + Classification)")
        print("="*60)
        
        # Create binary labels
        y_train_binary = self.create_binary_labels(y_train)
        y_val_binary = self.create_binary_labels(y_val)
        
        # Scale features
        X_train_scaled = self.scalers['features'].fit_transform(X_train)
        y_train_scaled = self.scalers['target'].fit_transform(y_train.reshape(-1, 1)).ravel()
        X_val_scaled = self.scalers['features'].transform(X_val)
        y_val_scaled = self.scalers['target'].transform(y_val.reshape(-1, 1)).ravel()
        
        # 1. Train LightGBM Regression
        print("\n1. Training LightGBM Regressor...")
        self.models['lightgbm_regression'] = lgb.LGBMRegressor(**self.lgb_params_regression)
        self.models['lightgbm_regression'].fit(
            X_train_scaled, y_train_scaled,
            eval_set=[(X_val_scaled, y_val_scaled)],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)]
        )
        
        # 2. Train LightGBM Classification
        print("\n2. Training LightGBM Classifier...")
        self.models['lightgbm_classification'] = lgb.LGBMClassifier(**self.lgb_params_classification)
        self.models['lightgbm_classification'].fit(
            X_train_scaled, y_train_binary,
            eval_set=[(X_val_scaled, y_val_binary)],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)]
        )
        
        # Feature importance (from regression model)
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.models['lightgbm_regression'].feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 15 Most Important Features (LightGBM):")
        print(feature_importance.head(15).to_string(index=False))
        
        # 3. Train HAELT
        print("\n3. Training HAELT (Multi-task)...")
        
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
            # Build model
            self.models['haelt'] = self.build_haelt(
                n_features=X_train.shape[1],
                sequence_length=self.sequence_length
            )
            
            # Callbacks with expert loss tracking
            class ExpertLossCallback(tf.keras.callbacks.Callback):
                def __init__(self, parent):
                    super().__init__()
                    self.parent = parent
                
                def on_epoch_end(self, epoch, logs=None):
                    # Extract expert losses
                    lstm_loss_reg = logs.get('val_lstm_expert_regression_loss', 0)
                    transformer_loss_reg = logs.get('val_transformer_expert_regression_loss', 0)
                    lstm_loss_clf = logs.get('val_lstm_expert_classification_loss', 0)
                    transformer_loss_clf = logs.get('val_transformer_expert_classification_loss', 0)
                    
                    # Update expert weights
                    self.parent.update_expert_weights(
                        lstm_loss_reg, transformer_loss_reg,
                        lstm_loss_clf, transformer_loss_clf
                    )
                    
                    if epoch % 10 == 0:
                        print(f"\nExpert weights - LSTM: {self.parent.expert_weights['lstm_path']:.3f}, "
                              f"Transformer: {self.parent.expert_weights['transformer_path']:.3f}")
            
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',  # Total loss
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
                    'best_haelt.keras',
                    monitor='val_loss',
                    save_best_only=True,
                    mode='min',
                    verbose=0
                ),
                ExpertLossCallback(self)
            ]
            
            # Train
            epochs = 100
            batch_size = 32
            
            # Prepare multi-output labels
            y_train_dict = {
                # Regression targets
                'final_regression': y_train_seq,
                'lstm_expert_regression': y_train_seq,
                'transformer_expert_regression': y_train_seq,
                # Classification targets
                'final_classification': y_train_seq_binary,
                'lstm_expert_classification': y_train_seq_binary,
                'transformer_expert_classification': y_train_seq_binary
            }
            y_val_dict = {
                # Regression targets
                'final_regression': y_val_seq,
                'lstm_expert_regression': y_val_seq,
                'transformer_expert_regression': y_val_seq,
                # Classification targets
                'final_classification': y_val_seq_binary,
                'lstm_expert_classification': y_val_seq_binary,
                'transformer_expert_classification': y_val_seq_binary
            }
            
            history = self.models['haelt'].fit(
                X_train_seq, y_train_dict,
                validation_data=(X_val_seq, y_val_dict),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
            
            # Load best model
            self.models['haelt'].load_weights('best_haelt.keras')
            
            print(f"\nFinal expert weights:")
            print(f"  LSTM path: {self.expert_weights['lstm_path']:.3f}")
            print(f"  Transformer path: {self.expert_weights['transformer_path']:.3f}")
            
            # Print model summary
            print("\nHAELT Model Summary:")
            print(f"Total parameters: {self.models['haelt'].count_params():,}")
        
        # 4. Train meta-learner
        print("\n4. Training Meta-Learner...")
        self._train_meta_learner(X_train_scaled, y_train_scaled, y_train_binary, 
                               X_val_scaled, y_val_scaled, y_val_binary)
        
        print("\nAll models trained successfully!")

    def _train_meta_learner(self, X_train, y_train_reg, y_train_clf, X_val, y_val_reg, y_val_clf):
        """Train meta-learner that combines both regression and classification predictions."""
        # Get predictions from base models
        train_preds = []
        val_preds = []
        
        # LightGBM regression predictions
        train_preds.append(self.models['lightgbm_regression'].predict(X_train))
        val_preds.append(self.models['lightgbm_regression'].predict(X_val))
        
        # LightGBM classification predictions (probabilities)
        train_preds.append(self.models['lightgbm_classification'].predict_proba(X_train)[:, 1])
        val_preds.append(self.models['lightgbm_classification'].predict_proba(X_val)[:, 1])
        
        # HAELT predictions
        X_train_seq, y_train_seq = self.prepare_sequences(X_train, y_train_reg)
        X_val_seq, y_val_seq = self.prepare_sequences(X_val, y_val_reg)
        
        if len(X_train_seq) > 0:
            # Get predictions from HAELT
            haelt_train_pred = self.models['haelt'].predict(X_train_seq, verbose=0)
            haelt_val_pred = self.models['haelt'].predict(X_val_seq, verbose=0)
            
            # Extract all predictions
            train_reg_final = haelt_train_pred['final_regression'].ravel()
            train_clf_final = haelt_train_pred['final_classification'].ravel()
            train_reg_lstm = haelt_train_pred['lstm_expert_regression'].ravel()
            train_clf_lstm = haelt_train_pred['lstm_expert_classification'].ravel()
            train_reg_transformer = haelt_train_pred['transformer_expert_regression'].ravel()
            train_clf_transformer = haelt_train_pred['transformer_expert_classification'].ravel()
            
            val_reg_final = haelt_val_pred['final_regression'].ravel()
            val_clf_final = haelt_val_pred['final_classification'].ravel()
            val_reg_lstm = haelt_val_pred['lstm_expert_regression'].ravel()
            val_clf_lstm = haelt_val_pred['lstm_expert_classification'].ravel()
            val_reg_transformer = haelt_val_pred['transformer_expert_regression'].ravel()
            val_clf_transformer = haelt_val_pred['transformer_expert_classification'].ravel()
            
            # Combine expert predictions
            haelt_train_reg_combined = (0.7 * train_reg_final + 
                                       0.3 * (self.expert_weights['lstm_path'] * train_reg_lstm + 
                                             self.expert_weights['transformer_path'] * train_reg_transformer))
            
            haelt_train_clf_combined = (0.7 * train_clf_final + 
                                       0.3 * (self.expert_weights['lstm_path'] * train_clf_lstm + 
                                             self.expert_weights['transformer_path'] * train_clf_transformer))
            
            haelt_val_reg_combined = (0.7 * val_reg_final + 
                                     0.3 * (self.expert_weights['lstm_path'] * val_reg_lstm + 
                                           self.expert_weights['transformer_path'] * val_reg_transformer))
            
            haelt_val_clf_combined = (0.7 * val_clf_final + 
                                     0.3 * (self.expert_weights['lstm_path'] * val_clf_lstm + 
                                           self.expert_weights['transformer_path'] * val_clf_transformer))
            
            # Align predictions
            for pred_type, train_pred, val_pred in [
                ('haelt_reg', haelt_train_reg_combined, haelt_val_reg_combined),
                ('haelt_clf', haelt_train_clf_combined, haelt_val_clf_combined)
            ]:
                full_train_pred = np.zeros(len(y_train_reg))
                full_val_pred = np.zeros(len(y_val_reg))
                
                full_train_pred[self.sequence_length:self.sequence_length+len(train_pred)] = train_pred
                full_val_pred[self.sequence_length:self.sequence_length+len(val_pred)] = val_pred
                
                # Forward fill
                for i in range(len(full_train_pred)):
                    if full_train_pred[i] == 0 and i > 0:
                        full_train_pred[i] = full_train_pred[i-1]
                for i in range(len(full_val_pred)):
                    if full_val_pred[i] == 0 and i > 0:
                        full_val_pred[i] = full_val_pred[i-1]
                
                train_preds.append(full_train_pred)
                val_preds.append(full_val_pred)
        
        # Stack predictions (now includes: lgb_reg, lgb_clf, haelt_reg, haelt_clf)
        train_meta_features = np.column_stack(train_preds)
        val_meta_features = np.column_stack(val_preds)
        
        # Add ALL original features
        train_meta_features = np.hstack([train_meta_features, X_train])
        val_meta_features = np.hstack([val_meta_features, X_val])
        
        # Build meta-learner for regression
        input_dim = train_meta_features.shape[1]
        
        inputs = Input(shape=(input_dim,))
        x = Dense(128, activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Dense(32, activation='relu')(x)
        outputs = Dense(1)(x)
        
        self.meta_learner = Model(inputs=inputs, outputs=outputs)
        self.meta_learner.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
        
        # Train
        self.meta_learner.fit(
            train_meta_features, y_train_reg,
            validation_data=(val_meta_features, y_val_reg),
            epochs=100,
            batch_size=32,
            callbacks=[
                EarlyStopping(patience=15, restore_best_weights=True),
                ReduceLROnPlateau(patience=5, factor=0.5)
            ],
            verbose=0
        )
        
        # Calculate combined performance metrics
        # Regression performance
        lgb_reg_mse = mean_squared_error(y_val_reg, val_preds[0])
        haelt_reg_mse = mean_squared_error(y_val_reg[self.sequence_length:], 
                                          val_preds[2][self.sequence_length:])
        
        # Classification performance (using AUC)
        lgb_clf_auc = roc_auc_score(y_val_clf, val_preds[1])
        haelt_clf_auc = roc_auc_score(y_val_clf[self.sequence_length:], 
                                      val_preds[3][self.sequence_length:])
        
        # Combined score (lower is better, so we use 1-AUC for classification)
        lgb_combined_score = 0.5 * lgb_reg_mse + 0.5 * (1 - lgb_clf_auc)
        haelt_combined_score = 0.5 * haelt_reg_mse + 0.5 * (1 - haelt_clf_auc)
        
        # Weights inversely proportional to combined score
        total_inv_score = 1/lgb_combined_score + 1/haelt_combined_score
        self.model_weights['lightgbm'] = (1/lgb_combined_score) / total_inv_score
        self.model_weights['haelt'] = (1/haelt_combined_score) / total_inv_score
        
        print(f"\nPerformance metrics:")
        print(f"  LightGBM - Regression MSE: {lgb_reg_mse:.4f}, Classification AUC: {lgb_clf_auc:.3f}")
        print(f"  HAELT - Regression MSE: {haelt_reg_mse:.4f}, Classification AUC: {haelt_clf_auc:.3f}")
        print(f"\nInitial model weights - LightGBM: {self.model_weights['lightgbm']:.3f}, "
              f"HAELT: {self.model_weights['haelt']:.3f}")

    def predict(self, X_test, use_meta_learner=True):
        """Make predictions with the ensemble, returning both regression and classification results."""
        X_test_scaled = self.scalers['features'].transform(X_test)
        
        predictions = {}
        
        # LightGBM predictions
        predictions['lightgbm_regression'] = self.models['lightgbm_regression'].predict(X_test_scaled)
        predictions['lightgbm_classification'] = self.models['lightgbm_classification'].predict_proba(X_test_scaled)[:, 1]
        
        # HAELT predictions
        X_test_seq, _ = self.prepare_sequences(X_test_scaled)
        if len(X_test_seq) > 0:
            haelt_pred = self.models['haelt'].predict(X_test_seq, verbose=0)
            
            # Extract all predictions
            reg_final = haelt_pred['final_regression'].ravel()
            clf_final = haelt_pred['final_classification'].ravel()
            reg_lstm = haelt_pred['lstm_expert_regression'].ravel()
            clf_lstm = haelt_pred['lstm_expert_classification'].ravel()
            reg_transformer = haelt_pred['transformer_expert_regression'].ravel()
            clf_transformer = haelt_pred['transformer_expert_classification'].ravel()
            
            # Apply dynamic expert weights
            weighted_reg = (self.expert_weights['lstm_path'] * reg_lstm + 
                           self.expert_weights['transformer_path'] * reg_transformer)
            weighted_clf = (self.expert_weights['lstm_path'] * clf_lstm + 
                           self.expert_weights['transformer_path'] * clf_transformer)
            
            # Combine with final predictions
            haelt_reg_combined = 0.7 * reg_final + 0.3 * weighted_reg
            haelt_clf_combined = 0.7 * clf_final + 0.3 * weighted_clf
            
            # Align predictions
            for pred_type, combined in [
                ('haelt_regression', haelt_reg_combined),
                ('haelt_classification', haelt_clf_combined)
            ]:
                full_pred = np.zeros(len(X_test))
                full_pred[self.sequence_length:self.sequence_length+len(combined)] = combined
                
                # Forward fill
                for i in range(len(full_pred)):
                    if full_pred[i] == 0 and i > 0:
                        full_pred[i] = full_pred[i-1]
                
                predictions[pred_type] = full_pred
        else:
            predictions['haelt_regression'] = np.zeros(len(X_test))
            predictions['haelt_classification'] = np.zeros(len(X_test))
        
        # Ensemble prediction using meta-learner
        if use_meta_learner and self.meta_learner is not None:
            # Prepare features for meta-learner
            pred_matrix = np.column_stack([
                predictions['lightgbm_regression'],
                predictions['lightgbm_classification'],
                predictions['haelt_regression'],
                predictions['haelt_classification']
            ])
            
            # Add ALL original features
            meta_features = np.hstack([pred_matrix, X_test_scaled])
            
            ensemble_pred = self.meta_learner.predict(meta_features, verbose=0).ravel()
        else:
            # Weighted average of regression predictions only
            weights = np.array([self.model_weights['lightgbm'], self.model_weights['haelt']])
            pred_matrix = np.column_stack([predictions['lightgbm_regression'], 
                                         predictions['haelt_regression']])
            ensemble_pred = np.average(pred_matrix, weights=weights, axis=1)
        
        # Inverse transform regression predictions
        ensemble_pred = self.scalers['target'].inverse_transform(ensemble_pred.reshape(-1, 1)).ravel()
        
        predictions['lightgbm_regression'] = self.scalers['target'].inverse_transform(
            predictions['lightgbm_regression'].reshape(-1, 1)
        ).ravel()
        predictions['haelt_regression'] = self.scalers['target'].inverse_transform(
            predictions['haelt_regression'].reshape(-1, 1)
        ).ravel()
        
        # Calculate combined confidence score
        # This combines both regression confidence (based on prediction variance) and classification probability
        lgb_confidence = predictions['lightgbm_classification']
        haelt_confidence = predictions['haelt_classification']
        
        # Weighted confidence
        ensemble_confidence = (self.model_weights['lightgbm'] * lgb_confidence + 
                             self.model_weights['haelt'] * haelt_confidence)
        
        # Create a composite prediction object
        composite_predictions = {
            'ensemble_price': ensemble_pred,
            'ensemble_confidence': ensemble_confidence,
            'ensemble_direction': (ensemble_confidence > 0.5).astype(int),
            'lightgbm': predictions['lightgbm_regression'],
            'haelt': predictions['haelt_regression'],
            'lightgbm_prob': predictions['lightgbm_classification'],
            'haelt_prob': predictions['haelt_classification']
        }
        
        return ensemble_pred, predictions, composite_predictions

    def update_model_weights_online(self, predictions, actual_price):
        """Update model weights based on combined regression and classification performance."""
        # Calculate actual direction
        if hasattr(self, 'last_price'):
            actual_direction = int(actual_price > self.last_price)
        else:
            actual_direction = 1  # Default for first prediction
        
        self.last_price = actual_price
        
        errors = {}
        
        # Calculate regression errors
        lgb_reg_error = abs(predictions['lightgbm'] - actual_price)
        haelt_reg_error = abs(predictions['haelt'] - actual_price)
        
        # Calculate classification errors (log loss)
        lgb_clf_pred = np.clip(predictions.get('lightgbm_prob', 0.5), 1e-7, 1-1e-7)
        haelt_clf_pred = np.clip(predictions.get('haelt_prob', 0.5), 1e-7, 1-1e-7)
        
        lgb_clf_error = -actual_direction * np.log(lgb_clf_pred) - (1-actual_direction) * np.log(1-lgb_clf_pred)
        haelt_clf_error = -actual_direction * np.log(haelt_clf_pred) - (1-actual_direction) * np.log(1-haelt_clf_pred)
        
        # Combined error (normalized)
        max_price = 10000  # Normalize regression error
        errors['lightgbm'] = 0.5 * (lgb_reg_error / max_price) + 0.5 * lgb_clf_error
        errors['haelt'] = 0.5 * (haelt_reg_error / max_price) + 0.5 * haelt_clf_error
        
        # Update performance history
        for name, error in errors.items():
            self.performance_history[name].append(error)
        
        # Calculate recent performance
        recent_performance = {}
        for name in ['lightgbm', 'haelt']:
            if len(self.performance_history[name]) > 0:
                recent_errors = list(self.performance_history[name])
                weights = np.exp(-0.1 * np.arange(len(recent_errors)))
                weights = weights / weights.sum()
                recent_performance[name] = np.average(recent_errors, weights=weights)
            else:
                recent_performance[name] = 1.0
        
        # Update weights
        performance_scores = {}
        for name in ['lightgbm', 'haelt']:
            if recent_performance[name] > 0:
                performance_scores[name] = 1.0 / recent_performance[name]
            else:
                performance_scores[name] = 1.0
        
        # Apply momentum
        new_weights = {}
        total_score = sum(performance_scores.values())
        
        for name in ['lightgbm', 'haelt']:
            target_weight = performance_scores[name] / total_score if total_score > 0 else 0.5
            current_weight = self.model_weights.get(name, 0.5)
            new_weight = (self.weight_momentum * current_weight + 
                         (1 - self.weight_momentum) * target_weight)
            new_weight = max(new_weight, self.min_weight)
            new_weights[name] = new_weight
        
        # Normalize
        total_weight = sum(new_weights.values())
        self.model_weights = {name: w/total_weight for name, w in new_weights.items()}
        
        self.weight_history.append(self.model_weights.copy())
        
        return self.model_weights

    def simulate_real_time_forecast(self, X_test, y_test, n_steps=None, verbose=True):
        """Simulate real-time prediction with both price and direction tracking."""
        if n_steps is None:
            n_steps = min(len(X_test) - 1, 168)
        else:
            n_steps = min(n_steps, len(X_test) - 1)
        
        if verbose:
            print(f"\nSimulating real-time forecast for {n_steps} steps...")
            print("="*60)
        
        predictions = []
        actual_values = []
        direction_predictions = []
        confidence_scores = []
        all_model_predictions = {'lightgbm': [], 'haelt': [], 
                               'lightgbm_prob': [], 'haelt_prob': []}
        weight_evolution = []
        errors = []
        direction_accuracies = []
        
        for step in range(n_steps):
            # Current features
            current_features = X_test[step].reshape(1, -1)
            
            # Prediction
            pred, model_preds, composite = self.predict(current_features, use_meta_learner=True)
            
            # Save predictions
            predictions.append(composite['ensemble_price'][0])
            actual_values.append(y_test[step])
            direction_predictions.append(composite['ensemble_direction'][0])
            confidence_scores.append(composite['ensemble_confidence'][0])
            
            # Save individual model predictions
            all_model_predictions['lightgbm'].append(composite['lightgbm'][0])
            all_model_predictions['haelt'].append(composite['haelt'][0])
            all_model_predictions['lightgbm_prob'].append(composite['lightgbm_prob'][0])
            all_model_predictions['haelt_prob'].append(composite['haelt_prob'][0])
            
            # Calculate price error
            error = abs(pred[0] - y_test[step])
            errors.append(error)
            
            # Calculate direction accuracy
            if step > 0:
                actual_direction = int(y_test[step] > y_test[step-1])
                predicted_direction = composite['ensemble_direction'][0]
                direction_correct = int(actual_direction == predicted_direction)
                direction_accuracies.append(direction_correct)
            
            # Update weights
            if step > 0:
                pred_dict = {
                    'lightgbm': composite['lightgbm'][0],
                    'haelt': composite['haelt'][0],
                    'lightgbm_prob': composite['lightgbm_prob'][0],
                    'haelt_prob': composite['haelt_prob'][0]
                }
                self.update_model_weights_online(pred_dict, y_test[step])
            
            weight_evolution.append(self.model_weights.copy())
            
            # Progress
            if verbose and (step + 1) % 24 == 0:
                recent_mae = np.mean(errors[-24:])
                recent_dir_acc = np.mean(direction_accuracies[-24:]) if direction_accuracies else 0
                avg_confidence = np.mean(confidence_scores[-24:])
                
                print(f"\nStep {step + 1}/{n_steps}:")
                print(f"  Price MAE: ${recent_mae:.2f}")
                print(f"  Direction Accuracy: {recent_dir_acc*100:.1f}%")
                print(f"  Average Confidence: {avg_confidence:.3f}")
                print(f"  Weights - LightGBM: {self.model_weights['lightgbm']:.3f}, "
                      f"HAELT: {self.model_weights['haelt']:.3f}")
                print(f"  Expert Weights - LSTM: {self.expert_weights['lstm_path']:.3f}, "
                      f"Transformer: {self.expert_weights['transformer_path']:.3f}")
        
        # Final metrics
        predictions = np.array(predictions)
        actual_values = np.array(actual_values)
        direction_predictions = np.array(direction_predictions)
        confidence_scores = np.array(confidence_scores)
        
        mae = mean_absolute_error(actual_values, predictions)
        rmse = np.sqrt(mean_squared_error(actual_values, predictions))
        mape = np.mean(np.abs((actual_values - predictions) / actual_values)) * 100
        
        # Direction metrics
        direction_accuracy = np.mean(direction_accuracies) if direction_accuracies else 0
        avg_confidence = np.mean(confidence_scores)
        
        # Calculate profit factor (for trading)
        profits = []
        for i in range(1, len(predictions)):
            actual_return = actual_values[i] - actual_values[i-1]
            predicted_direction = direction_predictions[i-1]
            confidence = confidence_scores[i-1]
            
            # Simulated P&L: trade in predicted direction with confidence as position size
            if predicted_direction == 1:  # Predicted up
                profit = actual_return * confidence
            else:  # Predicted down
                profit = -actual_return * confidence
            
            profits.append(profit)
        
        total_profit = np.sum(profits)
        profitable_trades = sum(1 for p in profits if p > 0)
        profit_factor = sum(p for p in profits if p > 0) / -sum(p for p in profits if p < 0) if any(p < 0 for p in profits) else np.inf
        
        if verbose:
            print("\n" + "="*60)
            print("REAL-TIME SIMULATION RESULTS")
            print("="*60)
            print(f"Total steps: {len(predictions)}")
            print("\nPrice Prediction Metrics:")
            print(f"  MAE: ${mae:.2f}")
            print(f"  RMSE: ${rmse:.2f}")
            print(f"  MAPE: {mape:.2f}%")
            print("\nDirection Prediction Metrics:")
            print(f"  Direction Accuracy: {direction_accuracy*100:.1f}%")
            print(f"  Average Confidence: {avg_confidence:.3f}")
            print("\nTrading Simulation:")
            print(f"  Total P&L: ${total_profit:.2f}")
            print(f"  Win Rate: {profitable_trades/len(profits)*100:.1f}%")
            print(f"  Profit Factor: {profit_factor:.2f}")
            print("\nFinal Model Weights:")
            print(f"  LightGBM: {self.model_weights['lightgbm']:.3f}")
            print(f"  HAELT: {self.model_weights['haelt']:.3f}")
            print("\nFinal Expert Weights:")
            print(f"  LSTM Path: {self.expert_weights['lstm_path']:.3f}")
            print(f"  Transformer Path: {self.expert_weights['transformer_path']:.3f}")
        
        return {
            'predictions': predictions,
            'actual_values': actual_values,
            'direction_predictions': direction_predictions,
            'confidence_scores': confidence_scores,
            'all_model_predictions': all_model_predictions,
            'weight_evolution': weight_evolution,
            'errors': errors,
            'direction_accuracies': direction_accuracies,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'direction_accuracy': direction_accuracy,
            'avg_confidence': avg_confidence,
            'total_profit': total_profit,
            'profit_factor': profit_factor
        }