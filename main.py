from utils.data import scrape_candles_to_dataframe
from utils.features import create_robust_features
from utils.model import EnhancedHierarchicalMetaLearner, simulate_enhanced_real_time_forecast# New import
from utils.plots import plot_results, plot_classification_results
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# Configuration
OPTIMIZE_HYPERPARAMS = False  
SYMBOL = 'SOL/USDT'  # or 'BTC/USDT'
USE_ENHANCED_MODEL = True  # Toggle between original and enhanced model


def main():
    # 1. Fetch data
    print("1. Fetching data...")
    data = scrape_candles_to_dataframe('binance', 3, SYMBOL, '1h', '2025-01-01T00:00:00Z', 1000)
    print(f"Total data: {len(data)}")

    # 2. Create features
    print("\n2. Creating features...")
    df = create_robust_features(data)

    # 3. Initialize model
    if USE_ENHANCED_MODEL:
        print("\n3. Initializing Enhanced Hierarchical Meta-Learning Model...")
        model = EnhancedHierarchicalMetaLearner(sequence_length=168, forecast_horizon=1)
    
    X, y = model.prepare_data(df)

    # 4. Split data
    print("\n4. Splitting data...")
    test_size = 730  # 730 hours ~ 1 month
    val_size = 730

    X_train = X[:-test_size-val_size]
    y_train = y[:-test_size-val_size]
    X_val = X[-test_size-val_size:-test_size]
    y_val = y[-test_size-val_size:-test_size]
    X_test = X[-test_size:]
    y_test = y[-test_size:]

    print(f"Train: {len(X_train)} samples")
    print(f"Val: {len(X_val)} samples")
    print(f"Test: {len(X_test)} samples")

    # 5. Train models
    print("\n5. Training models...")

    model.train(X_train, y_train, X_val, y_val)


    # 6. Run simulation
    print("\n6. Running real-time simulation...")
    n_simulation_steps = len(X_test) - model.sequence_length
    test_df = df.iloc[-test_size:].copy()


    simulation_result = simulate_enhanced_real_time_forecast(
            model, 
            test_df, 
            model.scalers['features'], 
            model.scalers['target'], 
            forecast_horizon=n_simulation_steps,  # Correct parameter name
            update_interval=10
    )

    # 7. Visualize results
    print("\n7. Plotting results...")
    
    # Create enhanced plots for hierarchical model
    if USE_ENHANCED_MODEL and 'base_regression' in simulation_result['all_model_predictions']:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # Plot 1: Final predictions vs actual
        ax1 = axes[0]
        ax1.plot(simulation_result['actual_values'], label='Actual', color='black', linewidth=2)
        ax1.plot(simulation_result['predictions'], label='Final Prediction', color='red', linewidth=2)
        ax1.plot(simulation_result['all_model_predictions']['level1_regression'], 
                label='Level 1 Regression', color='blue', alpha=0.7, linestyle='--')
        ax1.set_title('Hierarchical Predictions vs Actual Prices')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Base model predictions
        ax2 = axes[1]
        ax2.plot(simulation_result['actual_values'], label='Actual', color='black', linewidth=2, alpha=0.8)
        
        # Plot top 5 base models
        model_maes = {}
        for model_name in simulation_result['all_model_predictions']['base_regression']:
            model_preds = np.array(simulation_result['all_model_predictions']['base_regression'][model_name])
            model_mae = np.mean(np.abs(model_preds - simulation_result['actual_values']))
            model_maes[model_name] = model_mae
        
        top_models = sorted(model_maes.items(), key=lambda x: x[1])[:5]
        colors = plt.cm.rainbow(np.linspace(0, 1, len(top_models)))
        
        for (model_name, mae), color in zip(top_models, colors):
            preds = simulation_result['all_model_predictions']['base_regression'][model_name]
            ax2.plot(preds, label=f'{model_name} (MAE: ${mae:.1f})', 
                    color=color, alpha=0.7, linewidth=1)
        
        ax2.set_title('Top 5 Base Model Predictions')
        ax2.set_ylabel('Price')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Confidence scores and direction accuracy
        ax3 = axes[2]
        ax3_twin = ax3.twinx()
        
        # Plot confidence scores
        ax3.plot(simulation_result['confidence_scores'], label='Confidence', 
                color='green', alpha=0.7)
        ax3.set_ylabel('Confidence Score', color='green')
        ax3.tick_params(axis='y', labelcolor='green')
        ax3.set_ylim([0, 1])
        
        # Plot cumulative direction accuracy
        cumulative_accuracy = np.cumsum(simulation_result['direction_accuracies']) / np.arange(1, len(simulation_result['direction_accuracies'])+1)
        ax3_twin.plot(cumulative_accuracy, label='Cumulative Direction Accuracy', 
                     color='orange', linewidth=2)
        ax3_twin.set_ylabel('Direction Accuracy', color='orange')
        ax3_twin.tick_params(axis='y', labelcolor='orange')
        ax3_twin.set_ylim([0, 1])
        
        ax3.set_xlabel('Time Steps')
        ax3.set_title('Confidence Score and Direction Accuracy')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        

if __name__ == "__main__":
    main()