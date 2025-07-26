import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_results(simulation_result):
        """Visualizar resultados de la simulación."""
        fig = plt.figure(figsize=(20, 10))
        
        predictions = simulation_result['predictions']
        actual_values = simulation_result['actual_values']
        errors = simulation_result['errors']
        weight_evolution = simulation_result['weight_evolution']
        
        # 1. Predictions vs Actual
        ax1 = plt.subplot(2, 3, 1)
        x = np.arange(len(predictions))
        ax1.plot(x, actual_values, 'b-', label='Actual', linewidth=2)
        ax1.plot(x, predictions, 'r-', label='Predicted', linewidth=2, alpha=0.8)
        ax1.fill_between(x, predictions - np.array(errors), predictions + np.array(errors),
                        alpha=0.2, color='red', label='Error bounds')
        ax1.set_xlabel('Time Steps (hours)')
        ax1.set_ylabel('Price (USD)')
        ax1.set_title('Real-Time Predictions vs Actual')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Model Weight Evolution
        ax2 = plt.subplot(2, 3, 2)
        weight_df = pd.DataFrame(weight_evolution)
        ax2.plot(weight_df['lightgbm'], label='LightGBM', linewidth=2, color='blue')
        ax2.plot(weight_df['helformer'], label='Helformer', linewidth=2, color='green')
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Weight')
        ax2.set_title('Model Weight Evolution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # 3. Error Distribution
        ax3 = plt.subplot(2, 3, 3)
        ax3.hist(errors, bins=30, density=True, alpha=0.7, color='purple', edgecolor='black')
        ax3.axvline(x=np.mean(errors), color='r', linestyle='--', 
                   label=f'Mean: ${np.mean(errors):.2f}')
        ax3.axvline(x=np.median(errors), color='g', linestyle='--', 
                   label=f'Median: ${np.median(errors):.2f}')
        ax3.set_xlabel('Error (USD)')
        ax3.set_ylabel('Density')
        ax3.set_title('Error Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Individual Model Predictions
        ax4 = plt.subplot(2, 3, 4)
        all_preds = simulation_result['all_model_predictions']
        ax4.plot(x, actual_values, 'k-', label='Actual', linewidth=2)
        ax4.plot(x, all_preds['lightgbm'], 'b--', label='LightGBM', alpha=0.7)
        ax4.plot(x, all_preds['helformer'], 'g--', label='Helformer', alpha=0.7)
        ax4.plot(x, predictions, 'r-', label='Ensemble', linewidth=2)
        ax4.set_xlabel('Time Steps')
        ax4.set_ylabel('Price (USD)')
        ax4.set_title('Model Predictions Comparison')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Cumulative Error
        ax5 = plt.subplot(2, 3, 5)
        cumulative_error = np.cumsum(errors)
        ax5.plot(cumulative_error, 'orange', linewidth=2)
        ax5.set_xlabel('Time Steps')
        ax5.set_ylabel('Cumulative Error (USD)')
        ax5.set_title('Cumulative Prediction Error')
        ax5.grid(True, alpha=0.3)
        
        # 6. Rolling MAE
        ax6 = plt.subplot(2, 3, 6)
        window = 24
        rolling_mae = pd.Series(errors).rolling(window=window, min_periods=1).mean()
        ax6.plot(rolling_mae, 'green', linewidth=2)
        ax6.axhline(y=simulation_result['mae'], color='r', linestyle='--', 
                   label=f'Overall MAE: ${simulation_result["mae"]:.2f}')
        ax6.set_xlabel('Time Steps')
        ax6.set_ylabel('MAE (USD)')
        ax6.set_title(f'{window}-Hour Rolling MAE')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Imprimir estadísticas adicionales
        print("\n" + "="*60)
        print("DETAILED STATISTICS")
        print("="*60)
        
        # Por modelo
        for model in ['lightgbm', 'helformer']:
            model_errors = [abs(all_preds[model][i] - actual_values[i]) 
                           for i in range(len(actual_values))]
            print(f"\n{model.upper()} Performance:")
            print(f"  MAE: ${np.mean(model_errors):.2f}")
            print(f"  RMSE: ${np.sqrt(np.mean(np.array(model_errors)**2)):.2f}")
            print(f"  Max Error: ${np.max(model_errors):.2f}")
            print(f"  Min Error: ${np.min(model_errors):.2f}")

