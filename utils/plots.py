import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_results(simulation_result):
        """Visualizar resultados de la simulación."""
        fig = plt.figure(figsize=(20, 10))
        
        predictions = simulation_result['predictions']
        actual_values = simulation_result['actual_values']
        errors = simulation_result['errors']
        
        # 1. Predictions vs Actual
        
        x = np.arange(len(predictions))
        fig.plot(x, actual_values, 'b-', label='Actual', linewidth=2)
        fig.plot(x, predictions, 'r-', label='Predicted', linewidth=2, alpha=0.8)
        fig.fill_between(x, predictions - np.array(errors), predictions + np.array(errors),
                        alpha=0.2, color='red', label='Error bounds')
        fig.set_xlabel('Time Steps (hours)')
        fig.set_ylabel('Price (USD)')
        fig.set_title('Real-Time Predictions vs Actual')
        fig.legend()
        fig.grid(True, alpha=0.3)
        


