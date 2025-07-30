import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

def plot_results(simulation_result):
    """Enhanced plotting function that handles both original and hierarchical model outputs"""
    
    # Check if this is hierarchical model output
    is_hierarchical = 'base_regression' in simulation_result.get('all_model_predictions', {})
    
    if is_hierarchical:
        plot_hierarchical_results(simulation_result)
    else:
        plot_standard_results(simulation_result)

def plot_hierarchical_results(simulation_result):
    """Plot results for hierarchical meta-learning model"""
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
    
    # 1. Main prediction plot
    ax1 = fig.add_subplot(gs[0, :])
    plot_main_predictions(ax1, simulation_result)
    
    # 2. Model hierarchy visualization
    ax2 = fig.add_subplot(gs[1, 0])
    plot_model_hierarchy(ax2, simulation_result)
    
    # 3. Base model performance
    ax3 = fig.add_subplot(gs[1, 1:])
    plot_base_model_performance(ax3, simulation_result)
    
    # 4. Error distribution
    ax4 = fig.add_subplot(gs[2, 0])
    plot_error_distribution(ax4, simulation_result)
    
    # 5. Confidence analysis
    ax5 = fig.add_subplot(gs[2, 1])
    plot_confidence_analysis(ax5, simulation_result)
    
    # 6. Trading metrics
    ax6 = fig.add_subplot(gs[2, 2])
    plot_trading_metrics(ax6, simulation_result)
    
    # 7. Model contribution over time
    ax7 = fig.add_subplot(gs[3, :])
    plot_model_contributions(ax7, simulation_result)
    
    plt.suptitle('Enhanced Hierarchical Meta-Learning Model Results', fontsize=16, y=0.995)
    plt.tight_layout()
    plt.show()

def plot_main_predictions(ax, result):
    """Plot main predictions with confidence bands"""
    
    actual = result['actual_values']
    predictions = result['predictions']
    confidence = result['confidence_scores']
    
    # Create confidence bands
    upper_band = predictions + (1 - confidence) * np.std(actual - predictions) * 2
    lower_band = predictions - (1 - confidence) * np.std(actual - predictions) * 2
    
    # Plot
    ax.plot(actual, label='Actual', color='black', linewidth=2, alpha=0.8)
    ax.plot(predictions, label='Final Prediction', color='red', linewidth=2)
    
    # Confidence bands
    ax.fill_between(range(len(predictions)), lower_band, upper_band, 
                    alpha=0.2, color='red', label='Confidence Band')
    
    # Level 1 predictions if available
    if 'level1_regression' in result['all_model_predictions']:
        level1_pred = result['all_model_predictions']['level1_regression']
        ax.plot(level1_pred, label='Level 1 Meta', color='blue', 
                alpha=0.7, linestyle='--', linewidth=1.5)
    
    # Highlight high confidence predictions
    high_conf_mask = confidence > 0.8
    if np.any(high_conf_mask):
        ax.scatter(np.where(high_conf_mask)[0], predictions[high_conf_mask], 
                  color='green', s=50, alpha=0.6, label='High Confidence', zorder=5)
    
    ax.set_title('Hierarchical Model Predictions with Confidence Bands', fontsize=14)
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Price')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

def plot_model_hierarchy(ax, result):
    """Visualize the model hierarchy"""
    
    # Create hierarchy diagram
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    
    # Base models
    base_models = list(result['all_model_predictions']['base_regression'].keys())
    n_base = len(base_models)
    
    # Draw base model boxes
    base_y = 1
    base_spacing = 8 / n_base
    base_positions = []
    
    for i, model in enumerate(base_models):
        x = 1 + i * base_spacing
        rect = Rectangle((x-0.4, base_y-0.3), 0.8, 0.6, 
                        facecolor='lightblue', edgecolor='black')
        ax.add_patch(rect)
        ax.text(x, base_y, model[:6], ha='center', va='center', fontsize=8)
        base_positions.append((x, base_y))
    
    # Level 1 meta-learners
    level1_y = 4
    reg_rect = Rectangle((2.5, level1_y-0.3), 2, 0.6, 
                        facecolor='lightgreen', edgecolor='black')
    clf_rect = Rectangle((5.5, level1_y-0.3), 2, 0.6, 
                        facecolor='lightcoral', edgecolor='black')
    ax.add_patch(reg_rect)
    ax.add_patch(clf_rect)
    ax.text(3.5, level1_y, 'L1 Regression', ha='center', va='center', fontsize=10)
    ax.text(6.5, level1_y, 'L1 Classification', ha='center', va='center', fontsize=10)
    
    # Level 2 final
    level2_y = 7
    final_rect = Rectangle((4, level2_y-0.3), 2, 0.6, 
                          facecolor='gold', edgecolor='black')
    ax.add_patch(final_rect)
    ax.text(5, level2_y, 'L2 Final', ha='center', va='center', fontsize=12, weight='bold')
    
    # Draw connections
    # Base to Level 1
    for x, y in base_positions:
        ax.plot([x, 3.5], [y+0.3, level1_y-0.3], 'k-', alpha=0.3, linewidth=1)
        ax.plot([x, 6.5], [y+0.3, level1_y-0.3], 'k-', alpha=0.3, linewidth=1)
    
    # Level 1 to Level 2
    ax.plot([3.5, 5], [level1_y+0.3, level2_y-0.3], 'k-', alpha=0.5, linewidth=2)
    ax.plot([6.5, 5], [level1_y+0.3, level2_y-0.3], 'k-', alpha=0.5, linewidth=2)
    
    ax.set_title('Model Hierarchy', fontsize=12)
    ax.axis('off')

def plot_base_model_performance(ax, result):
    """Plot base model performance comparison"""
    
    actual = result['actual_values']
    base_preds = result['all_model_predictions']['base_regression']
    
    # Calculate metrics for each model
    model_metrics = {}
    for model_name, preds in base_preds.items():
        preds_array = np.array(preds)
        mae = np.mean(np.abs(preds_array - actual))
        rmse = np.sqrt(np.mean((preds_array - actual) ** 2))
        corr = np.corrcoef(preds_array, actual)[0, 1]
        model_metrics[model_name] = {'MAE': mae, 'RMSE': rmse, 'Correlation': corr}
    
    # Create DataFrame for plotting
    metrics_df = pd.DataFrame(model_metrics).T
    
    # Normalize metrics for radar chart
    metrics_norm = metrics_df.copy()
    metrics_norm['MAE'] = 1 - (metrics_norm['MAE'] - metrics_norm['MAE'].min()) / (metrics_norm['MAE'].max() - metrics_norm['MAE'].min())
    metrics_norm['RMSE'] = 1 - (metrics_norm['RMSE'] - metrics_norm['RMSE'].min()) / (metrics_norm['RMSE'].max() - metrics_norm['RMSE'].min())
    
    # Plot
    x = np.arange(len(metrics_df))
    width = 0.25
    
    ax.bar(x - width, metrics_norm['MAE'], width, label='MAE (inverted)', color='skyblue')
    ax.bar(x, metrics_norm['RMSE'], width, label='RMSE (inverted)', color='lightcoral')
    ax.bar(x + width, metrics_norm['Correlation'], width, label='Correlation', color='lightgreen')
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Normalized Score')
    ax.set_title('Base Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df.index, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add actual values as text
    for i, model in enumerate(metrics_df.index):
        mae_val = metrics_df.loc[model, 'MAE']
        ax.text(i-width, 0.05, f'${mae_val:.1f}', ha='center', va='bottom', fontsize=8)

def plot_error_distribution(ax, result):
    """Plot error distribution analysis"""
    
    errors = np.array(result['errors'])
    
    # Create bins for histogram
    bins = np.linspace(0, np.percentile(errors, 95), 30)
    
    # Plot histogram
    n, bins, patches = ax.hist(errors, bins=bins, alpha=0.7, color='steelblue', edgecolor='black')
    
    # Add statistics
    mean_error = np.mean(errors)
    median_error = np.median(errors)
    std_error = np.std(errors)
    
    # Add vertical lines for statistics
    ax.axvline(mean_error, color='red', linestyle='--', linewidth=2, label=f'Mean: ${mean_error:.2f}')
    ax.axvline(median_error, color='green', linestyle='--', linewidth=2, label=f'Median: ${median_error:.2f}')
    
    # Add text box with statistics
    textstr = f'Std Dev: ${std_error:.2f}\n95th %ile: ${np.percentile(errors, 95):.2f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.7, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    ax.set_xlabel('Absolute Error ($)')
    ax.set_ylabel('Frequency')
    ax.set_title('Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

def plot_confidence_analysis(ax, result):
    """Analyze relationship between confidence and accuracy"""
    
    confidence = np.array(result['confidence_scores'])
    errors = np.array(result['errors'])
    
    # Bin confidence scores
    n_bins = 10
    confidence_bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (confidence_bins[:-1] + confidence_bins[1:]) / 2
    
    # Calculate mean error for each confidence bin
    mean_errors = []
    error_stds = []
    counts = []
    
    for i in range(n_bins):
        mask = (confidence >= confidence_bins[i]) & (confidence < confidence_bins[i+1])
        if np.sum(mask) > 0:
            mean_errors.append(np.mean(errors[mask]))
            error_stds.append(np.std(errors[mask]))
            counts.append(np.sum(mask))
        else:
            mean_errors.append(np.nan)
            error_stds.append(np.nan)
            counts.append(0)
    
    # Plot
    valid_mask = ~np.isnan(mean_errors)
    ax.errorbar(bin_centers[valid_mask], np.array(mean_errors)[valid_mask], 
                yerr=np.array(error_stds)[valid_mask], 
                marker='o', capsize=5, capthick=2, linewidth=2, markersize=8)
    
    # Add sample counts
    for i, (x, y, n) in enumerate(zip(bin_centers[valid_mask], 
                                     np.array(mean_errors)[valid_mask], 
                                     np.array(counts)[valid_mask])):
        ax.text(x, y + np.array(error_stds)[valid_mask][i] + 5, f'n={n}', 
                ha='center', fontsize=8)
    
    ax.set_xlabel('Confidence Score')
    ax.set_ylabel('Mean Absolute Error ($)')
    ax.set_title('Confidence vs Accuracy')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)

def plot_trading_metrics(ax, result):
    """Plot trading performance metrics"""
    
    # Calculate cumulative returns
    actual = result['actual_values']
    predictions = result['predictions']
    confidence = result['confidence_scores']
    
    returns = []
    cumulative_pnl = [0]
    
    for i in range(1, len(actual)):
        actual_return = actual[i] - actual[i-1]
        pred_direction = 1 if predictions[i-1] > actual[i-1] else -1
        position_size = confidence[i-1]
        
        pnl = pred_direction * actual_return * position_size
        returns.append(pnl)
        cumulative_pnl.append(cumulative_pnl[-1] + pnl)
    
    # Plot cumulative P&L
    ax.plot(cumulative_pnl, linewidth=2, color='darkgreen')
    ax.fill_between(range(len(cumulative_pnl)), 0, cumulative_pnl, 
                    where=np.array(cumulative_pnl) > 0, alpha=0.3, color='green', label='Profit')
    ax.fill_between(range(len(cumulative_pnl)), 0, cumulative_pnl, 
                    where=np.array(cumulative_pnl) <= 0, alpha=0.3, color='red', label='Loss')
    
    # Add metrics
    total_pnl = cumulative_pnl[-1]
    sharpe = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252*24) if returns else 0
    max_dd = np.min(np.minimum.accumulate(cumulative_pnl) - cumulative_pnl)
    
    textstr = f'Total P&L: ${total_pnl:.2f}\nSharpe: {sharpe:.2f}\nMax DD: ${max_dd:.2f}'
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.7)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Cumulative P&L ($)')
    ax.set_title('Trading Performance')
    ax.grid(True, alpha=0.3)
    ax.legend()

def plot_model_contributions(ax, result):
    """Plot how different models contribute over time"""
    
    # This is a placeholder - in practice, you'd need to track model weights
    # For now, we'll show the performance of different models over time
    
    actual = result['actual_values']
    base_preds = result['all_model_predictions']['base_regression']
    
    # Calculate rolling MAE for each model
    window = 24  # 1 day
    time_steps = range(window, len(actual))
    
    rolling_mae = {}
    for model_name, preds in base_preds.items():
        preds_array = np.array(preds)
        mae_series = []
        for i in time_steps:
            window_mae = np.mean(np.abs(preds_array[i-window:i] - actual[i-window:i]))
            mae_series.append(window_mae)
        rolling_mae[model_name] = mae_series
    
    # Plot top 5 models
    avg_mae = {k: np.mean(v) for k, v in rolling_mae.items()}
    top_models = sorted(avg_mae.items(), key=lambda x: x[1])[:5]
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(top_models)))
    
    for (model_name, _), color in zip(top_models, colors):
        ax.plot(time_steps, rolling_mae[model_name], label=model_name, 
                color=color, linewidth=1.5, alpha=0.8)
    
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Rolling MAE (24h window)')
    ax.set_title('Model Performance Over Time')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

def plot_standard_results(simulation_result):
    """Original plotting function for non-hierarchical models"""
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    # 1. Predictions vs Actual
    ax = axes[0, 0]
    ax.plot(simulation_result['actual_values'], label='Actual', color='black', linewidth=2)
    ax.plot(simulation_result['predictions'], label='Ensemble', color='red', linewidth=2)
    
    if 'lightgbm' in simulation_result['all_model_predictions']:
        ax.plot(simulation_result['all_model_predictions']['lightgbm'], 
                label='LightGBM', alpha=0.5, linestyle='--')
    if 'haelt' in simulation_result['all_model_predictions']:
        ax.plot(simulation_result['all_model_predictions']['haelt'], 
                label='HAELT', alpha=0.5, linestyle='--')
    
    ax.set_title('Price Predictions vs Actual')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Prediction Errors
    ax = axes[0, 1]
    ax.plot(simulation_result['errors'], color='red', alpha=0.7)
    ax.axhline(y=np.mean(simulation_result['errors']), color='black', 
               linestyle='--', label=f'Mean: ${np.mean(simulation_result["errors"]):.2f}')
    ax.fill_between(range(len(simulation_result['errors'])), 
                    0, simulation_result['errors'], alpha=0.3, color='red')
    ax.set_title('Absolute Prediction Errors')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Absolute Error ($)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Model Weights Evolution
    ax = axes[1, 0]
    if simulation_result['weight_evolution']:
        weights_df = pd.DataFrame(simulation_result['weight_evolution'])
        for col in weights_df.columns:
            ax.plot(weights_df[col], label=col, linewidth=2)
        ax.set_title('Model Weights Evolution')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Weight')
        ax.legend()
        ax.set_ylim([0, 1])
    else:
        ax.text(0.5, 0.5, 'No weight evolution data', ha='center', va='center')
        ax.set_title('Model Weights Evolution')
    ax.grid(True, alpha=0.3)
    
    # 4. Cumulative Performance
    ax = axes[1, 1]
    cumulative_actual = np.cumsum(np.diff(simulation_result['actual_values'], prepend=simulation_result['actual_values'][0]))
    cumulative_pred = np.cumsum(np.diff(simulation_result['predictions'], prepend=simulation_result['predictions'][0]))
    
    ax.plot(cumulative_actual, label='Actual Returns', color='black', linewidth=2)
    ax.plot(cumulative_pred, label='Strategy Returns', color='green', linewidth=2)
    ax.fill_between(range(len(cumulative_pred)), cumulative_actual, cumulative_pred,
                    where=cumulative_pred >= cumulative_actual, alpha=0.3, color='green', label='Outperformance')
    ax.fill_between(range(len(cumulative_pred)), cumulative_actual, cumulative_pred,
                    where=cumulative_pred < cumulative_actual, alpha=0.3, color='red', label='Underperformance')
    ax.set_title('Cumulative Returns')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Cumulative Return ($)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Error Distribution
    ax = axes[2, 0]
    ax.hist(simulation_result['errors'], bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(x=np.mean(simulation_result['errors']), color='red', linestyle='--', 
               label=f'Mean: ${np.mean(simulation_result["errors"]):.2f}')
    ax.axvline(x=np.median(simulation_result['errors']), color='green', linestyle='--', 
               label=f'Median: ${np.median(simulation_result["errors"]):.2f}')
    ax.set_title('Error Distribution')
    ax.set_xlabel('Absolute Error ($)')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Performance Metrics Summary
    ax = axes[2, 1]
    metrics_text = f"""
    Performance Metrics:
    
    MAE: ${simulation_result['mae']:.2f}
    RMSE: ${simulation_result['rmse']:.2f}
    MAPE: {simulation_result['mape']:.2f}%
    
    Direction Accuracy: {simulation_result.get('direction_accuracy', 0)*100:.1f}%
    Average Confidence: {simulation_result.get('avg_confidence', 0):.3f}
    
    Total P&L: ${simulation_result.get('total_profit', 0):.2f}
    Profit Factor: {simulation_result.get('profit_factor', 0):.2f}
    """
    
    ax.text(0.1, 0.5, metrics_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.axis('off')
    
    plt.suptitle('Real-Time Forecasting Simulation Results', fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_classification_results(simulation_result):
    """Enhanced classification results plotting"""
    
    if 'direction_predictions' not in simulation_result:
        print("No classification data available")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Direction Predictions Over Time
    ax = axes[0, 0]
    
    # Calculate actual directions
    actual_directions = []
    for i in range(1, len(simulation_result['actual_values'])):
        actual_directions.append(1 if simulation_result['actual_values'][i] > simulation_result['actual_values'][i-1] else 0)
    
    # Align arrays
    min_len = min(len(actual_directions), len(simulation_result['direction_predictions'][1:]))
    actual_dir_aligned = actual_directions[:min_len]
    pred_dir_aligned = simulation_result['direction_predictions'][1:min_len+1]
    
    ax.plot(actual_dir_aligned, label='Actual Direction', alpha=0.7, linewidth=1)
    ax.plot(pred_dir_aligned, label='Predicted Direction', alpha=0.7, linewidth=1)
    
    # Add accuracy overlay
    correct_predictions = np.array(actual_dir_aligned) == np.array(pred_dir_aligned)
    ax.fill_between(range(len(correct_predictions)), 0, 1, 
                    where=correct_predictions, alpha=0.2, color='green', label='Correct')
    
    ax.set_title('Direction Predictions')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Direction (0=Down, 1=Up)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Rolling Accuracy
    ax = axes[0, 1]
    
    if 'direction_accuracies' in simulation_result and len(simulation_result['direction_accuracies']) > 0:
        window = min(24, len(simulation_result['direction_accuracies']))  # 24 hours
        rolling_acc = pd.Series(simulation_result['direction_accuracies']).rolling(window).mean()
        
        ax.plot(rolling_acc, linewidth=2, color='blue')
        ax.axhline(y=0.5, color='red', linestyle='--', label='Random Baseline')
        ax.fill_between(range(len(rolling_acc)), 0.5, rolling_acc, 
                       where=rolling_acc >= 0.5, alpha=0.3, color='green')
        ax.fill_between(range(len(rolling_acc)), 0.5, rolling_acc, 
                       where=rolling_acc < 0.5, alpha=0.3, color='red')
        
        ax.set_title(f'Rolling Direction Accuracy ({window}h window)')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Accuracy')
        ax.set_ylim([0, 1])
        ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Confidence Distribution
    ax = axes[1, 0]
    
    confidence = simulation_result['confidence_scores']
    ax.hist(confidence, bins=30, alpha=0.7, color='purple', edgecolor='black')
    ax.axvline(x=np.mean(confidence), color='red', linestyle='--', 
               label=f'Mean: {np.mean(confidence):.3f}')
    ax.axvline(x=0.5, color='black', linestyle=':', label='Neutral (0.5)')
    
    ax.set_title('Confidence Score Distribution')
    ax.set_xlabel('Confidence Score')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Confidence vs Accuracy
    ax = axes[1, 1]
    
    if len(simulation_result['direction_accuracies']) > 0:
        # Bin confidence and calculate accuracy for each bin
        n_bins = 10
        conf_bins = np.linspace(0, 1, n_bins + 1)
        bin_centers = (conf_bins[:-1] + conf_bins[1:]) / 2
        
        bin_accuracies = []
        bin_counts = []
        
        for i in range(n_bins):
            mask = (confidence[1:len(simulation_result['direction_accuracies'])+1] >= conf_bins[i]) & \
                   (confidence[1:len(simulation_result['direction_accuracies'])+1] < conf_bins[i+1])
            if np.sum(mask) > 0:
                bin_acc = np.mean(np.array(simulation_result['direction_accuracies'])[mask])
                bin_accuracies.append(bin_acc)
                bin_counts.append(np.sum(mask))
            else:
                bin_accuracies.append(np.nan)
                bin_counts.append(0)
        
        # Plot
        valid_mask = ~np.isnan(bin_accuracies)
        ax.bar(bin_centers[valid_mask], np.array(bin_accuracies)[valid_mask], 
               width=0.08, alpha=0.7, color='orange', edgecolor='black')
        
        # Add count labels
        for i, (x, y, n) in enumerate(zip(bin_centers[valid_mask], 
                                         np.array(bin_accuracies)[valid_mask], 
                                         np.array(bin_counts)[valid_mask])):
            ax.text(x, y + 0.02, f'n={n}', ha='center', fontsize=8)
        
        ax.axhline(y=0.5, color='red', linestyle='--', label='Random Baseline')
        ax.set_xlabel('Confidence Bin')
        ax.set_ylabel('Direction Accuracy')
        ax.set_title('Accuracy by Confidence Level')
        ax.set_ylim([0, 1])
        ax.legend()
    
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Classification & Direction Prediction Results', fontsize=16)
    plt.tight_layout()
    plt.show()