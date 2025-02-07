# helpers.py
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from io import BytesIO
import base64
from collections import Counter
import xgboost as xgb
from xgboost import XGBClassifier

def preprocess_sensor_data_hydraulic(sensor_df):
    """Convert tab-separated string data into numeric arrays."""
    numeric_data = sensor_df[0].str.split('\t', expand=True).astype(float)
    return numeric_data

# def load_and_preprocess_data_hydraulic(file_paths):
#     """Load and preprocess the sensor data."""
#     data = {}
#     for sensor_type in ['CE', 'CP', 'SE', 'PS6', 'PS2', 'FS1', 'FS2', 'EPS1', 'TS4', 'TS2']:
#         raw_data = pd.read_csv(f'{file_paths}/{sensor_type}.txt', header=None)
#         data[sensor_type] = preprocess_sensor_data_hydraulic(raw_data)

#     features = []
#     for sensor_name, sensor_data in data.items():
#         stats = pd.DataFrame({
#             f'{sensor_name}_mean': sensor_data.mean(axis=1),
#             f'{sensor_name}_std': sensor_data.std(axis=1),
#             f'{sensor_name}_max': sensor_data.max(axis=1),
#             f'{sensor_name}_min': sensor_data.min(axis=1),
#             f'{sensor_name}_range': sensor_data.max(axis=1) - sensor_data.min(axis=1),
#             f'{sensor_name}_rms': np.sqrt((sensor_data ** 2).mean(axis=1))
#         })
#         features.append(stats)

#     X = pd.concat(features, axis=1)
#     return X

def hydraulic_test(X):
    """Use the saved model and scaler to make predictions."""
    # data = load_and_preprocess_data_hydraulic(file_folder)

    model_path = "models/Hydraulic/hydraulic_model.pkl"
    scaler_path = "models/Hydraulic/scaler.pkl"

    # Load the scaler and model
    with open(scaler_path, 'rb') as file:
        loaded_scaler = pickle.load(file)

    with open(model_path, 'rb') as file:
        loaded_hydraulic_model = pickle.load(file)

    new_data_scaled = loaded_scaler.transform(X)
    predictions = loaded_hydraulic_model.predict(new_data_scaled)

    return predictions

def plot_predictions(predictions):
    """Generate time-series plots of the predictions."""

    components = ['Cooler condition', 'Valve condition', 'Pump leakage', 'Accumulator pressure', 'Stable flag']
    component_values = [
        ['3%: Close to total failure', '20%: Reduced efficiency', '100%: Full efficiency'],
        ['100%: Optimal switching behavior', '90%: Small lag', '80%: Severe lag', '73%: Close to total failure'],
        ['0: No leakage', '1: Weak leakage', '2: Severe leakage'],
        ['130: Optimal pressure', '115: Slightly reduced pressure', '100: Severely reduced pressure', '90: Close to total failure'],
        ['0: Conditions were stable', '1: Static conditions might not have been reached yet']
    ]
    
    # plt.figure(figsize=(6, 4))  # Adjust figure size as needed
    # # plt.plot(range(len(predictions)), predictions)
    # if len(predictions.shape) == 2:  # Assuming predictions is 2D: shape (n_samples, n_components)
    #     for i, component in enumerate(components):
    #         plt.plot(range(len(predictions)), predictions[:, i], label=component)
    # else:
    #     plt.plot(range(len(predictions)), predictions, label='Predictions')
    # plt.xlabel("Index")
    # plt.ylabel("Prediction Value")
    # plt.title("Predictions vs. Index")
    # plt.grid(True)
    # plt.legend(title="Components")
    # # plt.show()


    colors = ['b', 'g', 'r', 'c', 'm']  # Blue, Green, Red, Cyan, Magenta
    
    fig, axes = plt.subplots(5, 1, figsize=(10, 10))
    for i, ax in enumerate(axes):
        ax.plot(predictions[:, i], color=colors[i])
        ax.set_title(components[i])
        ax.set_xlabel('Time')
        ax.set_ylabel('Prediction')

        # Add the component values as legends on the right side of each plot
        # ax.legend(component_values[i], loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)

        # Add the component value indicators as text next to each plot
        # Manually place each indicator on the right of the plot (using x and y coordinates)
        y_pos = 0.9  # Starting position for the text (adjust as needed)
        for value in component_values[i]:
            ax.text(1.05, y_pos, value, transform=ax.transAxes, ha='left', va='top', fontsize=9)
            y_pos -=  -.1 # Adjust the spacing between lines of text
    
    
    # Adjust layout to make room for the legend
    plt.subplots_adjust(right=0.8)  # Reduce plot area to allow for legend space
    
    # Save the plot as a PNG image in memory
    img = BytesIO()
    plt.tight_layout()
    plt.savefig(img, format='png')
    img.seek(0)

    # Convert the image to base64 string to display in HTML
    img_base64 = base64.b64encode(img.getvalue()).decode('utf8')
    return img_base64


# """## Ball Bearing"""

# def load_and_preprocess_data(file_paths):
#   data = pd.read_csv(f"{file_paths}/feature_time_48k_2048_load_1.csv")
#   features = ['max', 'min', 'mean', 'sd', 'rms', 'skewness', 'kurtosis', 'crest', 'form']
#   return data[features]

def bearing_test(X):
  model = "models/Bearing/bearing_model.pkl"
  scaler = "models/Bearing/scaler.pkl"

  # Load the scaler
  with open(scaler, 'rb') as file:
      loaded_scaler = pickle.load(file)

  # Load the nn_model
  with open(model, 'rb') as file:
      loaded_bearing_model = pickle.load(file)

  new_data_scaled = loaded_scaler.transform(X)
  predictions = loaded_bearing_model.predict(new_data_scaled)

  label_encoder = f"models/Bearing/label_encoder.pkl"
  with open(label_encoder, 'rb') as file:
      loaded_label_encoder = pickle.load(file)

  labelled_predictions = loaded_label_encoder.inverse_transform(predictions)

  return predictions, labelled_predictions


def plot_predictions_bearing(labelled_predictions):
    # Convert predictions to numerical values for plotting purposes
    class_mapping = {label: idx for idx, label in enumerate(set(labelled_predictions))}
    numeric_predictions = [class_mapping[label] for label in labelled_predictions]

    plt.figure(figsize=(14, 6))
    # plt.plot(numeric_predictions, color='gray', label='Classification')
    # plt.xlabel('Time (Index)')
    # plt.ylabel('Class Labels')
    # plt.title('Time-Series Classification Results')

    # # Add a legend to show which class corresponds to which label
    # handles = [plt.Line2D([0], [0], color=plt.cm.tab10(i/10), lw=2) for i in range(len(class_mapping))]
    # labels = [f"{label}" for label in class_mapping]
    # plt.legend(handles, labels, title="Class Labels", bbox_to_anchor=(1.05, 1), loc='upper left')

    # # Save the plot as an image in memory
    # img = BytesIO()
    # plt.tight_layout()
    # plt.savefig(img, format='png')
    # img.seek(0)

    ###############
    # # Scatter plot where each point's color corresponds to the class label
    # colors = [plt.cm.tab10(idx / len(class_mapping)) for idx in numeric_predictions]  # Colors based on class
    # plt.scatter(range(len(numeric_predictions)), numeric_predictions, c=colors, label='Classification', s=50)

    # # Add labels and title
    # plt.xlabel('Time (Index)')
    # plt.ylabel('Class Labels')
    # plt.title('Time-Series Classification Results')

    # # Add a legend to show which class corresponds to which label
    # handles = [plt.Line2D([0], [0], color=plt.cm.tab10(i / len(class_mapping)), lw=2) for i in range(len(class_mapping))]
    # labels = [f"{label}" for label in class_mapping]
    # plt.legend(handles, labels, title="Class Labels", bbox_to_anchor=(1.05, 1), loc='upper left')
    ###########

    # Plot a thin dotted grey line over the scatter plot
    plt.plot(numeric_predictions, color='gray', linestyle=':', linewidth=1, label='Dotted Grey Line')

    # Scatter plot where each point's color corresponds to the class label
    colors = [plt.cm.tab10(idx / len(class_mapping)) for idx in numeric_predictions]  # Colors based on class
    plt.scatter(range(len(numeric_predictions)), numeric_predictions, c=colors, label='Classification', s=50)

    # Add labels and title
    plt.xlabel('Time (Index)')
    plt.ylabel('Class Labels')
    plt.title('Time-Series Classification Results')

    # Add a legend to show which class corresponds to which label
    handles = [plt.Line2D([0], [0], color=plt.cm.tab10(i / len(class_mapping)), lw=2) for i in range(len(class_mapping))]
    labels = [f"{label}" for label in class_mapping]
    plt.legend(handles, labels, title="Class Labels", bbox_to_anchor=(1.05, 1), loc='upper left')



    # Save the plot as an image in memory
    img = BytesIO()
    plt.tight_layout()
    plt.savefig(img, format='png')
    img.seek(0)

    # Convert the image to base64 to display in HTML
    img_base64_1 = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()  # Close the plot to avoid display on Flask server logs

    """Generate a pie chart of class distribution and return as base64."""
    class_counts = Counter(labelled_predictions)
    # Create a color palette using "tab20"
    colors = plt.cm.get_cmap('Spectral', 9)

    # Plot the pie chart
    plt.figure(figsize=(6, 6))
    plt.pie(class_counts.values(), labels=class_counts.keys(), autopct='%1.1f%%', startangle=90,  colors=colors(range(9)))
    plt.title('Class Distribution in Time-Series Data')
    plt.axis('equal')  # Equal aspect ratio ensures that pie chart is circular

    # Save the pie chart as an image in memory
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    # Convert the image to base64 to display in HTML
    img_base64_2 = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()  # Close the plot to avoid display on Flask server logs
    return img_base64_1, img_base64_2, class_counts


# """## Maintenance and Processing Eqp"""

# def load_and_preprocess_data(file_paths, loaded_label_encoder):
#   data = pd.read_csv(f"{file_paths}/manufacturing_data.csv")
#   # data = pd.read_csv(f"predictive_maintenance.csv")
#   features = ['UDI', 'Type', 'Air_temperature_K', 'Process_temperature_K', 'Rotational_speed_rpm', 'Torque_Nm', 'Tool_wear_min']
#   df = data[features]
#   df['Type'] = loaded_label_encoder.transform(df['Type'])
#   return df

def manufacturing_test(X):
  X.columns = X.columns.str.replace(r'[<>[\]]', '', regex=True)
  X.columns = X.columns.str.replace(r'\s+', '_', regex=True)

  model = f"models/Manufacturing/manufacturing_model.pkl"
  # scaler = f"{models_dir}/Manufacturing/scaler.pkl"

  # # Load the scaler
  # with open(scaler, 'rb') as file:
  #     loaded_scaler = pickle.load(file)

  # Load the nn_model
  with open(model, 'rb') as file:
      loaded_manufacturing_model = pickle.load(file)

  new_data_scaled = X
  predictions = loaded_manufacturing_model.predict(new_data_scaled)
  return predictions

def plot_predictions_manufacturing(predictions):
    # Time points (same length as model_outputs)
    time_points = np.arange(len(predictions))

    # Create a 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Time-Series Plot (Failure over time)
    # # axes[0, 0].scatter(time_points, predictions, label='Failure', color='red', marker='o', linestyle='-', markersize=4)
    # axes[0, 0].scatter(time_points, predictions, label='Failure', color='red', marker='o', linestyle='-')
    # axes[0, 0].set_title('Time-Series of Failure (1) and Non-Failure (0)')
    # axes[0, 0].set_xlabel('Time Index')
    # axes[0, 0].set_ylabel('Failure (1) / No Failure (0)')
    # axes[0, 0].set_yticks([0, 1])
    # axes[0, 0].set_yticklabels(['No Failure', 'Failure'])
    # axes[0, 0].legend()

    window_size = 200  # You can adjust this value
    rolling_avg = np.convolve(predictions, np.ones(window_size)/window_size, mode='valid')

    axes[0, 0].plot(time_points[window_size-1:], rolling_avg, label='Rolling Average of Failures', color='green', linestyle='-', marker='o', markersize=4)
    axes[0, 0].set_title('Rolling Average of Failures Over Time')
    axes[0, 0].set_xlabel('Time Index')
    axes[0, 0].set_ylabel('Rolling Average of Failure')
    axes[0, 0].legend()

    # Plot 2: Histogram of Failures vs No Failures
    axes[0, 1].hist(predictions, bins=2, color='skyblue', edgecolor='black', rwidth=0.8)
    axes[0, 1].set_title('Distribution of Failures vs No Failures')
    axes[0, 1].set_xlabel('Failure (1) / No Failure (0)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_xticks([0, 1])
    axes[0, 1].set_xticklabels(['No Failure', 'Failure'])
    axes[0, 1].grid(True)

    # Plot 3: Pie Chart of Failures vs No Failures
    failure_count = np.sum(predictions)
    no_failure_count = len(predictions) - failure_count
    labels = ['No Failure', 'Failure']
    sizes = [no_failure_count, failure_count]
    colors = ['lightgreen', 'lightcoral']

    axes[1, 0].pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    axes[1, 0].set_title('Proportion of Failures vs No Failures')
    axes[1, 0].axis('equal')  # Equal aspect ratio ensures the pie chart is circular

    # Plot 4: Cumulative Sum of Failures
    cumulative_failures = np.cumsum(predictions)

    axes[1, 1].plot(time_points, cumulative_failures, label='Cumulative Failures', color='purple')
    axes[1, 1].set_title('Cumulative Failures Over Time')
    axes[1, 1].set_xlabel('Time Index')
    axes[1, 1].set_ylabel('Cumulative Failures')
    axes[1, 1].grid(True)
    axes[1, 1].legend()

    # Adjust layout to prevent overlapping elements
    plt.tight_layout()
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    # Convert the image to base64 to display in HTML
    img_base64 = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()  # Close the plot to avoid display on Flask server logs
    return img_base64



def naval_test(X):
  model = f"models/Naval/naval_model.pkl"
  scaler = f"models/Naval/scaler.pkl"

  # Load the scaler
  with open(scaler, 'rb') as file:
      loaded_scaler = pickle.load(file)

  # Load the nn_model
  with open(model, 'rb') as file:
      loaded_naval_model = pickle.load(file)

  print(X.columns)
#   noise_std = 0.00005
#   numeric_cols = X.select_dtypes(include=[np.number]).columns
#   noisy_df = X.copy()
#   noisy_df[numeric_cols] += np.random.normal(loc=0, scale=noise_std, size=noisy_df[numeric_cols].shape)
#   augmented_df = pd.concat([X, noisy_df], ignore_index=True)

  new_data_scaled = loaded_scaler.transform(X)
  predictions = loaded_naval_model.predict(new_data_scaled)
  return predictions

def plot_predictions_naval(predictions):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, predictions.shape[0] + 1), predictions[:, 0], label='GT Compressor Decay State Coefficient')
    plt.plot(range(1, predictions.shape[0] + 1), predictions[:, 1], label='GT Turbine Decay State Coefficient')
    plt.xlabel('Datapoint Index')
    plt.ylabel('Predicted Value')
    plt.title('Predictions for Each Datapoint')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    # Convert the image to base64 to display in HTML
    img_base64 = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()  # Close the plot to avoid display on Flask server logs
    return img_base64