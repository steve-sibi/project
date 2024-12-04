## Recommended to Run on Google Colab since it is computationally expensive otherwise
# Install these packages beforehand
# !pip install tensorflow-privacy
# !pip install --upgrade tensorflow-estimator
# !pip install --upgrade tensorflow==2.14.0

import gc
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow_privacy
from sklearn.metrics import auc, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Suppress warnings
warnings.filterwarnings("ignore")

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def create_styled_plots():
    """Set consistent style for all plots"""
    plt.style.use('default')  # Using default matplotlib style instead of seaborn
    plt.rcParams['figure.figsize'] = [10, 6]
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    # Set color cycle for consistent colors across plots
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])

def load_and_preprocess_data(sample_fraction=1.0):
    """
    Load and preprocess the Iris dataset.

    Args:
        sample_fraction (float): Fraction of the dataset to use.

    Returns:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        X_test (np.ndarray): Testing features.
        y_test (np.ndarray): Testing labels.
        input_dim (int): Number of features.
    """
    from sklearn.datasets import load_iris

    iris = load_iris()
    X = iris.data
    y = iris.target

    # Convert to binary classification problem (Class 0 vs Classes 1 & 2)
    y = (y == 0).astype(np.float32)  # Class 0 as positive class

    # Optionally sample a fraction of the data
    if sample_fraction < 1.0:
        np.random.seed(42)
        indices = np.random.choice(len(X), int(len(X) * sample_fraction), replace=False)
        X = X[indices]
        y = y[indices]

    # Feature scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    input_dim = X_train.shape[1]

    return (
        X_train.astype(np.float32),
        y_train.astype(np.float32),
        X_test.astype(np.float32),
        y_test.astype(np.float32),
        input_dim,
    )

def create_logistic_regression_model(input_dim):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(1, activation="sigmoid", input_shape=(input_dim,))
    ])
    # Add more specific optimizer parameters
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    return model

def create_dp_logistic_regression_model(
    input_dim, l2_norm_clip, noise_multiplier, microbatches, learning_rate
):
    """
    Create a logistic regression model with differential privacy.

    Args:
        input_dim (int): Number of input features.
        l2_norm_clip (float): Clipping norm.
        noise_multiplier (float): Noise multiplier for DP-SGD.
        microbatches (int): Number of microbatches.
        learning_rate (float): Learning rate.

    Returns:
        model (tf.keras.Model): Compiled DP logistic regression model.
    """
    optimizer = tensorflow_privacy.DPKerasSGDOptimizer(
        l2_norm_clip=l2_norm_clip,
        noise_multiplier=noise_multiplier,
        num_microbatches=microbatches,
        learning_rate=learning_rate,
    )
    loss = tf.keras.losses.BinaryCrossentropy(
        from_logits=False, reduction=tf.losses.Reduction.NONE
    )

    model = tf.keras.models.Sequential(
        [tf.keras.layers.Dense(1, activation="sigmoid", input_shape=(input_dim,))]
    )
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    return model

def plot_training_history(history, title):
    """
    Plot training and validation accuracy over epochs.

    Args:
        history: Training history object.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_confusion_matrices(baseline_model, dp_model, X_test, y_test, noise_multiplier):
    """Plot confusion matrices for both models side by side"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Baseline model predictions
    y_pred_baseline = (baseline_model.predict(X_test) > 0.5).astype(int)
    cm_baseline = confusion_matrix(y_test, y_pred_baseline)
    
    # DP model predictions
    y_pred_dp = (dp_model.predict(X_test) > 0.5).astype(int)
    cm_dp = confusion_matrix(y_test, y_pred_dp)
    
    # Plot matrices
    sns.heatmap(cm_baseline, annot=True, fmt='d', ax=ax1, cmap='Blues')
    ax1.set_title('Baseline Model\nConfusion Matrix')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    
    sns.heatmap(cm_dp, annot=True, fmt='d', ax=ax2, cmap='Blues')
    ax2.set_title(f'DP Model (noise={noise_multiplier})\nConfusion Matrix')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    
    plt.tight_layout()
    plt.show()

def plot_prediction_distributions(baseline_model, dp_model, X_test, noise_multiplier):
    """Plot prediction probability distributions for both models"""
    baseline_preds = baseline_model.predict(X_test).flatten()
    dp_preds = dp_model.predict(X_test).flatten()
    
    plt.figure(figsize=(12, 6))
    sns.kdeplot(baseline_preds, label='Baseline Model', fill=True, alpha=0.3)
    sns.kdeplot(dp_preds, label=f'DP Model (noise={noise_multiplier})', fill=True, alpha=0.3)
    plt.title('Prediction Probability Distributions')
    plt.xlabel('Prediction Probability')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.show()

def membership_inference_attack(model, X_train, y_train, X_test, y_test):
    """
    Perform a membership inference attack on the model.

    Args:
        model (tf.keras.Model): Trained model.
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        X_test (np.ndarray): Testing features.
        y_test (np.ndarray): Testing labels.

    Returns:
        attack_auc (float): Attack AUC score.
    """
    # Get model predictions
    train_preds = model.predict(X_train, batch_size=64)
    test_preds = model.predict(X_test, batch_size=64)

    # Compute per-sample losses
    loss_fn = tf.keras.losses.BinaryCrossentropy(
        from_logits=False, reduction=tf.losses.Reduction.NONE
    )
    train_losses = loss_fn(y_train, train_preds).numpy()
    test_losses = loss_fn(y_test, test_preds).numpy()

    # Prepare data for attack model
    attack_X = np.concatenate([train_losses, test_losses])
    attack_y = np.concatenate(
        [np.ones(len(train_losses)), np.zeros(len(test_losses))]
    )

    # Reshape data
    attack_X = attack_X.reshape(-1, 1)

    # Train attack model
    attack_model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(8, activation="relu", input_shape=(1,)),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    attack_model.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
    )
    attack_model.fit(attack_X, attack_y, epochs=50, batch_size=16, verbose=0)

    # Evaluate attack model
    attack_preds = attack_model.predict(attack_X)
    fpr, tpr, thresholds = roc_curve(attack_y, attack_preds)
    attack_auc = auc(fpr, tpr)

    # Clear variables to save memory
    del train_preds, test_preds, train_losses, test_losses, attack_X, attack_y, attack_model
    gc.collect()

    return attack_auc

def create_results_table(results_df, baseline_results):
    """Create and display a styled results table"""
    # Create styled DataFrame
    styled_df = results_df.style.set_properties(**{
        'text-align': 'center',
        'padding': '10px'
    }).set_table_styles([{
        'selector': 'th',
        'props': [('text-align', 'center'), ('background-color', '#f2f2f2')]
    }])
    
    # Add baseline results for comparison
    baseline_row = pd.DataFrame({
        'Noise Multiplier': ['Baseline'],
        'Epsilon': ['∞'],
        'DP Model Accuracy': [f"{baseline_results['accuracy']:.2f}%"],
        'Attack AUC': [f"{baseline_results['attack_auc']:.4f}"],
        'Training Time (s)': [f"{baseline_results['training_time']:.2f}s"]
    })
    
    # Combine and display
    full_results = pd.concat([baseline_row, results_df]).reset_index(drop=True)
    return full_results

def plot_privacy_utility_tradeoff(valid_epsilons, valid_dp_accuracies, baseline_accuracy):
    """Create an enhanced privacy-utility trade-off plot"""
    plt.figure(figsize=(10, 6))
    
    # Plot points and line
    plt.plot(valid_epsilons, valid_dp_accuracies, 'o-', linewidth=2, markersize=8)
    
    # Add baseline
    plt.axhline(y=baseline_accuracy, color='r', linestyle='--', label='Baseline Accuracy')
    
    # Customize plot
    plt.xlabel('Privacy Budget (ε)')
    plt.ylabel('Model Accuracy')
    plt.title('Privacy-Utility Trade-off')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add annotations
    for i, (eps, acc) in enumerate(zip(valid_epsilons, valid_dp_accuracies)):
        plt.annotate(f'ε={eps:.2f}\nacc={acc:.2%}', 
                    (eps, acc), 
                    xytext=(10, 10),
                    textcoords='offset points')
    
    plt.show()

def main():
    # Set plot style
    create_styled_plots()

    # Load and preprocess data
    X_train, y_train, X_test, y_test, input_dim = load_and_preprocess_data()

    # Privacy parameters
    l2_norm_clip = 1.0
    noise_multipliers = [0.5, 1.0, 1.5, 2.0]
    microbatches = 1  # Set microbatches to 1 since dataset is small
    learning_rate = 0.05
    epochs = 50
    batch_size = 16  # Adjust batch size as needed

    # Create tf.data.Dataset for training and validation
    train_dataset = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .shuffle(buffer_size=128)
        .batch(batch_size, drop_remainder=False)
    )
    val_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(
        batch_size
    )

    # Train baseline model without DP
    print("Training baseline model without differential privacy...")
    baseline_model = create_logistic_regression_model(input_dim)
    start_time = time.time()
    baseline_history = baseline_model.fit(
      train_dataset, 
      epochs=epochs, 
      validation_data=val_dataset, 
      verbose=0,
      callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
    )
    baseline_training_time = time.time() - start_time
    print(f"Baseline model training time: {baseline_training_time:.2f} seconds")

    # Evaluate baseline model
    baseline_eval = baseline_model.evaluate(val_dataset, verbose=0)
    print(f"Baseline model accuracy: {baseline_eval[1] * 100:.2f}%")

    # Plot baseline training history
    plot_training_history(baseline_history, "Baseline Model Training History")

    # Membership inference attack on baseline model
    print("\nPerforming membership inference attack on baseline model...")
    baseline_attack_auc = membership_inference_attack(
        baseline_model, X_train, y_train, X_test, y_test
    )
    print(
        f"Baseline model Membership Inference Attack AUC: {baseline_attack_auc:.4f}"
    )

    # Clear variables to save memory
    gc.collect()

    # Train models with differential privacy
    dp_accuracies = []
    dp_attack_aucs = []
    dp_training_times = []
    epsilons = []

    # Import the compute_dp_sgd_privacy function
    from tensorflow_privacy.privacy.analysis.compute_dp_sgd_privacy_lib import \
        compute_dp_sgd_privacy

    for noise_multiplier in noise_multipliers:
        print(
            f"\nTraining DP model with noise_multiplier = {noise_multiplier}..."
        )
        dp_model = create_dp_logistic_regression_model(
            input_dim,
            l2_norm_clip,
            noise_multiplier,
            microbatches,
            learning_rate,
        )

        # Create datasets
        train_dataset_dp = train_dataset  # Use the same dataset
        val_dataset_dp = val_dataset

        start_time = time.time()
        dp_history = dp_model.fit(
            train_dataset_dp,
            epochs=epochs,
            validation_data=val_dataset_dp,
            verbose=0,
        )
        training_time = time.time() - start_time
        dp_eval = dp_model.evaluate(val_dataset_dp, verbose=0)
        dp_accuracy = dp_eval[1]

        print(f"DP model accuracy: {dp_accuracy * 100:.2f}%")
        print(f"DP model training time: {training_time:.2f} seconds")

        dp_accuracies.append(dp_accuracy)
        dp_training_times.append(training_time)

        # Compute privacy budget (epsilon)
        delta = 1e-5

        try:
            eps, _ = compute_dp_sgd_privacy(
                n=len(y_train),
                batch_size=batch_size,
                noise_multiplier=noise_multiplier,
                epochs=epochs,
                delta=delta,
            )
            epsilons.append(eps)
            print(
                f"DP-SGD with noise_multiplier = {noise_multiplier} achieves (ε = {eps:.2f}, δ = {delta})-DP"
            )
        except Exception as e:
            print(
                f"Could not compute epsilon for noise_multiplier={noise_multiplier}: {e}"
            )
            epsilons.append(None)

        # Membership inference attack
        attack_auc = membership_inference_attack(
            dp_model, X_train, y_train, X_test, y_test
        )
        dp_attack_aucs.append(attack_auc)
        print(f"Membership Inference Attack AUC: {attack_auc:.4f}")

        # Plot DP model training history
        plot_training_history(
            dp_history,
            f"DP Model Training History (noise_multiplier={noise_multiplier})",
        )

        # Add confusion matrices and prediction distributions
        plot_confusion_matrices(baseline_model, dp_model, X_test, y_test, noise_multiplier)
        plot_prediction_distributions(baseline_model, dp_model, X_test, noise_multiplier)

        # Example predictions
        print(f"\nPrediction Examples (noise_multiplier={noise_multiplier}):")
        sample_indices = np.random.choice(len(X_test), 5)
        for idx in sample_indices:
            baseline_pred = baseline_model.predict(X_test[idx:idx+1])[0][0]
            dp_pred = dp_model.predict(X_test[idx:idx+1])[0][0]
            actual = y_test[idx]
            print(f"Sample {idx}:")
            print(f"  Actual: {actual}")
            print(f"  Baseline prediction: {baseline_pred:.4f}")
            print(f"  DP prediction: {dp_pred:.4f}")
            print(f"  Prediction difference: {abs(baseline_pred - dp_pred):.4f}")
            print()

        # Clear variables to save memory
        del dp_model, dp_history
        gc.collect()

    # Filter out None values from epsilons
    valid_indices = [i for i, eps in enumerate(epsilons) if eps is not None]
    valid_epsilons = [epsilons[i] for i in valid_indices]
    valid_dp_accuracies = [dp_accuracies[i] for i in valid_indices]
    valid_dp_attack_aucs = [dp_attack_aucs[i] for i in valid_indices]

    if valid_epsilons:
        # Plot Privacy-Utility Trade-off
        plt.figure(figsize=(10, 6))
        plt.plot(
            valid_epsilons,
            valid_dp_accuracies,
            marker="o",
            label="DP Model Accuracy",
        )
        plt.hlines(
            baseline_eval[1],
            xmin=min(valid_epsilons),
            xmax=max(valid_epsilons),
            colors="r",
            linestyles="dashed",
            label="Baseline Model Accuracy",
        )
        plt.xlabel("Epsilon (Privacy Budget)")
        plt.ylabel("Accuracy")
        plt.title("Accuracy vs. Epsilon (Privacy Budget)")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Plot Attack AUC vs Epsilon
        plt.figure(figsize=(10, 6))
        plt.plot(
            valid_epsilons,
            valid_dp_attack_aucs,
            marker="o",
            label="DP Model Attack AUC",
        )
        plt.hlines(
            baseline_attack_auc,
            xmin=min(valid_epsilons),
            xmax=max(valid_epsilons),
            colors="r",
            linestyles="dashed",
            label="Baseline Model Attack AUC",
        )
        plt.xlabel("Epsilon (Privacy Budget)")
        plt.ylabel("Attack AUC")
        plt.title("Attack AUC vs. Epsilon (Privacy Budget)")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Enhanced privacy-utility trade-off plot
        plot_privacy_utility_tradeoff(valid_epsilons, valid_dp_accuracies, baseline_eval[1])
    else:
        print("No valid epsilon values were computed; skipping plots involving epsilon.")

    # Privacy-Utility Trade-off Analysis
    print("\nPrivacy-Utility Trade-off Analysis:")
    print(
        "As the privacy budget (epsilon) decreases (stronger privacy), model accuracy may decrease."
    )
    print(
        "Similarly, the membership inference attack AUC decreases, indicating better privacy protection."
    )

    # Display Results in a Table
    results_df = pd.DataFrame(
        {
            "Noise Multiplier": noise_multipliers,
            "Epsilon": epsilons,
            "DP Model Accuracy": [acc * 100 for acc in dp_accuracies],
            "Attack AUC": dp_attack_aucs,
            "Training Time (s)": dp_training_times,
        }
    )
    results_df["DP Model Accuracy"] = results_df["DP Model Accuracy"].apply(
        lambda x: f"{x:.2f}%"
    )
    results_df["Attack AUC"] = results_df["Attack AUC"].apply(
        lambda x: f"{x:.4f}"
    )
    results_df["Training Time (s)"] = results_df["Training Time (s)"].apply(
        lambda x: f"{x:.2f}s"
    )
    results_df["Epsilon"] = results_df["Epsilon"].apply(
        lambda x: f"{x:.2f}" if x is not None else "N/A"
    )

    # Create enhanced results table
    baseline_results = {
        'accuracy': baseline_eval[1] * 100,
        'attack_auc': baseline_attack_auc,
        'training_time': baseline_training_time
    }
    full_results = create_results_table(results_df, baseline_results)
    
    print("\nComplete Results Comparison:")
    print(full_results.to_string(index=False))

    # Baseline results
    print("\nBaseline Model:")
    print(f"Accuracy: {baseline_eval[1] * 100:.2f}%")
    print(f"Attack AUC: {baseline_attack_auc:.4f}")
    print(f"Training Time: {baseline_training_time:.2f}s")

if __name__ == "__main__":
    main()