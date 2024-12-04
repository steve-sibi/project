# Implementing Privacy-Preserving Techniques in Machine Learning Using Differential Privacy

This project demonstrates the implementation of differential privacy in machine learning using TensorFlow Privacy. It includes a comparison between standard logistic regression and differentially private logistic regression models using the Iris dataset.

## Prerequisites on Google Colab

```bash
pip install tensorflow-privacy
pip install --upgrade tensorflow-estimator
pip install --upgrade tensorflow==2.14.0
```

Additional requirements:
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- tensorflow

## Features

- Implementation of both standard and differentially private logistic regression
- Comprehensive privacy analysis and visualization
- Membership inference attack evaluation
- Privacy-utility trade-off analysis
- Multiple noise multiplier configurations
- Detailed performance metrics and comparisons

## Key Components

1. **Data Preprocessing**
   - Standardization of features
   - Train-test split
   - Binary classification conversion

2. **Model Implementation**
   - Baseline logistic regression model
   - DP-SGD (Differentially Private Stochastic Gradient Descent) model
   - Configurable privacy parameters

3. **Privacy Analysis**
   - Epsilon calculation for different noise multipliers
   - Membership inference attack implementation
   - Privacy budget analysis

4. **Visualization**
   - Training history plots
   - Confusion matrices
   - Prediction distribution comparisons
   - Privacy-utility trade-off plots

## Usage

1. **Load and Preprocess Data**
```python
X_train, y_train, X_test, y_test, input_dim = load_and_preprocess_data()
```

2. **Train Baseline Model**
```python
baseline_model = create_logistic_regression_model(input_dim)
baseline_history = baseline_model.fit(train_dataset, epochs=epochs, validation_data=val_dataset)
```

3. **Train DP Model**
```python
dp_model = create_dp_logistic_regression_model(
    input_dim,
    l2_norm_clip,
    noise_multiplier,
    microbatches,
    learning_rate
)
dp_history = dp_model.fit(train_dataset_dp, epochs=epochs, validation_data=val_dataset_dp)
```

## Key Functions

- `create_styled_plots()`: Sets consistent style for visualizations
- `load_and_preprocess_data()`: Handles data preparation
- `create_logistic_regression_model()`: Creates baseline model
- `create_dp_logistic_regression_model()`: Creates DP model
- `membership_inference_attack()`: Implements privacy attack testing
- `plot_privacy_utility_tradeoff()`: Visualizes privacy-utility balance

## Privacy Parameters

The implementation includes several configurable privacy parameters:
- `l2_norm_clip`: Gradient clipping norm
- `noise_multiplier`: Amount of noise added for privacy
- `microbatches`: Number of microbatches for gradient computation
- `learning_rate`: Model learning rate
- `delta`: Privacy failure probability

## Results Analysis

The code provides comprehensive analysis including:
- Model accuracy comparisons
- Training time measurements
- Attack effectiveness metrics
- Privacy budget calculations
- Visualization of trade-offs between privacy and utility

## Output

The implementation generates:
1. Training history visualizations
2. Confusion matrices
3. Prediction distribution plots
4. Privacy-utility trade-off analysis
5. Detailed results table with:
   - Noise multiplier values
   - Epsilon values
   - Model accuracies
   - Attack AUC scores
   - Training times

## Notes

- The implementation uses the Iris dataset converted to a binary classification problem
- Early stopping is implemented to prevent overfitting
- Memory management is handled through garbage collection
- The privacy budget (ε) is computed using TensorFlow Privacy's analysis tools
