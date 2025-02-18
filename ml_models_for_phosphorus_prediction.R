# Random Forest
library(randomForest)  # Loads the library for Random Forest, used to create the model

# Example with Abygayle's data: Predict Modified Morgan P from Mehlich-3 P
set.seed(123)  # Ensures reproducible results
# What does this mean? When the computer generates random numbers, it starts from an initial point called a seed.
# Setting set.seed(123) ensures the computer always starts this sequence of numbers from the same point, guaranteeing identical results whenever the code is run again.

data <- read.csv("your_file.csv")  # Replace with the correct file path
# What is a CSV file? It's a file where data is organized in columns separated by commas, similar to an Excel table.

# Train a Random Forest model to predict Modified Morgan P
rf_model <- randomForest(Modified_Morgan_P ~ Mehlich3_P + ., data = data, ntree = 500)  # The model learns using 500 decision trees
print(rf_model)  # Displays the trained model with basic metrics

# What is Random Forest? It is a model that creates multiple decision trees and then combines their results to make more accurate predictions.
# Usage instructions:
# - Install the library if needed: install.packages("randomForest")
# - Replace 'Modified_Morgan_P' with the target variable and 'Mehlich3_P' with the predictor variables
# - Add or remove columns as needed
# - The ntree parameter controls the number of trees in the forest

# Causal Forest
library(grf)  # Loads the library for causal forest models

# Prepare data for the causal model
set.seed(42)  # Ensures reproducible results
# Just like the previous set.seed, this line ensures the model will always yield the same results when run again.
X <- as.matrix(data[, c("Mehlich3_P", "other_variables")])  # Select the correct columns
Y <- data$Modified_Morgan_P  # Target variable
W <- rep(1, nrow(data))  # If no specific treatment is provided
c_forest <- causal_forest(X, Y, W)  # Train the causal forest model
preds <- predict(c_forest)$predictions  # Make predictions

# What is Causal Forest? It is a model used to understand the impact of one variable (treatment) on another (outcome).
# Usage instructions:
# - Replace 'Mehlich3_P' and other variables with the correct columns from your dataset
# - The variable W should indicate a treatment if one exists; otherwise, use 1

# XGBoost
library(xgboost)  # Loads the library for the XGBoost model

# Prepare the data for the model
labels <- data$Modified_Morgan_P  # Define the target variable
data_matrix <- xgb.DMatrix(data = as.matrix(data[, c("Mehlich3_P", "other_variables")]), label = labels)
params <- list(objective = "reg:squarederror", booster = "gbtree")
xgb_model <- xgb.train(params = params, data = data_matrix, nrounds = 100)  # Train the model with 100 iterations

# What is XGBoost? It is a tree-based model that learns from previous errors to improve prediction accuracy.
# Usage instructions:
# - Replace the columns in xgb.DMatrix with the appropriate ones from your dataset
# - Adjust the nrounds parameter for more or fewer iterations

# Neural Network (Multi-Layer Perceptron - MLP)
library(nnet)  # Loads the library for simple neural networks

# Train a neural network to predict Modified Morgan P
nn_model <- nnet(Modified_Morgan_P ~ Mehlich3_P + ., data = data, size = 5, linout = TRUE)  # Neural network with one hidden layer and 5 neurons
summary(nn_model)

# What is a Neural Network? It is a model inspired by the human brain that learns from data by adjusting internal connections called weights.
# This code uses a Multi-Layer Perceptron (MLP) architecture, a common type of neural network for regression tasks.
# Usage instructions:
# - Adjust the columns to match your dataset
# - The size parameter defines the number of neurons in the hidden layer

# Example prediction with new data
new_data <- data[1:5, c("Mehlich3_P", "other_variables")]  # Select the correct columns
predict(nn_model, new_data)  # Make predictions with the trained model

# What is a prediction? The model uses the learned pattern to estimate the target variable's value (Modified Morgan P) based on the provided values.

# General notes:
# - Make sure the data file is loaded correctly.
# - Replace 'Modified_Morgan_P' and 'Mehlich3_P' with the exact column names from your dataset.
# - The code uses the requested models to predict Modified Morgan P from Mehlich-3 P and other provided variables.
# - The neural network used here is a Multi-Layer Perceptron (MLP) implemented via the 'nnet' package.
# - The concepts have been explained in simple terms to help those unfamiliar with machine learning understand better.
