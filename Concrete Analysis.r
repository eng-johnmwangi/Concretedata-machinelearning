# Load necessary libraries
library(tidyverse)     # For data manipulation and visualization
library(caret)         # For machine learning and preprocessing
library(randomForest)  # For Random Forest
library(gbm)           # For Gradient Boosting Machine (GBM)

# Load the dataset
file_path <- "C:/Users/ENG. JMK/OneDrive/Desktop/CONCRETE DATA/Concrete_Data_Yeh.csv"
data <- read.csv(file_path)

# Inspect the data
glimpse(data)
summary(data)

# Check for missing values
colSums(is.na(data))

# Basic statistics and correlations
cor_matrix <- cor(data)
print(cor_matrix)

# Visualize the correlation matrix
library(ggcorrplot)
ggcorrplot(cor_matrix, hc.order = TRUE, type = "lower", lab = TRUE)

# Pairwise plots to explore feature relationships
pairs(data, col = "blue", pch = 20)

# Train-test split
set.seed(42)
train_index <- createDataPartition(data$csMPa, p = 0.8, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# Prepare features and target
train_x <- train_data %>% select(-csMPa)
train_y <- train_data$csMPa
test_x <- test_data %>% select(-csMPa)
test_y <- test_data$csMPa

# Train Gradient Boosting Machine (GBM)
set.seed(42)
gbm_model <- gbm(
  formula = csMPa ~ .,
  data = train_data,
  distribution = "gaussian",
  n.trees = 1000,
  interaction.depth = 4,
  shrinkage = 0.01,
  cv.folds = 5,
  verbose = FALSE
)

# Optimal number of trees
best_trees <- gbm.perf(gbm_model, method = "cv")
print(best_trees)

# Make predictions
predictions <- predict(gbm_model, test_x, n.trees = best_trees)

# Evaluate the model
mse <- mean((test_y - predictions)^2)
rmse <- sqrt(mse)
r2 <- 1 - (sum((test_y - predictions)^2) / sum((test_y - mean(test_y))^2))

cat("Mean Squared Error (MSE):", mse, "\n")
cat("Root Mean Squared Error (RMSE):", rmse, "\n")
cat("R² Score:", r2, "\n")

# Feature Importance
summary(gbm_model)

# Visualize feature importance
importance <- summary(gbm_model, plotit = FALSE)
ggplot(importance, aes(x = reorder(var, rel.inf), y = rel.inf)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Feature Importance (GBM)", x = "Features", y = "Relative Importance")
