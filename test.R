library(mlbench)  # Contains the Boston Housing Dataset
library(dplyr)    # Basic manipulation functions
library(ggplot2)  # Graphs and plots
library(reshape2) # To reshape data
library(caret)    # Creating train test sets

data() # View all available datasets
# Load the Boston Housing dataset in the object named 
# 'BostonHousing'
data('BostonHousing')
# For simplicity, lets call it 'housing'
housing = BostonHousing


ggplot(housing, aes(x = medv)) +
  stat_density() +
  labs(x = 'Median Value x $1000', y = 'Density', title = 'Density Plot of Median Value House Price') +
  theme_minimal()


set.seed(90)
train.idx = createDataPartition(y = housing$medv, p = 0.80, list = FALSE)
test.idx =createDataPartition(y=housing$medv, p=0.20,list=FALSE)
train = housing[train.idx, ]
test = housing[test.idx, ]


model = lm( medv ~ crim + rm + tax + lstat, data = train)


summary(model) # Obtain coefficients, Residuals and statistics
rsquare = summary(model)$r.squared # R-squared value



predictions = predict(model, newdata = test)
predicted.vs.original = data.frame(predicted = predictions, original = test$medv)   # Create a new data frame
ggplot(predicted.vs.original, aes(x = predicted, y = original)) +
  geom_point() +
  geom_smooth(color='blue') +
  labs(x = 'Predicted Values', y = 'Original Values', title = 'Predicted vs. Original Values') +
  theme_minimal()


# Decision Tree
model_decision_tree <- rpart(medv ~ crim + rm + tax + lstat, data = train)

# Predict on the test set
predictions_decision_tree <- predict(model_decision_tree, newdata = test)

# Calculate R-squared
rsquare_decision_tree <- cor(predictions_decision_tree, test$medv)^2
rsquare_decision_tree


# Gradient Boosting
model_gradient_boosting <- gbm(medv ~ crim + rm + tax + lstat, data = train, n.trees = 100, interaction.depth = 3, shrinkage = 0.1)

# Predict on the test set
predictions_gradient_boosting <- predict(model_gradient_boosting, newdata = test, n.trees = 100)

# Calculate R-squared
rsquare_gradient_boosting <- cor(predictions_gradient_boosting, test$medv)^2
rsquare_gradient_boosting

# Random Forest
model_random_forest <- randomForest(medv ~ crim + rm + tax + lstat, data = train, ntree = 100)

# Predict on the test set
predictions_random_forest <- predict(model_random_forest, newdata = test)

# Calculate R-squared
rsquare_random_forest <- cor(predictions_random_forest, test$medv)^2
rsquare_random_forest

print(c(
  Random_Forest = rsquare_random_forest,
  Gradient_Boosting = rsquare_gradient_boosting,
  Decision_Tree = rsquare_decision_tree,
  Linear_Regression = rsquare
))
