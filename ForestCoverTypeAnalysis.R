### MULTIVARIABLE PROBLEM ###


# LOAD PACKAGES -----------------------------------------------------------

library(tidyverse) # Data Wrangling
library(DataExplorer) # Data visualization
library(vroom) # Import data
library(tidymodels) # Modeling
library(rpart) # For random forest
library(reshape2) # To be able to melt my table
library(skimr) # for skim function
library(stacks) # For stack model


# LOAD DATA ---------------------------------------------------------------

# setwd('~/College/Stat348/ForestCoverType')
trainSet <- vroom('./train.csv')
testSet <- vroom('./test.csv')
trainSet$Cover_Type <- as.factor(trainSet$Cover_Type) # Convert Cover_Type as a factor

# # EDA ---------------------------------------------------------------------
# 
# # dplyr::glimpse(trainSet)
# # summary(trainSet)
# # skimr::skim(trainSet)
# # I have a few zero variabce predictors
# 
# # What is the Response Variable (Cover_Type with 7 categories)
# unique(trainSet$Cover_Type)
# 
# # Count per type 
# summary(trainSet$Cover_Type) # They all have the same amount 2160
# 
# # Look at distributions from 2 to 11 per category
# colnames(trainSet)[2:11]
# 
# ggplot(trainSet, aes(x = Elevation, fill = Cover_Type)) +
#   geom_density(alpha = 0.5) +
#   labs(title = "Elevation by Cover_Type")
# # Elevation is very different per type
# ggplot(trainSet, aes(x = Aspect, fill = Cover_Type)) +
#   geom_density(alpha = 0.5) +
#   labs(title = "Aspect by Cover_Type")
# # Aspect is a little different
# ggplot(trainSet, aes(x = Slope, fill = Cover_Type)) +
#   geom_density(alpha = 0.5) +
#   labs(title = "Slope by Cover_Type")
# # Slope is almost the same
# ggplot(trainSet, aes(x = Slope, fill = Cover_Type)) +
#   geom_density(alpha = 0.5) +
#   labs(title = "Slope by Cover_Type")
# 


# STACKING -----------------------------------------------------------

# Recipe
my_recipe <- recipe(Cover_Type~., data=trainSet) %>% 
  step_rm('Id') %>%
  step_zv(all_predictors()) %>%# remove all zero variance predictors
  step_normalize(all_numeric_predictors())  # normalized all numeric predictors
# glm target encoding encoding precitors

## Split data for CV
folds <- vfold_cv(trainSet, v = 5, repeats=1)

## Control Settings for Stacking models
untunedModel <- control_stack_grid()
tunedModel <- control_stack_resamples()

# RANDOM FOREST -----------------------------------------------------------

# Model
rf_mod <- rand_forest(mtry = tune(),
                      min_n=tune(),
                      trees=300) %>% #Type of model
  set_engine("ranger") %>% # What R function to use
  set_mode("classification")

## Workflow
rf_wf <- workflow() %>% 
  add_recipe(my_recipe) %>%
  add_model(rf_mod)


## Set up grid of tuning values
tuning_grid <- grid_regular(mtry(range = c(1,52)),
                            levels = 5, 
                            min_n()) # Maybe don't use levels

# Cross Validation
# rf_models <- rf_wf %>%
#   tune_grid(resamples=folds,
#             grid=tuning_grid,
#             metrics=metric_set(accuracy),
#             control=untunedModel)

load("FCT.Rdata")
# save(rf_models, file = "FCTrf.Rdata")

# BOOST TREES -------------------------------------------------------------

boost_model <- boost_tree(tree_depth=tune(),
                          trees=tune(),
                          learn_rate=tune()) %>%
  set_engine("lightgbm") %>% #or "xgboost" but lightgbm is faster
  set_mode("classification")

# Workflow
bt_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(boost_model)

# Tune
tuneGrid <- grid_regular(tree_depth(),
                            trees(),
                            learn_rate(),
                            levels = 5)

# Cross Validation
bst_models <- bt_wf %>%
  tune_grid(resamples=folds,
            grid=tuneGrid,
            metrics=metric_set(accuracy),
            control = untunedModel)

# load("FCTbst.Rdata")
save(bst_models, file = "FCTbst.Rdata")

# BACK IN THE STACKING ----------------------------------------------------

## Specify which models to include
forest_stack <- stacks() %>%
  add_candidates(rf_models) %>%
  add_candidates(bst_models) 

# Fit the stacked model
fitted_forest_stack <-  forest_stack %>%
  blend_predictions() %>%
  fit_members()

# Predict
forest_pred_stack <- fitted_forest_stack %>%
  predict(new_data = testSet)


# Format table
testSet$Cover_Type <- forest_pred_stack$.pred_class
results <- testSet %>%
  select(Id, Cover_Type)

# get csv file
vroom_write(results, 'submissions.csv', delim = ",")








