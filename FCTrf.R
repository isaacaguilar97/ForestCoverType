library(tidyverse) # Data Wrangling
library(DataExplorer) # Data visualization
library(vroom) # Import data
library(tidymodels) # Modeling
library(rpart) # For random forest
library(reshape2) # To be able to melt my table
library(skimr) # for skim function
library(stacks) # For stack model
library(lightgbm) # For xgboost model
library(bonsai) # For xgboost model


# LOAD DATA ---------------------------------------------------------------

# setwd('~/College/Stat348/ForestCoverType')
trainSet <- vroom('./train.csv')
testSet <- vroom('./test.csv')
trainSet$Cover_Type <- as.factor(trainSet$Cover_Type) # Convert Cover_Type as a factor

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
rf_models <- rf_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(accuracy),
            control=untunedModel)

# load("FCT.Rdata")
save(rf_models, file = "FCTrf.Rdata")


