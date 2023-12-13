### MULTIVARIABLE PROBLEM ###
library(doParallel)

num_cores <- parallel::detectCores() #How many cores do I have?
cl <- makePSOCKcluster(num_cores)
registerDoParallel(cl)

# LOAD PACKAGES -----------------------------------------------------------

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
# Look at distributions from 2 to 11 per category
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
rf_mod <- rand_forest(mtry = 26,
                      min_n=2,
                      trees=300) %>% #Type of model
  set_engine("ranger") %>% # What R function to use
  set_mode("classification")

## Workflow
rf_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(rf_mod)

# Cross-validation 
rf_results <- rf_wf %>%
  fit_resamples(resamples = folds,
                metrics = metric_set(roc_auc),
                control=untunedModel)


# ## Set up grid of tuning values
# tuning_grid <- grid_regular(mtry(range = c(1,51)),
#                             levels = 5, 
#                             min_n()) # Maybe don't use levels
# 
# # Cross Validation
# rf_models <- rf_wf %>%
#   tune_grid(resamples=folds,
#             grid=tuning_grid,
#             metrics=metric_set(accuracy),
#             control=tunedModel)


# load("FCTrf.Rdata")
# save(rf_models, file = "FCTrf.Rdata")


# PENALIZED LOGISTIC REGRESSION -------------------------------------------

plg_mod <- multinom_reg(mixture=0, penalty=1) %>%
  set_engine("glmnet", family = "multinomial")

# Workflow
plg_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(plg_mod)

plg_results <- plg_wf %>%
  fit_resamples(resamples = folds,
                metrics = metric_set(roc_auc),
                control=untunedModel)

# ## Grid of values to tune over
# tuning_grid <- grid_regular(penalty(),
#                             mixture(),
#                             levels = 5) 
# 
# ## Run the CV
# plg_models <- plg_wf %>%
#   tune_grid(resamples=folds,
#           grid=tuning_grid,
#           metrics=metric_set(accuracy),
#           control=tunedModel)
# 
# 
# 
# load("FCTplg.Rdata")
# save(plg_models, file = "FCTplg.Rdata")


# BOOST TREES -------------------------------------------------------------

boost_model <- boost_tree(tree_depth=6,
                          trees=50,
                          learn_rate=0.1) %>%
  set_engine("lightgbm") %>% 
  set_mode("classification")

# Workflow
bt_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(boost_model)

bt_results <- bt_wf %>%
  fit_resamples(resamples = folds,
                metrics = metric_set(roc_auc),
                control=untunedModel)

# # Tune
# tuneGrid <- grid_regular(learn_rate(),
#                          levels = 3)
# 
# # Cross Validation
# bst_models <- bt_wf %>%
#   tune_grid(resamples=folds,
#             grid=tuneGrid,
#             metrics=metric_set(accuracy),
#             control = untunedModel)

# load("FCTbst.Rdata")
# save(bst_models, file = "FCTbst.Rdata")


# BACK IN THE STACKING ----------------------------------------------------

## Specify which models to include
forest_stack <- stacks() %>%
  add_candidates(rf_results) %>%
  add_candidates(bt_results) %>%
  add_candidates(plg_results)
  

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

stopCluster(cl)







