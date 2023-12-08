### MULTIVARIABLE PROBLEM ###


# LOAD PACKAGES -----------------------------------------------------------

library(tidyverse) # Data Wrangling
library(DataExplorer) # Data visualization
library(vroom) # Import data
library(tidymodels) # Modeling
library(rpart) # For random forest
library(reshape2) # To be able to melt my table


# LOAD DATA ---------------------------------------------------------------

# setwd('~/College/Stat348/ForestCoverType')
trainSet <- vroom('./train.csv')
testSet <- vroom('./test3.csv')
trainSet$Cover_Type <- as.factor(trainSet$Cover_Type) # Convert Cover_Type as a factor

# EDA ---------------------------------------------------------------------

dplyr::glimpse(trainSet)
#summary(trainSet)
skimr::skim(trainSet)
# I have a few zero variabce predictors

# What is the Response Variable (Cover_Type with 7 categories)
unique(trainSet$Cover_Type)

# Count per type 
summary(trainSet$Cover_Type) # They all have the same amount 2160

# Look at distributions from 2 to 11 per category
colnames(trainSet)[2:11]

ggplot(trainSet, aes(x = Elevation, fill = Cover_Type)) +
  geom_density(alpha = 0.5) +
  labs(title = "Elevation by Cover_Type")
# Elevation is very different per type
ggplot(trainSet, aes(x = Aspect, fill = Cover_Type)) +
  geom_density(alpha = 0.5) +
  labs(title = "Aspect by Cover_Type")
# Aspect is a little different
ggplot(trainSet, aes(x = Slope, fill = Cover_Type)) +
  geom_density(alpha = 0.5) +
  labs(title = "Slope by Cover_Type")
# Slope is almost the same
ggplot(trainSet, aes(x = Slope, fill = Cover_Type)) +
  geom_density(alpha = 0.5) +
  labs(title = "Slope by Cover_Type")


# RANDOM FOREST -----------------------------------------------------------


# Model
rf_mod <- rand_forest(mtry = tune(),
                      min_n=tune(),
                      trees=500) %>% #Type of model
  set_engine("ranger") %>% # What R function to use
  set_mode("classification")

## Recipe
my_recipe <- recipe(Cover_Type~., data=trainSet) %>% 
  step_rm('Id') %>%
  step_other(-all_outcomes(), threshold = 0) %>%# remove all zero variance predictors
  step_normalize(all_numeric_predictors())  # normalized all numeric predictors
  # glm target encoding encoding precitors

## Workflow
rf_wf <- workflow() %>% 
  add_recipe(my_recipe) %>%
  add_model(rf_mod)



## Set up grid of tuning values
tuning_grid <- grid_regular(mtry(range = c(1,54)),
                            min_n()) # Maybe don't use levels

## Set up K-fold CV (This usually takes sometime)
folds <- vfold_cv(trainSet, v = 5, repeats=1)

CV_results <- rf_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(accuracy))

## Find best tuning parameters
bestTune <- CV_results %>% 
  select_best("accuracy")

## Finalize workflow and predict

final_wf <- rf_wf %>% 
  finalize_workflow(bestTune) %>% 
  fit(data=trainSet)

preds <- final_wf %>%
  predict(new_data = testSet, type = 'class')

# Format table
testSet$Cover_Type <- preds$.pred_class
results <- testSet %>%
  select(Id, Coverype)

# get csv file
vroom_write(results, 'submissions.csv', delim = ",")



