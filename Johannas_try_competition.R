# Title: "1636 Competition"
# Subtitle: "Data Science and Machine Learning 1827"
# File:    Johannas try Competition
# Project: CO2 Emissions Project
# Author:  Boulat, Kusmenko, Schmidtner, Wentges
# Date:    2023-02-08

# Load and Install Packages -----------------------------------------------

if (suppressWarnings(!require("pacman"))){
  install.packages("pacman")} 

p_load(caret)
p_load(here)
p_load(randomForest)
p_load(tidyverse)
p_load(caret)
p_load(ranger)
p_load(tuneRanger)
p_load(here)
p_load(rgdal) # needed for spTransform
p_load(rmapshaper) # needed for ms_simplfy
p_load(sf) #  encode spatial vector data
p_load(tidyverse)
p_load(zoo)
p_load(fastDummies)
library(plyr)
library(tidyverse)
library(data.table)
library(fst)
library(ranger)
library(tuneRanger)
library(xgboost)
library(caret)
library(viridis)
library(readr)
library(zoo)
library(fastDummies)

suppressWarnings(suppressMessages(library(lubridate)))  # to be able to load here function

# Load Data ---------------------------------------------------------------

# https://www.r-bloggers.com/2015/06/variable-importance-plot-and-variable-selection/

# https://r-graph-gallery.com/176-custom-choropleth-map-cartography-pkg.html


Final_testing <- read_csv("input_data/holdout_data.csv")
With_NA_training <- read_csv("input_data/trainig_test_data.csv")

## reordering

Final_testing <- Final_testing[, c(1,10,11,5,8,12,7,29,6,26,28,24,25,14,3,16,17,18,19,20,21,22,23,27,4,15,9,13,2)]
With_NA_training <- With_NA_training[, c(2,11,12,1,6,9,13,8,30,7,27,29,25,26,15,4,17,18,19,20,21,22,23,24,28,5,16,10,14,3)]

## check if there are NAs
check_NAs <- With_NA_training %>% 
  filter_all(any_vars(is.na(.)))


#  Data imputation  --------------------------------------------------

imputed_training <- With_NA_training

# linear interpolation, if NAs not at beginning or end of time series             --> this method can not be applied as we dont have a linear trend / no panel data
# https://stackoverflow.com/questions/33696795/r-interpolation-of-nas-by-group  
#final_ipol <-imputed_training %>% group_by(geo) %>% 
#  mutate(HEALTH_PRS_THS_ipol = na.approx(HEALTH_PRS_THS, na.rm = FALSE)) %>% 
#  fill(HEALTH_PRS_THS_ipol, .direction = 'down') %>% 
#  fill(HEALTH_PRS_THS_ipol, .direction = 'up')



# split up the dummies with multiple options to be binary

imputed_training <- dummy_cols(imputed_training, select_columns = "sex")
imputed_training <- dummy_cols(imputed_training, select_columns = "ethnicity")
imputed_training <- dummy_cols(imputed_training, select_columns = "race")
imputed_training <- dummy_cols(imputed_training, select_columns = "education")
imputed_training <- dummy_cols(imputed_training, select_columns = "marriage")
imputed_training <- dummy_cols(imputed_training, select_columns = "family_relation")
imputed_training <- dummy_cols(imputed_training, select_columns = "family_reference")
imputed_training <- dummy_cols(imputed_training, select_columns = "child_info")
imputed_training <- dummy_cols(imputed_training, select_columns = "status_last_week")
imputed_training <- dummy_cols(imputed_training, select_columns = "paidbyhour")
imputed_training <- dummy_cols(imputed_training, select_columns = "income_a")
imputed_training <- dummy_cols(imputed_training, select_columns = "income_b")
imputed_training <- dummy_cols(imputed_training, select_columns = "income_c")
imputed_training <- dummy_cols(imputed_training, select_columns = "income_d")
imputed_training <- dummy_cols(imputed_training, select_columns = "union")
imputed_training <- dummy_cols(imputed_training, select_columns = "industry")
imputed_training <- dummy_cols(imputed_training, select_columns = "occupation")
imputed_training <- dummy_cols(imputed_training, select_columns = "class")
imputed_training <- dummy_cols(imputed_training, select_columns = "class2")
imputed_training <- dummy_cols(imputed_training, select_columns = "veteran")
imputed_training <- dummy_cols(imputed_training, select_columns = "line")

# delete the initial collumns with the multiple options in the dummy collumn 
only_split_up_dummies <- subset(imputed_training, select = -c(sex, ethnicity, race,education,
                                                              marriage,family_relation, family_reference,
                                                              child_info, status_last_week,paidbyhour,
                                                              income_a, income_b, income_c, income_d, 
                                                              union, industry, occupation, class, class2,
                                                              veteran, line))



# ideas:
#  work hours 1 and 2; use either one if only one; take average if both and differing



# Random Forest Analysis --------------------------------------------------

## Train the model on the training dataset
final_training_subset = Final_training[,4:30] # why do we need to exclude the GEO data (at least done like this in the tutorial,.. however could als be useful, couldnt it?)

rf_1 = ranger(data = final_training_subset, dependent.variable.name = "income", importance = "impurity")



####### adapted until here, first need to do the imputation 
control <- trainControl(method = "cv", number=5)
tuning_grid <- expand.grid(mtry=seq(23,33, by=5), splitrule="variance", min.node.size= seq(1,5, by=2))
rf_caret = caret::train(data = final_training_subset , tCO2_pCAP ~ ., method = "ranger", tuneGrid = tuning_grid, trControl = control, importance="impurity")
rf_caret
rf_2 = rf_caret$finalModel

########### try different tuning grids

#mtry=c(1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,6,6,7,7,7,7,7,7,7,7,7,7,8,8,8,8,8,8,8,8,8,8,9,9,9,9,9,9,9,9,9,9,10,10,10,10,10,10,10,10,10,10,11,11,11,11,11,11,11,11,11,11,12,12,12,12,12,12,12,12,12,12,13,13,13,13,13,13,13,13,13,13,14,14,14,14,14,14,14,14,14,14,15,15,15,15,15,15,15,15,15,15,16,16,16,16,16,16,16,16,16,16,17,17,17,17,17,17,17,17,17,17,18,18,18,18,18,18,18,18,18,18,19,19,19,19,19,19,19,19,19,19,20,20,20,20,20,20,20,20,20,20,21,21,21,21,21,21,21,21,21,21,22,22,22,22,22,22,22,22,22,22,23,23,23,23,23,23,23,23,23,23,24,24,24,24,24,24,24,24,24,24,25,25,25,25,25,25,25,25,25,25,26,26,26,26,26,26,26,26,26,26,27,27,27,27,27,27,27,27,27,27,28,28,28,28,28,28,28,28,28,28,29,29,29,29,29,29,29,29,29,29,30,30,30,30,30,30,30,30,30,30,31,31,31,31,31,31,31,31,31,31,32,32,32,32,32,32,32,32,32,32,33,33,33,33,33,33,33,33,33,33,34,34,34,34,34,34,34,34,34,34,35,35,35,35,35,35,35,35,35,35,36,36,36,36,36,36,36,36,36,36,37,37,37,37,37,37,37,37,37,37)
#spr=c("V",”V”,”V”,”V”,”V”,”E”,”E”,”E”,”E”,”E”,“V”,”V”,”V”,”V”,”V”,”E”,”E”,”E”,”E”,”E”,“V”,”V”,”V”,”V”,”V”,”E”,”E”,”E”,”E”,”E”,“V”,”V”,”V”,”V”,”V”,”E”,”E”,”E”,”E”,”E”,“V”,”V”,”V”,”V”,”V”,”E”,”E”,”E”,”E”,”E”,“V”,”V”,”V”,”V”,”V”,”E”,”E”,”E”,”E”,”E”,“V”,”V”,”V”,”V”,”V”,”E”,”E”,”E”,”E”,”E”,“V”,”V”,”V”,”V”,”V”,”E”,”E”,”E”,”E”,”E”,“V”,”V”,”V”,”V”,”V”,”E”,”E”,”E”,”E”,”E”,“V”,”V”,”V”,”V”,”V”,”E”,”E”,”E”,”E”,”E”,“V”,”V”,”V”,”V”,”V”,”E”,”E”,”E”,”E”,”E”,“V”,”V”,”V”,”V”,”V”,”E”,”E”,”E”,”E”,”E”,“V”,”V”,”V”,”V”,”V”,”E”,”E”,”E”,”E”,”E”,“V”,”V”,”V”,”V”,”V”,”E”,”E”,”E”,”E”,”E”,“V”,”V”,”V”,”V”,”V”,”E”,”E”,”E”,”E”,”E”,“V”,”V”,”V”,”V”,”V”,”E”,”E”,”E”,”E”,”E”,“V”,”V”,”V”,”V”,”V”,”E”,”E”,”E”,”E”,”E”,“V”,”V”,”V”,”V”,”V”,”E”,”E”,”E”,”E”,”E”,“V”,”V”,”V”,”V”,”V”,”E”,”E”,”E”,”E”,”E”,“V”,”V”,”V”,”V”,”V”,”E”,”E”,”E”,”E”,”E”,“V”,”V”,”V”,”V”,”V”,”E”,”E”,”E”,”E”,”E”,“V”,”V”,”V”,”V”,”V”,”E”,”E”,”E”,”E”,”E”,“V”,”V”,”V”,”V”,”V”,”E”,”E”,”E”,”E”,”E”,“V”,”V”,”V”,”V”,”V”,”E”,”E”,”E”,”E”,”E”,“V”,”V”,”V”,”V”,”V”,”E”,”E”,”E”,”E”,”E”,“V”,”V”,”V”,”V”,”V”,”E”,”E”,”E”,”E”,”E”,“V”,”V”,”V”,”V”,”V”,”E”,”E”,”E”,”E”,”E”,“V”,”V”,”V”,”V”,”V”,”E”,”E”,”E”,”E”,”E”,“V”,”V”,”V”,”V”,”V”,”E”,”E”,”E”,”E”,”E”,“V”,”V”,”V”,”V”,”V”,”E”,”E”,”E”,”E”,”E”,“V”,”V”,”V”,”V”,”V”,”E”,”E”,”E”,”E”,”E”,“V”,”V”,”V”,”V”,”V”,”E”,”E”,”E”,”E”,”E”,“V”,”V”,”V”,”V”,”V”,”E”,”E”,”E”,”E”,”E”,“V”,”V”,”V”,”V”,”V”,”E”,”E”,”E”,”E”,”E”,“V”,”V”,”V”,”V”,”V”,”E”,”E”,”E”,”E”,”E”,“V”,”V”,”V”,”V”,”V”,”E”,”E”,”E”,”E”,”E”,“V”,”V”,”V”,”V”,”V”,”E”,”E”,”E”,”E”,”E”)
#mns=c(1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5)

#cbind(mtry, spr,mns)

#mtry <- c(1:37)
#splitrule <- c("variance", "extratrees")
#min.node.side <- c(1:5)
#a <- data.frame(mtry, splitrule, min.node.side)

# create more tuning grids, to get different models
# 2. higher minimimum node size to prevent model over fitting
tuning_grid_2 = expand.grid(mtry = seq(20, 30, by = 5), splitrule = "variance", min.node.size = seq(4, 8, by = 2))


# 3. lower number of minimimum node size to get a more complex model
tuning_grid_3 = expand.grid(mtry = seq(20, 30, by = 5), splitrule = "variance", min.node.size = seq(1, 3, by = 2))
rf_3 = caret::train(data = final_training_subset , tCO2_pCAP ~ ., method = "ranger", tuneGrid = tuning_grid_3, trControl = control, importance="impurity")
rf_3 = rf_3$finalModel


# 3. lower number of minimimum node size to get a more complex model
tuning_grid_4 = expand.grid(mtry = seq(20, 30, by = 5), splitrule = "variance", min.node.size = seq(1, 3, by = 2))
rf_4 = caret::train(data = final_training_subset , tCO2_pCAP ~ ., method = "ranger", tuneGrid = tuning_grid_4, trControl = control, importance="impurity")
rf_4 = rf_4$finalModel
rf_4

## best one 
tuning_grid_5 = expand.grid(mtry = seq(10, 35, by = 5), splitrule = "variance", min.node.size =1)
rf_caret_5 = caret::train(data = final_training_subset , tCO2_pCAP ~ ., method = "ranger", tuneGrid = tuning_grid_5, trControl = control, importance="impurity")
rf_caret_5
rf_5 = rf_caret_5$finalModel
rf_5

pred_rf_1 = predict(rf_1, Final_testing[,7:44])
pred_rf_2 = predict(rf_2, Final_testing[,7:44])
pred_rf_5 = predict(rf_5, Final_testing[,7:44])

combined_pred_rf_1 = cbind(Final_testing[, 1:7], pred_rf_1$predictions)
combined_pred_rf_2 = cbind(Final_testing[, 1:7], pred_rf_2$predictions)
combined_pred_rf_5 = cbind(Final_testing[, 1:7], pred_rf_5$predictions)

combined_pred_rf_1 <- combined_pred_rf_1[, c(1, 2, 7, 8, 4)]
combined_pred_rf_2 <- combined_pred_rf_2[, c(1, 2, 7, 8, 4)]
combined_pred_rf_5 <- combined_pred_rf_5[, c(1, 2, 7, 8, 4)]

#didn't work (from the homework)
#names(combined_pred_rf_1) = c(names(Final_testing[1:4]), "rf_1")
#names(combined_pred_rf_2) = c(names(Final_testing[1:4]), "rf_2")


###### prediction error and variable importance for rf_2 

rf_2$prediction.error

importance_rf_2 = data.frame(importance(rf_2)[order(importance(rf_2), decreasing=TRUE)])

# adjust the row and column names of the data set
names(importance_rf_2) = "importance"
importance_rf_2$var_name = rownames(importance_rf_2)

# select 10 most important covariates
importance_rf_2 = importance_rf_2[1:10,]

# plot variable importance scores
ggplot(importance_rf_2, aes(x = reorder(var_name, importance, mean), y = importance)) +
  geom_point() +
  labs(title = "Random Forest variable importance for C02 emissions", subtitle = "Training between 2016 and 2018 to predict 2019", x = "", y = "Mean decrease in sum of squared residuals \nresiduals when a variable is included in a tree split") +
  coord_flip() +
  theme(axis.text.y = element_text(hjust = 0))

###### prediction error and variable importance for rf_5 

rf_5$prediction.error

importance_rf_5 = data.frame(importance(rf_5)[order(importance(rf_5), decreasing=TRUE)])

# adjust the row and column names of the data set
names(importance_rf_5) = "importance"
importance_rf_5$var_name = rownames(importance_rf_5)

# select 10 most important covariates
importance_rf_5 = importance_rf_5[1:10,]

# plot variable importance scores
ggplot(importance_rf_5, aes(x = reorder(var_name, importance, mean), y = importance)) +
  geom_point() +
  labs(title = "Random Forest variable  importance for C02 emissions", subtitle = "Training between 2016 and 2018 to predict 2019", x = "", y = "Mean decrease in sum of squared residuals \nresiduals when a variable is included in a tree split") +
  coord_flip() +
  theme(axis.text.y = element_text(hjust = 0))


# save rf model output

#saveRDS(combined_pred_rf_2, file = "combined_pred_rf_2.Rds")   # Johanna (08.02.2023): i deactivated those, as i cannot find the saved data anywhere, but included a new saving structure 
#saveRDS(combined_pred_rf_5, file = "combined_pred_rf_5.Rds")

#  Johanna (08.02.2023): we should do it like this, but somehow there are only error messages,.. 
#library(conflicted)
#conflicts_prefer(here::here)
#saveRDS(combined_pred_rf_1, file = here("data", "03_output_data", "Model_output", "rf", "combined_pred_rf_1.rds"), compress = TRUE)
#saveRDS(combined_pred_rf_2, file = here("data", "03_output_data", "Model_output", "rf", "combined_pred_rf_2.rds"), compress = TRUE)
#saveRDS(combined_pred_rf_5, file = here("data", "03_output_data", "Model_output", "rf", "combined_pred_rf_5.rds"), compress = TRUE)

#  set WD manually!!!!!!!  and save 
saveRDS(combined_pred_rf_1, "combined_pred_rf_1.rds")
saveRDS(combined_pred_rf_2, "combined_pred_rf_2.rds")
saveRDS(combined_pred_rf_5, "combined_pred_rf_5.rds")


################# XGB

control = trainControl(method = "cv", number = 5)
tuning_grid_xbg = expand.grid(nrounds = seq(50, 150, by = 50), max_depth = 6, eta = seq(0.2, 0.4, by = 0.1), gamma = seq(0.01, 0.03, by = 0.01), colsample_bytree = 1, min_child_weight = 1, subsample = 1)
gb_caret = caret::train(data = final_training_subset, tCO2_pCAP ~ ., method = "xgbTree", trControl = control, tuneGrid = tuning_grid_xbg, verbosity = 0)
gb_1 <- gb_caret$finalModel
gb_1

tuning_grid_xbg_2 = expand.grid(nrounds = seq(75, 175, by = 25), max_depth = 5, eta = seq(0.15, 0.35, by = 0.1), gamma = seq(0.01, 0.04, by = 0.01), colsample_bytree = 1, min_child_weight = 1, subsample = 1)
gb_caret_2 = caret::train(data = final_training_subset, tCO2_pCAP ~ ., method = "xgbTree", trControl = control, tuneGrid = tuning_grid_xbg_2, verbosity = 0)
gb_caret_2
gb_2 <- gb_caret_2$finalModel
gb_2

pred_gb_2 = predict(gb_2, newdata = as.matrix(Final_testing[,8:44]))
combined_pred_gb_2 = cbind(Final_testing[,1:8], pred_gb_2)
combined_pred_gb_2 <- combined_pred_gb_2[, c(1, 2, 4, 7, 9)]

###### variable importance for gb_2

xgb.ggplot.importance(xgb.importance(model = gb_2)[1:10,])

#### save xgb model output 
#  set WD manually!!!!!!!

saveRDS(combined_pred_gb_2, file = "combined_pred_gb_2.rds")


# Clean Up ----------------------------------------------------------------

# Clear environment
rm(list = ls()) 

# Clear packages
# p_unload(all)  # Remove all add-ons
#detach("package:datasets", unload = TRUE)  # For base

# Garbage collection
gc()

# Clear console
cat("\014")  # ctrl+L

