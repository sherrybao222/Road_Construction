library("broom")
library("ramify")
### Data loading 
numParticipants = 9
numPuzzle = 92
numCities <- read.csv('numCities.csv',header = FALSE)
mas <- read.csv('mas.csv',header = FALSE)
nos <- read.csv('nos.csv',header = FALSE)
undo_c <- read.csv('undo_c.csv',header = FALSE)
leftover <- read.csv('leftover.csv',header = FALSE)
numError <- read.csv('numError.csv',header = FALSE)
sumSeverityErrors <- read.csv('sumSeverityErrors.csv',header = FALSE)
numUNDO <- read.csv('numUNDO.csv',header = FALSE)
TT <- read.csv('TT.csv',header = FALSE)

### make dirs
path_root <- 'puzzle-level'
n_numCities <- 'numCities'
n_mas <- 'mas'
n_nos <- 'nos'
n_undo_c <- 'undo_c'
n_leftover <- 'leftover'
n_numError <- 'numError'
n_sumSeverityErrors <- 'sumSeverityErrors'
n_numUNDO <- 'numUNDO'
n_TT <- 'TT'

### Analysis
### Puzzle-level
# Predicting the number of connected cities
# Model: Number of connected cities(puzzle, subject) = intercept(random effect of subject) (1 | subject) + w1 * MAS(puzzle) + w2 *number of optimal solutions (puzzle) + w3 * UNDO (puzzle) + 3 interaction
Betas = data.frame()
models = list()
for (i in 1:numParticipants){
  data_ <- data.frame(MAS = mas[1:numPuzzle,i], NOS = nos[1:numPuzzle,i], UNDO = undo_c[1:numPuzzle,i])
  fit1 <- glm(numCities[1:numPuzzle,i] ~ .^2, data=data_, family = poisson())
  write.csv( tidy( fit1 ) , paste(path_root, n_numCities, paste("fit_Sub", i, '.csv', sep=""), sep='/') )
  sink(paste(path_root, n_numCities, paste("fit_Sub", i, '.txt', sep=""), sep='/'))
  print(summary(fit1))
  sink()  # returns output to the console
  models[i] <- fit1
  
  if (i == 1) {
    one_data_  = data.frame(MAS = mas[1:numPuzzle,i], NOS = nos[1:numPuzzle,i], UNDO = undo_c[1:numPuzzle,i])
    one_numCities = data.frame(numCities[1:numPuzzle,i])
  } else{
    one_data_<-rbind(one_data_,data.frame(MAS = mas[1:numPuzzle,i], NOS = nos[1:numPuzzle,i], UNDO = undo_c[1:numPuzzle,i]))
    one_numCities<-rbind(one_numCities, data.frame(numCities[1:numPuzzle,i]))
  }
}
save(models, file=paste(path_root, n_numCities,"full_list_perSub_models.RData", sep='/'))

fit1 <- glm(as.numeric(unlist(one_numCities)) ~ .^2, data=one_data_, family = poisson())
sink(paste(path_root, n_numCities, paste("fit_one_model", '.txt', sep=""), sep='/'))
print(summary(fit1))
sink()  # returns output to the console

# Predicting the number of undo in the UNDO condition
# Model: Number of undo(puzzle, subject) = intercept(random effect of subject) (1 | subject) + w1 * MAS(puzzle) + w2 * budget left after the maximum number of cities has been connected (puzzle) + w3*number of optimal solutions (puzzle) + w4*number of errors in a puzzle (puzzle) + interaction
Betas = data.frame()
models = list()
for (i in 1:numParticipants){
  data_ <- data.frame(MAS = mas[1:numPuzzle,i], LEFTOVER = leftover[1:numPuzzle,i], NOS = nos[1:numPuzzle,i], NUMERROR = numError[1:numPuzzle,i])
  fit1 <- glm(numUNDO[1:numPuzzle,i] ~ .^2, data=data_, family = poisson())
  write.csv( tidy( fit1 ) , paste(path_root, n_numUNDO, paste("fit_Sub", i, '.csv', sep=""), sep='/') )
  sink(paste(path_root, n_numUNDO, paste("fit_Sub", i, '.txt', sep=""), sep='/'))
  print(summary(fit1))
  sink()  # returns output to the console
  models[i] <- fit1
  
  if (i == 1) {
    one_data_  = data.frame(MAS = mas[1:numPuzzle,i], LEFTOVER = leftover[1:numPuzzle,i], NOS = nos[1:numPuzzle,i], NUMERROR = numError[1:numPuzzle,i])
    one_numUNDO = data.frame(numUNDO[1:numPuzzle,i])
  } else{
    one_data_<-rbind(one_data_,data.frame(MAS = mas[1:numPuzzle,i], LEFTOVER = leftover[1:numPuzzle,i], NOS = nos[1:numPuzzle,i], NUMERROR = numError[1:numPuzzle,i]))
    one_numUNDO<-rbind(one_numUNDO, data.frame(numUNDO[1:numPuzzle,i]))
  }
}
save(models, file=paste(path_root, n_numUNDO,"full_list_perSub_models.RData", sep='/'))

fit1 <- glm(as.numeric(unlist(one_numUNDO)) ~ .^2, data=one_data_, family = poisson())
sink(paste(path_root, n_numUNDO, paste("fit_one_model", '.txt', sep=""), sep='/'))
print(summary(fit1))
sink()  # returns output to the console




# Predicting the sum of severity of all errors
# Model: sum of severity of errors(puzzle, subject) = intercept(random effect of subject) (1 | subject) + w1 * MAS(puzzle) + w2 * budget left after the maximum number of cities has been connected (puzzle)  + w3*number of optimal solutions (puzzle) + w4*UNDO condition (puzzle) + interaction 
Betas = data.frame()
models = list()
for (i in 1:numParticipants){
  data_ <- data.frame(MAS = mas[1:numPuzzle,i], LEFTOVER = leftover[1:numPuzzle,i], NOS = nos[1:numPuzzle,i], UNDO = undo_c[1:numPuzzle,i])
  fit1 <- glm(sumSeverityErrors[1:numPuzzle,i] ~ .^2, data=data_, family = poisson())
  write.csv( tidy( fit1 ) , paste(path_root, n_sumSeverityErrors, paste("fit_Sub", i, '.csv', sep=""), sep='/') )
  sink(paste(path_root, n_sumSeverityErrors, paste("fit_Sub", i, '.txt', sep=""), sep='/'))
  print(summary(fit1))
  sink()  # returns output to the console
  models[i] <- fit1
  
  if (i == 1) {
    one_data_  = data.frame(MAS = mas[1:numPuzzle,i], LEFTOVER = leftover[1:numPuzzle,i], NOS = nos[1:numPuzzle,i], NUMERROR = numError[1:numPuzzle,i])
    one_sumSeverityErrors = data.frame(sumSeverityErrors[1:numPuzzle,i])
  } else{
    one_data_<-rbind(one_data_,data.frame(MAS = mas[1:numPuzzle,i], LEFTOVER = leftover[1:numPuzzle,i], NOS = nos[1:numPuzzle,i], NUMERROR = numError[1:numPuzzle,i]))
    one_sumSeverityErrors<-rbind(one_sumSeverityErrors, data.frame(sumSeverityErrors[1:numPuzzle,i]))
  }
}
save(models, file=paste(path_root, n_sumSeverityErrors,"full_list_perSub_models.RData", sep='/'))

fit1 <- glm(as.numeric(unlist(one_sumSeverityErrors)) ~ .^2, data=one_data_, family = poisson())
sink(paste(path_root, n_sumSeverityErrors, paste("fit_one_model", '.txt', sep=""), sep='/'))
print(summary(fit1))
sink()  # returns output to the console




# Predicting RT for a trial
# Model: RT(puzzle, subject) = intercept(random effect of subject) (1 | subject) + w1 * MAS(puzzle) + w2*number of optimal solutions (puzzle) + w3*UNDO condition (puzzle) + interaction 
Betas = data.frame()
models = list()
for (i in 1:numParticipants){
  data_ <- data.frame(MAS = mas[1:numPuzzle,i], NOS = nos[1:numPuzzle,i], UNDO = undo_c[1:numPuzzle,i])
  fit1 <- glm(TT[1:numPuzzle,i] ~ .^2, data=data_, family = inverse.gaussian())
  write.csv( tidy( fit1 ) , paste(path_root, n_TT, paste("fit_Sub", i, '.csv', sep=""), sep='/') )
  sink(paste(path_root, n_TT, paste("fit_Sub", i, '.txt', sep=""), sep='/'))
  print(summary(fit1))
  sink()  # returns output to the console
  models[i] <- fit1
  
  if (i == 1) {
    one_data_  = data.frame(MAS = mas[1:numPuzzle,i], LEFTOVER = leftover[1:numPuzzle,i], NOS = nos[1:numPuzzle,i], NUMERROR = numError[1:numPuzzle,i])
    one_TT = data.frame(TT[1:numPuzzle,i])
  } else{
    one_data_<-rbind(one_data_,data.frame(MAS = mas[1:numPuzzle,i], LEFTOVER = leftover[1:numPuzzle,i], NOS = nos[1:numPuzzle,i], NUMERROR = numError[1:numPuzzle,i]))
    one_TT<-rbind(one_TT, data.frame(TT[1:numPuzzle,i]))
  }
}
save(models, file=paste(path_root, n_TT,"full_list_perSub_models.RData", sep='/'))

fit1 <- glm(as.numeric(unlist(one_TT)) ~ .^2, data=one_data_, family = poisson())
sink(paste(path_root, n_TT, paste("fit_one_model", '.txt', sep=""), sep='/'))
print(summary(fit1))
sink()  # returns output to the console
