### Data loading part should be added in here somewhere.

### data <- csv.read('filename.csv')
### For now, the code is written using the simulation data.

numParticipants = 100
numPuzzle = 92
numSession = 4

# numChoice = 20 # MAX number of choices per Puzzle

### random data generation it will be data<-csv.read()
## Variables
set.seed(2021)

#### Puzzle level 
#1. numCities/ number of cities connected
numCities <- matrix(, nrow = numPuzzle, ncol = 0)
numCities_seed <- sample(x=5:11, size=numPuzzle, replace=T)
for (i in 1:numParticipants){
  randperm_idx <- sample(x=1:numPuzzle, size=numPuzzle)
  numCities <- cbind(numCities, numCities_seed[randperm_idx])# sample(x=5:11, size=numPuzzle, replac=T) # randomized for each subject
}

dim(numCities) # [92,100] which is [numPuzzle, numParticipants]

#2. MAS/max achievable score 
mas <- matrix(, nrow = numPuzzle, ncol = 0)
mas_seed <- sample(x=5:11, size=numPuzzle, replace=T)
for (i in 1:numParticipants){
  randperm_idx <- sample(x=1:numPuzzle, size=numPuzzle)
  mas <- cbind(mas, mas_seed[randperm_idx])# sample(x=5:11, size=numPuzzle, replac=T) # randomized for each subject
}

dim(mas) # [92,100] which is [numPuzzle, numParticipants]

#3. NOS/number of optimal solutions
nos <- matrix(, nrow = numPuzzle, ncol = 0)
nos_seed <- sample(x=1:3, size=numPuzzle, replace=T)
for (i in 1:numParticipants){
  randperm_idx <- sample(x=1:numPuzzle, size=numPuzzle)
  nos <- cbind(nos, nos_seed[randperm_idx])# sample(x=5:11, size=numPuzzle, replac=T) # randomized for each subject
}

dim(nos) # [92,100] which is [numPuzzle, numParticipants]

#4. UNDO condition/ 1 for with undo 0 for without undo. numPuzzle/numSession
undo_c <- matrix(, nrow = numPuzzle, ncol = 0)
undo_c_seed <- c(0,0,1,1)
for (i in 1:numParticipants){
  # undo_c_ <- c(matrix(, nrow =0, ncol = 0))
  undo_c_ <- c()
  randperm_idx <- sample(x=1:numSession, size=numSession)
  undo_c_seed <- undo_c_seed[randperm_idx]
  
  for (sei in 1:numSession){
    undo_c_ <- c(undo_c_, rep(1,numPuzzle/numSession) * undo_c_seed[sei])
  }
  
  undo_c <- cbind(undo_c, undo_c_)
}

dim(undo_c) # [92,100] which is [numPuzzle, numParticipants] 0 / 1

#5. leftover/ budget left after the maximum number of cities has been connected 
leftover <- matrix(, nrow = numPuzzle, ncol = 0)
for (i in 1:numParticipants){
  leftover <- cbind(leftover, runif(n=numPuzzle, min=1e-12, max=19.9999999999))#
}

dim(leftover) # [92, 100] which is [numPuzzle, numParticipants]

#6. numError/ number of errors in a puzzle
numError <- matrix(, nrow=numPuzzle, ncol=0)
for (i in 1:numParticipants){
  numError <- cbind(numError, sample(x=0:11, size=numPuzzle, replace=T))
}

dim(numError) # [92, 100] which is [numPuzzle, numParticipants]

#*. sumSeverityErrors/ sum of severity of errors
sumSeverityErrors <- matrix(, nrow=numPuzzle, ncol=0)
for (i in 1:numParticipants){
  sumSeverityErrors <- cbind(sumSeverityErrors, sample(x=0:20, size=numPuzzle, replace=T))
}

dim(sumSeverityErrors) # [92, 100] which is [numPuzzle, numParticipants]


#*. numUNDO/ Number of undo
numUNDO <- matrix(, nrow=numPuzzle, ncol=0)
for (i in 1:numParticipants){
  numUNDO <- cbind(numUNDO, sample(x=0:20, size=numPuzzle, replace=T))
}

dim(numUNDO) # [92, 100] which is [numPuzzle, numParticipants]


#*. TT/ time taken for a trial 
TT <- matrix(, nrow = numPuzzle, ncol = 0)
for (i in 1:numParticipants){
  TT <- cbind(TT, runif(n=numPuzzle, min=1e-12, max=19.9999999999))#
}

dim(TT) # [92, 100] which is [numPuzzle, numParticipants]


#### Choice level 
#7. currNumCities/ current number of connected cities 
currNumCitiesCell <- matrix(list(), nrow = 1, ncol =numParticipants)

for (i in 1:numParticipants){
  for (j in 1:numPuzzle){
    Tr_ <- sample(x=10:30, size = 1)
    randPerData <- c(1:Tr_)
    currNumCitiesCell[[1,i]] <- c(currNumCitiesCell[[1,i]], randPerData)
  }
}

dim(currNumCitiesCell) # [1,100] which is [1, numParticipants]

#8. cumErrorSeverity/ sum of Severity til now
cumErrorSeverity <- matrix(list(), nrow = 1, ncol =numParticipants)

for (i in 1:numParticipants){
  for (j in 1:numPuzzle){
    Tr_ <- sample(x=10:30, size = 1)
    tr <- cumsum(order(sample(x=0:5, size=Tr_, replace=T)))
    randPerData <- c(tr)
    cumErrorSeverity[[1,i]] <- c(cumErrorSeverity[[1,i]], randPerData)
  }
}

dim(cumErrorSeverity) # [1,100] which is [1, numParticipants]


#9. errorSeverity/ Severity of error
errorSeverity <- matrix(list(), nrow = 1, ncol =numParticipants)

for (i in 1:numParticipants){
  for (j in 1:numPuzzle){
    Tr_ <- sample(x=10:30, size = 1)
    tr <- sample(x=0:5, size=Tr_, replace=T)
    randPerData <- c(tr)
    errorSeverity[[1,i]] <- c(errorSeverity[[1,i]], randPerData)
  }
}

dim(errorSeverity) # [1,100] which is [1, numParticipants]

#10. numCitisReach/ number of cities within reach
numCitisReach <- matrix(list(), nrow = 1, ncol =numParticipants)

for (i in 1:numParticipants){
  for (j in 1:numPuzzle){
    Tr_ <- sample(x=10:30, size = 1)
    tr <- sample(x=3:5, size=Tr_, replace=T)
    randPerData <- c(tr)
    numCitisReach[[1,i]] <- c(numCitisReach[[1,i]], randPerData)
  }
}

dim(numCitisReach) # [1,100] which is [1, numParticipants]

#11. response time
RTs <- matrix(list(), nrow = 1, ncol =numParticipants)

for (i in 1:numParticipants){
  for (j in 1:numPuzzle){
    Tr_ <- sample(x=10:30, size = 1)
    tr <- runif(n=Tr_, min=1e-12, max=19.9999999999)
    randPerData <- c(tr)
    RTs[[1,i]] <- c(RTs[[1,i]], randPerData)
  }
}

dim(RTs) # [1,100] which is [1, numParticipants]

#12. UNDO response time
RTs_undo <- matrix(list(), nrow = 1, ncol =numParticipants)

for (i in 1:numParticipants){
  for (j in 1:numPuzzle){
    Tr_ <- sample(x=2:15, size = 1)
    tr <- runif(n=Tr_, min=1e-12, max=19.9999999999)
    randPerData <- c(tr)
    RTs_undo[[1,i]] <- c(RTs_undo[[1,i]], randPerData)
  }
}

dim(RTs_undo) # [1,100] which is [1, numParticipants]


#*. undoOrNot/ Undo or not 
undoOrNot  <- matrix(list(), nrow = 1, ncol =numParticipants)

for (i in 1:numParticipants){
  for (j in 1:numPuzzle){
    Tr_ <- sample(x=10:30, size = 1)
    tr <- sample(x=0:1, size=Tr_, replace=T)
    randPerData <- c(tr)
    undoOrNot[[1,i]] <- c(undoOrNot[[1,i]], randPerData)
  }
}

dim(undoOrNot) # [1,100] which is [1, numParticipants]

#########################################################
#########################################################
#########################################################
#########################################################
#########################################################

### Puzzle-level

# Predicting the number of connected cities
# Model: Number of connected cities(puzzle, subject) = intercept(random effect of subject) (1 | subject) + w1 * MAS(puzzle) + w2 *number of optimal solutions (puzzle) + w3 * UNDO (puzzle) + 3 interaction
Betas = data.frame()
for (i in 1:numParticipants){
  data <- data.frame(MAS = mas[1:numPuzzle,i], NOS = nos[1:numPuzzle,i], UNDO = undo_c[1:numPuzzle,i])
  fit1 <- glm(numCities[1:numPuzzle,i] ~ .^2, data=data, family = poisson())
}

# Predicting the number of undo in the UNDO condition
# Model: Number of undo(puzzle, subject) = intercept(random effect of subject) (1 | subject) + w1 * MAS(puzzle) + w2 * budget left after the maximum number of cities has been connected (puzzle) + w3*number of optimal solutions (puzzle) + w4*number of errors in a puzzle (puzzle) + interaction
Betas = data.frame()
for (i in 1:numParticipants){
  data <- data.frame(MAS = mas[1:numPuzzle,i], LEFTOVER = leftover[1:numPuzzle,i], NOS = nos[1:numPuzzle,i], NUMERROR = numError[1:numPuzzle,i])
  fit1 <- glm(numUNDO[1:numPuzzle,i] ~ .^2, data=data, family = poisson())
}

# Predicting the sum of severity of all errors
# Model: sum of severity of errors(puzzle, subject) = intercept(random effect of subject) (1 | subject) + w1 * MAS(puzzle) + w2 * budget left after the maximum number of cities has been connected (puzzle)  + w3*number of optimal solutions (puzzle) + w4*UNDO condition (puzzle) + interaction 
Betas = data.frame()
for (i in 1:numParticipants){
  data <- data.frame(MAS = mas[1:numPuzzle,i], LEFTOVER = leftover[1:numPuzzle,i], NOS = nos[1:numPuzzle,i], UNDO = undo_c[1:numPuzzle,i])
  fit1 <- glm(sumSeverityErrors[1:numPuzzle,i] ~ .^2, data=data, family = poisson())
}

# Predicting RT for a trial
# Model: RT(puzzle, subject) = intercept(random effect of subject) (1 | subject) + w1 * MAS(puzzle) + w2*number of optimal solutions (puzzle) + w3*UNDO condition (puzzle) + interaction 
Betas = data.frame()
for (i in 1:numParticipants){
  data <- data.frame(MAS = mas[1:numPuzzle,i], NOS = nos[1:numPuzzle,i], UNDO = undo_c[1:numPuzzle,i])
  fit1 <- glm(TT[1:numPuzzle,i] ~ .^2, data=data, family = inverse.gaussian())
}

### Choice-level

# Predicting whether they undo or not
# Model: Undo or not (choice, subject) = intercept + w1 * current number of connected cities (choice) + w2 * sum of Severity of error until current step (choice) + w3* Severity of error (choice) + w4 * number of cities within reach (choice) + w5* number of optimal solutions (puzzle) + w6 * response time (choice) + interaction.
Betas = data.frame()
for (i in 1:numParticipants){
  data <- data.frame(CURRNUMCITIES = currNumCities[[1,i]], CUMERRORSEVERITY = cumErrorSeverity[[1,i]], ERRORSEVERITY = errorSeverity[[1,i]], NUMCITIESREACH  = numCitisReach[[1,i]], RT = RTs[[1,i]], NOS = nos[1:numPuzzle,i])
  fit1 <- glm(undoOrNot[[1,i]] ~ .^2, data=data, family = binomial())
}

# Predicting the severity of error
# Model: the severity of error(choice, subject) = intercept + w1 * current number of connected cities (choice) + w2 * number of cities within reach (choice) + w3 number of optimal solutions (puzzle) + w4 * response time (choice) +w5*UNDO condition (puzzle) + interaction.
Betas = data.frame()
for (i in 1:numParticipants){
  data <- data.frame(CURRNUMCITIES = currNumCities[[1,i]], NUMCITIESREACH  = numCitisReach[[1,i]], RT = RTs[[1,i]], NOS = nos[1:numPuzzle,i], UNDO = undo_c[1:numPuzzle,i])
  fit1 <- glm(errorSeverity[[1,i]] ~ .^2, data=data, family = poisson())
}


# Predicting error or not
# Model: error or not(choice, subject) = intercept + w1 * current number of connected cities (choice) + w2 * number of cities within reach (choice) + w3 * number of optimal solutions (puzzle) + w4 * response time (choice) + w5*UNDO condition (puzzle)+ interaction.
Betas = data.frame()
for (i in 1:numParticipants){
  data <- data.frame(CURRNUMCITIES = currNumCities[[1,i]], CUMERRORSEVERITY = cumErrorSeverity[[1,i]], ERRORSEVERITY = errorSeverity[[1,i]], NUMCITIESREACH  = numCitisReach[[1,i]], RT = RTs[[1,i]], NOS = nos[1:numPuzzle,i])
  fit1 <- glm(undoOrNot[[1,i]] ~ .^2, data=data, family = binomial())
}


# Predicting choice RT
# Model: RT(choice, subject) = intercept + w1 * current number of connected cities (choice) + w2 * number of cities within reach (choice) + w3*number of optimal solutions (puzzle) + w4*UNDO condition (puzzle)+ interaction.
Betas = data.frame()
for (i in 1:numParticipants){
  data <- data.frame(CURRNUMCITIES = currNumCities[[1,i]], CUMERRORSEVERITY = cumErrorSeverity[[1,i]], ERRORSEVERITY = errorSeverity[[1,i]], NUMCITIESREACH  = numCitisReach[[1,i]])
  fit1 <- glm(RTs[[1,i]] ~ .^2, data=data, family = inverse.gaussian())
}


# Predicting undo RT
# Model: undo RT(choice, subject) = intercept + w1 * current number of connected cities (choice) + w2 * number of cities within reach (choice) + w3 * sum of Severity of error until current step (choice) + w4* Severity of error (choice) + interaction.
Betas = data.frame()
for (i in 1:numParticipants){
  data <- data.frame(CURRNUMCITIES = currNumCities[[1,i]], CUMERRORSEVERITY = cumErrorSeverity[[1,i]], ERRORSEVERITY = errorSeverity[[1,i]], NUMCITIESREACH  = numCitisReach[[1,i]])
  fit1 <- glm(RTs_undo[[1,i]] ~ .^2, data=data, family = inverse.gaussian())
}



