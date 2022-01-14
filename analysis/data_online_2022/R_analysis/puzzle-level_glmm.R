# load libraries   ---------

library("lme4")



# load data        ---------

data_csv <- read.csv("data.csv")
path_root <- 'puzzle-level-glmm'
head(data_csv)
attach(data_csv)
# subjects	numCities	mas	nos	undo_c	leftover	numError	sumSeverityErrors	numUNDO	TT



# one GLM template ---------

glm1 = glmer(numCities ~ mas + nos + undo_c + (1 | subjects), family = poisson())
glm2 = glmer(numUNDO ~ mas + nos + leftover + numError + (1 | subjects), family = poisson())
glm3 = glmer(sumSeverityErrors ~ mas + nos + leftover + undo_c + (1 | subjects), family = poisson())
glm4 = glmer(TT ~ mas + nos + undo_c + (1 | subjects), family = inverse.gaussian())



# GLMs             ---------

# GLM 1: Number of connected cities(puzzle, subject) = intercept(random effect of subject) (1 | subject) + w1 * MAS(puzzle) + w2 *number of optimal solutions (puzzle) + w3 * UNDO (puzzle) + 3 interaction
GLM1_RESULTS <- list()
GLM1_MODELS <- list()

glm1_01     = glmer(numCities ~ mas + nos + undo_c + (1 | subjects) + (1 | puzzleID) + (1 | subjects:puzzleID), family = poisson())
GLM1_RESULTS <-rbind(GLM1_RESULTS , extractAIC(glm1_01))
GLM1_MODELS  <-rbind(GLM1_MODELS, glm1_01)
glm1_02     = glmer(numCities ~ mas + nos + undo_c + (1 | subjects) + (1 | subjects:puzzleID), family = poisson())
GLM1_RESULTS <-rbind(GLM1_RESULTS , extractAIC(glm1_02))
GLM1_MODELS  <-rbind(GLM1_MODELS, glm1_02)
glm1_03     = glmer(numCities ~ mas + nos + undo_c + (1 | subjects) + (1 | puzzleID), family = poisson())
GLM1_RESULTS <-rbind(GLM1_RESULTS , extractAIC(glm1_03))
GLM1_MODELS  <-rbind(GLM1_MODELS, glm1_03)
glm1_04       = glmer(numCities ~ mas + nos + undo_c + (1 | subjects), family = poisson())
GLM1_RESULTS <-rbind(GLM1_RESULTS , extractAIC(glm1_04))
GLM1_MODELS  <-rbind(GLM1_MODELS, glm1_04)

glm1_05       = glmer(numCities ~ mas + nos + undo_c + (mas | puzzleID) + (nos | puzzleID) + (undo_c | puzzleID) + (1 | subjects) + (1 | puzzleID) + (1 | subjects:puzzleID), family = poisson())
GLM1_RESULTS <-rbind(GLM1_RESULTS , extractAIC(glm1_05))
GLM1_MODELS  <-rbind(GLM1_MODELS, glm1_05)

glm1_0       = glmer(numCities ~ mas + nos + undo_c + (1 | subjects), family = poisson())




glm1_0       = glmer(numCities ~ mas + nos + undo_c + (1 | subjects), family = poisson())
glm1_1       = glmer(numCities ~ mas + nos + undo_c + mas:undo_c + (1 | subjects), family = poisson())
glm1_2       = glmer(numCities ~ mas + nos + undo_c + undo_c:nos + (1 | subjects), family = poisson())
glm1_3       = glmer(numCities ~ mas + nos + undo_c + mas:nos + mas:undo_c + (1 | subjects), family = poisson())
glm1_4       = glmer(numCities ~ mas + nos + undo_c + undo_c:nos + mas:undo_c + (1 | subjects), family = poisson())
glm1_5       = glmer(numCities ~ mas + nos + undo_c + undo_c:nos + mas:undo_c + (1 | subjects), family = poisson())
glm1_maximal = glmer(numCities ~ mas + nos + undo_c + mas:nos + mas:undo_c + undo_c:nos + (1 | subjects), family = poisson())

# GLM 2: Number of undo(puzzle, subject) = intercept(random effect of subject) (1 | subject) + w1 * MAS(puzzle) + w2 * budget left after the maximum number of cities has been connected (puzzle) + w3*number of optimal solutions (puzzle) + w4*number of errors in a puzzle (puzzle) + interaction
glm2 = glmer(numUNDO ~ mas + nos + leftover + numError + (1 | subjects), family = poisson())

glm2_0       = glmer(numUNDO ~ mas + nos + leftover + (1+leftover|subjects) + (1|leftover) + (1 | subjects), family = poisson())
glm2_1       = glmer(numUNDO ~ mas + nos + (0+leftover|subjects) + mas:undo_c + (1 | subjects), family = poisson())
glm2_2       = glmer(numUNDO ~ mas + nos + (0+leftover|subjects) + undo_c:nos + (1 | subjects), family = poisson())
glm2_3       = glmer(numUNDO ~ mas + nos + (0+leftover|subjects) + mas:nos + mas:undo_c + (1 | subjects), family = poisson())
glm2_4       = glmer(numUNDO ~ mas + nos + (0+leftover|subjects) + undo_c:nos + mas:undo_c + (1 | subjects), family = poisson())
glm2_5       = glmer(numUNDO ~ mas + nos + (0+leftover|subjects) + undo_c:nos + mas:undo_c + (1 | subjects), family = poisson())
glm2_maximal = glmer(numUNDO ~ mas + nos + (0+leftover|subjects) + mas:nos + mas:undo_c + undo_c:nos + (1 | subjects), family = poisson())





glm1_maximal = glmer(numCities ~ mas + nos + undo_c + mas:nos + mas:undo_c + undo_c:nos + 
               (0+mas|subjects) + (0+nos|subjects) + (0+undo_c|subjects) + 
               (0+mas:nos|subjects) + (0+undo_c:nos|subjects) + (0+undo_c:mas|subjects) + (1 | subjects), family = poisson())



fit1 <- glm(TT[1:numPuzzle,i] ~ .^2, data=data_, family = inverse.gaussian())

glm1 = glmer(numCities ~ mas + nos + undo_c + mas:nos + mas:undo_c + undo_c:nos + (0+mas*nos|subjects) + (0+nos|subjects) + (0+undo_c|subjects) + (0+mas|subjects) + (0+nos|subjects) + (0+undo_c|subjects) + (1 | subjects), family = poisson())
glm1 = glmer(numCities ~ mas + nos + undo_c + mas:nos + mas:undo_c + undo_c:nos + (1 | subjects), family = poisson())
glm1 = glmer(numCities ~ mas + nos + undo_c + mas:nos + mas:undo_c + undo_c:nos + (1 | subjects), family = poisson())




# Predicting the number of connected cities
# Model: Number of connected cities(puzzle, subject) = intercept(random effect of subject) (1 | subject) + w1 * MAS(puzzle) + w2 *number of optimal solutions (puzzle) + w3 * UNDO (puzzle) + 3 interaction
save(models, file=paste(path_root, n_numCities,"full_list_perSub_models.RData", sep='/'))
data_ <- data.frame(MAS = mas[1:numPuzzle,i], NOS = nos[1:numPuzzle,i], UNDO = undo_c[1:numPuzzle,i])
fit1 <- glm(as.numeric(unlist(one_numCities)) ~ .^2, data=one_data_, family = poisson())
sink(paste(path_root, n_numCities, paste("fit_one_model", '.txt', sep=""), sep='/'))
print(summary(fit1))
sink()  # returns output to the console





# examples -----------------------------------
install.packages("lme4") 
install.packages("car") 
library("lme4")
library("car")

dichotic = read.table("http://www.utstat.toronto.edu/~brunner/data/legal/HandEar.data.txt")
#dichotic_csv = read.csv("./HandEar.data.csv")
head(dichotic)
#head(dichotic_csv)
attach(dichotic)
#attach(dichotic_csv)

#dichotic = lmer(rtime ~ handed*ear + (1 | subject))
dichotic = glmer(rtime ~ handed*ear + (1 | subject), family = poisson())
summary(dichotic)

# Install packages ---------------------------------
pkgs_CRAN <- c("lme4","MCMCglmm","blme",
               "pbkrtest","coda","aods3","bbmle","ggplot2",
               "reshape2","plyr","numDeriv","Hmisc",
               "plotMCMC","gridExtra","R2admb",
               "broom.mixed","dotwhisker")
install.packages(pkgs_CRAN)
rr <- "http://www.math.mcmaster.ca/bolker/R"
install.packages("glmmADMB",type="source",repos=rr)


library("devtools")
print('Install packages completed')

# Load files ---------------------------------



# Load libraries ---------------------------------
## primary GLMM-fitting packages:
library("lme4")
library("glmmADMB")      ## (not on CRAN)
library("glmmTMB")
library("MCMCglmm")
library("blme")
library("MASS")          ## for glmmPQL (base R)
library("nlme")          ## for intervals(), tundra example (base R)

# Residuals ---------------------------------

## auxiliary
library("ggplot2")       ## for pretty plots generally
## ggplot customization:
theme_set(theme_bw())
scale_colour_discrete <- function(...,palette="Set1") {
  scale_colour_brewer(...,palette=palette)
}
scale_colour_orig <- ggplot2::scale_colour_discrete
scale_fill_discrete <- function(...,palette="Set1") {
  scale_fill_brewer(...,palette=palette)
}
## to squash facets together ...
zmargin <- theme(panel.spacing=grid::unit(0,"lines"))
library("gridExtra")     ## for grid.arrange()
library("broom.mixed")
## n.b. as of 25 Sep 2018, need bbolker github version of dotwhisker ...
library("dotwhisker")
library("coda")      ## MCMC diagnostics
library("aods3")     ## overdispersion diagnostics
library("plotMCMC") ## pretty plots from MCMC fits
library("bbmle")     ## AICtab
library("pbkrtest")  ## parametric bootstrap
library("Hmisc")
## for general-purpose reshaping and data manipulation:
library("reshape2")
library("plyr")
## for illustrating effects of observation-level variance in binary data:
library("numDeriv")
print('Load packages completed')

