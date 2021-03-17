library(tidyverse)
library(rstanarm)
library(rstan)
library(glue)
library(glmnet)
library(caret)
library(future)
library(furrr)
library(xgboost)



# Read in regular season results
regResults <- read_csv("MRegularSeasonDetailedResults.csv")
regResults$WTeamID <- as.factor(regResults$WTeamID)
regResults$LTeamID <- as.factor(regResults$LTeamID)
regResults$Season <- as.factor(regResults$Season)

head(regResults)

# Read in tournament results
tourneyResults <- read_csv("MNCAATourneyDetailedResults.csv")
tourneyResults <- tourneyResults %>% 
  mutate(team1 = ifelse(WTeamID < LTeamID, WTeamID, LTeamID),
         team2 = ifelse(WTeamID > LTeamID, WTeamID, LTeamID),
         team1Score = ifelse(WTeamID < LTeamID, WScore, LScore),
         team2Score = ifelse(WTeamID > LTeamID, WScore, LScore),
         team1Win = ifelse(team1Score > team2Score, 1, 0))
tourneyResults$Season <- as.factor(tourneyResults$Season)
tourneyResults$team1 <- as.factor(tourneyResults$team1)
tourneyResults$team2 <- as.factor(tourneyResults$team2)

head(tourneyResults)
seasons <- read_csv('MSeasons.csv')
seasons %>% head()

seeds <- read_csv('MNCAATourneySeeds.csv')
seeds$Season <- as.factor(seeds$Season)
seeds$TeamID <- as.factor(seeds$TeamID)

## Massey ordinals
massey <- read_csv('MMasseyOrdinals.csv')
sort(unique(massey$SystemName))

# Summarise a select few ordinals
# Obtain starting rank, final rank and SD
masseySumm <- massey %>% group_by(Season, TeamID) %>% 
  filter(SystemName %in% c("NET", "POM", 'MAS', 'EBP', 'SRS', 'TPR', 'USA', 'AP') & RankingDayNum <= 128) %>% 
  arrange(RankingDayNum) %>% 
  summarise(avgFinishRank = mean(case_when(RankingDayNum>=114 ~ OrdinalRank), na.rm=T),
            sdFinishRank = sd(case_when(RankingDayNum>=114 ~ OrdinalRank), na.rm=T)) %>% 
  left_join(massey %>% group_by(Season, TeamID, SystemName) %>% filter(SystemName %in% c("NET", "POM", 'MAS', 'EBP', 'SRS', 'TPR', 'USA', 'AP') & RankingDayNum <= 128) %>% 
              arrange(RankingDayNum) %>% 
              summarise(StartRank = first(OrdinalRank),
                        FinalRank = last(OrdinalRank),
                        rise = StartRank - FinalRank) %>% 
              group_by(Season, TeamID) %>% 
              summarise(avgStartRank = mean(StartRank, na.rm=T),
                        avgFinalRank = mean(FinalRank, na.rm=T),
                        avgRise = mean(rise, na.rm=T)))
masseySumm$Season <- as.factor(masseySumm$Season)
masseySumm$TeamID <- as.factor(masseySumm$TeamID)


# Create eFG variable
regResults <- regResults %>% mutate(WeFG = (WFGM + .5*WFGM3)*100/(WFGA+WFGA3),
                      LeFG = (LFGM + .5*LFGM3)*100/(LFGA+LFGA3))

# Create datasets to model for SOS-adjusted metrics

# Offensive and Defensive Rating
ortgData <- bind_rows(regResults %>% 
                        select(team1 = WTeamID,
                               team2 = LTeamID,
                               points = WScore,
                               Season),
                      regResults %>% 
                        select(team1 = LTeamID,
                               team2 = WTeamID,
                               points = LScore,
                               Season))

# eFG % and eFG% Allowed
eFGData <- bind_rows(regResults %>% 
                       select(team1 = WTeamID,
                              team2 = LTeamID,
                              eFG = WeFG,
                              Season),
                     regResults %>% 
                       select(team1 = LTeamID,
                              team2 = WTeamID,
                              eFG = LeFG,
                              Season))

# tov efficiency (for and against)
tovData <- bind_rows(regResults %>% 
                       select(team1 = WTeamID,
                              team2 = LTeamID,
                              tov = WTO,
                              Season),
                     regResults %>% 
                       select(team1 = LTeamID,
                              team2 = WTeamID,
                              tov = LTO,
                              Season))
# Free Throw rate
FTData <- bind_rows(regResults %>% 
                      select(team1 = WTeamID,
                             team2 = LTeamID,
                             ft = WFTM,
                             Season),
                    regResults %>% 
                      select(team1 = LTeamID,
                             team2 = WTeamID,
                             ft = LFTM,
                             Season))

# Rebounding efficiency
RebData <- bind_rows(regResults %>% mutate(WRebs = WOR + WDR) %>% 
                      select(team1 = WTeamID,
                             team2 = LTeamID,
                             rebs = WRebs,
                             Season),
                    regResults %>% mutate(LRebs = LOR + LDR) %>% 
                      select(team1 = LTeamID,
                             team2 = WTeamID,
                             rebs = LRebs,
                             Season))

# Assists metric
AstData <- bind_rows(regResults  %>% 
                       select(team1 = WTeamID,
                              team2 = LTeamID,
                              ast = WAst,
                              Season),
                     regResults  %>% 
                       select(team1 = LTeamID,
                              team2 = WTeamID,
                              ast = LAst,
                              Season))


# Steal metric
StlData <- bind_rows(regResults %>% 
                       select(team1 = WTeamID,
                              team2 = LTeamID,
                              steals = WStl,
                              Season),
                     regResults  %>% 
                       select(team1 = LTeamID,
                              team2 = WTeamID,
                              steals = LStl,
                              Season))

# Block Metric
BlockData <- bind_rows(regResults %>% 
                         select(team1 = WTeamID,
                                team2 = LTeamID,
                                blocks = WBlk,
                                Season),
                       regResults  %>% 
                         select(team1 = LTeamID,
                                team2 = WTeamID,
                                blocks = LBlk,
                                Season))


# Data for min/max of FGA (proxy for pace)
paceData <- bind_rows(regResults %>% 
                        select(TeamID = WTeamID,
                               fga = WFGA,
                               fgaAllowed = LFGA,
                               Season),
                      regResults %>% 
                        select(TeamID = LTeamID,
                               fga = LFGA,
                               fgaAllowed = WFGA,
                               Season)) %>% 
            group_by(Season, TeamID) %>% 
            summarise(fgaMin = min(fga),
                      fgaMax = max(fga),
                      fgaAllowedMin = min(fgaAllowed),
                      fgaAllowedMax = max(fgaAllowed),
                      fgaRange = fgaMax-fgaMin,
                      fgaAllowedRange = fgaAllowedMax - fgaAllowedMin)



### Data for wins in final 14 days ####
lateWins <- bind_rows(regResults %>% 
            select(TeamID = WTeamID,
                   DayNum,
                   Season) %>% 
            mutate(Win = 1),
          regResults %>% 
            select(TeamID = LTeamID,
                   DayNum,
                   Season) %>% 
            mutate(Win = 0)) %>% 
  filter(DayNum > 114) %>% 
  group_by(Season, TeamID) %>% 
  summarise(finalWinPerc = mean(Win))


lambdas <- 10^seq(-3, 3, by=.1)

plan(multisession, workers=4)
# Create models for SOS-adjusted efficiency metrics
# Then extract coefficients and organize as a data frame

################### Model for Offensive and Defensive Rating #############################################################
sosRtg <- ortgData %>% 
  split(.$Season) %>% 
  future_map_dfr(~ cv.glmnet(.x %>% select(-points, -Season) %>% dummyVars("~.", .) %>% predict(., newdata=.x %>% select(-points, -Season)) %>% data.matrix(),
                  .x$points,
                  alpha=0,
                  lambda = lambdas) %>% 
        coef() %>% as.matrix() %>% 
          as.data.frame() %>% 
          rownames_to_column(., var = 'param') %>% 
          select(value=`1`, param) %>% 
          filter(param != '(Intercept)') %>% 
          mutate(paramA = param %>% str_split(., '\\.', simplify=T) %>% .[,1],
                 paramB = param %>% str_split(., '\\.', simplify=T) %>% .[,2]) %>% 
          select(-param) %>% 
          pivot_wider(names_from=paramA, values_from = value),
        .id = "Season") %>% 
  select(Season, TeamID = paramB, ORtg = team1, DRtg = team2)



######################## Model for eFG% (for and against) ###################################################################
soseFG <- eFGData %>% 
  split(.$Season) %>% 
  future_map_dfr(~ cv.glmnet(.x %>% select(-eFG, -Season) %>% dummyVars("~.", .) %>% predict(., newdata=.x %>% select(-eFG, -Season)) %>% data.matrix(),
                             .x$eFG,
                             alpha=0,
                             lambda = lambdas) %>% 
                   coef() %>% as.matrix() %>% 
                   as.data.frame() %>% 
                   rownames_to_column(., var = 'param') %>% 
                   select(value=`1`, param) %>% 
                   filter(param != '(Intercept)') %>% 
                   mutate(paramA = param %>% str_split(., '\\.', simplify=T) %>% .[,1],
                          paramB = param %>% str_split(., '\\.', simplify=T) %>% .[,2]) %>% 
                   select(-param) %>% 
                   pivot_wider(names_from=paramA, values_from = value),
                 .id = "Season") %>% 
  select(Season, TeamID = paramB, eFG = team1, eFGAllowed = team2)


######################## Model for turnover efficiency #####################################################################
sosTOV <- tovData %>% 
  split(.$Season) %>% 
  future_map_dfr(~ cv.glmnet(.x %>% select(-tov, -Season) %>% dummyVars("~.", .) %>% predict(., newdata=.x %>% select(-tov, -Season)) %>% data.matrix(),
                             .x$tov,
                             alpha=0,
                             lambda = lambdas) %>% 
                   coef() %>% as.matrix() %>% 
                   as.data.frame() %>% 
                   rownames_to_column(., var = 'param') %>% 
                   select(value=`1`, param) %>% 
                   filter(param != '(Intercept)') %>% 
                   mutate(paramA = param %>% str_split(., '\\.', simplify=T) %>% .[,1],
                          paramB = param %>% str_split(., '\\.', simplify=T) %>% .[,2]) %>% 
                   select(-param) %>% 
                   pivot_wider(names_from=paramA, values_from = value),
                 .id = "Season") %>% 
  select(Season, TeamID = paramB, tov = team1, tovForced = team2)


########################### Model for Free Throw efficiency #######################################################################
sosFT <- FTData %>% 
  split(.$Season) %>% 
  future_map_dfr(~ cv.glmnet(.x %>% select(-ft, -Season) %>% dummyVars("~.", .) %>% predict(., newdata=.x %>% select(-ft, -Season)) %>% data.matrix(),
                             .x$ft,
                             alpha=0,
                             lambda = lambdas) %>% 
                   coef() %>% as.matrix() %>% 
                   as.data.frame() %>% 
                   rownames_to_column(., var = 'param') %>% 
                   select(value=`1`, param) %>% 
                   filter(param != '(Intercept)') %>% 
                   mutate(paramA = param %>% str_split(., '\\.', simplify=T) %>% .[,1],
                          paramB = param %>% str_split(., '\\.', simplify=T) %>% .[,2]) %>% 
                   select(-param) %>% 
                   pivot_wider(names_from=paramA, values_from = value),
                 .id = "Season") %>% 
  select(Season, TeamID = paramB, FT = team1, FTAllowed = team2)

############################### Model for Rebound efficiency ######################################################################
sosRebs <- RebData %>% 
  split(.$Season) %>% 
  future_map_dfr(~ cv.glmnet(.x %>% select(-rebs, -Season) %>% dummyVars("~.", .) %>% predict(., newdata=.x %>% select(-rebs, -Season)) %>% data.matrix(),
                             .x$rebs,
                             alpha=0,
                             lambda = lambdas) %>% 
                   coef() %>% as.matrix() %>% 
                   as.data.frame() %>% 
                   rownames_to_column(., var = 'param') %>% 
                   select(value=`1`, param) %>% 
                   filter(param != '(Intercept)') %>% 
                   mutate(paramA = param %>% str_split(., '\\.', simplify=T) %>% .[,1],
                          paramB = param %>% str_split(., '\\.', simplify=T) %>% .[,2]) %>% 
                   select(-param) %>% 
                   pivot_wider(names_from=paramA, values_from = value),
                 .id = "Season") %>% 
  select(Season, TeamID = paramB, Rebs = team1, RebAllowed = team2)

############################### Model for assists #######################################################################
sosAst <- AstData %>% 
  split(.$Season) %>% 
  future_map_dfr(~ cv.glmnet(.x %>% select(-ast, -Season) %>% dummyVars("~.", .) %>% predict(., newdata=.x %>% select(-ast, -Season)) %>% data.matrix(),
                             .x$ast,
                             alpha=0,
                             lambda = lambdas) %>% 
                   coef() %>% as.matrix() %>% 
                   as.data.frame() %>% 
                   rownames_to_column(., var = 'param') %>% 
                   select(value=`1`, param) %>% 
                   filter(param != '(Intercept)') %>% 
                   mutate(paramA = param %>% str_split(., '\\.', simplify=T) %>% .[,1],
                          paramB = param %>% str_split(., '\\.', simplify=T) %>% .[,2]) %>% 
                   select(-param) %>% 
                   pivot_wider(names_from=paramA, values_from = value),
                 .id = "Season") %>% 
  select(Season, TeamID = paramB, Ast = team1, AstAllowed = team2)

################################# Model for steal efficiency ###################################################################
sosStl <- StlData %>% 
  split(.$Season) %>% 
  future_map_dfr(~ cv.glmnet(.x %>% select(-steals, -Season) %>% dummyVars("~.", .) %>% predict(., newdata=.x %>% select(-steals, -Season)) %>% data.matrix(),
                             .x$steals,
                             alpha=0,
                             lambda = lambdas,
                             family='poisson') %>% 
                   coef() %>% as.matrix() %>% 
                   as.data.frame() %>% 
                   rownames_to_column(., var = 'param') %>% 
                   select(value=`1`, param) %>% 
                   filter(param != '(Intercept)') %>% 
                   mutate(paramA = param %>% str_split(., '\\.', simplify=T) %>% .[,1],
                          paramB = param %>% str_split(., '\\.', simplify=T) %>% .[,2]) %>% 
                   select(-param) %>% 
                   pivot_wider(names_from=paramA, values_from = value),
                 .id = "Season") %>% 
  select(Season, TeamID = paramB, Steals = team1, StealsAllowed = team2)

######################################## Model for block efficiency #################################################################
sosBlocks <- BlockData %>% 
  split(.$Season) %>% 
  future_map_dfr(~ cv.glmnet(.x %>% dplyr::select(-blocks, -Season) %>% dummyVars("~.", .) %>% predict(., newdata=.x %>% dplyr::select(-blocks, -Season)) %>% data.matrix(),
                             .x$blocks,
                             alpha=0,
                             lambda = lambdas,
                             family='poisson') %>% 
                   coef() %>% as.matrix() %>% 
                   as.data.frame() %>% 
                   rownames_to_column(., var = 'param') %>% 
                   dplyr::select(value=`1`, param) %>% 
                   filter(param != '(Intercept)') %>% 
                   mutate(paramA = param %>% str_split(., '\\.', simplify=T) %>% .[,1],
                          paramB = param %>% str_split(., '\\.', simplify=T) %>% .[,2]) %>% 
                   dplyr::select(-param) %>% 
                   pivot_wider(names_from=paramA, values_from = value),
                 .id = "Season") %>% 
  dplyr::select(Season, TeamID = paramB, Blocks = team1, BlocksAllowed = team2)





# Create dataframe that includes each tournament matchup and the seed ranking for each team
# Double length of dataset by allowing for team1ID > team2ID
tournSeedSumm <- bind_rows(tourneyResults %>% select(Season, team1, team2, team1Win, team1Score, team2Score) %>% 
  left_join(seeds, by=c("Season", 'team1'='TeamID')) %>% 
  left_join(seeds, by=c("Season", 'team2' = 'TeamID'), suffix=c('1', '2')) %>% 
  mutate(Seed1 = str_extract(Seed1, '\\d+') %>% as.numeric(),
         Seed2 = str_extract(Seed2, '\\d+') %>% as.numeric(),
         SeedDiff = Seed1-Seed2),
  tourneyResults %>% select(Season, team1=team2, team2=team1, team1Win, team1Score=team2Score, team2Score=team1Score) %>% 
    mutate(team1Win = 1 - team1Win) %>% 
    left_join(seeds, by=c("Season", 'team1'='TeamID')) %>% 
    left_join(seeds, by=c("Season", 'team2' = 'TeamID'), suffix=c('1', '2')) %>% 
    mutate(Seed1 = str_extract(Seed1, '\\d+') %>% as.numeric(),
           Seed2 = str_extract(Seed2, '\\d+') %>% as.numeric(),
           SeedDiff = Seed1-Seed2))



ggplot(data = tournSeedSumm,
       aes(x = SeedDiff, y = team1Win)) + geom_smooth()

# Create a logistic regression that uses the cubed difference in seed ranking to predict whether team 1 win's or not
# Chose the cubed difference to ensure that seed differentials in the middle third are more similar, while differences in the tails have more extreme effects
baseGlm <- glm(team1Win ~ I(SeedDiff^3), data = tournSeedSumm, family=binomial)
summary(baseGlm)
# After fitting GLM, add in modeled predictions to use as 'base rates'
tournSeedSumm <- tournSeedSumm %>% mutate(preds = predict(baseGlm, newdata = tournSeedSumm, type='response'))


ggplot(data = tournSeedSumm,
       aes(x = SeedDiff,
           y = preds)) + geom_point()


# Combine modeled efficiency ratings into one data frame
metricDf <- sosRtg %>% left_join(soseFG, by=c('Season', 'TeamID')) %>% left_join(sosTOV, by=c("Season", 'TeamID')) %>% left_join(sosFT, by=c("Season", "TeamID")) %>% 
            left_join(sosRebs, by=c("Season", 'TeamID')) %>% 
            left_join(sosStl, by=c("Season", "TeamID")) %>% 
            left_join(sosBlocks, by=c("Season", 'TeamID')) %>% 
            left_join(sosAst, by=c("Season", "TeamID")) %>% 
            left_join(paceData, by=c("Season", 'TeamID')) %>% 
            left_join(lateWins, by=c("Season", 'TeamID')) %>% 
            left_join(masseySumm, by=c('Season', 'TeamID'))

# Now join modeled efficiency data frame to dataframe containing tournament match ups
tournModelDf <- tournSeedSumm %>% left_join(metricDf, by=c("Season", 'team1' = 'TeamID')) %>% left_join(metricDf, by=c("Season", 'team2' = "TeamID"), suffix = c("1", "2"))
tournModelDf <- tournModelDf %>% mutate(Response = factor(team1Win, levels=c(0,1), labels=c("Loss", "Win")))


# Feature engineering
tournModelDf <- tournModelDf %>% 
                mutate(effRtg = ORtg1 - DRtg1 - ORtg2 + DRtg2,
                       eFgRtg = eFG1 - eFGAllowed1 - eFG2 + eFGAllowed2,
                       tovEff = tov1 - tovForced1 - tov2 + tovForced2,
                       ftEff = FT1 - FTAllowed1 - FT2 + FTAllowed2,
                       rebEff = Rebs1 - RebAllowed1 - Rebs2 + RebAllowed2,
                       stlEff = Steals1 - StealsAllowed1 - Steals2 + StealsAllowed2,
                       blkEff = Blocks1 - BlocksAllowed1 - Blocks2 + BlocksAllowed2,
                       defMetric = tovEff + stlEff + blkEff,
                       astEff = Ast1 - AstAllowed1 - Ast2 + AstAllowed2,
                       paceDiff = fgaRange1 - fgaRange2,
                       sdRankDiff = sqrt(sdFinishRank1) - sqrt(sdFinishRank2),
                       rankRiseDiff = sqrt(abs(avgRise1)) - sqrt(abs(avgRise2)),
                       logRankDiff = log(avgFinalRank1) - log(avgFinalRank2),
                       pointDiff = team1Score - team2Score)

tournModelDf %>% filter(avgRise1==0)

ggplot(data = tournModelDf,
       aes(x = log(avgFinalRank1) - log(avgFinalRank2),
           y = team1Win)) + geom_smooth() + geom_point() 

# train/test split
splitter <- sample(1:nrow(tournModelDf), .7*nrow(tournModelDf), replace=F)
trainDf <- tournModelDf[splitter,]
testDf <- tournModelDf[-splitter,]




ctrl <- trainControl(method='cv', number=10, search='random', savePredictions = TRUE, summaryFunction = mnLogLoss, classProbs = T, verbose=T)

xgbModel <- train(Response ~ ORtg1 + ORtg2 +
                             DRtg1 + DRtg2 +
                             eFG1 + eFG2 +
                             eFGAllowed1 + eFGAllowed2 +
                             tov1 + tov2 +
                             tovForced1 + tovForced2 +
                             FT1 + FT2 +
                             FTAllowed1 + FTAllowed2 +
                             effRtg +
                             eFgRtg +
                             tovEff +
                             ftEff +
                             rebEff +
                             stlEff +
                             blkEff +
                             defMetric +
                             paceDiff +
                             astEff +
                             sdRankDiff +
                             rankRiseDiff +
                             logRankDiff +
                             avgFinishRank1 +
                             avgFinishRank2 +
                             sdFinishRank1 +
                             sdFinishRank2,
                  data = trainDf,
                  method = 'xgbTree',
                  trControl = ctrl,
                  tuneLength = 30,
                  metric = 'logLoss')

xgbModel$results
varImp(xgbModel)
xgbModel$bestTune


# Expand grid to search around optimal parameters from random search
gridSearch <- expand.grid(nrounds = 1050,
                          max_depth = 2,
                          eta = .01,
                          gamma = 7.55,
                          colsample_bytree = .575,
                          min_child_weight = 20,
                          subsample = .925)

ctrlTune <- trainControl(method='loocv', savePredictions = TRUE, summaryFunction = mnLogLoss, classProbs = T, verbose=T)

xgbModelTune <- train(Response ~ ORtg1 + ORtg2 +
                        DRtg1 + DRtg2 +
                        eFG1 + eFG2 +
                        eFGAllowed1 + eFGAllowed2 +
                        tov1 + tov2 +
                        tovForced1 + tovForced2 +
                        FT1 + FT2 +
                        FTAllowed1 + FTAllowed2 +
                        effRtg +
                        eFgRtg +
                        tovEff +
                        ftEff +
                        rebEff +
                        stlEff +
                        blkEff +
                        defMetric +
                        paceDiff +
                        astEff +
                        sdRankDiff +
                        rankRiseDiff +
                        logRankDiff +
                        avgFinishRank1 +
                        avgFinishRank2 +
                        sdFinishRank1 +
                        sdFinishRank2,
                      data = trainDf,
                      method = 'xgbTree',
                      trControl = ctrlTune,
                      tuneGrid = gridSearch,
                      metric = 'logLoss')
xgbModelTune$results
varImp(xgbModelTune)

trainDf <- trainDf %>% mutate(xgbPreds = xgbModelTune$pred$Win)


# Model point differential for ensemble model
pointDiffModel <- stan_lm(pointDiff ~ effRtg,
                          data = trainDf,
                          chains = 4,
                          iter = 3500,
                          warmup = 1000,
                          cores=4,
                          prior = R2(location=.3, what=c('median')))
summary(pointDiffModel)
pairs(pointDiffModel)
pointDiffDf <- as.data.frame(pointDiffModel)

trainDf$stanPreds <- trainDf$effRtg %>% map_dbl(., function(x){
  y <- x * pointDiffDf$effRtg + pointDiffDf$`(Intercept)`
  perc <- sum(ifelse(y > 0, 1, 0))/length(y)
  return(perc)
})

ggplot(data = trainDf,
       aes(x = effRtg,
           y = pointDiff)) + geom_point() + geom_smooth()

ggplot(data = trainDf,
       aes(x = stanPreds,
           y = team1Win)) + geom_point() + geom_smooth()

## Ridge regression for point diff
ridgePointDiff <- cv.glmnet(trainDf %>% select(effRtg, eFgRtg, tovEff, ftEff, rebEff, stlEff, blkEff, logRankDiff, sdRankDiff) %>% as.matrix(),
                            trainDf %>% pull(pointDiff),
                            alpha=0,
                            lambda = lambdas)

ridgePointDiff$lambda.min

trainDf$ridgePreds <- predict(ridgePointDiff, newx=trainDf %>% select(effRtg, eFgRtg, tovEff, ftEff, rebEff, stlEff, blkEff, logRankDiff, sdRankDiff) %>% as.matrix())

# Create GLM to stack base rates, xgboost preds, stan model preds
stackGlm <- glm(team1Win ~ preds + stanPreds + ridgePreds, data = trainDf, family = 'binomial')
summary(stackGlm)

trainDf <- trainDf %>% mutate(stackPreds = predict(stackGlm, newdata=trainDf, type='response'))

MLmetrics::LogLoss((trainDf$stackPreds + trainDf$xgbPreds)/2, trainDf$team1Win)

testDf <- testDf %>% mutate(xgbPreds = predict(xgbModelTune, newdata=testDf, type='prob')[,2])

testDf$stanPreds <- testDf$effRtg %>% map_dbl(., function(x){
  y <- x * pointDiffDf$effRtg + pointDiffDf$`(Intercept)`
  perc <- sum(ifelse(y > 0, 1, 0))/length(y)
  return(perc)})

testDf$ridgePreds <- predict(ridgePointDiff, newx=testDf %>% select(effRtg, eFgRtg, tovEff, ftEff, rebEff, stlEff, blkEff, logRankDiff, sdRankDiff) %>% as.matrix())

testDf <- testDf %>% mutate(stackPreds = predict(stackGlm, newdata=testDf, type='response'))

testDf <- testDf %>% mutate(finalPreds = (xgbPreds + stackPreds)/2)

MLmetrics::LogLoss((testDf$stackPreds+testDf$xgbPreds)/2, testDf$team1Win)

ggplot(data = testDf,
       aes(x = (stackPreds + xgbPreds)/2,
           y = team1Win)) + geom_point() + geom_smooth()





##### Read in stage 2 data for scoring
stageTwoRegResults <- read_csv('MDataFiles_Stage2/MRegularSeasonDetailedResults.csv') %>% filter(Season==2021)
stageTwoRegResults$WTeamID <- as.factor(stageTwoRegResults$WTeamID)
stageTwoRegResults$LTeamID <- as.factor(stageTwoRegResults$LTeamID)
stageTwoRegResults$Season <- as.factor(stageTwoRegResults$Season)


stageTwoSeeds <- read_csv('MDataFiles_Stage2/MNCAATourneySeeds.csv') %>% filter(Season==2021)
stageTwoSeeds$Season <- as.factor(stageTwoSeeds$Season)

stageTwoMassey <- read_csv('MDataFiles_Stage2/MMasseyOrdinals.csv') %>% filter(Season==2021)



stageTwoMasseySumm <- stageTwoMassey %>% group_by(TeamID) %>% 
  filter(SystemName %in% c("NET", "POM", 'MAS', 'EBP', 'SRS', 'TPR', 'USA', 'AP') & RankingDayNum <= 128) %>% 
  arrange(RankingDayNum) %>% 
  summarise(avgFinishRank = mean(case_when(RankingDayNum>=114 ~ OrdinalRank), na.rm=T),
            sdFinishRank = sd(case_when(RankingDayNum>=114 ~ OrdinalRank), na.rm=T)) %>% 
  left_join(stageTwoMassey %>% group_by(TeamID, SystemName) %>% filter(SystemName %in% c("NET", "POM", 'MAS', 'EBP', 'SRS', 'TPR', 'USA', 'AP') & RankingDayNum <= 128) %>% 
              arrange(RankingDayNum) %>% 
              summarise(StartRank = first(OrdinalRank),
                        FinalRank = last(OrdinalRank),
                        rise = StartRank - FinalRank) %>% 
              group_by(TeamID) %>% 
              summarise(avgStartRank = mean(StartRank, na.rm=T),
                        avgFinalRank = mean(FinalRank, na.rm=T),
                        avgRise = mean(rise, na.rm=T)))
stageTwoMasseySumm$TeamID <- as.factor(stageTwoMasseySumm$TeamID)


stageTwoRegResults <- stageTwoRegResults %>% mutate(WeFG = (WFGM + .5*WFGM3)*100/(WFGA+WFGA3),
                                    LeFG = (LFGM + .5*LFGM3)*100/(LFGA+LFGA3))

# Create datasets to model for SOS-adjusted metrics

# Offensive and Defensive Rating
ortgData2 <- bind_rows(stageTwoRegResults %>% 
                        select(team1 = WTeamID,
                               team2 = LTeamID,
                               points = WScore,
                               Season),
                      stageTwoRegResults %>% 
                        select(team1 = LTeamID,
                               team2 = WTeamID,
                               points = LScore,
                               Season))

# eFG % and eFG% Allowed
eFGData2 <- bind_rows(stageTwoRegResults %>% 
                       select(team1 = WTeamID,
                              team2 = LTeamID,
                              eFG = WeFG,
                              Season),
                     stageTwoRegResults %>% 
                       select(team1 = LTeamID,
                              team2 = WTeamID,
                              eFG = LeFG,
                              Season))

# tov efficiency (for and against)
tovData2 <- bind_rows(stageTwoRegResults %>% 
                       select(team1 = WTeamID,
                              team2 = LTeamID,
                              tov = WTO,
                              Season),
                     stageTwoRegResults %>% 
                       select(team1 = LTeamID,
                              team2 = WTeamID,
                              tov = LTO,
                              Season))
# Free Throw rate
FTData2 <- bind_rows(stageTwoRegResults %>% 
                      select(team1 = WTeamID,
                             team2 = LTeamID,
                             ft = WFTM,
                             Season),
                    stageTwoRegResults %>% 
                      select(team1 = LTeamID,
                             team2 = WTeamID,
                             ft = LFTM,
                             Season))

# Rebounding efficiency
RebData2 <- bind_rows(stageTwoRegResults %>% mutate(WRebs = WOR + WDR) %>% 
                       select(team1 = WTeamID,
                              team2 = LTeamID,
                              rebs = WRebs,
                              Season),
                     stageTwoRegResults %>% mutate(LRebs = LOR + LDR) %>% 
                       select(team1 = LTeamID,
                              team2 = WTeamID,
                              rebs = LRebs,
                              Season))

# Assists metric
AstData2 <- bind_rows(stageTwoRegResults  %>% 
                       select(team1 = WTeamID,
                              team2 = LTeamID,
                              ast = WAst,
                              Season),
                     stageTwoRegResults  %>% 
                       select(team1 = LTeamID,
                              team2 = WTeamID,
                              ast = LAst,
                              Season))


# Steal metric
StlData2 <- bind_rows(stageTwoRegResults %>% 
                       select(team1 = WTeamID,
                              team2 = LTeamID,
                              steals = WStl,
                              Season),
                     stageTwoRegResults  %>% 
                       select(team1 = LTeamID,
                              team2 = WTeamID,
                              steals = LStl,
                              Season))

# Block Metric
BlockData2 <- bind_rows(stageTwoRegResults %>% 
                         select(team1 = WTeamID,
                                team2 = LTeamID,
                                blocks = WBlk,
                                Season),
                       stageTwoRegResults  %>% 
                         select(team1 = LTeamID,
                                team2 = WTeamID,
                                blocks = LBlk,
                                Season))


# Data for min/max of FGA (proxy for pace)
paceData2 <- bind_rows(stageTwoRegResults %>% 
                        select(TeamID = WTeamID,
                               fga = WFGA,
                               fgaAllowed = LFGA,
                               Season),
                      stageTwoRegResults %>% 
                        select(TeamID = LTeamID,
                               fga = LFGA,
                               fgaAllowed = WFGA,
                               Season)) %>% 
  group_by(TeamID) %>% 
  summarise(fgaMin = min(fga),
            fgaMax = max(fga),
            fgaAllowedMin = min(fgaAllowed),
            fgaAllowedMax = max(fgaAllowed),
            fgaRange = fgaMax-fgaMin,
            fgaAllowedRange = fgaAllowedMax - fgaAllowedMin)

lateWins2 <- bind_rows(stageTwoRegResults %>% 
                        select(TeamID = WTeamID,
                               DayNum,
                               Season) %>% 
                        mutate(Win = 1),
                      stageTwoRegResults %>% 
                        select(TeamID = LTeamID,
                               DayNum,
                               Season) %>% 
                        mutate(Win = 0)) %>% 
  filter(DayNum > 114) %>% 
  group_by(TeamID) %>% 
  summarise(finalWinPerc = mean(Win))




######### SOS Models for test data #####################################################################################
################### Model for Offensive and Defensive Rating #############################################################
sosRtg2 <- cv.glmnet(ortgData2 %>% select(-points, -Season) %>% dummyVars("~.", .) %>% predict(., newdata=ortgData2 %>% select(-points, -Season)) %>% data.matrix(),
                             ortgData2$points,
                             alpha=0,
                             lambda = lambdas) %>% 
                   coef() %>% as.matrix() %>% 
                   as.data.frame() %>% 
                   rownames_to_column(., var = 'param') %>% 
                   select(value=`1`, param) %>% 
                   filter(param != '(Intercept)') %>% 
                   mutate(paramA = param %>% str_split(., '\\.', simplify=T) %>% .[,1],
                          paramB = param %>% str_split(., '\\.', simplify=T) %>% .[,2]) %>% 
                   select(-param) %>% 
                   pivot_wider(names_from=paramA, values_from = value) %>% 
  select(TeamID = paramB, ORtg = team1, DRtg = team2)



######################## Model for eFG% (for and against) ###################################################################
soseFG2 <- cv.glmnet(eFGData2 %>% select(-eFG, -Season) %>% dummyVars("~.", .) %>% predict(., newdata=eFGData2 %>% select(-eFG, -Season)) %>% data.matrix(),
                     eFGData2$eFG,
                             alpha=0,
                             lambda = lambdas) %>% 
                   coef() %>% as.matrix() %>% 
                   as.data.frame() %>% 
                   rownames_to_column(., var = 'param') %>% 
                   select(value=`1`, param) %>% 
                   filter(param != '(Intercept)') %>% 
                   mutate(paramA = param %>% str_split(., '\\.', simplify=T) %>% .[,1],
                          paramB = param %>% str_split(., '\\.', simplify=T) %>% .[,2]) %>% 
                   select(-param) %>% 
                   pivot_wider(names_from=paramA, values_from = value) %>% 
  select(TeamID = paramB, eFG = team1, eFGAllowed = team2)


######################## Model for turnover efficiency #####################################################################
sosTOV2 <- cv.glmnet(tovData2 %>% select(-tov, -Season) %>% dummyVars("~.", .) %>% predict(., newdata=tovData2 %>% select(-tov, -Season)) %>% data.matrix(),
                     tovData2$tov,
                             alpha=0,
                             lambda = lambdas) %>% 
                   coef() %>% as.matrix() %>% 
                   as.data.frame() %>% 
                   rownames_to_column(., var = 'param') %>% 
                   select(value=`1`, param) %>% 
                   filter(param != '(Intercept)') %>% 
                   mutate(paramA = param %>% str_split(., '\\.', simplify=T) %>% .[,1],
                          paramB = param %>% str_split(., '\\.', simplify=T) %>% .[,2]) %>% 
                   select(-param) %>% 
                   pivot_wider(names_from=paramA, values_from = value) %>% 
  select(TeamID = paramB, tov = team1, tovForced = team2)


########################### Model for Free Throw efficiency #######################################################################
sosFT2 <- cv.glmnet(FTData2 %>% select(-ft, -Season) %>% dummyVars("~.", .) %>% predict(., newdata=FTData2 %>% select(-ft, -Season)) %>% data.matrix(),
                    FTData2$ft,
                             alpha=0,
                             lambda = lambdas) %>% 
                   coef() %>% as.matrix() %>% 
                   as.data.frame() %>% 
                   rownames_to_column(., var = 'param') %>% 
                   select(value=`1`, param) %>% 
                   filter(param != '(Intercept)') %>% 
                   mutate(paramA = param %>% str_split(., '\\.', simplify=T) %>% .[,1],
                          paramB = param %>% str_split(., '\\.', simplify=T) %>% .[,2]) %>% 
                   select(-param) %>% 
                   pivot_wider(names_from=paramA, values_from = value) %>% 
  select(TeamID = paramB, FT = team1, FTAllowed = team2)

############################### Model for Rebound efficiency ######################################################################
sosRebs2 <- cv.glmnet(RebData2 %>% select(-rebs, -Season) %>% dummyVars("~.", .) %>% predict(., newdata=RebData2 %>% select(-rebs, -Season)) %>% data.matrix(),
                      RebData2$rebs,
                             alpha=0,
                             lambda = lambdas) %>% 
                   coef() %>% as.matrix() %>% 
                   as.data.frame() %>% 
                   rownames_to_column(., var = 'param') %>% 
                   select(value=`1`, param) %>% 
                   filter(param != '(Intercept)') %>% 
                   mutate(paramA = param %>% str_split(., '\\.', simplify=T) %>% .[,1],
                          paramB = param %>% str_split(., '\\.', simplify=T) %>% .[,2]) %>% 
                   select(-param) %>% 
                   pivot_wider(names_from=paramA, values_from = value) %>% 
  select(TeamID = paramB, Rebs = team1, RebAllowed = team2)

############################### Model for assists #######################################################################
sosAst2 <- cv.glmnet(AstData2 %>% select(-ast, -Season) %>% dummyVars("~.", .) %>% predict(., newdata=AstData2 %>% select(-ast, -Season)) %>% data.matrix(),
                     AstData2$ast,
                             alpha=0,
                             lambda = lambdas) %>% 
                   coef() %>% as.matrix() %>% 
                   as.data.frame() %>% 
                   rownames_to_column(., var = 'param') %>% 
                   select(value=`1`, param) %>% 
                   filter(param != '(Intercept)') %>% 
                   mutate(paramA = param %>% str_split(., '\\.', simplify=T) %>% .[,1],
                          paramB = param %>% str_split(., '\\.', simplify=T) %>% .[,2]) %>% 
                   select(-param) %>% 
                   pivot_wider(names_from=paramA, values_from = value) %>% 
  select(TeamID = paramB, Ast = team1, AstAllowed = team2)

################################# Model for steal efficiency ###################################################################
sosStl2 <- cv.glmnet(StlData2 %>% select(-steals, -Season) %>% dummyVars("~.", .) %>% predict(., newdata=StlData2 %>% select(-steals, -Season)) %>% data.matrix(),
                     StlData2$steals,
                             alpha=0,
                             lambda = lambdas,
                             family='poisson') %>% 
                   coef() %>% as.matrix() %>% 
                   as.data.frame() %>% 
                   rownames_to_column(., var = 'param') %>% 
                   select(value=`1`, param) %>% 
                   filter(param != '(Intercept)') %>% 
                   mutate(paramA = param %>% str_split(., '\\.', simplify=T) %>% .[,1],
                          paramB = param %>% str_split(., '\\.', simplify=T) %>% .[,2]) %>% 
                   select(-param) %>% 
                   pivot_wider(names_from=paramA, values_from = value) %>% 
  select(TeamID = paramB, Steals = team1, StealsAllowed = team2)

######################################## Model for block efficiency #################################################################
sosBlocks2 <- cv.glmnet(BlockData2 %>% dplyr::select(-blocks, -Season) %>% dummyVars("~.", .) %>% predict(., newdata=BlockData2 %>% dplyr::select(-blocks, -Season)) %>% data.matrix(),
                        BlockData2$blocks,
                             alpha=0,
                             lambda = lambdas,
                             family='poisson') %>% 
                   coef() %>% as.matrix() %>% 
                   as.data.frame() %>% 
                   rownames_to_column(., var = 'param') %>% 
                   dplyr::select(value=`1`, param) %>% 
                   filter(param != '(Intercept)') %>% 
                   mutate(paramA = param %>% str_split(., '\\.', simplify=T) %>% .[,1],
                          paramB = param %>% str_split(., '\\.', simplify=T) %>% .[,2]) %>% 
                   dplyr::select(-param) %>% 
                   pivot_wider(names_from=paramA, values_from = value) %>% 
  dplyr::select(TeamID = paramB, Blocks = team1, BlocksAllowed = team2)




# Create synthetic dataframe for all possible tournament matchups
tourn21 <- expand.grid(Season = "2021", team1 = stageTwoSeeds$TeamID, team2 = stageTwoSeeds$TeamID)
tourn21 <- tourn21 %>% filter(team1 < team2)
tourn21$team1 <- as.factor(tourn21$team1)
tourn21$team2 <- as.factor(tourn21$team2)
stageTwoSeeds$TeamID <- as.factor(stageTwoSeeds$TeamID)

tourn21 <- tourn21 %>% left_join(stageTwoSeeds %>% select(-Season), by=c('team1' = 'TeamID')) %>% left_join(stageTwoSeeds %>% select(-Season), by = c('team2' = 'TeamID'), suffix = c('1', '2'))
tourn21 <- tourn21 %>% mutate(Seed1 = str_extract(Seed1, '\\d+') %>% as.numeric(),
                              Seed2 = str_extract(Seed2, '\\d+') %>% as.numeric(),
                              SeedDiff = Seed1-Seed2)


# base rate predictions based on seed only
tourn21 <- tourn21 %>% mutate(preds = predict(baseGlm, newdata=tourn21, type='response'))


# Modeling metric data frame
metricDf2 <- sosRtg2 %>% left_join(soseFG2, by=c('TeamID')) %>% left_join(sosTOV2, by=c('TeamID')) %>% left_join(sosFT2, by=c("TeamID")) %>% 
  left_join(sosRebs2, by=c('TeamID')) %>% 
  left_join(sosStl2, by=c("TeamID")) %>% 
  left_join(sosBlocks2, by=c('TeamID')) %>% 
  left_join(sosAst2, by=c("TeamID")) %>% 
  left_join(paceData2, by=c('TeamID')) %>% 
  left_join(lateWins2, by=c('TeamID')) %>% 
  left_join(stageTwoMasseySumm, by=c('TeamID'))




# Now join modeled efficiency data frame to dataframe containing tournament match ups
tournModelDf2 <- tourn21 %>% left_join(metricDf2, by=c('team1' = 'TeamID')) %>% left_join(metricDf2, by=c('team2' = "TeamID"), suffix = c("1", "2"))
tournModelDf2 <- tournModelDf2 %>% mutate(Response = factor(team1Win, levels=c(0,1), labels=c("Loss", "Win")))


# Feature engineering
tournModelDf2 <- tournModelDf2 %>% 
  mutate(effRtg = ORtg1 - DRtg1 - ORtg2 + DRtg2,
         eFgRtg = eFG1 - eFGAllowed1 - eFG2 + eFGAllowed2,
         tovEff = tov1 - tovForced1 - tov2 + tovForced2,
         ftEff = FT1 - FTAllowed1 - FT2 + FTAllowed2,
         rebEff = Rebs1 - RebAllowed1 - Rebs2 + RebAllowed2,
         stlEff = Steals1 - StealsAllowed1 - Steals2 + StealsAllowed2,
         blkEff = Blocks1 - BlocksAllowed1 - Blocks2 + BlocksAllowed2,
         defMetric = tovEff + stlEff + blkEff,
         astEff = Ast1 - AstAllowed1 - Ast2 + AstAllowed2,
         paceDiff = fgaRange1 - fgaRange2,
         sdRankDiff = sqrt(sdFinishRank1) - sqrt(sdFinishRank2),
         rankRiseDiff = sqrt(abs(avgRise1)) - sqrt(abs(avgRise2)),
         logRankDiff = log(avgFinalRank1) - log(avgFinalRank2))


tournModelDf2 <- tournModelDf2 %>% mutate(xgbPreds = predict(xgbModelTune, newdata=tournModelDf2, type='prob')[,2])


tournModelDf2$stanPreds <- tournModelDf2$effRtg %>% map_dbl(., function(x){
  y <- x * pointDiffDf$effRtg + pointDiffDf$`(Intercept)`
  perc <- sum(ifelse(y > 0, 1, 0))/length(y)
  return(perc)
})

tournModelDf2$ridgePreds <- predict(ridgePointDiff, newx=tournModelDf2 %>% select(effRtg, eFgRtg, tovEff, ftEff, rebEff, stlEff, blkEff, logRankDiff, sdRankDiff) %>% as.matrix())


tournModelDf2$stackPreds <- predict(stackGlm, newdata = tournModelDf2, type="response")

tournModelDf2 <- tournModelDf2 %>% mutate(finalPreds = (xgbPreds + stackPreds)/2)

tournModelDf2 <- tournModelDf2 %>% mutate(ID = paste('2021', team1, team2, sep='_'))

submissionDf <- tournModelDf2 %>% select(ID, Pred = finalPreds)

teamIDDf <- read_csv('MDataFiles_Stage2/MTeams.csv')
teamIDDf$TeamID <- as.factor(teamIDDf$TeamID)

viewingDf <- tournModelDf2 %>% select(ID, team1, team2, Pred = finalPreds) %>%
            left_join(teamIDDf %>% select(TeamID, TeamName), by=c("team1" = 'TeamID')) %>% left_join(teamIDDf %>% select(TeamID, TeamName), by=c("team2" = 'TeamID')) %>% 
            arrange(team1, team2)


write.csv(submissionDf, 'submission_2021_1.csv')
