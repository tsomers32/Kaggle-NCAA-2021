library(tidyverse)
library(rstanarm)
library(rstan)
library(glue)
library(glmnet)
library(caret)
library(future)
library(furrr)
library(xgboost)
library(MASS)


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


# Add metrics for 'four factors' in regular season data
regResults <- regResults %>% mutate(WeFG = (WFGM + .5*WFGM3)*100/(WFGA+WFGA3),
                      LeFG = (LFGM + .5*LFGM3)*100/(LFGA+LFGA3))

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

oRtgDataY <- ortgData$points

oRtgDataX <- ortgData %>% select(-points, -Season) %>% dummyVars("~.", .) %>% predict(., newdata=ortgData %>% select(-points, -Season)) %>% data.frame(., Season = ortgData$Season)

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
eFGDataX <- eFGData %>% select(-eFG, -Season) %>% dummyVars("~.", .) %>% predict(., newdata=eFGData %>% select(-eFG, -Season)) %>% data.frame(., Season = eFGData$Season)
eFGDataY <- eFGData$eFG


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
tovDataX <- tovData %>% select(-tov, -Season) %>% dummyVars("~.", .) %>% predict(., newdata=tovData %>% select(-tov, -Season)) %>% data.frame(., Season = tovData$Season)
tovDataY <- tovData$tov

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
ftDataX <- FTData %>% select(-ft, -Season) %>% dummyVars("~.", .) %>% predict(., newdata=FTData %>% select(-ft, -Season)) %>% data.frame(., Season = FTData$Season)
ftDataY <- FTData$ft

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


lambdas <- 10^seq(-3, 3, by=.1)

plan(multisession, workers=4)

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




tournSeedSumm <- tourneyResults %>% select(Season, team1, team2, team1Win) %>% 
  left_join(seeds, by=c("Season", 'team1'='TeamID')) %>% 
  left_join(seeds, by=c("Season", 'team2' = 'TeamID'), suffix=c('1', '2')) %>% 
  mutate(Seed1 = str_extract(Seed1, '\\d+') %>% as.numeric(),
         Seed2 = str_extract(Seed2, '\\d+') %>% as.numeric(),
         SeedDiff = Seed1-Seed2)

ggplot(data = tournSeedSumm,
       aes(x = SeedDiff, y = team1Win)) + geom_smooth()


baseGlm <- glm(team1Win ~ I(SeedDiff^3), data = tournSeedSumm, family=binomial)
summary(baseGlm)
tournSeedSumm <- tournSeedSumm %>% mutate(preds = predict(baseGlm, newdata = tournSeedSumm, type='response'))

ggplot(data = tournSeedSumm,
       aes(x = SeedDiff,
           y = preds)) + geom_point()


metricDf <- sosRtg %>% left_join(soseFG, by=c('Season', 'TeamID')) %>% left_join(sosTOV, by=c("Season", 'TeamID')) %>% left_join(sosFT, by=c("Season", "TeamID")) %>% 
            left_join(sosRebs, by=c("Season", 'TeamID')) %>% 
            left_join(sosStl, by=c("Season", "TeamID")) %>% 
            left_join(sosBlocks, by=c("Season", 'TeamID'))

tournModelDf <- tournSeedSumm %>% left_join(metricDf, by=c("Season", 'team1' = 'TeamID')) %>% left_join(metricDf, by=c("Season", 'team2' = "TeamID"), suffix = c("1", "2"))
tournModelDf <- tournModelDf %>% mutate(Response = factor(team1Win, levels=c(0,1), labels=c("Loss", "Win")))

ggplot(data = tournModelDf,
       aes(x = Steals1 - StealsAllowed1 - Steals2 + StealsAllowed2,
           y = team1Win)) + geom_smooth()



## Next step: Create modified predictor variables

ctrl <- trainControl(method='cv', number=10, search='random', savePredictions = TRUE, summaryFunction = mnLogLoss, classProbs = T, verbose=T)

xgbModel <- train(Response ~ ORtg1 + ORtg2 +
                             DRtg1 + DRtg2 +
                             eFG1 + eFG2 +
                             eFGAllowed1 + eFGAllowed2 +
                             tov1 + tov2 +
                             tovForced1 + tovForced2 +
                             FT1 + FT2 +
                             FTAllowed1 + FTAllowed2,
                  data = tournModelDf,
                  method = 'xgbTree',
                  trControl = ctrl,
                  tuneLength = 30,
                  metric = 'logLoss')

xgbModel$results
varImp(xgbModel)






