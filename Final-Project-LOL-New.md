Final Project LOL
================
Due: Wednesday 12/19 at 11:59am, Canvas submission

## The Data

The dataset I have chosen to conduct my regression analysis is the
[(LoL) League of Legends Ranked Games
dataset](https://www.kaggle.com/datasets/datasnaek/league-of-legends/?select=games.csv).
This data looks at ranked League of Legends games played during season 9
in the EUW region. This dataset was collected using the Riot Games API,
which makes it easy to look up and collect information on a users ranked
history and collect their games.

``` r
# Load packages
library(ggplot2)
library(rstanarm)
library(bayesplot)
library(bayesrules)
library(tidyverse)
library(tidybayes)
library(dplyr)
library(broom.mixed)
library(interactions)
```

To preface this analysis, one must first understand the game of League
of Legends, or LOL for short. In LOL, two 5 person teams fight to
destroy each others nexus and get to choose and ban a champion, each
with a different set of skills and abilities, and 2 summoner spells to
do so. Along the way, the players must both fight each other, destroy
objectives such as towers and inhibitors, and strategize effectively to
reach the opponent’s nexus. There are also several monsters that teams
can defeat to buff them as well, such as Rift Herald, Baron Nashor, and
an assorted set of Dragons.

We will use `lol_data` to build various models of League of Legends
`gameDuration`. Throughout, we’ll utilize weakly informative priors and
a basic understanding that LOL games usually are [30
minutes](https://www.leagueofgraphs.com/stats/game-durations) but can
range from 25-35 minutes. We will asses an array of different predictors
and combinations to determine what can provide us the best fit for
predicting the game duration of a League of Legends game. A base
criteria for these games is there at least needs to be one tower, one
inhibitor, and one champion kill for each game. This is to mitigate the
amount of games that teams may have forfeited due to a player leaving or
trolling.

``` r
lol_data <- read.csv("games.csv",
                    sep=",",
                    na.strings=c(""," ","NA","N/A")
)
head(lol_data)
```

    ##       gameId creationTime gameDuration seasonId winner firstBlood firstTower
    ## 1 3326086514 1.504279e+12         1949        9      1          2          1
    ## 2 3229566029 1.497849e+12         1851        9      1          1          1
    ## 3 3327363504 1.504360e+12         1493        9      1          2          1
    ## 4 3326856598 1.504349e+12         1758        9      1          1          1
    ## 5 3330080762 1.504554e+12         2094        9      1          2          1
    ## 6 3287435705 1.501668e+12         2059        9      1          2          2
    ##   firstInhibitor firstBaron firstDragon firstRiftHerald t1_champ1id
    ## 1              1          1           1               2           8
    ## 2              1          0           1               1         119
    ## 3              1          1           2               0          18
    ## 4              1          1           1               0          57
    ## 5              1          1           1               0          19
    ## 6              1          1           2               0          40
    ##   t1_champ1_sum1 t1_champ1_sum2 t1_champ2id t1_champ2_sum1 t1_champ2_sum2
    ## 1             12              4         432              3              4
    ## 2              7              4          39             12              4
    ## 3              4              7         141             11              4
    ## 4              4             12          63              4             14
    ## 5              4             12          29             11              4
    ## 6              3              4         141             11              4
    ##   t1_champ3id t1_champ3_sum1 t1_champ3_sum2 t1_champ4id t1_champ4_sum1
    ## 1          96              4              7          11             11
    ## 2          76              4              3          10              4
    ## 3         267              3              4          68              4
    ## 4          29              4              7          61              4
    ## 5          40              4              3         119              4
    ## 6          24             12              4          45              3
    ##   t1_champ4_sum2 t1_champ5id t1_champ5_sum1 t1_champ5_sum2 t1_towerKills
    ## 1              6         112              4             14            11
    ## 2             14          35              4             11            10
    ## 3             12          38             12              4             8
    ## 4              1          36             11              4             9
    ## 5              7         134              7              4             9
    ## 6              4          67              4              7             8
    ##   t1_inhibitorKills t1_baronKills t1_dragonKills t1_riftHeraldKills t1_ban1
    ## 1                 1             2              3                  0      92
    ## 2                 4             0              2                  1      51
    ## 3                 1             1              1                  0     117
    ## 4                 2             1              2                  0     238
    ## 5                 2             1              3                  0      90
    ## 6                 1             1              1                  0     117
    ##   t1_ban2 t1_ban3 t1_ban4 t1_ban5 t2_champ1id t2_champ1_sum1 t2_champ1_sum2
    ## 1      40      69     119     141         104             11              4
    ## 2     122      17     498      19          54              4             12
    ## 3      40      29      16      53          69              4              7
    ## 4      67     516     114      31          90             14              4
    ## 5      64     412      25      31          37              3              4
    ## 6       6     238     122     105          92              4             12
    ##   t2_champ2id t2_champ2_sum1 t2_champ2_sum2 t2_champ3id t2_champ3_sum1
    ## 1         498              4              7         122              6
    ## 2          25              4             14         120             11
    ## 3         412             14              4         126              4
    ## 4          19             11              4         412              4
    ## 5          59              4             12         141             11
    ## 6          15              4              7         245             12
    ##   t2_champ3_sum2 t2_champ4id t2_champ4_sum1 t2_champ4_sum2 t2_champ5id
    ## 1              4         238             14              4         412
    ## 2              4         157              4             14          92
    ## 3             12          24              4             11          22
    ## 4              3          92              4             14          22
    ## 5              4          38              4             12          51
    ## 6              4           2              4             11          12
    ##   t2_champ5_sum1 t2_champ5_sum2 t2_towerKills t2_inhibitorKills t2_baronKills
    ## 1              4              3             5                 0             0
    ## 2              4              7             2                 0             0
    ## 3              7              4             2                 0             0
    ## 4              4              7             0                 0             0
    ## 5              4              7             3                 0             0
    ## 6              4             14             6                 0             0
    ##   t2_dragonKills t2_riftHeraldKills t2_ban1 t2_ban2 t2_ban3 t2_ban4 t2_ban5
    ## 1              1                  1     114      67      43      16      51
    ## 2              0                  0      11      67     238      51     420
    ## 3              1                  0     157     238     121      57      28
    ## 4              0                  0     164      18     141      40      51
    ## 5              1                  0      86      11     201     122      18
    ## 6              3                  0     119     134     154      63      31

## Setting Up the Data

Transforming data to fit baseline of at least one kill, tower, and
inhibitor

``` r
lol_data <- lol_data[!(lol_data$firstBlood == 0 | lol_data$firstTower == 0 | lol_data$firstInhibitor == 0), ]
```

``` r
nrow(lol_data)
```

    ## [1] 45214

``` r
colnames(lol_data)
```

    ##  [1] "gameId"             "creationTime"       "gameDuration"      
    ##  [4] "seasonId"           "winner"             "firstBlood"        
    ##  [7] "firstTower"         "firstInhibitor"     "firstBaron"        
    ## [10] "firstDragon"        "firstRiftHerald"    "t1_champ1id"       
    ## [13] "t1_champ1_sum1"     "t1_champ1_sum2"     "t1_champ2id"       
    ## [16] "t1_champ2_sum1"     "t1_champ2_sum2"     "t1_champ3id"       
    ## [19] "t1_champ3_sum1"     "t1_champ3_sum2"     "t1_champ4id"       
    ## [22] "t1_champ4_sum1"     "t1_champ4_sum2"     "t1_champ5id"       
    ## [25] "t1_champ5_sum1"     "t1_champ5_sum2"     "t1_towerKills"     
    ## [28] "t1_inhibitorKills"  "t1_baronKills"      "t1_dragonKills"    
    ## [31] "t1_riftHeraldKills" "t1_ban1"            "t1_ban2"           
    ## [34] "t1_ban3"            "t1_ban4"            "t1_ban5"           
    ## [37] "t2_champ1id"        "t2_champ1_sum1"     "t2_champ1_sum2"    
    ## [40] "t2_champ2id"        "t2_champ2_sum1"     "t2_champ2_sum2"    
    ## [43] "t2_champ3id"        "t2_champ3_sum1"     "t2_champ3_sum2"    
    ## [46] "t2_champ4id"        "t2_champ4_sum1"     "t2_champ4_sum2"    
    ## [49] "t2_champ5id"        "t2_champ5_sum1"     "t2_champ5_sum2"    
    ## [52] "t2_towerKills"      "t2_inhibitorKills"  "t2_baronKills"     
    ## [55] "t2_dragonKills"     "t2_riftHeraldKills" "t2_ban1"           
    ## [58] "t2_ban2"            "t2_ban3"            "t2_ban4"           
    ## [61] "t2_ban5"

We can clean up our data by omitting unnecessary columns

``` r
lol_data <- subset(lol_data, select = -c(t2_ban1, t2_ban2, t2_ban3, t2_ban4, t2_ban5, t1_ban1, t1_ban2, t1_ban3, t1_ban4, t1_ban5, t1_champ1_sum1, t1_champ2_sum1, t1_champ3_sum1, t1_champ4_sum1, t1_champ5_sum1, t2_champ5_sum1, t2_champ1_sum1, t2_champ2_sum1, t2_champ3_sum1, t2_champ4_sum1, t2_champ5_sum1, t1_champ1_sum2, t1_champ2_sum2, t1_champ3_sum2, t1_champ4_sum2, t1_champ5_sum2, t2_champ1_sum2, t2_champ1_sum2, t2_champ2_sum2, t2_champ3_sum2, t2_champ4_sum2, t2_champ5_sum2, t1_champ1id, t1_champ2id, t1_champ3id, t1_champ4id, t1_champ5id, t2_champ1id, t2_champ2id, t2_champ3id, t2_champ4id, t2_champ5id))

colnames(lol_data)
```

    ##  [1] "gameId"             "creationTime"       "gameDuration"      
    ##  [4] "seasonId"           "winner"             "firstBlood"        
    ##  [7] "firstTower"         "firstInhibitor"     "firstBaron"        
    ## [10] "firstDragon"        "firstRiftHerald"    "t1_towerKills"     
    ## [13] "t1_inhibitorKills"  "t1_baronKills"      "t1_dragonKills"    
    ## [16] "t1_riftHeraldKills" "t2_towerKills"      "t2_inhibitorKills" 
    ## [19] "t2_baronKills"      "t2_dragonKills"     "t2_riftHeraldKills"

We can also covert some of the columns to factors for team 1 and team 2
for ease of visualization and interpretation

``` r
lol_data$winner <- as.factor(lol_data$winner)
lol_data$firstBlood <- as.factor(lol_data$firstBlood)
lol_data$firstTower <- as.factor(lol_data$firstTower)
lol_data$firstInhibitor <- as.factor(lol_data$firstInhibitor)
lol_data$firstBaron <- as.factor(lol_data$firstBaron)
lol_data$firstDragon <- as.factor(lol_data$firstDragon)
lol_data$firstRiftHerald <- as.factor(lol_data$firstRiftHerald)
```

``` r
head(lol_data)
```

    ##       gameId creationTime gameDuration seasonId winner firstBlood firstTower
    ## 1 3326086514 1.504279e+12         1949        9      1          2          1
    ## 2 3229566029 1.497849e+12         1851        9      1          1          1
    ## 3 3327363504 1.504360e+12         1493        9      1          2          1
    ## 4 3326856598 1.504349e+12         1758        9      1          1          1
    ## 5 3330080762 1.504554e+12         2094        9      1          2          1
    ## 6 3287435705 1.501668e+12         2059        9      1          2          2
    ##   firstInhibitor firstBaron firstDragon firstRiftHerald t1_towerKills
    ## 1              1          1           1               2            11
    ## 2              1          0           1               1            10
    ## 3              1          1           2               0             8
    ## 4              1          1           1               0             9
    ## 5              1          1           1               0             9
    ## 6              1          1           2               0             8
    ##   t1_inhibitorKills t1_baronKills t1_dragonKills t1_riftHeraldKills
    ## 1                 1             2              3                  0
    ## 2                 4             0              2                  1
    ## 3                 1             1              1                  0
    ## 4                 2             1              2                  0
    ## 5                 2             1              3                  0
    ## 6                 1             1              1                  0
    ##   t2_towerKills t2_inhibitorKills t2_baronKills t2_dragonKills
    ## 1             5                 0             0              1
    ## 2             2                 0             0              0
    ## 3             2                 0             0              1
    ## 4             0                 0             0              0
    ## 5             3                 0             0              1
    ## 6             6                 0             0              3
    ##   t2_riftHeraldKills
    ## 1                  1
    ## 2                  0
    ## 3                  0
    ## 4                  0
    ## 5                  0
    ## 6                  0

## Summary Analysis

Here are some visualizations and statistics to help better understand
our data and what we are working with.

``` r
summary(lol_data)
```

    ##      gameId           creationTime        gameDuration     seasonId winner   
    ##  Min.   :3.215e+09   Min.   :1.497e+12   Min.   : 477   Min.   :9   1:22867  
    ##  1st Qu.:3.292e+09   1st Qu.:1.502e+12   1st Qu.:1632   1st Qu.:9   2:22347  
    ##  Median :3.320e+09   Median :1.504e+12   Median :1895   Median :9            
    ##  Mean   :3.306e+09   Mean   :1.503e+12   Mean   :1930   Mean   :9            
    ##  3rd Qu.:3.327e+09   3rd Qu.:1.504e+12   3rd Qu.:2196   3rd Qu.:9            
    ##  Max.   :3.332e+09   Max.   :1.505e+12   Max.   :4728   Max.   :9            
    ##  firstBlood firstTower firstInhibitor firstBaron firstDragon firstRiftHerald
    ##  1:23151    1:23248    1:23054        0:14617    0:  434     0:21922        
    ##  2:22063    2:21966    2:22160        1:14469    1:22258     1:11933        
    ##                                       2:16128    2:22522     2:11359        
    ##                                                                             
    ##                                                                             
    ##                                                                             
    ##  t1_towerKills    t1_inhibitorKills t1_baronKills    t1_dragonKills 
    ##  Min.   : 0.000   Min.   : 0.000    Min.   :0.0000   Min.   :0.000  
    ##  1st Qu.: 3.000   1st Qu.: 0.000    1st Qu.:0.0000   1st Qu.:0.000  
    ##  Median : 7.000   Median : 1.000    Median :0.0000   Median :1.000  
    ##  Mean   : 6.173   Mean   : 1.159    Mean   :0.4173   Mean   :1.483  
    ##  3rd Qu.:10.000   3rd Qu.: 2.000    3rd Qu.:1.0000   3rd Qu.:2.000  
    ##  Max.   :11.000   Max.   :10.000    Max.   :5.0000   Max.   :6.000  
    ##  t1_riftHeraldKills t2_towerKills    t2_inhibitorKills t2_baronKills   
    ##  Min.   :0.0000     Min.   : 0.000   Min.   : 0.000    Min.   :0.0000  
    ##  1st Qu.:0.0000     1st Qu.: 2.000   1st Qu.: 0.000    1st Qu.:0.0000  
    ##  Median :0.0000     Median : 7.000   Median : 1.000    Median :0.0000  
    ##  Mean   :0.2639     Mean   : 6.014   Mean   : 1.122    Mean   :0.4641  
    ##  3rd Qu.:1.0000     3rd Qu.:10.000   3rd Qu.: 2.000    3rd Qu.:1.0000  
    ##  Max.   :1.0000     Max.   :11.000   Max.   :10.000    Max.   :4.0000  
    ##  t2_dragonKills  t2_riftHeraldKills
    ##  Min.   :0.000   Min.   :0.0000    
    ##  1st Qu.:0.000   1st Qu.:0.0000    
    ##  Median :1.000   Median :0.0000    
    ##  Mean   :1.506   Mean   :0.2512    
    ##  3rd Qu.:2.000   3rd Qu.:1.0000    
    ##  Max.   :6.000   Max.   :1.0000

Some conclusions we can make here:

- The mean game time is around 1930 seconds (roughly 32 minutes)

- Game duration lies mostly between 1632 and 2196

- Team 1 has won more games, has more first bloods, towers, inhibitors,
  and rift heralds

- Team 2 has more first barons and dragons.

- Team 1’s tower, inhibitor, and herald kills are higher than Team 2’s
  which is reflected from them getting the first of each more often than
  not.

- Team 2’s dragon and baron kills are higher than Team 1’s which is
  reflected from them getting the first of each more often than not.

## Visualizations

We can use ggplot do visualize some of our findings

### Response Distribution

``` r
ggplot(lol_data, aes(x = gameDuration)) +
  geom_histogram(binwidth = 60, fill = "blue", color = "black") +
  labs(x = "Game Duration (seconds)", y = "Frequency", title = "Distribution of Game Duration") +
  theme_minimal()
```

![](Final-Project-LOL-New_files/figure-gfm/unnamed-chunk-9-1.png)<!-- -->

Game duration seems to be a bit right-skewed

### First Comparisons

``` r
ggplot(lol_data, aes(x = winner, fill = winner)) + 
      geom_bar() + 
      theme(text = element_text(size=9)) +
      labs(y = "Winners")
```

![](Final-Project-LOL-New_files/figure-gfm/unnamed-chunk-10-1.png)<!-- -->

``` r
ggplot(lol_data, aes(x = firstBlood, fill = firstBlood)) + 
      geom_bar() + 
      theme(text = element_text(size=9)) +
      labs(y = "First Bloods")
```

![](Final-Project-LOL-New_files/figure-gfm/unnamed-chunk-11-1.png)<!-- -->

``` r
ggplot(lol_data, aes(x = firstTower, fill = firstTower)) + 
      geom_bar() + 
      theme(text = element_text(size=9)) +
      labs(y = "First Towers")
```

![](Final-Project-LOL-New_files/figure-gfm/unnamed-chunk-12-1.png)<!-- -->

``` r
ggplot(lol_data, aes(x = firstInhibitor, fill = firstInhibitor)) + 
      geom_bar() + 
      theme(text = element_text(size=9)) +
      labs(y = "First Inhibitors")
```

![](Final-Project-LOL-New_files/figure-gfm/unnamed-chunk-13-1.png)<!-- -->

``` r
ggplot(lol_data, aes(x = firstDragon, fill = firstDragon)) + 
      geom_bar() + 
      theme(text = element_text(size=9)) +
      labs(y = "First Dragons")
```

![](Final-Project-LOL-New_files/figure-gfm/unnamed-chunk-14-1.png)<!-- -->

``` r
ggplot(lol_data, aes(x = firstRiftHerald, fill = firstRiftHerald)) + 
      geom_bar() + 
      theme(text = element_text(size=9)) +
      labs(y = "First Rift Heralds")
```

![](Final-Project-LOL-New_files/figure-gfm/unnamed-chunk-15-1.png)<!-- -->

``` r
ggplot(lol_data, aes(x = firstBaron, fill = firstBaron)) + 
      geom_bar() + 
      theme(text = element_text(size=9)) +
      labs(y = "First Barons")
```

![](Final-Project-LOL-New_files/figure-gfm/unnamed-chunk-16-1.png)<!-- -->

These all are consistent with our previous summary findings about first
categories

### Total Objectives Comparisons

``` r
ggplot(lol_data, aes(x = t1_towerKills)) + 
      geom_bar() + 
      theme(text = element_text(size=10)) +
      labs(y = "Count")
```

![](Final-Project-LOL-New_files/figure-gfm/unnamed-chunk-17-1.png)<!-- -->

``` r
ggplot(lol_data, aes(x = t2_towerKills)) + 
      geom_bar() + 
      theme(text = element_text(size=10)) +
      labs(y = "Count")
```

![](Final-Project-LOL-New_files/figure-gfm/unnamed-chunk-18-1.png)<!-- -->

``` r
tower_kills <- data.frame(Team = c("T1", "T2"),
                          Kills = c(sum(lol_data$t1_towerKills, na.rm = TRUE), 
                                    sum(lol_data$t2_towerKills, na.rm = TRUE)))

ggplot(tower_kills, aes(x = Team, y = Kills, fill = Team)) +
  geom_bar(stat = "identity") +
  labs(x = "Team", y = "Total Tower Kills", title = "Comparison of Tower Kills between T1 and T2")
```

![](Final-Project-LOL-New_files/figure-gfm/unnamed-chunk-19-1.png)<!-- -->

More often than not, both teams tend to kill \>6 towers per game

``` r
ggplot(lol_data, aes(x = t1_inhibitorKills)) + 
      geom_bar() + 
      theme(text = element_text(size=10)) +
      labs(y = "Count")
```

![](Final-Project-LOL-New_files/figure-gfm/unnamed-chunk-20-1.png)<!-- -->

``` r
ggplot(lol_data, aes(x = t2_inhibitorKills)) + 
      geom_bar() + 
      theme(text = element_text(size=10)) +
      labs(y = "Count")
```

![](Final-Project-LOL-New_files/figure-gfm/unnamed-chunk-21-1.png)<!-- -->

``` r
# Calculate the total number of inhibitor kills for each team
inhibitor_kills <- data.frame(Team = c("T1", "T2"),
                              Kills = c(sum(lol_data$t1_inhibitorKills, na.rm = TRUE), 
                                        sum(lol_data$t2_inhibitorKills, na.rm = TRUE)))

ggplot(inhibitor_kills, aes(x = Team, y = Kills, fill = Team)) +
  geom_bar(stat = "identity") +
  labs(x = "Team", y = "Total Inhibitor Kills", title = "Comparison of Inhibitor Kills between T1 and T2")
```

![](Final-Project-LOL-New_files/figure-gfm/unnamed-chunk-22-1.png)<!-- -->

Games usually end with 0 to 2 inhibitors being destroyed, likely due to
late game forfeits or steam rolls (which is often the case in many
games)

``` r
ggplot(lol_data, aes(x = t1_dragonKills)) + 
      geom_bar() + 
      theme(text = element_text(size=10)) +
      labs(y = "Count")
```

![](Final-Project-LOL-New_files/figure-gfm/unnamed-chunk-23-1.png)<!-- -->

``` r
ggplot(lol_data, aes(x = t2_dragonKills)) + 
      geom_bar() + 
      theme(text = element_text(size=10)) +
      labs(y = "Count")
```

![](Final-Project-LOL-New_files/figure-gfm/unnamed-chunk-24-1.png)<!-- -->

``` r
# Calculate the total number of dragon kills for each team
dragon_kills <- data.frame(Team = c("T1", "T2"),
                           Kills = c(sum(lol_data$t1_dragonKills, na.rm = TRUE), 
                                     sum(lol_data$t2_dragonKills, na.rm = TRUE)))

ggplot(dragon_kills, aes(x = Team, y = Kills, fill = Team)) +
  geom_bar(stat = "identity") +
  labs(x = "Team", y = "Total Dragon Kills", title = "Comparison of Dragon Kills between T1 and T2")
```

![](Final-Project-LOL-New_files/figure-gfm/unnamed-chunk-25-1.png)<!-- -->

Dragon kills comfortably lie within the 0-2 range

``` r
ggplot(lol_data, aes(x = t1_riftHeraldKills)) + 
      geom_bar() + 
      theme(text = element_text(size=10)) +
      labs(y = "Count")
```

![](Final-Project-LOL-New_files/figure-gfm/unnamed-chunk-26-1.png)<!-- -->

``` r
ggplot(lol_data, aes(x = t2_riftHeraldKills)) + 
      geom_bar() + 
      theme(text = element_text(size=10)) +
      labs(y = "Count")
```

![](Final-Project-LOL-New_files/figure-gfm/unnamed-chunk-27-1.png)<!-- -->

``` r
# Calculate the total number of rift herald kills for each team
riftHerald_kills <- data.frame(Team = c("T1", "T2"),
                               Kills = c(sum(lol_data$t1_riftHeraldKills, na.rm = TRUE), 
                                         sum(lol_data$t2_riftHeraldKills, na.rm = TRUE)))

ggplot(riftHerald_kills, aes(x = Team, y = Kills, fill = Team)) +
  geom_bar(stat = "identity") +
  labs(x = "Team", y = "Total Rift Herald Kills", title = "Comparison of Rift Herald Kills between T1 and T2")
```

![](Final-Project-LOL-New_files/figure-gfm/unnamed-chunk-28-1.png)<!-- -->

Rift herald only spawns from 9:50 - 19:45 during the game and is most
often only taken once if at all

``` r
ggplot(lol_data, aes(x = t1_baronKills)) + 
      geom_bar() + 
      theme(text = element_text(size=10)) +
      labs(y = "Count")
```

![](Final-Project-LOL-New_files/figure-gfm/unnamed-chunk-29-1.png)<!-- -->

``` r
ggplot(lol_data, aes(x = t2_baronKills)) + 
      geom_bar() + 
      theme(text = element_text(size=10)) +
      labs(y = "Count")
```

![](Final-Project-LOL-New_files/figure-gfm/unnamed-chunk-30-1.png)<!-- -->

``` r
# Calculate the total number of baron kills for each team
baron_kills <- data.frame(Team = c("T1", "T2"),
                          Kills = c(sum(lol_data$t1_baronKills, na.rm = TRUE), 
                                    sum(lol_data$t2_baronKills, na.rm = TRUE)))

ggplot(baron_kills, aes(x = Team, y = Kills, fill = Team)) +
  geom_bar(stat = "identity") +
  labs(x = "Team", y = "Total Baron Kills", title = "Comparison of Baron Kills between T1 and T2")
```

![](Final-Project-LOL-New_files/figure-gfm/unnamed-chunk-31-1.png)<!-- -->

Baron spawns after the rift herald and usually is game over if taken and
used properly, hence why the amount taken is usually 0 or 1 as sometimes
games end before it can be taken

## Base Model and Prior Modeling

Lets start off with a base model to see if we can get insight from just
the winner of the match

``` r
base <- stan_glm(gameDuration ~ winner,
  data = lol_data, family = gaussian, 
  prior_intercept = normal(1800, 150, autoscale = TRUE),
  prior = normal(0, 2.5, autoscale = TRUE), 
  prior_aux = exponential(1, autoscale = TRUE),
  chains = 4, iter = 5000*2, seed = 84735)
```

    ## 
    ## SAMPLING FOR MODEL 'continuous' NOW (CHAIN 1).
    ## Chain 1: 
    ## Chain 1: Gradient evaluation took 0.00202 seconds
    ## Chain 1: 1000 transitions using 10 leapfrog steps per transition would take 20.2 seconds.
    ## Chain 1: Adjust your expectations accordingly!
    ## Chain 1: 
    ## Chain 1: 
    ## Chain 1: Iteration:    1 / 10000 [  0%]  (Warmup)
    ## Chain 1: Iteration: 1000 / 10000 [ 10%]  (Warmup)
    ## Chain 1: Iteration: 2000 / 10000 [ 20%]  (Warmup)
    ## Chain 1: Iteration: 3000 / 10000 [ 30%]  (Warmup)
    ## Chain 1: Iteration: 4000 / 10000 [ 40%]  (Warmup)
    ## Chain 1: Iteration: 5000 / 10000 [ 50%]  (Warmup)
    ## Chain 1: Iteration: 5001 / 10000 [ 50%]  (Sampling)
    ## Chain 1: Iteration: 6000 / 10000 [ 60%]  (Sampling)
    ## Chain 1: Iteration: 7000 / 10000 [ 70%]  (Sampling)
    ## Chain 1: Iteration: 8000 / 10000 [ 80%]  (Sampling)
    ## Chain 1: Iteration: 9000 / 10000 [ 90%]  (Sampling)
    ## Chain 1: Iteration: 10000 / 10000 [100%]  (Sampling)
    ## Chain 1: 
    ## Chain 1:  Elapsed Time: 0.199 seconds (Warm-up)
    ## Chain 1:                13.585 seconds (Sampling)
    ## Chain 1:                13.784 seconds (Total)
    ## Chain 1: 
    ## 
    ## SAMPLING FOR MODEL 'continuous' NOW (CHAIN 2).
    ## Chain 2: 
    ## Chain 2: Gradient evaluation took 1.5e-05 seconds
    ## Chain 2: 1000 transitions using 10 leapfrog steps per transition would take 0.15 seconds.
    ## Chain 2: Adjust your expectations accordingly!
    ## Chain 2: 
    ## Chain 2: 
    ## Chain 2: Iteration:    1 / 10000 [  0%]  (Warmup)
    ## Chain 2: Iteration: 1000 / 10000 [ 10%]  (Warmup)
    ## Chain 2: Iteration: 2000 / 10000 [ 20%]  (Warmup)
    ## Chain 2: Iteration: 3000 / 10000 [ 30%]  (Warmup)
    ## Chain 2: Iteration: 4000 / 10000 [ 40%]  (Warmup)
    ## Chain 2: Iteration: 5000 / 10000 [ 50%]  (Warmup)
    ## Chain 2: Iteration: 5001 / 10000 [ 50%]  (Sampling)
    ## Chain 2: Iteration: 6000 / 10000 [ 60%]  (Sampling)
    ## Chain 2: Iteration: 7000 / 10000 [ 70%]  (Sampling)
    ## Chain 2: Iteration: 8000 / 10000 [ 80%]  (Sampling)
    ## Chain 2: Iteration: 9000 / 10000 [ 90%]  (Sampling)
    ## Chain 2: Iteration: 10000 / 10000 [100%]  (Sampling)
    ## Chain 2: 
    ## Chain 2:  Elapsed Time: 0.209 seconds (Warm-up)
    ## Chain 2:                13.481 seconds (Sampling)
    ## Chain 2:                13.69 seconds (Total)
    ## Chain 2: 
    ## 
    ## SAMPLING FOR MODEL 'continuous' NOW (CHAIN 3).
    ## Chain 3: 
    ## Chain 3: Gradient evaluation took 1.6e-05 seconds
    ## Chain 3: 1000 transitions using 10 leapfrog steps per transition would take 0.16 seconds.
    ## Chain 3: Adjust your expectations accordingly!
    ## Chain 3: 
    ## Chain 3: 
    ## Chain 3: Iteration:    1 / 10000 [  0%]  (Warmup)
    ## Chain 3: Iteration: 1000 / 10000 [ 10%]  (Warmup)
    ## Chain 3: Iteration: 2000 / 10000 [ 20%]  (Warmup)
    ## Chain 3: Iteration: 3000 / 10000 [ 30%]  (Warmup)
    ## Chain 3: Iteration: 4000 / 10000 [ 40%]  (Warmup)
    ## Chain 3: Iteration: 5000 / 10000 [ 50%]  (Warmup)
    ## Chain 3: Iteration: 5001 / 10000 [ 50%]  (Sampling)
    ## Chain 3: Iteration: 6000 / 10000 [ 60%]  (Sampling)
    ## Chain 3: Iteration: 7000 / 10000 [ 70%]  (Sampling)
    ## Chain 3: Iteration: 8000 / 10000 [ 80%]  (Sampling)
    ## Chain 3: Iteration: 9000 / 10000 [ 90%]  (Sampling)
    ## Chain 3: Iteration: 10000 / 10000 [100%]  (Sampling)
    ## Chain 3: 
    ## Chain 3:  Elapsed Time: 0.215 seconds (Warm-up)
    ## Chain 3:                14.101 seconds (Sampling)
    ## Chain 3:                14.316 seconds (Total)
    ## Chain 3: 
    ## 
    ## SAMPLING FOR MODEL 'continuous' NOW (CHAIN 4).
    ## Chain 4: 
    ## Chain 4: Gradient evaluation took 1.6e-05 seconds
    ## Chain 4: 1000 transitions using 10 leapfrog steps per transition would take 0.16 seconds.
    ## Chain 4: Adjust your expectations accordingly!
    ## Chain 4: 
    ## Chain 4: 
    ## Chain 4: Iteration:    1 / 10000 [  0%]  (Warmup)
    ## Chain 4: Iteration: 1000 / 10000 [ 10%]  (Warmup)
    ## Chain 4: Iteration: 2000 / 10000 [ 20%]  (Warmup)
    ## Chain 4: Iteration: 3000 / 10000 [ 30%]  (Warmup)
    ## Chain 4: Iteration: 4000 / 10000 [ 40%]  (Warmup)
    ## Chain 4: Iteration: 5000 / 10000 [ 50%]  (Warmup)
    ## Chain 4: Iteration: 5001 / 10000 [ 50%]  (Sampling)
    ## Chain 4: Iteration: 6000 / 10000 [ 60%]  (Sampling)
    ## Chain 4: Iteration: 7000 / 10000 [ 70%]  (Sampling)
    ## Chain 4: Iteration: 8000 / 10000 [ 80%]  (Sampling)
    ## Chain 4: Iteration: 9000 / 10000 [ 90%]  (Sampling)
    ## Chain 4: Iteration: 10000 / 10000 [100%]  (Sampling)
    ## Chain 4: 
    ## Chain 4:  Elapsed Time: 0.218 seconds (Warm-up)
    ## Chain 4:                13.179 seconds (Sampling)
    ## Chain 4:                13.397 seconds (Total)
    ## Chain 4:

``` r
prior_summary(base) 
```

    ## Priors for model 'base' 
    ## ------
    ## Intercept (after predictors centered)
    ##   Specified prior:
    ##     ~ normal(location = 1800, scale = 150)
    ##   Adjusted prior:
    ##     ~ normal(location = 1800, scale = 64093)
    ## 
    ## Coefficients
    ##   Specified prior:
    ##     ~ normal(location = 0, scale = 2.5)
    ##   Adjusted prior:
    ##     ~ normal(location = 0, scale = 2137)
    ## 
    ## Auxiliary (sigma)
    ##   Specified prior:
    ##     ~ exponential(rate = 1)
    ##   Adjusted prior:
    ##     ~ exponential(rate = 0.0023)
    ## ------
    ## See help('prior_summary.stanreg') for more details

A prior predictive check can be used to see what we can potentially
expect from the data alone and what the stan model viewed as appropriate
for our predictions

``` r
base_priors <- update(base, prior_PD = TRUE)
```

    ## 
    ## SAMPLING FOR MODEL 'continuous' NOW (CHAIN 1).
    ## Chain 1: 
    ## Chain 1: Gradient evaluation took 1.1e-05 seconds
    ## Chain 1: 1000 transitions using 10 leapfrog steps per transition would take 0.11 seconds.
    ## Chain 1: Adjust your expectations accordingly!
    ## Chain 1: 
    ## Chain 1: 
    ## Chain 1: Iteration:    1 / 10000 [  0%]  (Warmup)
    ## Chain 1: Iteration: 1000 / 10000 [ 10%]  (Warmup)
    ## Chain 1: Iteration: 2000 / 10000 [ 20%]  (Warmup)
    ## Chain 1: Iteration: 3000 / 10000 [ 30%]  (Warmup)
    ## Chain 1: Iteration: 4000 / 10000 [ 40%]  (Warmup)
    ## Chain 1: Iteration: 5000 / 10000 [ 50%]  (Warmup)
    ## Chain 1: Iteration: 5001 / 10000 [ 50%]  (Sampling)
    ## Chain 1: Iteration: 6000 / 10000 [ 60%]  (Sampling)
    ## Chain 1: Iteration: 7000 / 10000 [ 70%]  (Sampling)
    ## Chain 1: Iteration: 8000 / 10000 [ 80%]  (Sampling)
    ## Chain 1: Iteration: 9000 / 10000 [ 90%]  (Sampling)
    ## Chain 1: Iteration: 10000 / 10000 [100%]  (Sampling)
    ## Chain 1: 
    ## Chain 1:  Elapsed Time: 3.812 seconds (Warm-up)
    ## Chain 1:                0.134 seconds (Sampling)
    ## Chain 1:                3.946 seconds (Total)
    ## Chain 1: 
    ## 
    ## SAMPLING FOR MODEL 'continuous' NOW (CHAIN 2).
    ## Chain 2: 
    ## Chain 2: Gradient evaluation took 7e-06 seconds
    ## Chain 2: 1000 transitions using 10 leapfrog steps per transition would take 0.07 seconds.
    ## Chain 2: Adjust your expectations accordingly!
    ## Chain 2: 
    ## Chain 2: 
    ## Chain 2: Iteration:    1 / 10000 [  0%]  (Warmup)
    ## Chain 2: Iteration: 1000 / 10000 [ 10%]  (Warmup)
    ## Chain 2: Iteration: 2000 / 10000 [ 20%]  (Warmup)
    ## Chain 2: Iteration: 3000 / 10000 [ 30%]  (Warmup)
    ## Chain 2: Iteration: 4000 / 10000 [ 40%]  (Warmup)
    ## Chain 2: Iteration: 5000 / 10000 [ 50%]  (Warmup)
    ## Chain 2: Iteration: 5001 / 10000 [ 50%]  (Sampling)
    ## Chain 2: Iteration: 6000 / 10000 [ 60%]  (Sampling)
    ## Chain 2: Iteration: 7000 / 10000 [ 70%]  (Sampling)
    ## Chain 2: Iteration: 8000 / 10000 [ 80%]  (Sampling)
    ## Chain 2: Iteration: 9000 / 10000 [ 90%]  (Sampling)
    ## Chain 2: Iteration: 10000 / 10000 [100%]  (Sampling)
    ## Chain 2: 
    ## Chain 2:  Elapsed Time: 2.355 seconds (Warm-up)
    ## Chain 2:                0.119 seconds (Sampling)
    ## Chain 2:                2.474 seconds (Total)
    ## Chain 2: 
    ## 
    ## SAMPLING FOR MODEL 'continuous' NOW (CHAIN 3).
    ## Chain 3: 
    ## Chain 3: Gradient evaluation took 6e-06 seconds
    ## Chain 3: 1000 transitions using 10 leapfrog steps per transition would take 0.06 seconds.
    ## Chain 3: Adjust your expectations accordingly!
    ## Chain 3: 
    ## Chain 3: 
    ## Chain 3: Iteration:    1 / 10000 [  0%]  (Warmup)
    ## Chain 3: Iteration: 1000 / 10000 [ 10%]  (Warmup)
    ## Chain 3: Iteration: 2000 / 10000 [ 20%]  (Warmup)
    ## Chain 3: Iteration: 3000 / 10000 [ 30%]  (Warmup)
    ## Chain 3: Iteration: 4000 / 10000 [ 40%]  (Warmup)
    ## Chain 3: Iteration: 5000 / 10000 [ 50%]  (Warmup)
    ## Chain 3: Iteration: 5001 / 10000 [ 50%]  (Sampling)
    ## Chain 3: Iteration: 6000 / 10000 [ 60%]  (Sampling)
    ## Chain 3: Iteration: 7000 / 10000 [ 70%]  (Sampling)
    ## Chain 3: Iteration: 8000 / 10000 [ 80%]  (Sampling)
    ## Chain 3: Iteration: 9000 / 10000 [ 90%]  (Sampling)
    ## Chain 3: Iteration: 10000 / 10000 [100%]  (Sampling)
    ## Chain 3: 
    ## Chain 3:  Elapsed Time: 4.099 seconds (Warm-up)
    ## Chain 3:                0.137 seconds (Sampling)
    ## Chain 3:                4.236 seconds (Total)
    ## Chain 3: 
    ## 
    ## SAMPLING FOR MODEL 'continuous' NOW (CHAIN 4).
    ## Chain 4: 
    ## Chain 4: Gradient evaluation took 1.1e-05 seconds
    ## Chain 4: 1000 transitions using 10 leapfrog steps per transition would take 0.11 seconds.
    ## Chain 4: Adjust your expectations accordingly!
    ## Chain 4: 
    ## Chain 4: 
    ## Chain 4: Iteration:    1 / 10000 [  0%]  (Warmup)
    ## Chain 4: Iteration: 1000 / 10000 [ 10%]  (Warmup)
    ## Chain 4: Iteration: 2000 / 10000 [ 20%]  (Warmup)
    ## Chain 4: Iteration: 3000 / 10000 [ 30%]  (Warmup)
    ## Chain 4: Iteration: 4000 / 10000 [ 40%]  (Warmup)
    ## Chain 4: Iteration: 5000 / 10000 [ 50%]  (Warmup)
    ## Chain 4: Iteration: 5001 / 10000 [ 50%]  (Sampling)
    ## Chain 4: Iteration: 6000 / 10000 [ 60%]  (Sampling)
    ## Chain 4: Iteration: 7000 / 10000 [ 70%]  (Sampling)
    ## Chain 4: Iteration: 8000 / 10000 [ 80%]  (Sampling)
    ## Chain 4: Iteration: 9000 / 10000 [ 90%]  (Sampling)
    ## Chain 4: Iteration: 10000 / 10000 [100%]  (Sampling)
    ## Chain 4: 
    ## Chain 4:  Elapsed Time: 4.636 seconds (Warm-up)
    ## Chain 4:                0.132 seconds (Sampling)
    ## Chain 4:                4.768 seconds (Total)
    ## Chain 4:

``` r
# 200 prior model lines
lol_data %>%
  add_fitted_draws(base, n = 200) %>%
  ggplot(aes(x = winner, y = gameDuration)) +
    geom_line(aes(y = .value, group = .draw), alpha = 0.05)
```

    ## Warning: `fitted_draws` and `add_fitted_draws` are deprecated as their names were confusing.
    ## - Use [add_]epred_draws() to get the expectation of the posterior predictive.
    ## - Use [add_]linpred_draws() to get the distribution of the linear predictor.
    ## - For example, you used [add_]fitted_draws(..., scale = "response"), which
    ##   means you most likely want [add_]epred_draws(...).
    ## NOTE: When updating to the new functions, note that the `model` parameter is now
    ##   named `object` and the `n` parameter is now named `ndraws`.

![](Final-Project-LOL-New_files/figure-gfm/unnamed-chunk-34-1.png)<!-- -->

``` r
# 4 prior simulated datasets
set.seed(3)
lol_data %>%
  add_predicted_draws(base, n = 4) %>%
  ggplot(aes(x = winner, y = gameDuration)) +
    geom_point(aes(y = .prediction, group = .draw)) + 
    facet_wrap(~ .draw)
```

    ## Warning: 
    ## In add_predicted_draws(): The `n` argument is a deprecated alias for `ndraws`.
    ## Use the `ndraws` argument instead.
    ## See help("tidybayes-deprecated").

![](Final-Project-LOL-New_files/figure-gfm/unnamed-chunk-34-2.png)<!-- -->

Prior distribution expects the games to range from 1910 - 1945 seconds
as expected

Game duration lies between 500-3500 seconds which falls in line with our
summary

Prior distribution expects games to last longer if team 2 wins as
compared to team 1

``` r
mcmc_trace(base, size = .1)
```

![](Final-Project-LOL-New_files/figure-gfm/unnamed-chunk-35-1.png)<!-- -->

Plots look fast mixing and consistent

``` r
rhat(base)
```

    ## (Intercept)     winner2       sigma 
    ##   0.9998587   0.9999295   0.9999225

Rhat close to 1 and not \>1.05, good

``` r
neff_ratio(base)
```

    ## (Intercept)     winner2       sigma 
    ##     0.92220     0.95970     0.95575

neff ratio is a little high but still acceptable, \>.10 which is good

``` r
tidy(base, effects = c("fixed", "aux"),
     conf.int = TRUE, conf.level = 0.95)
```

    ## # A tibble: 4 × 5
    ##   term        estimate std.error conf.low conf.high
    ##   <chr>          <dbl>     <dbl>    <dbl>     <dbl>
    ## 1 (Intercept)   1918.       2.86   1912.     1924. 
    ## 2 winner2         23.9      4.02     16.1      31.7
    ## 3 sigma          427.       1.41    424.      430. 
    ## 4 mean_PPD      1930.       2.85   1924.     1935.

Overall the model could work, but more investigation should be done to
improve what we have

## Interaction Investigation

Lets see if there are some interactions we can expect

Speaking from experience in playing the game myself, I predict the
dragon and baron kills will have a big impact on the game duration due
to the buffs they give each member of the team being crucial to winning

``` r
int <- stan_glm(gameDuration ~ t1_dragonKills:t2_dragonKills,
  data = lol_data, family = gaussian, 
  prior_intercept = normal(1800, 150, autoscale = TRUE),
  prior = normal(0, 2.5, autoscale = TRUE), 
  prior_aux = exponential(1, autoscale = TRUE),
  chains = 4, iter = 5000*2, seed = 84735)
```

    ## 
    ## SAMPLING FOR MODEL 'continuous' NOW (CHAIN 1).
    ## Chain 1: 
    ## Chain 1: Gradient evaluation took 0.003132 seconds
    ## Chain 1: 1000 transitions using 10 leapfrog steps per transition would take 31.32 seconds.
    ## Chain 1: Adjust your expectations accordingly!
    ## Chain 1: 
    ## Chain 1: 
    ## Chain 1: Iteration:    1 / 10000 [  0%]  (Warmup)
    ## Chain 1: Iteration: 1000 / 10000 [ 10%]  (Warmup)
    ## Chain 1: Iteration: 2000 / 10000 [ 20%]  (Warmup)
    ## Chain 1: Iteration: 3000 / 10000 [ 30%]  (Warmup)
    ## Chain 1: Iteration: 4000 / 10000 [ 40%]  (Warmup)
    ## Chain 1: Iteration: 5000 / 10000 [ 50%]  (Warmup)
    ## Chain 1: Iteration: 5001 / 10000 [ 50%]  (Sampling)
    ## Chain 1: Iteration: 6000 / 10000 [ 60%]  (Sampling)
    ## Chain 1: Iteration: 7000 / 10000 [ 70%]  (Sampling)
    ## Chain 1: Iteration: 8000 / 10000 [ 80%]  (Sampling)
    ## Chain 1: Iteration: 9000 / 10000 [ 90%]  (Sampling)
    ## Chain 1: Iteration: 10000 / 10000 [100%]  (Sampling)
    ## Chain 1: 
    ## Chain 1:  Elapsed Time: 0.25 seconds (Warm-up)
    ## Chain 1:                13.571 seconds (Sampling)
    ## Chain 1:                13.821 seconds (Total)
    ## Chain 1: 
    ## 
    ## SAMPLING FOR MODEL 'continuous' NOW (CHAIN 2).
    ## Chain 2: 
    ## Chain 2: Gradient evaluation took 1.6e-05 seconds
    ## Chain 2: 1000 transitions using 10 leapfrog steps per transition would take 0.16 seconds.
    ## Chain 2: Adjust your expectations accordingly!
    ## Chain 2: 
    ## Chain 2: 
    ## Chain 2: Iteration:    1 / 10000 [  0%]  (Warmup)
    ## Chain 2: Iteration: 1000 / 10000 [ 10%]  (Warmup)
    ## Chain 2: Iteration: 2000 / 10000 [ 20%]  (Warmup)
    ## Chain 2: Iteration: 3000 / 10000 [ 30%]  (Warmup)
    ## Chain 2: Iteration: 4000 / 10000 [ 40%]  (Warmup)
    ## Chain 2: Iteration: 5000 / 10000 [ 50%]  (Warmup)
    ## Chain 2: Iteration: 5001 / 10000 [ 50%]  (Sampling)
    ## Chain 2: Iteration: 6000 / 10000 [ 60%]  (Sampling)
    ## Chain 2: Iteration: 7000 / 10000 [ 70%]  (Sampling)
    ## Chain 2: Iteration: 8000 / 10000 [ 80%]  (Sampling)
    ## Chain 2: Iteration: 9000 / 10000 [ 90%]  (Sampling)
    ## Chain 2: Iteration: 10000 / 10000 [100%]  (Sampling)
    ## Chain 2: 
    ## Chain 2:  Elapsed Time: 0.243 seconds (Warm-up)
    ## Chain 2:                14.727 seconds (Sampling)
    ## Chain 2:                14.97 seconds (Total)
    ## Chain 2: 
    ## 
    ## SAMPLING FOR MODEL 'continuous' NOW (CHAIN 3).
    ## Chain 3: 
    ## Chain 3: Gradient evaluation took 1e-05 seconds
    ## Chain 3: 1000 transitions using 10 leapfrog steps per transition would take 0.1 seconds.
    ## Chain 3: Adjust your expectations accordingly!
    ## Chain 3: 
    ## Chain 3: 
    ## Chain 3: Iteration:    1 / 10000 [  0%]  (Warmup)
    ## Chain 3: Iteration: 1000 / 10000 [ 10%]  (Warmup)
    ## Chain 3: Iteration: 2000 / 10000 [ 20%]  (Warmup)
    ## Chain 3: Iteration: 3000 / 10000 [ 30%]  (Warmup)
    ## Chain 3: Iteration: 4000 / 10000 [ 40%]  (Warmup)
    ## Chain 3: Iteration: 5000 / 10000 [ 50%]  (Warmup)
    ## Chain 3: Iteration: 5001 / 10000 [ 50%]  (Sampling)
    ## Chain 3: Iteration: 6000 / 10000 [ 60%]  (Sampling)
    ## Chain 3: Iteration: 7000 / 10000 [ 70%]  (Sampling)
    ## Chain 3: Iteration: 8000 / 10000 [ 80%]  (Sampling)
    ## Chain 3: Iteration: 9000 / 10000 [ 90%]  (Sampling)
    ## Chain 3: Iteration: 10000 / 10000 [100%]  (Sampling)
    ## Chain 3: 
    ## Chain 3:  Elapsed Time: 0.23 seconds (Warm-up)
    ## Chain 3:                13.639 seconds (Sampling)
    ## Chain 3:                13.869 seconds (Total)
    ## Chain 3: 
    ## 
    ## SAMPLING FOR MODEL 'continuous' NOW (CHAIN 4).
    ## Chain 4: 
    ## Chain 4: Gradient evaluation took 1.2e-05 seconds
    ## Chain 4: 1000 transitions using 10 leapfrog steps per transition would take 0.12 seconds.
    ## Chain 4: Adjust your expectations accordingly!
    ## Chain 4: 
    ## Chain 4: 
    ## Chain 4: Iteration:    1 / 10000 [  0%]  (Warmup)
    ## Chain 4: Iteration: 1000 / 10000 [ 10%]  (Warmup)
    ## Chain 4: Iteration: 2000 / 10000 [ 20%]  (Warmup)
    ## Chain 4: Iteration: 3000 / 10000 [ 30%]  (Warmup)
    ## Chain 4: Iteration: 4000 / 10000 [ 40%]  (Warmup)
    ## Chain 4: Iteration: 5000 / 10000 [ 50%]  (Warmup)
    ## Chain 4: Iteration: 5001 / 10000 [ 50%]  (Sampling)
    ## Chain 4: Iteration: 6000 / 10000 [ 60%]  (Sampling)
    ## Chain 4: Iteration: 7000 / 10000 [ 70%]  (Sampling)
    ## Chain 4: Iteration: 8000 / 10000 [ 80%]  (Sampling)
    ## Chain 4: Iteration: 9000 / 10000 [ 90%]  (Sampling)
    ## Chain 4: Iteration: 10000 / 10000 [100%]  (Sampling)
    ## Chain 4: 
    ## Chain 4:  Elapsed Time: 0.222 seconds (Warm-up)
    ## Chain 4:                13.507 seconds (Sampling)
    ## Chain 4:                13.729 seconds (Total)
    ## Chain 4:

``` r
int2 <- stan_glm(gameDuration ~ t1_baronKills:t2_baronKills,
  data = lol_data, family = gaussian, 
  prior_intercept = normal(1800, 150, autoscale = TRUE),
  prior = normal(0, 2.5, autoscale = TRUE), 
  prior_aux = exponential(1, autoscale = TRUE),
  chains = 4, iter = 5000*2, seed = 84735)
```

    ## 
    ## SAMPLING FOR MODEL 'continuous' NOW (CHAIN 1).
    ## Chain 1: 
    ## Chain 1: Gradient evaluation took 1.7e-05 seconds
    ## Chain 1: 1000 transitions using 10 leapfrog steps per transition would take 0.17 seconds.
    ## Chain 1: Adjust your expectations accordingly!
    ## Chain 1: 
    ## Chain 1: 
    ## Chain 1: Iteration:    1 / 10000 [  0%]  (Warmup)
    ## Chain 1: Iteration: 1000 / 10000 [ 10%]  (Warmup)
    ## Chain 1: Iteration: 2000 / 10000 [ 20%]  (Warmup)
    ## Chain 1: Iteration: 3000 / 10000 [ 30%]  (Warmup)
    ## Chain 1: Iteration: 4000 / 10000 [ 40%]  (Warmup)
    ## Chain 1: Iteration: 5000 / 10000 [ 50%]  (Warmup)
    ## Chain 1: Iteration: 5001 / 10000 [ 50%]  (Sampling)
    ## Chain 1: Iteration: 6000 / 10000 [ 60%]  (Sampling)
    ## Chain 1: Iteration: 7000 / 10000 [ 70%]  (Sampling)
    ## Chain 1: Iteration: 8000 / 10000 [ 80%]  (Sampling)
    ## Chain 1: Iteration: 9000 / 10000 [ 90%]  (Sampling)
    ## Chain 1: Iteration: 10000 / 10000 [100%]  (Sampling)
    ## Chain 1: 
    ## Chain 1:  Elapsed Time: 0.228 seconds (Warm-up)
    ## Chain 1:                13.476 seconds (Sampling)
    ## Chain 1:                13.704 seconds (Total)
    ## Chain 1: 
    ## 
    ## SAMPLING FOR MODEL 'continuous' NOW (CHAIN 2).
    ## Chain 2: 
    ## Chain 2: Gradient evaluation took 1.5e-05 seconds
    ## Chain 2: 1000 transitions using 10 leapfrog steps per transition would take 0.15 seconds.
    ## Chain 2: Adjust your expectations accordingly!
    ## Chain 2: 
    ## Chain 2: 
    ## Chain 2: Iteration:    1 / 10000 [  0%]  (Warmup)
    ## Chain 2: Iteration: 1000 / 10000 [ 10%]  (Warmup)
    ## Chain 2: Iteration: 2000 / 10000 [ 20%]  (Warmup)
    ## Chain 2: Iteration: 3000 / 10000 [ 30%]  (Warmup)
    ## Chain 2: Iteration: 4000 / 10000 [ 40%]  (Warmup)
    ## Chain 2: Iteration: 5000 / 10000 [ 50%]  (Warmup)
    ## Chain 2: Iteration: 5001 / 10000 [ 50%]  (Sampling)
    ## Chain 2: Iteration: 6000 / 10000 [ 60%]  (Sampling)
    ## Chain 2: Iteration: 7000 / 10000 [ 70%]  (Sampling)
    ## Chain 2: Iteration: 8000 / 10000 [ 80%]  (Sampling)
    ## Chain 2: Iteration: 9000 / 10000 [ 90%]  (Sampling)
    ## Chain 2: Iteration: 10000 / 10000 [100%]  (Sampling)
    ## Chain 2: 
    ## Chain 2:  Elapsed Time: 0.215 seconds (Warm-up)
    ## Chain 2:                13.293 seconds (Sampling)
    ## Chain 2:                13.508 seconds (Total)
    ## Chain 2: 
    ## 
    ## SAMPLING FOR MODEL 'continuous' NOW (CHAIN 3).
    ## Chain 3: 
    ## Chain 3: Gradient evaluation took 1.6e-05 seconds
    ## Chain 3: 1000 transitions using 10 leapfrog steps per transition would take 0.16 seconds.
    ## Chain 3: Adjust your expectations accordingly!
    ## Chain 3: 
    ## Chain 3: 
    ## Chain 3: Iteration:    1 / 10000 [  0%]  (Warmup)
    ## Chain 3: Iteration: 1000 / 10000 [ 10%]  (Warmup)
    ## Chain 3: Iteration: 2000 / 10000 [ 20%]  (Warmup)
    ## Chain 3: Iteration: 3000 / 10000 [ 30%]  (Warmup)
    ## Chain 3: Iteration: 4000 / 10000 [ 40%]  (Warmup)
    ## Chain 3: Iteration: 5000 / 10000 [ 50%]  (Warmup)
    ## Chain 3: Iteration: 5001 / 10000 [ 50%]  (Sampling)
    ## Chain 3: Iteration: 6000 / 10000 [ 60%]  (Sampling)
    ## Chain 3: Iteration: 7000 / 10000 [ 70%]  (Sampling)
    ## Chain 3: Iteration: 8000 / 10000 [ 80%]  (Sampling)
    ## Chain 3: Iteration: 9000 / 10000 [ 90%]  (Sampling)
    ## Chain 3: Iteration: 10000 / 10000 [100%]  (Sampling)
    ## Chain 3: 
    ## Chain 3:  Elapsed Time: 0.226 seconds (Warm-up)
    ## Chain 3:                13.293 seconds (Sampling)
    ## Chain 3:                13.519 seconds (Total)
    ## Chain 3: 
    ## 
    ## SAMPLING FOR MODEL 'continuous' NOW (CHAIN 4).
    ## Chain 4: 
    ## Chain 4: Gradient evaluation took 1.8e-05 seconds
    ## Chain 4: 1000 transitions using 10 leapfrog steps per transition would take 0.18 seconds.
    ## Chain 4: Adjust your expectations accordingly!
    ## Chain 4: 
    ## Chain 4: 
    ## Chain 4: Iteration:    1 / 10000 [  0%]  (Warmup)
    ## Chain 4: Iteration: 1000 / 10000 [ 10%]  (Warmup)
    ## Chain 4: Iteration: 2000 / 10000 [ 20%]  (Warmup)
    ## Chain 4: Iteration: 3000 / 10000 [ 30%]  (Warmup)
    ## Chain 4: Iteration: 4000 / 10000 [ 40%]  (Warmup)
    ## Chain 4: Iteration: 5000 / 10000 [ 50%]  (Warmup)
    ## Chain 4: Iteration: 5001 / 10000 [ 50%]  (Sampling)
    ## Chain 4: Iteration: 6000 / 10000 [ 60%]  (Sampling)
    ## Chain 4: Iteration: 7000 / 10000 [ 70%]  (Sampling)
    ## Chain 4: Iteration: 8000 / 10000 [ 80%]  (Sampling)
    ## Chain 4: Iteration: 9000 / 10000 [ 90%]  (Sampling)
    ## Chain 4: Iteration: 10000 / 10000 [100%]  (Sampling)
    ## Chain 4: 
    ## Chain 4:  Elapsed Time: 0.206 seconds (Warm-up)
    ## Chain 4:                13.271 seconds (Sampling)
    ## Chain 4:                13.477 seconds (Total)
    ## Chain 4:

``` r
summary(int)
```

    ## 
    ## Model Info:
    ##  function:     stan_glm
    ##  family:       gaussian [identity]
    ##  formula:      gameDuration ~ t1_dragonKills:t2_dragonKills
    ##  algorithm:    sampling
    ##  sample:       20000 (posterior sample size)
    ##  priors:       see help('prior_summary')
    ##  observations: 45214
    ##  predictors:   2
    ## 
    ## Estimates:
    ##                                 mean   sd     10%    50%    90% 
    ## (Intercept)                   1728.5    2.0 1725.9 1728.5 1731.1
    ## t1_dragonKills:t2_dragonKills  147.5    0.9  146.3  147.5  148.6
    ## sigma                          340.5    1.1  339.0  340.5  341.9
    ## 
    ## Fit Diagnostics:
    ##            mean   sd     10%    50%    90% 
    ## mean_PPD 1929.8    2.3 1926.9 1929.8 1932.7
    ## 
    ## The mean_ppd is the sample average posterior predictive distribution of the outcome variable (for details see help('summary.stanreg')).
    ## 
    ## MCMC diagnostics
    ##                               mcse Rhat n_eff
    ## (Intercept)                   0.0  1.0  18279
    ## t1_dragonKills:t2_dragonKills 0.0  1.0  21144
    ## sigma                         0.0  1.0  18662
    ## mean_PPD                      0.0  1.0  18790
    ## log-posterior                 0.0  1.0   9402
    ## 
    ## For each parameter, mcse is Monte Carlo standard error, n_eff is a crude measure of effective sample size, and Rhat is the potential scale reduction factor on split chains (at convergence Rhat=1).

``` r
summary(int2)
```

    ## 
    ## Model Info:
    ##  function:     stan_glm
    ##  family:       gaussian [identity]
    ##  formula:      gameDuration ~ t1_baronKills:t2_baronKills
    ##  algorithm:    sampling
    ##  sample:       20000 (posterior sample size)
    ##  priors:       see help('prior_summary')
    ##  observations: 45214
    ##  predictors:   2
    ## 
    ## Estimates:
    ##                               mean   sd     10%    50%    90% 
    ## (Intercept)                 1884.0    1.9 1881.6 1884.0 1886.4
    ## t1_baronKills:t2_baronKills  466.7    4.8  460.4  466.6  472.8
    ## sigma                        390.0    1.3  388.4  390.0  391.7
    ## 
    ## Fit Diagnostics:
    ##            mean   sd     10%    50%    90% 
    ## mean_PPD 1929.8    2.6 1926.5 1929.8 1933.1
    ## 
    ## The mean_ppd is the sample average posterior predictive distribution of the outcome variable (for details see help('summary.stanreg')).
    ## 
    ## MCMC diagnostics
    ##                             mcse Rhat n_eff
    ## (Intercept)                 0.0  1.0  17022
    ## t1_baronKills:t2_baronKills 0.0  1.0  20014
    ## sigma                       0.0  1.0  18691
    ## mean_PPD                    0.0  1.0  18228
    ## log-posterior               0.0  1.0   9914
    ## 
    ## For each parameter, mcse is Monte Carlo standard error, n_eff is a crude measure of effective sample size, and Rhat is the potential scale reduction factor on split chains (at convergence Rhat=1).

Summary stats look promising with both values not containing 0 for their
95% CI showing they are significant and can be useful

``` r
# Extract the posterior samples
posterior_samples <- as.matrix(int)
posterior_samples2 <- as.matrix(int2)

# Plot the posterior distribution of the interaction term
mcmc_hist(posterior_samples, pars = c("t1_dragonKills:t2_dragonKills"))
```

    ## `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.

![](Final-Project-LOL-New_files/figure-gfm/unnamed-chunk-43-1.png)<!-- -->

``` r
mcmc_hist(posterior_samples2, pars = c("t1_baronKills:t2_baronKills"))
```

    ## `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.

![](Final-Project-LOL-New_files/figure-gfm/unnamed-chunk-43-2.png)<!-- -->

The predictors don’t look skewed either, looks good.

We can canclude that there may be a potential interaction between dragon
kills and baron kills for each team

We can utilize this info later on upon building a better model

## Main Model

Let’s assess how much getting a jump start in the game does for
determining the length of the game

``` r
main <- stan_glm(gameDuration ~ winner + firstBlood + firstTower + firstDragon + firstBaron,
  data = lol_data, family = gaussian, 
  prior_intercept = normal(1800, 150, autoscale = TRUE),
  prior = normal(0, 2.5, autoscale = TRUE), 
  prior_aux = exponential(1, autoscale = TRUE),
  chains = 4, iter = 5000*2, seed = 84735)
```

    ## 
    ## SAMPLING FOR MODEL 'continuous' NOW (CHAIN 1).
    ## Chain 1: 
    ## Chain 1: Gradient evaluation took 1.8e-05 seconds
    ## Chain 1: 1000 transitions using 10 leapfrog steps per transition would take 0.18 seconds.
    ## Chain 1: Adjust your expectations accordingly!
    ## Chain 1: 
    ## Chain 1: 
    ## Chain 1: Iteration:    1 / 10000 [  0%]  (Warmup)
    ## Chain 1: Iteration: 1000 / 10000 [ 10%]  (Warmup)
    ## Chain 1: Iteration: 2000 / 10000 [ 20%]  (Warmup)
    ## Chain 1: Iteration: 3000 / 10000 [ 30%]  (Warmup)
    ## Chain 1: Iteration: 4000 / 10000 [ 40%]  (Warmup)
    ## Chain 1: Iteration: 5000 / 10000 [ 50%]  (Warmup)
    ## Chain 1: Iteration: 5001 / 10000 [ 50%]  (Sampling)
    ## Chain 1: Iteration: 6000 / 10000 [ 60%]  (Sampling)
    ## Chain 1: Iteration: 7000 / 10000 [ 70%]  (Sampling)
    ## Chain 1: Iteration: 8000 / 10000 [ 80%]  (Sampling)
    ## Chain 1: Iteration: 9000 / 10000 [ 90%]  (Sampling)
    ## Chain 1: Iteration: 10000 / 10000 [100%]  (Sampling)
    ## Chain 1: 
    ## Chain 1:  Elapsed Time: 0.886 seconds (Warm-up)
    ## Chain 1:                14.079 seconds (Sampling)
    ## Chain 1:                14.965 seconds (Total)
    ## Chain 1: 
    ## 
    ## SAMPLING FOR MODEL 'continuous' NOW (CHAIN 2).
    ## Chain 2: 
    ## Chain 2: Gradient evaluation took 1.5e-05 seconds
    ## Chain 2: 1000 transitions using 10 leapfrog steps per transition would take 0.15 seconds.
    ## Chain 2: Adjust your expectations accordingly!
    ## Chain 2: 
    ## Chain 2: 
    ## Chain 2: Iteration:    1 / 10000 [  0%]  (Warmup)
    ## Chain 2: Iteration: 1000 / 10000 [ 10%]  (Warmup)
    ## Chain 2: Iteration: 2000 / 10000 [ 20%]  (Warmup)
    ## Chain 2: Iteration: 3000 / 10000 [ 30%]  (Warmup)
    ## Chain 2: Iteration: 4000 / 10000 [ 40%]  (Warmup)
    ## Chain 2: Iteration: 5000 / 10000 [ 50%]  (Warmup)
    ## Chain 2: Iteration: 5001 / 10000 [ 50%]  (Sampling)
    ## Chain 2: Iteration: 6000 / 10000 [ 60%]  (Sampling)
    ## Chain 2: Iteration: 7000 / 10000 [ 70%]  (Sampling)
    ## Chain 2: Iteration: 8000 / 10000 [ 80%]  (Sampling)
    ## Chain 2: Iteration: 9000 / 10000 [ 90%]  (Sampling)
    ## Chain 2: Iteration: 10000 / 10000 [100%]  (Sampling)
    ## Chain 2: 
    ## Chain 2:  Elapsed Time: 0.968 seconds (Warm-up)
    ## Chain 2:                14.153 seconds (Sampling)
    ## Chain 2:                15.121 seconds (Total)
    ## Chain 2: 
    ## 
    ## SAMPLING FOR MODEL 'continuous' NOW (CHAIN 3).
    ## Chain 3: 
    ## Chain 3: Gradient evaluation took 1.4e-05 seconds
    ## Chain 3: 1000 transitions using 10 leapfrog steps per transition would take 0.14 seconds.
    ## Chain 3: Adjust your expectations accordingly!
    ## Chain 3: 
    ## Chain 3: 
    ## Chain 3: Iteration:    1 / 10000 [  0%]  (Warmup)
    ## Chain 3: Iteration: 1000 / 10000 [ 10%]  (Warmup)
    ## Chain 3: Iteration: 2000 / 10000 [ 20%]  (Warmup)
    ## Chain 3: Iteration: 3000 / 10000 [ 30%]  (Warmup)
    ## Chain 3: Iteration: 4000 / 10000 [ 40%]  (Warmup)
    ## Chain 3: Iteration: 5000 / 10000 [ 50%]  (Warmup)
    ## Chain 3: Iteration: 5001 / 10000 [ 50%]  (Sampling)
    ## Chain 3: Iteration: 6000 / 10000 [ 60%]  (Sampling)
    ## Chain 3: Iteration: 7000 / 10000 [ 70%]  (Sampling)
    ## Chain 3: Iteration: 8000 / 10000 [ 80%]  (Sampling)
    ## Chain 3: Iteration: 9000 / 10000 [ 90%]  (Sampling)
    ## Chain 3: Iteration: 10000 / 10000 [100%]  (Sampling)
    ## Chain 3: 
    ## Chain 3:  Elapsed Time: 0.888 seconds (Warm-up)
    ## Chain 3:                14.395 seconds (Sampling)
    ## Chain 3:                15.283 seconds (Total)
    ## Chain 3: 
    ## 
    ## SAMPLING FOR MODEL 'continuous' NOW (CHAIN 4).
    ## Chain 4: 
    ## Chain 4: Gradient evaluation took 1.4e-05 seconds
    ## Chain 4: 1000 transitions using 10 leapfrog steps per transition would take 0.14 seconds.
    ## Chain 4: Adjust your expectations accordingly!
    ## Chain 4: 
    ## Chain 4: 
    ## Chain 4: Iteration:    1 / 10000 [  0%]  (Warmup)
    ## Chain 4: Iteration: 1000 / 10000 [ 10%]  (Warmup)
    ## Chain 4: Iteration: 2000 / 10000 [ 20%]  (Warmup)
    ## Chain 4: Iteration: 3000 / 10000 [ 30%]  (Warmup)
    ## Chain 4: Iteration: 4000 / 10000 [ 40%]  (Warmup)
    ## Chain 4: Iteration: 5000 / 10000 [ 50%]  (Warmup)
    ## Chain 4: Iteration: 5001 / 10000 [ 50%]  (Sampling)
    ## Chain 4: Iteration: 6000 / 10000 [ 60%]  (Sampling)
    ## Chain 4: Iteration: 7000 / 10000 [ 70%]  (Sampling)
    ## Chain 4: Iteration: 8000 / 10000 [ 80%]  (Sampling)
    ## Chain 4: Iteration: 9000 / 10000 [ 90%]  (Sampling)
    ## Chain 4: Iteration: 10000 / 10000 [100%]  (Sampling)
    ## Chain 4: 
    ## Chain 4:  Elapsed Time: 0.913 seconds (Warm-up)
    ## Chain 4:                14.285 seconds (Sampling)
    ## Chain 4:                15.198 seconds (Total)
    ## Chain 4:

``` r
prior_summary(main) 
```

    ## Priors for model 'main' 
    ## ------
    ## Intercept (after predictors centered)
    ##   Specified prior:
    ##     ~ normal(location = 1800, scale = 150)
    ##   Adjusted prior:
    ##     ~ normal(location = 1800, scale = 64093)
    ## 
    ## Coefficients
    ##   Specified prior:
    ##     ~ normal(location = [0,0,0,...], scale = [2.5,2.5,2.5,...])
    ##   Adjusted prior:
    ##     ~ normal(location = [0,0,0,...], scale = [2136.54,2137.02,2137.26,...])
    ## 
    ## Auxiliary (sigma)
    ##   Specified prior:
    ##     ~ exponential(rate = 1)
    ##   Adjusted prior:
    ##     ~ exponential(rate = 0.0023)
    ## ------
    ## See help('prior_summary.stanreg') for more details

``` r
mcmc_trace(main, size = .1)
```

![](Final-Project-LOL-New_files/figure-gfm/unnamed-chunk-46-1.png)<!-- -->

Chains look normal

``` r
rhat(main)
```

    ##  (Intercept)      winner2  firstBlood2  firstTower2 firstDragon1 firstDragon2 
    ##    1.0001252    1.0000866    0.9999479    0.9999554    1.0000860    1.0001157 
    ##  firstBaron1  firstBaron2        sigma 
    ##    1.0000559    0.9999221    1.0000304

``` r
neff_ratio(main)
```

    ##  (Intercept)      winner2  firstBlood2  firstTower2 firstDragon1 firstDragon2 
    ##      0.56465      1.06070      1.40485      1.23750      0.54275      0.54175 
    ##  firstBaron1  firstBaron2        sigma 
    ##      1.05630      1.05235      1.12255

Rhat looks good, some of the neff ratio values are still high but
acceptable for now

``` r
tidy(main, effects = c("fixed", "aux"),
     conf.int = TRUE, conf.level = 0.95)
```

    ## # A tibble: 10 × 5
    ##    term         estimate std.error conf.low conf.high
    ##    <chr>           <dbl>     <dbl>    <dbl>     <dbl>
    ##  1 (Intercept)   1176.       18.3   1141.     1211.  
    ##  2 winner2         -1.92      4.51   -10.9       7.03
    ##  3 firstBlood2      2.04      3.64    -5.04      9.15
    ##  4 firstTower2      4.05      3.95    -3.82     11.7 
    ##  5 firstDragon1   486.       18.6    450.      521.  
    ##  6 firstDragon2   478.       18.6    442.      514.  
    ##  7 firstBaron1    399.        4.57   390.      408.  
    ##  8 firstBaron2    413.        4.59   404.      422.  
    ##  9 sigma          377.        1.25   375.      379.  
    ## 10 mean_PPD      1930.        2.49  1925.     1935.

From the tidy output, it looks like winner, firstBlood, and firstTower
lose their significance when adding other predictors since their 95% CI
range includes 0. We’ll use this info later for refining our model

``` r
newdata <- data.frame(winner = factor(1, levels = levels(lol_data$winner)),
                      firstBlood = factor(2, levels = levels(lol_data$firstBlood)),
                      firstTower = factor(2, levels = levels(lol_data$firstTower)),
                      firstDragon = factor(2, levels = levels(lol_data$firstDragon)),
                      firstBaron = factor(2, levels = levels(lol_data$firstBaron)))

main_predict <- posterior_predict(
  main, 
  newdata = newdata)
mcmc_areas(main_predict) +  xlab("Game Duration") +
  ggtitle('Predictive distribution of a League of Legends game where first blood, tower, dragon, baron was team 2 and team 1 won') +
  theme(plot.title = element_text(size = 7))
```

![](Final-Project-LOL-New_files/figure-gfm/unnamed-chunk-50-1.png)<!-- -->

Here’s an example output of where a game duration may lie if team 1 wins
despite being behind from the start. The game is a little later than
normal, but to be expected as the team would need time to make the
comeback in the first place. However, I believe this is undershooting
the true time we could expect from this scenario. We’ll verify if this
is true after refining our model

## More Predictors

We can add some numerical variables to see how much adding the amount of
each objective a team has taken to see how it affects the game duration

``` r
ext_main <- stan_glm(gameDuration ~ winner + firstBlood + firstTower + firstDragon + firstBaron + t1_towerKills + t1_inhibitorKills + t1_baronKills + t1_dragonKills + t2_towerKills + t2_inhibitorKills + t2_baronKills + t2_dragonKills,
  data = lol_data, family = gaussian, 
  prior_intercept = normal(1800, 150, autoscale = TRUE),
  prior = normal(0, 2.5, autoscale = TRUE), 
  prior_aux = exponential(1, autoscale = TRUE),
  chains = 4, iter = 5000*2, seed = 84735)
```

    ## 
    ## SAMPLING FOR MODEL 'continuous' NOW (CHAIN 1).
    ## Chain 1: 
    ## Chain 1: Gradient evaluation took 0.000127 seconds
    ## Chain 1: 1000 transitions using 10 leapfrog steps per transition would take 1.27 seconds.
    ## Chain 1: Adjust your expectations accordingly!
    ## Chain 1: 
    ## Chain 1: 
    ## Chain 1: Iteration:    1 / 10000 [  0%]  (Warmup)
    ## Chain 1: Iteration: 1000 / 10000 [ 10%]  (Warmup)
    ## Chain 1: Iteration: 2000 / 10000 [ 20%]  (Warmup)
    ## Chain 1: Iteration: 3000 / 10000 [ 30%]  (Warmup)
    ## Chain 1: Iteration: 4000 / 10000 [ 40%]  (Warmup)
    ## Chain 1: Iteration: 5000 / 10000 [ 50%]  (Warmup)
    ## Chain 1: Iteration: 5001 / 10000 [ 50%]  (Sampling)
    ## Chain 1: Iteration: 6000 / 10000 [ 60%]  (Sampling)
    ## Chain 1: Iteration: 7000 / 10000 [ 70%]  (Sampling)
    ## Chain 1: Iteration: 8000 / 10000 [ 80%]  (Sampling)
    ## Chain 1: Iteration: 9000 / 10000 [ 90%]  (Sampling)
    ## Chain 1: Iteration: 10000 / 10000 [100%]  (Sampling)
    ## Chain 1: 
    ## Chain 1:  Elapsed Time: 1.343 seconds (Warm-up)
    ## Chain 1:                14.578 seconds (Sampling)
    ## Chain 1:                15.921 seconds (Total)
    ## Chain 1: 
    ## 
    ## SAMPLING FOR MODEL 'continuous' NOW (CHAIN 2).
    ## Chain 2: 
    ## Chain 2: Gradient evaluation took 1.5e-05 seconds
    ## Chain 2: 1000 transitions using 10 leapfrog steps per transition would take 0.15 seconds.
    ## Chain 2: Adjust your expectations accordingly!
    ## Chain 2: 
    ## Chain 2: 
    ## Chain 2: Iteration:    1 / 10000 [  0%]  (Warmup)
    ## Chain 2: Iteration: 1000 / 10000 [ 10%]  (Warmup)
    ## Chain 2: Iteration: 2000 / 10000 [ 20%]  (Warmup)
    ## Chain 2: Iteration: 3000 / 10000 [ 30%]  (Warmup)
    ## Chain 2: Iteration: 4000 / 10000 [ 40%]  (Warmup)
    ## Chain 2: Iteration: 5000 / 10000 [ 50%]  (Warmup)
    ## Chain 2: Iteration: 5001 / 10000 [ 50%]  (Sampling)
    ## Chain 2: Iteration: 6000 / 10000 [ 60%]  (Sampling)
    ## Chain 2: Iteration: 7000 / 10000 [ 70%]  (Sampling)
    ## Chain 2: Iteration: 8000 / 10000 [ 80%]  (Sampling)
    ## Chain 2: Iteration: 9000 / 10000 [ 90%]  (Sampling)
    ## Chain 2: Iteration: 10000 / 10000 [100%]  (Sampling)
    ## Chain 2: 
    ## Chain 2:  Elapsed Time: 1.469 seconds (Warm-up)
    ## Chain 2:                15.113 seconds (Sampling)
    ## Chain 2:                16.582 seconds (Total)
    ## Chain 2: 
    ## 
    ## SAMPLING FOR MODEL 'continuous' NOW (CHAIN 3).
    ## Chain 3: 
    ## Chain 3: Gradient evaluation took 1.8e-05 seconds
    ## Chain 3: 1000 transitions using 10 leapfrog steps per transition would take 0.18 seconds.
    ## Chain 3: Adjust your expectations accordingly!
    ## Chain 3: 
    ## Chain 3: 
    ## Chain 3: Iteration:    1 / 10000 [  0%]  (Warmup)
    ## Chain 3: Iteration: 1000 / 10000 [ 10%]  (Warmup)
    ## Chain 3: Iteration: 2000 / 10000 [ 20%]  (Warmup)
    ## Chain 3: Iteration: 3000 / 10000 [ 30%]  (Warmup)
    ## Chain 3: Iteration: 4000 / 10000 [ 40%]  (Warmup)
    ## Chain 3: Iteration: 5000 / 10000 [ 50%]  (Warmup)
    ## Chain 3: Iteration: 5001 / 10000 [ 50%]  (Sampling)
    ## Chain 3: Iteration: 6000 / 10000 [ 60%]  (Sampling)
    ## Chain 3: Iteration: 7000 / 10000 [ 70%]  (Sampling)
    ## Chain 3: Iteration: 8000 / 10000 [ 80%]  (Sampling)
    ## Chain 3: Iteration: 9000 / 10000 [ 90%]  (Sampling)
    ## Chain 3: Iteration: 10000 / 10000 [100%]  (Sampling)
    ## Chain 3: 
    ## Chain 3:  Elapsed Time: 1.624 seconds (Warm-up)
    ## Chain 3:                15.036 seconds (Sampling)
    ## Chain 3:                16.66 seconds (Total)
    ## Chain 3: 
    ## 
    ## SAMPLING FOR MODEL 'continuous' NOW (CHAIN 4).
    ## Chain 4: 
    ## Chain 4: Gradient evaluation took 1.9e-05 seconds
    ## Chain 4: 1000 transitions using 10 leapfrog steps per transition would take 0.19 seconds.
    ## Chain 4: Adjust your expectations accordingly!
    ## Chain 4: 
    ## Chain 4: 
    ## Chain 4: Iteration:    1 / 10000 [  0%]  (Warmup)
    ## Chain 4: Iteration: 1000 / 10000 [ 10%]  (Warmup)
    ## Chain 4: Iteration: 2000 / 10000 [ 20%]  (Warmup)
    ## Chain 4: Iteration: 3000 / 10000 [ 30%]  (Warmup)
    ## Chain 4: Iteration: 4000 / 10000 [ 40%]  (Warmup)
    ## Chain 4: Iteration: 5000 / 10000 [ 50%]  (Warmup)
    ## Chain 4: Iteration: 5001 / 10000 [ 50%]  (Sampling)
    ## Chain 4: Iteration: 6000 / 10000 [ 60%]  (Sampling)
    ## Chain 4: Iteration: 7000 / 10000 [ 70%]  (Sampling)
    ## Chain 4: Iteration: 8000 / 10000 [ 80%]  (Sampling)
    ## Chain 4: Iteration: 9000 / 10000 [ 90%]  (Sampling)
    ## Chain 4: Iteration: 10000 / 10000 [100%]  (Sampling)
    ## Chain 4: 
    ## Chain 4:  Elapsed Time: 1.423 seconds (Warm-up)
    ## Chain 4:                15.343 seconds (Sampling)
    ## Chain 4:                16.766 seconds (Total)
    ## Chain 4:

``` r
prior_summary(ext_main) 
```

    ## Priors for model 'ext_main' 
    ## ------
    ## Intercept (after predictors centered)
    ##   Specified prior:
    ##     ~ normal(location = 1800, scale = 150)
    ##   Adjusted prior:
    ##     ~ normal(location = 1800, scale = 64093)
    ## 
    ## Coefficients
    ##   Specified prior:
    ##     ~ normal(location = [0,0,0,...], scale = [2.5,2.5,2.5,...])
    ##   Adjusted prior:
    ##     ~ normal(location = [0,0,0,...], scale = [2136.54,2137.02,2137.26,...])
    ## 
    ## Auxiliary (sigma)
    ##   Specified prior:
    ##     ~ exponential(rate = 1)
    ##   Adjusted prior:
    ##     ~ exponential(rate = 0.0023)
    ## ------
    ## See help('prior_summary.stanreg') for more details

``` r
mcmc_trace(ext_main, size = .1)
```

![](Final-Project-LOL-New_files/figure-gfm/unnamed-chunk-53-1.png)<!-- -->

Traces look good

``` r
rhat(ext_main)
```

    ##       (Intercept)           winner2       firstBlood2       firstTower2 
    ##         0.9999612         0.9999395         0.9999895         0.9998515 
    ##      firstDragon1      firstDragon2       firstBaron1       firstBaron2 
    ##         1.0000056         1.0000145         1.0001244         1.0001247 
    ##     t1_towerKills t1_inhibitorKills     t1_baronKills    t1_dragonKills 
    ##         0.9999698         1.0000633         1.0001761         0.9998990 
    ##     t2_towerKills t2_inhibitorKills     t2_baronKills    t2_dragonKills 
    ##         0.9998515         0.9999585         1.0003134         0.9999253 
    ##             sigma 
    ##         0.9998820

``` r
neff_ratio(ext_main)
```

    ##       (Intercept)           winner2       firstBlood2       firstTower2 
    ##           0.58290           0.79920           2.05680           1.44595 
    ##      firstDragon1      firstDragon2       firstBaron1       firstBaron2 
    ##           0.48310           0.48375           0.66920           0.65460 
    ##     t1_towerKills t1_inhibitorKills     t1_baronKills    t1_dragonKills 
    ##           0.71785           1.02960           0.65910           0.94240 
    ##     t2_towerKills t2_inhibitorKills     t2_baronKills    t2_dragonKills 
    ##           0.74235           1.01930           0.65770           0.91395 
    ##             sigma 
    ##           1.16840

Rhat looks good, some neff ratios (i.e. firstBlood, firstTower) are now
too high to keep and will need to be handled later

``` r
tidy(ext_main, effects = c("fixed", "aux"),
     conf.int = TRUE, conf.level = 0.95)
```

    ## # A tibble: 18 × 5
    ##    term              estimate std.error conf.low conf.high
    ##    <chr>                <dbl>     <dbl>    <dbl>     <dbl>
    ##  1 (Intercept)        719.       12.5     694.      743.  
    ##  2 winner2             -4.75      5.26    -15.1       5.54
    ##  3 firstBlood2          1.37      2.22     -3.04      5.68
    ##  4 firstTower2          0.523     2.61     -4.61      5.59
    ##  5 firstDragon1       -62.5      11.7     -84.8     -39.8 
    ##  6 firstDragon2       -65.3      11.7     -87.5     -42.5 
    ##  7 firstBaron1        -85.4       4.99    -95.1     -75.4 
    ##  8 firstBaron2        -81.5       4.80    -91.0     -72.0 
    ##  9 t1_towerKills       53.2       0.788    51.7      54.7 
    ## 10 t1_inhibitorKills    6.65      1.52      3.69      9.67
    ## 11 t1_baronKills      142.        3.80    135.      150.  
    ## 12 t1_dragonKills     182.        1.52    179.      185.  
    ## 13 t2_towerKills       54.6       0.790    53.1      56.1 
    ## 14 t2_inhibitorKills    4.72      1.58      1.68      7.74
    ## 15 t2_baronKills      142.        3.60    135.      149.  
    ## 16 t2_dragonKills     177.        1.53    174.      180.  
    ## 17 sigma              229.        0.760   228.      231.  
    ## 18 mean_PPD          1930.        1.54   1927.     1933.

Here, the predictors winner, firstBlood, and firstTower are not
significant due to their 95% CI including 0

``` r
# dataframe with the specified values
newdata_ext <- data.frame(
  winner = factor(1, levels = levels(lol_data$winner)),
  firstBlood = factor(2, levels = levels(lol_data$firstBlood)),
  firstTower = factor(2, levels = levels(lol_data$firstTower)),
  firstDragon = factor(2, levels = levels(lol_data$firstDragon)),
  firstBaron = factor(2, levels = levels(lol_data$firstBaron)),
  t1_towerKills = 7,
  t1_inhibitorKills = 2,
  t1_baronKills = 2,
  t1_dragonKills = 2,
  t2_towerKills = 3,
  t2_inhibitorKills = 2,
  t2_baronKills = 1,
  t2_dragonKills = 3
)

ext_main_predict <- posterior_predict(ext_main, newdata = newdata_ext)

mcmc_areas(ext_main_predict) +  
  xlab("Game Duration") +
  ggtitle('Predictive distribution of a League of Legends game where first blood, tower, dragon, baron was team 2 and team 1 won, with additional predictors') +
  theme(plot.title = element_text(size = 8))
```

![](Final-Project-LOL-New_files/figure-gfm/unnamed-chunk-57-1.png)<!-- -->

After adding a few more variables into play we can see the game gets
longer (and expectedly so). Team 1 comes from behind but ends up getting
2 barons and 7 towers making us believe this game will have to go on for
a while for the comeback to truly be complete

## Refined Model

Let’s now remove the high neff ratio terms from the extended model
(i.e. \> 1) and include the interaction terms we deemed useful
beforehand

``` r
ext_main_int <- stan_glm(gameDuration ~ firstDragon + firstBaron + t1_baronKills + t1_dragonKills + t2_baronKills + t2_dragonKills + t1_baronKills:t2_baronKills + t1_dragonKills:t2_dragonKills,
  data = lol_data, family = gaussian, 
  prior_intercept = normal(1800, 150, autoscale = TRUE),
  prior = normal(0, 2.5, autoscale = TRUE), 
  prior_aux = exponential(1, autoscale = TRUE),
  chains = 4, iter = 5000*2, seed = 84735)
```

    ## 
    ## SAMPLING FOR MODEL 'continuous' NOW (CHAIN 1).
    ## Chain 1: 
    ## Chain 1: Gradient evaluation took 1.7e-05 seconds
    ## Chain 1: 1000 transitions using 10 leapfrog steps per transition would take 0.17 seconds.
    ## Chain 1: Adjust your expectations accordingly!
    ## Chain 1: 
    ## Chain 1: 
    ## Chain 1: Iteration:    1 / 10000 [  0%]  (Warmup)
    ## Chain 1: Iteration: 1000 / 10000 [ 10%]  (Warmup)
    ## Chain 1: Iteration: 2000 / 10000 [ 20%]  (Warmup)
    ## Chain 1: Iteration: 3000 / 10000 [ 30%]  (Warmup)
    ## Chain 1: Iteration: 4000 / 10000 [ 40%]  (Warmup)
    ## Chain 1: Iteration: 5000 / 10000 [ 50%]  (Warmup)
    ## Chain 1: Iteration: 5001 / 10000 [ 50%]  (Sampling)
    ## Chain 1: Iteration: 6000 / 10000 [ 60%]  (Sampling)
    ## Chain 1: Iteration: 7000 / 10000 [ 70%]  (Sampling)
    ## Chain 1: Iteration: 8000 / 10000 [ 80%]  (Sampling)
    ## Chain 1: Iteration: 9000 / 10000 [ 90%]  (Sampling)
    ## Chain 1: Iteration: 10000 / 10000 [100%]  (Sampling)
    ## Chain 1: 
    ## Chain 1:  Elapsed Time: 1.363 seconds (Warm-up)
    ## Chain 1:                14.647 seconds (Sampling)
    ## Chain 1:                16.01 seconds (Total)
    ## Chain 1: 
    ## 
    ## SAMPLING FOR MODEL 'continuous' NOW (CHAIN 2).
    ## Chain 2: 
    ## Chain 2: Gradient evaluation took 1.7e-05 seconds
    ## Chain 2: 1000 transitions using 10 leapfrog steps per transition would take 0.17 seconds.
    ## Chain 2: Adjust your expectations accordingly!
    ## Chain 2: 
    ## Chain 2: 
    ## Chain 2: Iteration:    1 / 10000 [  0%]  (Warmup)
    ## Chain 2: Iteration: 1000 / 10000 [ 10%]  (Warmup)
    ## Chain 2: Iteration: 2000 / 10000 [ 20%]  (Warmup)
    ## Chain 2: Iteration: 3000 / 10000 [ 30%]  (Warmup)
    ## Chain 2: Iteration: 4000 / 10000 [ 40%]  (Warmup)
    ## Chain 2: Iteration: 5000 / 10000 [ 50%]  (Warmup)
    ## Chain 2: Iteration: 5001 / 10000 [ 50%]  (Sampling)
    ## Chain 2: Iteration: 6000 / 10000 [ 60%]  (Sampling)
    ## Chain 2: Iteration: 7000 / 10000 [ 70%]  (Sampling)
    ## Chain 2: Iteration: 8000 / 10000 [ 80%]  (Sampling)
    ## Chain 2: Iteration: 9000 / 10000 [ 90%]  (Sampling)
    ## Chain 2: Iteration: 10000 / 10000 [100%]  (Sampling)
    ## Chain 2: 
    ## Chain 2:  Elapsed Time: 1.234 seconds (Warm-up)
    ## Chain 2:                14.68 seconds (Sampling)
    ## Chain 2:                15.914 seconds (Total)
    ## Chain 2: 
    ## 
    ## SAMPLING FOR MODEL 'continuous' NOW (CHAIN 3).
    ## Chain 3: 
    ## Chain 3: Gradient evaluation took 1.7e-05 seconds
    ## Chain 3: 1000 transitions using 10 leapfrog steps per transition would take 0.17 seconds.
    ## Chain 3: Adjust your expectations accordingly!
    ## Chain 3: 
    ## Chain 3: 
    ## Chain 3: Iteration:    1 / 10000 [  0%]  (Warmup)
    ## Chain 3: Iteration: 1000 / 10000 [ 10%]  (Warmup)
    ## Chain 3: Iteration: 2000 / 10000 [ 20%]  (Warmup)
    ## Chain 3: Iteration: 3000 / 10000 [ 30%]  (Warmup)
    ## Chain 3: Iteration: 4000 / 10000 [ 40%]  (Warmup)
    ## Chain 3: Iteration: 5000 / 10000 [ 50%]  (Warmup)
    ## Chain 3: Iteration: 5001 / 10000 [ 50%]  (Sampling)
    ## Chain 3: Iteration: 6000 / 10000 [ 60%]  (Sampling)
    ## Chain 3: Iteration: 7000 / 10000 [ 70%]  (Sampling)
    ## Chain 3: Iteration: 8000 / 10000 [ 80%]  (Sampling)
    ## Chain 3: Iteration: 9000 / 10000 [ 90%]  (Sampling)
    ## Chain 3: Iteration: 10000 / 10000 [100%]  (Sampling)
    ## Chain 3: 
    ## Chain 3:  Elapsed Time: 1.549 seconds (Warm-up)
    ## Chain 3:                14.661 seconds (Sampling)
    ## Chain 3:                16.21 seconds (Total)
    ## Chain 3: 
    ## 
    ## SAMPLING FOR MODEL 'continuous' NOW (CHAIN 4).
    ## Chain 4: 
    ## Chain 4: Gradient evaluation took 1.6e-05 seconds
    ## Chain 4: 1000 transitions using 10 leapfrog steps per transition would take 0.16 seconds.
    ## Chain 4: Adjust your expectations accordingly!
    ## Chain 4: 
    ## Chain 4: 
    ## Chain 4: Iteration:    1 / 10000 [  0%]  (Warmup)
    ## Chain 4: Iteration: 1000 / 10000 [ 10%]  (Warmup)
    ## Chain 4: Iteration: 2000 / 10000 [ 20%]  (Warmup)
    ## Chain 4: Iteration: 3000 / 10000 [ 30%]  (Warmup)
    ## Chain 4: Iteration: 4000 / 10000 [ 40%]  (Warmup)
    ## Chain 4: Iteration: 5000 / 10000 [ 50%]  (Warmup)
    ## Chain 4: Iteration: 5001 / 10000 [ 50%]  (Sampling)
    ## Chain 4: Iteration: 6000 / 10000 [ 60%]  (Sampling)
    ## Chain 4: Iteration: 7000 / 10000 [ 70%]  (Sampling)
    ## Chain 4: Iteration: 8000 / 10000 [ 80%]  (Sampling)
    ## Chain 4: Iteration: 9000 / 10000 [ 90%]  (Sampling)
    ## Chain 4: Iteration: 10000 / 10000 [100%]  (Sampling)
    ## Chain 4: 
    ## Chain 4:  Elapsed Time: 1.22 seconds (Warm-up)
    ## Chain 4:                15.007 seconds (Sampling)
    ## Chain 4:                16.227 seconds (Total)
    ## Chain 4:

``` r
prior_summary(ext_main_int)
```

    ## Priors for model 'ext_main_int' 
    ## ------
    ## Intercept (after predictors centered)
    ##   Specified prior:
    ##     ~ normal(location = 1800, scale = 150)
    ##   Adjusted prior:
    ##     ~ normal(location = 1800, scale = 64093)
    ## 
    ## Coefficients
    ##   Specified prior:
    ##     ~ normal(location = [0,0,0,...], scale = [2.5,2.5,2.5,...])
    ##   Adjusted prior:
    ##     ~ normal(location = [0,0,0,...], scale = [2136.66,2136.42,2289.92,...])
    ## 
    ## Auxiliary (sigma)
    ##   Specified prior:
    ##     ~ exponential(rate = 1)
    ##   Adjusted prior:
    ##     ~ exponential(rate = 0.0023)
    ## ------
    ## See help('prior_summary.stanreg') for more details

``` r
mcmc_trace(ext_main_int, size = .1)
```

![](Final-Project-LOL-New_files/figure-gfm/unnamed-chunk-60-1.png)<!-- -->

So far so good

``` r
rhat(ext_main_int)
```

    ##                   (Intercept)                  firstDragon1 
    ##                     1.0003440                     1.0003428 
    ##                  firstDragon2                   firstBaron1 
    ##                     1.0003775                     1.0000760 
    ##                   firstBaron2                 t1_baronKills 
    ##                     1.0002465                     1.0000761 
    ##                t1_dragonKills                 t2_baronKills 
    ##                     0.9998796                     1.0002153 
    ##                t2_dragonKills   t1_baronKills:t2_baronKills 
    ##                     0.9999051                     1.0002525 
    ## t1_dragonKills:t2_dragonKills                         sigma 
    ##                     1.0000077                     1.0000297

Rhat looks satisfactory for all predictors

``` r
neff_ratio(ext_main_int)
```

    ##                   (Intercept)                  firstDragon1 
    ##                       0.58795                       0.50540 
    ##                  firstDragon2                   firstBaron1 
    ##                       0.50105                       0.56190 
    ##                   firstBaron2                 t1_baronKills 
    ##                       0.53910                       0.53865 
    ##                t1_dragonKills                 t2_baronKills 
    ##                       0.67185                       0.53005 
    ##                t2_dragonKills   t1_baronKills:t2_baronKills 
    ##                       0.67705                       0.66965 
    ## t1_dragonKills:t2_dragonKills                         sigma 
    ##                       0.82210                       1.12835

Neff ratios are much more reasonable now and a big improvement from
prior models with no value being \> .90

``` r
tidy(ext_main_int, effects = c("fixed", "aux"),
     conf.int = TRUE, conf.level = 0.95)
```

    ## # A tibble: 13 × 5
    ##    term                          estimate std.error conf.low conf.high
    ##    <chr>                            <dbl>     <dbl>    <dbl>     <dbl>
    ##  1 (Intercept)                     1211.     12.2     1187.     1234. 
    ##  2 firstDragon1                     -41.2    13.0      -66.2     -15.8
    ##  3 firstDragon2                     -45.5    13.0      -70.6     -20.3
    ##  4 firstBaron1                      -62.3     6.15     -74.2     -50.2
    ##  5 firstBaron2                      -58.2     5.81     -69.6     -46.9
    ##  6 t1_baronKills                    178.      4.86     168.      188. 
    ##  7 t1_dragonKills                   205.      1.85     202.      209. 
    ##  8 t2_baronKills                    178.      4.56     170.      187. 
    ##  9 t2_dragonKills                   202.      1.83     199.      206. 
    ## 10 t1_baronKills:t2_baronKills       39.3     4.52      30.5      48.2
    ## 11 t1_dragonKills:t2_dragonKills     24.1     0.959     22.3      26.0
    ## 12 sigma                            252.      0.832    251.      254. 
    ## 13 mean_PPD                        1930.      1.67    1927.     1933.

Every value looks significant from the 95% CI (even the interaction
terms!!) due to none of them including 0.

``` r
# dataframe with the specified values
newdata_ext_int <- data.frame(
  firstDragon = factor(2, levels = levels(lol_data$firstDragon)),
  firstBaron = factor(2, levels = levels(lol_data$firstBaron)),
  t1_baronKills = 2,
  t1_dragonKills = 2,
  t2_baronKills = 1,
  t2_dragonKills = 3
)

ext_main_int_predict <- posterior_predict(ext_main_int, newdata = newdata_ext_int)

mcmc_areas(ext_main_int_predict) +  
  xlab("Game Duration") +
  ggtitle('Predictive distribution of a League of Legends game where first dragon, baron was team 2 and team 1 had more baron kills with interaction') +
  theme(plot.title = element_text(size = 8))
```

![](Final-Project-LOL-New_files/figure-gfm/unnamed-chunk-64-1.png)<!-- -->

The games seem to drag on even longer with this new model. This is
expected since the dragon and baron respawn times are quite long so, as
said before, the comeback would realistically take longer than average.

Lets do some model comparisons to verify which model is best

## Model Comparisons

### PP Check

``` r
pp_check(main, nreps = 50) + xlab("Game Duration") +
  ggtitle('Main effects model')
```

![](Final-Project-LOL-New_files/figure-gfm/unnamed-chunk-65-1.png)<!-- -->

The main model gets a decent amount of area, yet it starts moving too
far right and undershoots at the peak

``` r
pp_check(ext_main, nreps = 50) + xlab("Game Duration") +
  ggtitle('Extended Main effects model')
```

![](Final-Project-LOL-New_files/figure-gfm/unnamed-chunk-66-1.png)<!-- -->

The extended model is objectively better than the main model, but still
undershoots area at the peak

``` r
pp_check(ext_main_int, nreps = 50) + xlab("Game Duration") +
  ggtitle('Refined Interaction effects model')
```

![](Final-Project-LOL-New_files/figure-gfm/unnamed-chunk-67-1.png)<!-- -->

The interaction model is very similar to the extended model which is
better than the main model.

The ppchecks show most of the are being covered for the last 2 models
with the interaction model doing better around the peak by a small
margin

### 10 Fold Cross-Validations

``` r
test_sample <- lol_data %>% head(10000)
nrow(test_sample)
```

    ## [1] 10000

``` r
set.seed(84735)

p_main <- prediction_summary(model = main, data = test_sample)
p_ext_main <- prediction_summary(model = ext_main, data = test_sample)
p_ext_main_int <- prediction_summary(model = ext_main_int, data = test_sample)
```

``` r
rbind(p_main, p_ext_main, p_ext_main_int)
```

    ##        mae mae_scaled within_50 within_95
    ## 1 255.6851  0.6784996    0.4974    0.9609
    ## 2 149.6301  0.6522156    0.5142    0.9514
    ## 3 166.5749  0.6602105    0.5099    0.9554

These are the raw MAE values based on a sample of 10000 observations
from the data, we can use these to determine the behavior and bias of
each model based on their cross-validation results

``` r
set.seed(84735)

cv_main <- prediction_summary_cv(
  model = main, data = lol_data, k = 10)

cv_extend <- prediction_summary_cv(
  model = ext_main, data = lol_data, k = 10)

cv_interact <- prediction_summary_cv(
  model = ext_main_int, data = lol_data, k = 10)
```

``` r
rbind(cv_main$cv, cv_extend$cv, cv_interact$cv)
```

    ##        mae mae_scaled within_50 within_95
    ## 1 255.8624  0.6787885 0.4971249 0.9586412
    ## 2 150.2308  0.6559153 0.5124297 0.9516519
    ## 3 166.2608  0.6589512 0.5095544 0.9526031

### Loo Diagnostics

``` r
set.seed(34521)
main_elpd <- loo(main)
ext_main_elpd <- loo(ext_main)
ext_main_interact_elpd <- loo(ext_main_int)

main_elpd$estimates
```

    ##               Estimate          SE
    ## elpd_loo -3.323804e+05 174.6839030
    ## p_loo     9.093789e+00   0.1196568
    ## looic     6.647608e+05 349.3678059

``` r
ext_main_elpd$estimates
```

    ##              Estimate          SE
    ## elpd_loo -309859.5959 173.9985196
    ## p_loo         19.2336   0.2843918
    ## looic     619719.1918 347.9970392

``` r
ext_main_interact_elpd$estimates
```

    ##               Estimate          SE
    ## elpd_loo -314197.32595 179.0596124
    ## p_loo         13.56236   0.2282117
    ## looic     628394.65191 358.1192249

``` r
c(main_elpd$estimates[1], ext_main_elpd$estimates[1], ext_main_interact_elpd$estimates[1])
```

    ## [1] -332380.4 -309859.6 -314197.3

``` r
loo_compare(main_elpd, ext_main_elpd, ext_main_interact_elpd)
```

    ##              elpd_diff se_diff 
    ## ext_main          0.0       0.0
    ## ext_main_int  -4337.7      96.8
    ## main         -22520.8     178.3

The extended model fairs better than the interaction model when
comparing their ELPD and MAE. The interaction model does 4337.6 points
“worse” for ELPD and is around 16 points higher in its MAE. However,
something to consider here is the idea of
[overfitting](https://statisticsbyjim.com/regression/overfitting-regression-models/)
and how more predictors and affect the model overall and skew our
metrics.

Despite the interaction model covering more area, the
[MAE](https://stephenallwright.com/good-mae-score/) and ELPD are both
worse. This may be due to these values being inflated by the larger
number of predictors in the extended model. Furthermore, the extended
model contains numerous values with higher than normal Neff Ratios which
is a [cause for
concern](https://stats.stackexchange.com/questions/296059/effective-sample-size-greater-than-actual-sample-size)
about the validity of the model along with predictors that were not
significant.

Despite having less predictors, the interaction model garners more area
in its predictive posterior distribution while having a similar MAE and
good ELPD score too (-309859.7 vs 314197.3) which is a minor (~1%)
difference given the size of these values. However, these interaction
terms could also be having an adverse effect on the model leading to
these greater errors.

### Overfitting

``` r
rbind(p_main, p_ext_main, p_ext_main_int) 
```

    ##        mae mae_scaled within_50 within_95
    ## 1 255.6851  0.6784996    0.4974    0.9609
    ## 2 149.6301  0.6522156    0.5142    0.9514
    ## 3 166.5749  0.6602105    0.5099    0.9554

``` r
rbind(cv_main$cv, cv_extend$cv, cv_interact$cv)
```

    ##        mae mae_scaled within_50 within_95
    ## 1 255.8624  0.6787885 0.4971249 0.9586412
    ## 2 150.2308  0.6559153 0.5124297 0.9516519
    ## 3 166.2608  0.6589512 0.5095544 0.9526031

Based on the difference between the raw MAE values and the
cross-validation MAE values, overfitting does not seem to pose a threat
to these models, however we can only truly know this if we are given new
data entirely to test the models. Furthermore, the extended model does
have a larger (though mostly negligible due to how small it is)
difference from the cross-validation MAE compared to the other models.

What is undoubtedly clear is that adding numerical predictors to the
model does help the predictions overall and is shown with both the
extended and interaction model fairing better than the main model in
every aspect.

## Regression Inference

Let’s see how the 2 best models fair when conducting a quick hypothesis
test to see the Posterior probability of a game around 40 minutes long

We’ll use the predictive posterior models we created and visualized
earlier to conduct the tests

$$
H_0: \pi \geq 2500
$$

$$
H_a: \pi < 2500
$$

``` r
# Extract the posterior samples
posterior_samples <- ext_main_int_predict

p_H0 <- mean(posterior_samples >= 2500)

p_Ha <- mean(posterior_samples < 2500)

# Print the results
cat("Posterior probability of H0 (π ≥ 2500):", p_H0)
```

    ## Posterior probability of H0 (π ≥ 2500): 0.9364

``` r
cat("Posterior probability of Ha (π < 2500):", p_Ha)
```

    ## Posterior probability of Ha (π < 2500): 0.0636

``` r
# Extract the posterior samples
posterior_samples <- ext_main_predict

p_H0 <- mean(posterior_samples >= 2500)

p_Ha <- mean(posterior_samples < 2500)

# Print the results
cat("Posterior probability of H0 (π ≥ 2500):", p_H0)
```

    ## Posterior probability of H0 (π ≥ 2500): 0.42085

``` r
cat("Posterior probability of Ha (π < 2500):", p_Ha)
```

    ## Posterior probability of Ha (π < 2500): 0.57915

Though the two models seemingly predict the actual distribution of the
data well, the 2 models have very different outcomes when conducting the
test.

The extended model favors the alternate hypothesis while the interaction
model clearly favors the null hypothesis. This both coincides with what
we saw before from the visualizations and tells us that the extended
model favors games that are shorter while the interaction model expects
games like this to take longer.

This is also something one should consider when choosing the model.
Whether or not the predictions themselves seem realistic given the
scenario/circumstances.

To back up what I’m saying, Baron Nashor spawns at 20 minutes and
respawns every 6 minutes. Since 3 Barons are killed in both scenarios,
the game would have to be at least 38 minutes and only if the teams kill
the Baron IMMEDIATELY (which is usually never the case). Therefore, one
would expect majority of the area to lie within the ~40 minute range at
the very least for when the game would end.

## Conclusion

In conclusion, it is a choice between whether one wants to take the
chance of playing with a model that may be susceptible to overfitting
for the sake of potentially less error or a slightly higher error model
that has interactions but would be less likely to be susceptible to
overfitting.

Personally, I say that the refined interaction model is the best model
to use for predicting the game duration of a LOL match given our data
due to its comparable MAE along with better posterior predictive
abilities and lack of evidence for overfitting. Furthermore, based on my
experience and game rules, the predictive values it comes up with end up
being much more realistic in the grand scheme of things and with other
factors such as respawn time considered.

## References

- <https://www.bayesrulesbook.com/>

- <https://stats.stackexchange.com/questions/296059/effective-sample-size-greater-than-actual-sample-size>

- <https://stephenallwright.com/good-mae-score/>

- <https://discourse.mc-stan.org/t/understanding-looic/13409/6>

- <https://discourse.mc-stan.org/t/projpred-elpd-goes-down-and-rmse-goes-up-after-x-variables/13153>

- <https://stats.stackexchange.com/questions/313564/how-does-bayesian-analysis-make-accurate-predictions-using-subjectively-chosen-p>

- <https://medium.com/@ooemma83/interpretation-of-evaluation-metrics-for-regression-analysis-mae-mse-rmse-mape-r-squared-and-5693b61a9833>

- <https://statisticsbyjim.com/regression/overfitting-regression-models/>

- <https://stats.stackexchange.com/questions/9053/how-does-cross-validation-overcome-the-overfitting-problem>

## Honor Pledge

On my honor, I have neither received nor given any unauthorized
assistance on this project

Signed: Thomas Christo (tjc260)
