remove(list = ls())
library(ggplot2)
library(reshape2)
setwd("/Users/hengzhang/Dropbox/My Projects/Xinfeng Private/TrawindShippingCostForecast")
load("clean_data_nov172018.RData")
df.base <- df.clean

# Get y 
regress.df <- df.clean %>% select(cost)

# Add ship_year and ship_month
df.base$ship_year <- as.numeric(format(as.Date(df.clean$ship_time), '%Y'))
df.base$ship_month <- as.numeric(format(as.Date(df.clean$ship_time), '%m'))

# Add long-term fixed effect
regress.df$trend <- (df.base$ship_year - 2016)*52 + df.base$ship_month

# Add month effect, vsl_company, total_weight
regress.df$ship_month <- as.factor(df.base$ship_month)
regress.df$port <- df.base$port
regress.df$total_weight <- df.base$total_weight
regress.df$vsl_type <- df.base$vsl_type

# Divide by training and testing
regress.train <- regress.df[1:409,]
regress.test <- regress.df[410:531,]
model1 <- lm(cost ~., data = regress.train)

# Summaize the result
tmp <- c(mean(abs(model1$residuals)), mean(abs(model1$residuals)/regress.train$cost),summary(model1)$r.squared)

# Prediction
regress.test$port[regress.test$port == 'JiaoJiang'] = 'JiaoJiangSSZD'
result_test_raw <- data.frame(true = regress.test$cost, pred = predict(model1, regress.test))
result_test_raw$error <- result_test_raw$true - result_test_raw$pred
tmp2 <- c(mean(abs(result_test_raw$error)), mean(abs(result_test_raw$error)/result_test_raw$true), NA)
summary_result_model1 <- data.frame(train = tmp, test = tmp2)
rownames(summary_result_model1) <- c("APE", "MAPE", "R-Squared")

# Plot the prediction and true values
regress.tmp <- regress.df
regress.tmp$port[regress.tmp$port == 'JiaoJiang'] = 'JiaoJiangSSZD'
result_tmp <- data.frame(true = regress.tmp$cost, pred = predict(model1, regress.tmp), ship_time = df.base$ship_time)
plot.df <- melt(result_tmp, id.vars = "ship_time")
p <- ggplot(plot.df, aes(x = ship_time, y = value, group = variable, color = variable)) + geom_line() + geom_point() + 
     geom_vline(xintercept = df.base$bid_time[410])
p


## Sanity check
#tmp <- regress.df %>% select(-1)
#tmp2 <- unique(tmp)
## Passed! No repetition of rows