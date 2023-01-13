MAXTIMEINICU <- 72

CLUSTERID <- Sys.getenv("CLUSTERID")
PROCID <- Sys.getenv("PROCID")
METHOD <- Sys.getenv("METHOD")
R_LIBS_USER <- Sys.getenv("R_LIBS_USER")

.libPaths(R_LIBS_USER)

install.packages("jsonlite")
install.packages("ggplot2")
install.packages("glue")
install.packages("here")
install.packages("logger")
install.packages("readr")
install.packages("reshape2")
install.packages("ROCR")

library("jsonlite")
library("glue")
library("here")
library("logger")
library("readr")
library("ggplot2")
library("reshape2")

base_base_loc <- here("data")
base_loc <- here(base_base_loc, "sepsis")
results <- here(base_loc, "results")

resfile <- here(glue::glue(CLUSTERID, "-", PROCID, "-", METHOD, ".json"))
csvfile <- here(glue::glue(CLUSTERID, "-", PROCID, "-", METHOD, ".csv"))
statframe <- data.frame()

curv_json <- read_json(resfile)

hours_list <- c("24hr", "12hr", "6hr", "3hr")

curv_data3 <- curv_json$`24hr`
aliv_data3 <- data.frame()
dead_data3 <- data.frame()

n <- length(curv_data3$truth)
for(i in 1:n) {
  status <- ifelse(curv_data3$truth[[i]] == 0, "alive", "dead")
  tmp <- unlist(curv_data3$iloc[[i]])
  tmp_df <- data.frame(x = as.numeric(names(tmp)), y = tmp)
  if(status == "alive") {
    aliv_data3 <- rbind(aliv_data3, t(tmp))
  } else {
    dead_data3 <- rbind(dead_data3, t(tmp))
  }
}

aliv_data3t <- t(t(colMeans(aliv_data3)))
aliv_data3t <- data.frame(x = as.numeric(rownames(aliv_data3t)), y = aliv_data3t)
dead_data3t <- t(t(colMeans(dead_data3)))
dead_data3t <- data.frame(x = as.numeric(rownames(dead_data3t)), y = dead_data3t)

p_avg <- ggplot() +
  geom_line(data = aliv_data3t, aes(x = x, y = y, color = "Alive")) +
  geom_line(data = dead_data3t, aes(x = x, y = y, color = "Dead")) +
  ylim(0, 1) + xlab("Time (hours)") + ylab("Probability of Survival") +
  scale_color_manual(breaks = c("Alive", "Dead"), values = c("black", "red"), name = "Status at 72 Hours") +
  theme(legend.position = "bottom")

ggsave(here(glue::glue(CLUSTERID, "-", PROCID, "-", METHOD, "-24hr-survival.pdf")), p_avg)

all_data24 <- rbind(data.frame(pred = aliv_data3$`71.0`, truth = 1),
                   data.frame(pred = dead_data3$`71.0`, truth = 0))

pred.time24 <- ROCR::prediction(all_data24$pred, all_data24$truth)
perf.time24 <- ROCR::performance(pred.time24 , measure = "tpr", x.measure = "fpr")
auc <- ROCR::performance(pred.time24, measure = "auc")
auc.time24 <- auc@y.values[[1]]
model.label.time24 <- paste("24 hours (auc: ", round(auc.time24, 2), ")", sep = "")

df <- data.frame(x = 0:1 , y = 0:1)
roc.time24 <- data.frame(pfa = unlist(perf.time24@x.values), pd = unlist(perf.time24@y.values), model = model.label.time24)

p_roc <- ggplot() +
    geom_line(data = roc.time24, aes(x=pfa, y=pd, color = model)) +
    xlab("False Positive Rate") + ylab("True Positive Rate") +
    geom_line(data = df, aes(x = x, y = y), linetype = "dotted") +
    labs(color = "Time of Prediction") + theme(legend.position="bottom")
ggsave(here(glue::glue(CLUSTERID, "-", PROCID, "-", METHOD, "-24hr-ROC.pdf")), p_roc)

tval <- t.test(aliv_data3[,10], dead_data3[,10], "greater")
statframe.tmp <- data.frame(clusterid = CLUSTERID, procid = PROCID, method = METHOD, period = "24",
                            xmean = tval$estimate[1], ymean =tval$estimate[2],
                            stderr = tval$stderr, tstat = tval$statistic,
                            df = tval$parameter[1], pval = tval$p.value, auc=auc.time24)
rownames(statframe.tmp) <- NULL

statframe <- rbind(statframe, statframe.tmp)

curv_data3 <- curv_json$`12hr`
aliv_data3 <- data.frame()
dead_data3 <- data.frame()

n <- length(curv_data3$truth)
for(i in 1:n) {
  status <- ifelse(curv_data3$truth[[i]] == 0, "alive", "dead")
  tmp <- unlist(curv_data3$iloc[[i]])
  tmp_df <- data.frame(x = as.numeric(names(tmp)), y = tmp)
  if(status == "alive") {
    aliv_data3 <- rbind(aliv_data3, t(tmp))
  } else {
    dead_data3 <- rbind(dead_data3, t(tmp))
  }
}

aliv_data3t <- t(t(colMeans(aliv_data3)))
aliv_data3t <- data.frame(x = as.numeric(rownames(aliv_data3t)), y = aliv_data3t)
dead_data3t <- t(t(colMeans(dead_data3)))
dead_data3t <- data.frame(x = as.numeric(rownames(dead_data3t)), y = dead_data3t)

p_avg <- ggplot() +
  geom_line(data = aliv_data3t, aes(x = x, y = y, color = "Alive")) +
  geom_line(data = dead_data3t, aes(x = x, y = y, color = "Dead")) +
  ylim(0, 1) + xlab("Time (hours)") + ylab("Probability of Survival") +
  scale_color_manual(breaks = c("Alive", "Dead"), values = c("black", "red"), name = "Status at 72 Hours") +
  theme(legend.position = "bottom")

ggsave(here(glue::glue(CLUSTERID, "-", PROCID, "-", METHOD, "-12hr-survival.pdf")), p_avg)

all_data12 <- rbind(data.frame(pred = aliv_data3$`71.0`, truth = 1),
                   data.frame(pred = dead_data3$`71.0`, truth = 0))

pred.time12 <- ROCR::prediction(all_data12$pred, all_data12$truth)
perf.time12 <- ROCR::performance(pred.time12 , measure = "tpr", x.measure = "fpr")
auc <- ROCR::performance(pred.time12, measure = "auc")
auc.time12 <- auc@y.values[[1]]
model.label.time12 <- paste("12 hours (auc: ", round(auc.time12, 2), ")", sep = "")

df <- data.frame(x = 0:1 , y = 0:1)
roc.time12 <- data.frame(pfa = unlist(perf.time12@x.values), pd = unlist(perf.time12@y.values), model = model.label.time12)

p_roc <- ggplot() +
    geom_line(data = roc.time12, aes(x=pfa, y=pd, color = model)) +
    xlab("False Positive Rate") + ylab("True Positive Rate") +
    geom_line(data = df, aes(x = x, y = y), linetype = "dotted") +
    labs(color = "Time of Prediction") + theme(legend.position="bottom")
ggsave(here(glue::glue(CLUSTERID, "-", PROCID, "-", METHOD, "-12hr-ROC.pdf")), p_roc)

tval <- t.test(aliv_data3[,10], dead_data3[,10], "greater")
statframe.tmp <- data.frame(clusterid = CLUSTERID, procid = PROCID, method = METHOD, period = "12",
                            xmean = tval$estimate[1], ymean =tval$estimate[2],
                            stderr = tval$stderr, tstat = tval$statistic,
                            df = tval$parameter[1], pval = tval$p.value, auc=auc.time12)
rownames(statframe.tmp) <- NULL

statframe <- rbind(statframe, statframe.tmp)

curv_data3 <- curv_json$`6hr`
aliv_data3 <- data.frame()
dead_data3 <- data.frame()

n <- length(curv_data3$truth)
for(i in 1:n) {
  status <- ifelse(curv_data3$truth[[i]] == 0, "alive", "dead")
  tmp <- unlist(curv_data3$iloc[[i]])
  tmp_df <- data.frame(x = as.numeric(names(tmp)), y = tmp)
  if(status == "alive") {
    aliv_data3 <- rbind(aliv_data3, t(tmp))
  } else {
    dead_data3 <- rbind(dead_data3, t(tmp))
  }
}

aliv_data3t <- t(t(colMeans(aliv_data3)))
aliv_data3t <- data.frame(x = as.numeric(rownames(aliv_data3t)), y = aliv_data3t)
dead_data3t <- t(t(colMeans(dead_data3)))
dead_data3t <- data.frame(x = as.numeric(rownames(dead_data3t)), y = dead_data3t)

p_avg <- ggplot() +
  geom_line(data = aliv_data3t, aes(x = x, y = y, color = "Alive")) +
  geom_line(data = dead_data3t, aes(x = x, y = y, color = "Dead")) +
  ylim(0, 1) + xlab("Time (hours)") + ylab("Probability of Survival") +
  scale_color_manual(breaks = c("Alive", "Dead"), values = c("black", "red"), name = "Status at 72 Hours") +
  theme(legend.position = "bottom")

ggsave(here(glue::glue(CLUSTERID, "-", PROCID, "-", METHOD, "-06hr-survival.pdf")), p_avg)

all_data6 <- rbind(data.frame(pred = aliv_data3$`71.0`, truth = 1),
                   data.frame(pred = dead_data3$`71.0`, truth = 0))

pred.time6 <- ROCR::prediction(all_data6$pred, all_data6$truth)
perf.time6 <- ROCR::performance(pred.time6 , measure = "tpr", x.measure = "fpr")
auc <- ROCR::performance(pred.time6, measure = "auc")
auc.time6 <- auc@y.values[[1]]
model.label.time6 <- paste("6 hours (auc: ", round(auc.time6, 2), ")", sep = "")

df <- data.frame(x = 0:1 , y = 0:1)
roc.time6 <- data.frame(pfa = unlist(perf.time6@x.values), pd = unlist(perf.time6@y.values), model = model.label.time6)

p_roc <- ggplot() +
    geom_line(data = roc.time6, aes(x=pfa, y=pd, color = model)) +
    xlab("False Positive Rate") + ylab("True Positive Rate") +
    geom_line(data = df, aes(x = x, y = y), linetype = "dotted") +
    labs(color = "Time of Prediction") + theme(legend.position="bottom")
ggsave(here(glue::glue(CLUSTERID, "-", PROCID, "-", METHOD, "-06hr-ROC.pdf")), p_roc)

tval <- t.test(aliv_data3[,10], dead_data3[,10], "greater")
statframe.tmp <- data.frame(clusterid = CLUSTERID, procid = PROCID, method = METHOD, period = "6",
                            xmean = tval$estimate[1], ymean =tval$estimate[2],
                            stderr = tval$stderr, tstat = tval$statistic,
                            df = tval$parameter[1], pval = tval$p.value, auc=auc.time6)
rownames(statframe.tmp) <- NULL

statframe <- rbind(statframe, statframe.tmp)

curv_data3 <- curv_json$`3hr`
aliv_data3 <- data.frame()
dead_data3 <- data.frame()

n <- length(curv_data3$truth)
for(i in 1:n) {
  status <- ifelse(curv_data3$truth[[i]] == 0, "alive", "dead")
  tmp <- unlist(curv_data3$iloc[[i]])
  tmp_df <- data.frame(x = as.numeric(names(tmp)), y = tmp)
  if(status == "alive") {
    aliv_data3 <- rbind(aliv_data3, t(tmp))
  } else {
    dead_data3 <- rbind(dead_data3, t(tmp))
  }
}

aliv_data3t <- t(t(colMeans(aliv_data3)))
aliv_data3t <- data.frame(x = as.numeric(rownames(aliv_data3t)), y = aliv_data3t)
dead_data3t <- t(t(colMeans(dead_data3)))
dead_data3t <- data.frame(x = as.numeric(rownames(dead_data3t)), y = dead_data3t)

p_avg <- ggplot() +
  geom_line(data = aliv_data3t, aes(x = x, y = y, color = "Alive")) +
  geom_line(data = dead_data3t, aes(x = x, y = y, color = "Dead")) +
  ylim(0, 1) + xlab("Time (hours)") + ylab("Probability of Survival") +
  scale_color_manual(breaks = c("Alive", "Dead"), values = c("black", "red"), name = "Status at 72 Hours") +
  theme(legend.position = "bottom")

ggsave(here(glue::glue(CLUSTERID, "-", PROCID, "-", METHOD, "-03hr-survival.pdf")), p_avg)

all_data3 <- rbind(data.frame(pred = aliv_data3$`71.0`, truth = 1),
                   data.frame(pred = dead_data3$`71.0`, truth = 0))

pred.time3 <- ROCR::prediction(all_data3$pred, all_data3$truth)
perf.time3 <- ROCR::performance(pred.time3 , measure = "tpr", x.measure = "fpr")
auc <- ROCR::performance(pred.time3, measure = "auc")
auc.time3 <- auc@y.values[[1]]
model.label.time3 <- paste("3 hours (auc: ", round(auc.time3, 2), ")", sep = "")

df <- data.frame(x = 0:1 , y = 0:1)
roc.time3 <- data.frame(pfa = unlist(perf.time3@x.values), pd = unlist(perf.time3@y.values), model = model.label.time3)

p_roc <- ggplot() +
    geom_line(data = roc.time3, aes(x=pfa, y=pd, color = model)) +
    xlab("False Positive Rate") + ylab("True Positive Rate") +
    geom_line(data = df, aes(x = x, y = y), linetype = "dotted") +
    labs(color = "Time of Prediction") + theme(legend.position="bottom")
ggsave(here(glue::glue(CLUSTERID, "-", PROCID, "-", METHOD, "-03hr-ROC.pdf")), p_roc)

tval <- t.test(aliv_data3[,10], dead_data3[,10], "greater")
statframe.tmp <- data.frame(clusterid = CLUSTERID, procid = PROCID, method = METHOD, period = "3",
                            xmean = tval$estimate[1], ymean =tval$estimate[2],
                            stderr = tval$stderr, tstat = tval$statistic,
                            df = tval$parameter[1], pval = tval$p.value, auc=auc.time3)
rownames(statframe.tmp) <- NULL

statframe <- rbind(statframe, statframe.tmp)
write.csv(statframe, csvfile, row.names = FALSE)

p_roc <- ggplot() +
    geom_line(data = roc.time3, aes(x=pfa, y=pd, color = model)) +
    geom_line(data = roc.time6, aes(x=pfa, y=pd, color = model)) +
    geom_line(data = roc.time12, aes(x=pfa, y=pd, color = model)) +
    geom_line(data = roc.time24, aes(x=pfa, y=pd, color = model)) +
    xlab("False Positive Rate") + ylab("True Positive Rate") +
    geom_line(data = df, aes(x = x, y = y), linetype = "dotted") +
    labs(color = "Time of Prediction") +
    scale_color_manual(values = c(model.label.time3, model.label.time6, model.label.time12, model.label.time24),
                       breaks = c(model.label.time3, model.label.time6, model.label.time12, model.label.time24)) +
  theme(legend.position="bottom")
print(p_roc)
ggsave(here(glue::glue(CLUSTERID, "-", PROCID, "-", METHOD, "-collected-ROC.pdf")), p_roc)

write.csv(roc.time3, here(glue::glue(CLUSTERID, "-", PROCID, "-", METHOD, "-24hr-ROC.csv")), row.names = FALSE)
write.csv(roc.time3, here(glue::glue(CLUSTERID, "-", PROCID, "-", METHOD, "-12hr-ROC.csv")), row.names = FALSE)
write.csv(roc.time3, here(glue::glue(CLUSTERID, "-", PROCID, "-", METHOD, "-06hr-ROC.csv")), row.names = FALSE)
write.csv(roc.time3, here(glue::glue(CLUSTERID, "-", PROCID, "-", METHOD, "-03hr-ROC.csv")), row.names = FALSE)
