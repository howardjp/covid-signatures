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

tval <- t.test(aliv_data3[,10], dead_data3[,10], "greater")
statframe.tmp <- data.frame(clusterid = CLUSTERID, procid = PROCID, method = METHOD, period = "24",
                            xmean = tval$estimate[1], ymean =tval$estimate[2],
                            stderr = tval$stderr, tstat = tval$statistic,
                            df = tval$parameter[1], pval = tval$p.value)
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

tval <- t.test(aliv_data3[,10], dead_data3[,10], "greater")
statframe.tmp <- data.frame(clusterid = CLUSTERID, procid = PROCID, method = METHOD, period = "12",
                            xmean = tval$estimate[1], ymean =tval$estimate[2],
                            stderr = tval$stderr, tstat = tval$statistic,
                            df = tval$parameter[1], pval = tval$p.value)
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

tval <- t.test(aliv_data3[,10], dead_data3[,10], "greater")
statframe.tmp <- data.frame(clusterid = CLUSTERID, procid = PROCID, method = METHOD, period = "6",
                            xmean = tval$estimate[1], ymean =tval$estimate[2],
                            stderr = tval$stderr, tstat = tval$statistic,
                            df = tval$parameter[1], pval = tval$p.value)
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

tval <- t.test(aliv_data3[,10], dead_data3[,10], "greater")
statframe.tmp <- data.frame(clusterid = CLUSTERID, procid = PROCID, method = METHOD, period = "3",
                            xmean = tval$estimate[1], ymean =tval$estimate[2],
                            stderr = tval$stderr, tstat = tval$statistic,
                            df = tval$parameter[1], pval = tval$p.value)
rownames(statframe.tmp) <- NULL

statframe <- rbind(statframe, statframe.tmp)

write.csv(statframe, csvfile, row.names = FALSE)