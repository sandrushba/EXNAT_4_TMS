rm(list=ls())
library(ggplot2)
library(data.table)
library(here)
data = read.csv(here::here("PCD/pcd_combined.csv"))
# data$PCD[data$PCD>1000] = NA
# data$PCD[data$PCD< -1000] = NA

data = data[complete.cases(data),]
data = data.table(data)
a = data[,.(mean=mean(PCD),median=median(PCD), max=max(PCD), min=min(PCD)), by=.(sub, session, target)]
max(data$PCD, na.rm = T)
ggplot(data, aes(x=sub, y=PCD, color=sub))+
  # geom_violin()+
  geom_point(alpha=.2)+
  facet_grid(session~target)+
  theme_bw()+
  theme(axis.text.x = element_text(angle=60, hjust = 1)) +
  coord_cartesian(ylim=c(-100,100))


ggplot(data[data$sub == "sub-05",], aes(x=idx, y=PCD, color=sub))+
  # geom_violin()+
  geom_line(alpha=1, se=F)+
  facet_grid(session~target)+
  theme_bw()+
  theme(axis.text.x = element_text(angle=60, hjust = 1)) +
  coord_cartesian(ylim=c(-100,100))
