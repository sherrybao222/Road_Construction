# plotting data 02/27/2020
plot(test_exp_3$mapid,test_exp_3$n_city)
con1 = mean(subset(test_exp_3, cond==2 & mapid)$n_city)
sd(subset(test_exp_3, cond==2)$n_city)
con2 = mean(subset(test_exp_3, cond==3 & mapid)$n_city)  
sd(subset(test_exp_3, cond==3)$n_city)

