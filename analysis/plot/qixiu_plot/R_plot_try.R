# plotting data 02/27/2020
d1 = data.frame(test_exp)
d2 = data.frame(test_exp_2)
d3 = data.frame(test_exp_3)

# RC cities list 
rc_all = c(subset(d1, cond==2)$n_city,subset(d2, cond==2)$n_city,subset(d3, cond==2)$n_city)
undo_all = c(subset(d1, cond==3)$n_city,subset(d2, cond==3)$n_city,subset(d3, cond==3)$n_city)
plot(rc_all,undo_all[1:75034])

con1 = mean(subset(test_exp_3, cond==2 & mapid)$n_city)
sd(subset(test_exp_3, cond==2)$n_city)
con2 = mean(subset(test_exp_3, cond==3 & mapid)$n_city)  
sd(subset(test_exp_3, cond==3)$n_city)

