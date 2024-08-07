import matplotlib.pyplot as pyplot

popsize = 10
f_name_cent = '../central_kshot40_lpop{}'.format(popsize)

# f_name_fl = 'fl_noniid_alpha1_kshot40_frac1_lpop{}'.format(popsize)
# f_name_fedavg = 'fedavg_noniid_alpha1_kshot40_frac1_lpop{}'.format(popsize)

f_name_fl = 'new_fl_iid_kshot40_frac1_lpop{}'.format(popsize)
f_name_fedavg = '../check_prompt/fedavg_iid_kshot40_frac1_lpop{}'.format(popsize)

sigma_cent = []
sigma_fl = []
sigma_fedavg = []

end = 150

with open(f_name_cent, encoding='utf-8') as f:
    for line in f.readlines():
        if line[:11] == "Check sigma":
            str_split = line.split(' ')
            acc = float(str_split[-1])
            sigma_cent.append(acc)

with open(f_name_fl, encoding='utf-8') as f:
    for line in f.readlines():
        if line[:17] == "Check sigma local":
            str_split = line.split(' ')
            acc = float(str_split[-1])
            sigma_fl.append(acc)


with open(f_name_fedavg, encoding='utf-8') as f:
    for line in f.readlines():
        if line[:11] == "Check sigma":
            str_split = line.split(' ')
            acc = float(str_split[-1])
            sigma_fedavg.append(acc)

print(max(sigma_cent))
print(max(sigma_fl))
print(max(sigma_fedavg))

# acc_cent = acc_cent[:end]
# acc_fl = acc_fl[:end]
sigma_fedavg = sigma_fedavg[:end]



pyplot.plot([i*10/8 for i in range(len(sigma_cent))], sigma_cent)
# pyplot.plot(acc_cent)
pyplot.plot(sigma_fedavg)
pyplot.plot(sigma_fl)

pyplot.legend(["Centralized", "FedAvg", "FedBBT"])
pyplot.ylabel("sigma")
pyplot.xlabel("communication round")
pyplot.savefig("new_sigma_iid_pop{}.jpg".format(popsize))