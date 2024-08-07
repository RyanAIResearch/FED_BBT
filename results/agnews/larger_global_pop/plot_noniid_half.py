import matplotlib.pyplot as pyplot

popsize = 5
f_name_cent = '../bak_realtime/central_kshot40_lpop{}'.format(popsize)

f_name_fl = 'halfpicked_new_fl_noniid_alpha1_kshot40_frac1_lpop{}'.format(popsize)
f_name_fedavg = '../fedavg_noniid_alpha1_kshot40_frac1_lpop{}'.format(popsize)

acc_cent = []
acc_fl = []
acc_fedavg = []

end = 150

with open(f_name_cent, encoding='utf-8') as f:
    for line in f.readlines():
        if line[:8] == "Test acc":
            str_split = line.split(' ')
            acc = float(str_split[-1])
            acc_cent.append(acc)

with open(f_name_fl, encoding='utf-8') as f:
    for line in f.readlines():
        if line[:15] == "Global test acc":
            str_split = line.split(' ')
            acc = float(str_split[-1])
            acc_fl.append(acc)


with open(f_name_fedavg, encoding='utf-8') as f:
    for line in f.readlines():
        if line[:15] == "Global test acc":
            str_split = line.split(' ')
            acc = float(str_split[-1])
            acc_fedavg.append(acc)

print(max(acc_cent))
print(max(acc_fl))
print(max(acc_fedavg))

# acc_cent = acc_cent[:end]
# acc_fl = acc_fl[:end]
# acc_fedavg = acc_fedavg[:end]


print(acc_fl)

pyplot.plot([i for i in range(len(acc_cent))], acc_cent)
# pyplot.plot(acc_cent)
pyplot.plot(acc_fedavg)
pyplot.plot(acc_fl)

pyplot.legend(["Centralized", "FedAvg", "FedBBT"])
pyplot.savefig("halfpicked_noniid_pop{}.jpg".format(popsize))