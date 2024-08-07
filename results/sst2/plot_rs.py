import matplotlib.pyplot as pyplot

popsize = 10
rs=50
f_name_cent = 'central_kshot40_lpop{}_rs{}'.format(popsize, rs)


acc_cent = []

end = 150

with open(f_name_cent, encoding='utf-8') as f:
    for line in f.readlines():
        if line[:8] == "Test acc":
            str_split = line.split(' ')
            acc = float(str_split[-1])
            acc_cent.append(acc)


print(max(acc_cent))


acc_cent = acc_cent[:end]




pyplot.plot([i for i in range(len(acc_cent))], acc_cent)


pyplot.savefig("pop{}_rs{}.jpg".format(popsize, rs))
# pyplot.savefig("pop{}.jpg".format(popsize))