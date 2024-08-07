import matplotlib.pyplot as pyplot


f_name_cent = 'fl_noniid_alpha1_kshot-1_lr1e-2_frac1'

# f_name_cent = 'fl_iid_kshot-1_lr1e-2_frac1'
acc_cent = []



with open(f_name_cent, encoding='utf-8') as f:
    for line in f.readlines():
        if line[:15] == "Global test acc":
            str_split = line.split(' ')
            acc = float(str_split[-1])
            acc_cent.append(acc)


print(max(acc_cent[:200]))





pyplot.plot([i for i in range(len(acc_cent))], acc_cent)


pyplot.savefig("noniid_k40.jpg")
# pyplot.savefig("pop{}.jpg".format(popsize))