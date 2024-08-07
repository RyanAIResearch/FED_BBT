import matplotlib.pyplot as pyplot

f_name_cent_5 = 'central_kshot40_lpop5'
f_name_cent_10 = 'central_kshot40_lpop10'

sigma_cent5 = []
sigma_cent10 = []

end = 150

with open(f_name_cent_5, encoding='utf-8') as f:
    for line in f.readlines():
        if line[:11] == "Check sigma":
            str_split = line.split(' ')
            acc = float(str_split[-1])
            sigma_cent5.append(acc)

with open(f_name_cent_10, encoding='utf-8') as f:
    for line in f.readlines():
        if line[:11] == "Check sigma":
            str_split = line.split(' ')
            acc = float(str_split[-1])
            sigma_cent10.append(acc)


print(max(sigma_cent5))
print(max(sigma_cent10))

# acc_cent = acc_cent[:end]
# acc_fl = acc_fl[:end]
# acc_fedavg = acc_fedavg[:end]



pyplot.plot([i*10/8 for i in range(len(sigma_cent5))], sigma_cent5)
pyplot.plot([i*10/8 for i in range(len(sigma_cent10))], sigma_cent10)


pyplot.legend(["Centralized 5", "Centralized 10"])
pyplot.ylabel("sigma")
pyplot.xlabel("communication round")
pyplot.savefig("sigma_centralized.jpg")