import cma
import copy
import numpy as np

x = np.array([1,1,1,1,2,2,2,2])

es_init = cma.CMAEvolutionStrategy(8 * [0], 1, {'seed': 2023, 'maxiter':4, 'popsize':4})
es = es_init._copy_light(inopts={'seed': 2023, 'maxiter':4, 'popsize':2})

for i in range(2):
    print(es.countiter)
    solutions = es.ask()
    es.tell(solutions, [1,0])
    print(solutions)

print("!!!!!!!!!!")
# es.countiter = 0
es.sigma =es.sigma*2
while not es.stop():
    print(es.countiter)
    solutions = es.ask()
    solutions = np.concatenate([solutions, solutions], axis=0)
    es.tell(solutions, [2,0, 2, 0])
    print(solutions)
print("###########")
# while not es_init.stop():
#     print(es_init.countiter)
#     solutions = es_init.ask()
#     es_init.tell(solutions, [2,0,5,1])
#     print(solutions)
# print("###########")

#es.opts.set({'maxiter': 100})

# while not es.stop():
#     solutions = es.ask()
#     es.tell(solutions, [1,0,0.5,1])
#     print(solutions)

# print(es_init.mean)

# es_init.ask()
# es_init.tell(solutions, [1,0,0.5,1])
# print(es_init.mean)
# es_init.ask()
# es_init.tell(solutions, [1,0,0.5,1])
# print(es_init.mean)


# # print(es.C)

# # print(es.B)
# # print(es.D)

# print(es.sm.transform(x))

# # es.B = es.B*2

# # print(es.sm.C)

# # print(es.sm.B)
# # print(es.sm.D)


# es.sm.C[0,0] = 0


# # es.sm.update_now()

# # es.sm.update([x for i in range(len(x))], x)

# es.sm.update_now()

# # es.sm._decompose_C()

# # es.mean = es.mean + 2

# print(es.sm.C)

# print(es.sm.B)
# print(es.sm.D)


# print(es.sm.transform(x))
