import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

lp = 5

label = np.load("lp{}_label.npy".format(lp))
pred = np.load("lp{}_pred.npy".format(lp))

confusion_matrix = metrics.confusion_matrix(label[-1], pred[-1])
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[0,1,2,3])

cm_display.plot()

plt.savefig("cm_lp{}.png".format(lp))