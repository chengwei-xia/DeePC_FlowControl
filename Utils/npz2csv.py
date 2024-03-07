import numpy as np

DF = np.load("/home/issay/UROP/DPC_Flow/Saved_Hankels/Hankel.npz")
print(DF.files)

arr=DF.files[0]
np.savetxt("Up.csv", DF[arr], delimiter=",")

arr=DF.files[1]
np.savetxt("Uf.csv", DF[arr], delimiter=",")

arr=DF.files[2]
np.savetxt("Yp.csv", DF[arr], delimiter=",")

arr=DF.files[3]
np.savetxt("Yf.csv", DF[arr], delimiter=",")

arr=DF.files[4]
np.savetxt("Uini.csv", DF[arr], delimiter=",")

arr=DF.files[5]
np.savetxt("Yini.csv", DF[arr], delimiter=",")