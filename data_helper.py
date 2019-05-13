from sklearn.datasets.samples_generator import make_blobs
from pandas import DataFrame
import numpy as np

def make_seperable():
	X, y = make_blobs(n_samples=1000, centers=5, n_features=117)
	b = np.ones((X.shape[0],X.shape[1]+1))
	b[:,1:] = X
	for thing in y:
		if thing % 2 == 0:
			thing = 0
		else:
			thing = 1
	b[:,0] = y

	df = DataFrame(b)
	df.to_csv ('not_so_seperable.csv',index=False,header=False) #Don't forget to add '.csv' at the end of the path

def count_values(column,filename):
	print("reading file")
	data = np.genfromtxt(filename,delimiter=',')

	print("processing...")
	values = dict()
	for val in data.T[column]:
		if val in values.keys():
			values[val] += 1
		else:
			values[val] = 1

	for key, value in sorted(values.items(), key=lambda x: x[0]):
		print("{} : {}".format(key, value))

def pca(filename):
	data = np.genfromtxt(filename,delimiter=',')
	data = data[:,4:]
	for col in data:
		col = col-col.mean()
	covar_matrix = (1/(data.shape[0]-1))* data.T@data

	eig_vals, eig_vecs = np.linalg.eig(covar_matrix)

	vals_dict = dict()
	pos = 0
	for val in eig_vals:
		vals_dict[pos] = (val/np.sum(eig_vals))*100
		#print(val, val/np.sum(eig_vals))
		pos = pos + 1
	
	total = 0
	for key, value in sorted(vals_dict.items(), key=lambda x: x[1])[::-1]:
		total = total + value
		print("{} : {}".format(key, value))
		if total > 70:
			break


	#print(covar_matrix)

#count_values(int(input("axis?\n")),"dota2ToyTest.csv")
pca("dota2Train.csv")