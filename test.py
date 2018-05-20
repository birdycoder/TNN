import numpy as np

def load_data(a):
	if a == 1:
		train = np.loadtxt('PA-B-train_01.dat')
		input = train[:,:2]
		output = train[:,2:]
		result = np.array([input,output]).transpose()
		return result
		
	if a == 2:
		train = np.loadtxt('PA-B-train_02.dat')
		input = train[:,:2]
		output = train[:,2:]
		result = np.array([input,output]).transpose()
		return result
	if a == 3:
		train = np.loadtxt('PA-B-train_03.dat')
		input = train[:,:4]
		output = train[:,4:]
		result = np.array([input,output]).transpose()
		return result

	if a == 4:
		train = np.loadtxt('PA-B-train_04.dat')
		input = train[:,:2]
		output = train[:,2:]
		result = np.array([input,output]).transpose()
		return result