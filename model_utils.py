from params import *
import numpy as np

def is_absorbing(coords, model_num, **kwargs):
	if model_num == 0:
		return coords[0] > R*np.cos(kwargs['theta'])
	elif model_num == 1:
		return coords[0] > A*np.cos(kwargs['theta'])
	elif model_num == 2:
		return coords[1] > B*np.cos(kwargs['theta'])
	#	f = ((coords - np.array([0, B, 0]))**2).sum() < B**2 * np.cos(kwargs['theta'])**2
	#	return f and coords[1] > 0
	elif model_num == 3:
		spots = kwargs['spots']
		for i in range(spots.shape[0]):
			if ((spots[i, :] - coords)**2).sum() < s**2:
				return True
		return False
	elif model_num == 4 or model_num == 5:
		return True
	else:
		raise 'Invalid model number'

def is_in_cluster(coords, model_num, **kwargs):
	if model_num == 0:
		return coords[0] > R*np.cos(kwargs['theta'])
	elif model_num == 1:
		return coords[0] > A*np.cos(kwargs['theta'])
	elif model_num == 2:
		return coords[1] > B*np.cos(kwargs['theta'])
	#	f = ((coords - np.array([0, B, 0]))**2).sum() < B**2 * np.cos(kwargs['theta'])**2
	#	return f and coords[1] > 0
	elif model_num == 3:
		spots = kwargs['spots']
		for i in range(spots.shape[0]):
			if ((spots[i, :] - coords)**2).sum() < s**2:
				return True
		return False
	elif model_num == 4:
		return coords[0] > A*np.cos(kwargs['theta'])
	elif model_num == 5:
		return coords[1] > B*np.cos(kwargs['theta'])
	else:
		raise 'Invalid model number'

generator = np.random.default_rng()
def gen_spots(num_of_spots):
	th = np.linspace(0, np.pi, 3000)
	prob = np.sin(th)*np.pi/6000
	prob[0] += 1 - prob.sum()
	theta = generator.choice(th, num_of_spots, p=prob)
	phi = 2*np.pi*generator.random(num_of_spots)
	return np.array([R*np.sin(theta)*np.cos(phi), R*np.sin(theta)*np.sin(phi), R*np.cos(theta)]).transpose()

