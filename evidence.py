import numpy as np
from scipy.stats import chi2
from scipy.stats import norm
import matplotlib.pyplot as plt

# read the evidence generated from GLOM
STAR_ID = 101501
file_dir = './initial_evidence/'
initial_evidence = np.zeros(7)

for i in range(7):
	initial_evidence[i] = np.loadtxt(file_dir + str(STAR_ID) + '_evidence_' + str(i+1) + '.txt')

# read the pca scores as activity indicators and calculate the likelihood of the 0 model
file_dir = "./pca/"

# Approach 1 - Chi-squared

logP = np.zeros(6)
for i in range(6):
	C = np.loadtxt(file_dir + str(STAR_ID) + '_C' + str(i+1) + '.txt')
	C_err = np.loadtxt(file_dir + str(STAR_ID) + '_err_C' + str(i+1) + '.txt')
	X = sum((C/C_err)**2) 			# X follows chi2 distribution?
	logP[i] = np.log(chi2.pdf(X, len(C)))

# Final_evidence = initial_evidence + the likelihood of models with zeros in the activity indicators
final_evidence = np.zeros(7)
for i in range(7):
	final_evidence[i] = initial_evidence[i] + sum(logP[i:])


# Approach 2 - Gaussian

logP2 = np.zeros(6)
for i in range(6):
	C = np.loadtxt(file_dir + str(STAR_ID) + '_C' + str(i+1) + '.txt')
	C_err = np.loadtxt(file_dir + str(STAR_ID) + '_err_C' + str(i+1) + '.txt')
	logP2[i] = sum(np.log(norm.pdf(0, C, C_err)))

final_evidence2 = np.zeros(7)
for i in range(7):
	final_evidence2[i] = initial_evidence[i] + sum(logP2[i:])

# Plots
num_indicator = np.arange(7)

plt.plot(num_indicator, initial_evidence, 'kd')
plt.xlabel('number of activity indicators')
plt.ylabel('initial evidence')
plt.savefig('initial_evidence.png')
plt.show()

plt.plot(num_indicator, final_evidence, 'o', label='chi2')
plt.plot(num_indicator, final_evidence2, 's', label='gaussian')
plt.legend()
plt.xlabel('number of activity indicators')
plt.ylabel('final evidence')
plt.savefig('final_evidence.png')
plt.show()

#------------------------------
# Maximum Likelihood Estimation
#------------------------------
N = 100
model_mean = np.linspace(-2, 2, N)

# Approach 1 - Chi-squared
logP = np.zeros((6,N))

for i in range(6):
	for k in range(N):
		C = np.loadtxt(file_dir + '101501_C' + str(i+1) + '.txt')
		C_err = np.loadtxt(file_dir + '101501_err_C' + str(i+1) + '.txt')
		X = sum(((C-model_mean[k])/C_err)**2)
		logP[i, k] = np.log(chi2.pdf(X, len(C)))

	plt.plot(model_mean, logP[i, :])
	plt.title('chi2_logP '+str(i+1))
	plt.xlabel('constant_model_value')
	plt.ylabel('log-likelihood')
	plt.axvline(x=np.average(C, weights=1/C_err**2))
	plt.savefig('chi2_logP'+str(i+1)+'.png')
	plt.close()
# The plots above show that the mean estimated by
# maximising the chi-squared log-likelihood function
# is consistent with the weighted mean of the data.

# Approach 2 - Gaussian
logP2 = np.zeros((6,N))

for i in range(6):
	for k in range(N):
		C = np.loadtxt(file_dir + '101501_C' + str(i+1) + '.txt')
		C_err = np.loadtxt(file_dir + '101501_err_C' + str(i+1) + '.txt')
		logP2[i, k] = sum(np.log(norm.pdf(model_mean[k], C, C_err)))
	plt.plot(model_mean, logP2[i, :])
	plt.title('gaussian_logP ' + str(i+1))
	plt.xlabel('constant_model_value')
	plt.ylabel('log-likelihood')
	plt.axvline(x=np.average(C, weights=1/C_err**2))
	plt.savefig('gaussian_logP'+str(i+1)+'.png')
	plt.close()
# The plots above show that the mean estimated by
# maximising the gaussian log-likelihood function
# is consistent with the weighted mean of the data.