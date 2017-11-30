#Plot data using scatter plot
import matplotlib.pyplot as plt

def plotdata(X, y, prediction = 0, line = False):
	#plt.figure(0)
	if line:
	    plt.plot(X,prediction)

	plt.scatter(X, y, marker = 'x')
	plt.xlabel('X')
	plt.ylabel('y')
	plt.title('Plot of X and y')

	plt.show()
