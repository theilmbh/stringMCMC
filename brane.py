import numpy
import scipy

def logQ(x, mean, precOvr2, logsqrtprecovr2pi):
	xRel = x - mean
	return logsqrtprecovr2pi - (xRel**2)*precOvr2

class MHString:

	def __init__(self, D, n_nodes, p_join, beta):
		self.n_nodes = n_nodes
		self.D = D
		self.X = np.zeros((D, n_nodes))
		self.p_join = p_join
		self.beta = beta 
		self.getNewAdjMatrix()

	def getNeighborCount(self, elementIndex):

	def getNumNodes(self):
		return self.n_nodes

	def getNeighbors(self, elementIndex):
		mat_row = self.Adj[elementIndex, :]
		mat_row[elementIndex] = 0
		return np.greater(mat_row, 0.5)

	def getNeighborCount(self, elementIndex):
		return np.sum(self.Adj[elementIndex, :]) - 1

	def getAdjMatrix(self):
		return self.Adj

	def getNewAdjMatrix(self):
		S = np.identity(self.n_nodes)
		S = np.random.permutation(S)

		A = np.random.rand(self.n_nodes, self.n_nodes)
		A = np.greater(self.p_join, A)
		A = (A + np.transpose(A))/2
		A = A - np.diagflat(np.diagonal(A)) + np.diagflat(np.ones(self.n_nodes))
		self.Adj = A
		return S*(A*S.I)

	def getProposal(self, updateIndex):

		neighborCount = self.getNeighborCount(updateIndex)
		neighbors = self.getNeighbors(updateIndex)
		beta = self.beta
		alpha = ((2-neighborCount)*beta + self.alpha0)
		precOvr2 = alpha+beta*neighborCount
		mean_norm = 1.0/precOvr2
		prec = 2.0*precOvr2
		sigma = 1.0/np.sqrt(prec)
		logsqrtprecovr2pi = 0.5*np.log(prec) - np.log(np.sqrt(2*np.pi))

		current = self.X[:, updateIndex]
		neighborValues = np.zeros((self.D, neighborCount))
		neighborValues = self.X[:, neighbors]

		propReverseE = 0
		propForwardE = 0
		proposal = np.zeros((self.D, 1))
		for d in range(self.D):
			currentD = current[d]
			preMeanSum = alhpa*currentD
			preMeanSum = preMeanSum + np.sum(beta*neighborValues[d, :])
			preMean = preMeanSum * mean_norm

			proposalValue = sigma*np.random.randn() + preMean 
			proposal[d] = proposalValue

			postMeanSum = alpha*proposalValue 
			postMeanSum = postMeanSum + np.sum(beta*neighborValues[d, :])
			postMean = postMeanSum*mean_norm
			propReverseE = propReverseE + -1.0*logQ(currentD, postMean, precOvr2, logsqrtprecovr2pi)
			propForwardE = propForwardE + -1.0*logQ(proposalValue, preMean, precOvr2, logsqrtprecovr2pi)

		self.MHUpdate(proposal, propReverseE, propForwardE)
