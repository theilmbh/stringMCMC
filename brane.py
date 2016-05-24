import numpy as np
import scipy as sp
import numpy.linalg as la

def logQ(x, mean, precOvr2, logsqrtprecovr2pi):
	xRel = x - mean
	return logsqrtprecovr2pi - (xRel**2)*precOvr2

class MHString:

	def __init__(self, D, n_nodes, p_join, beta, alpha0, logtarget_distrib):
		self.n_nodes = n_nodes
		self.D = D
		self.X = np.zeros((D, n_nodes))
		self.p_join = p_join
		self.beta = beta 
		self.getNewAdjMatrix()
		self.alpha0 = alpha0
		self.logtarget_distrib = logtarget_distrib

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
		A = 0.5*np.identity(self.n_nodes)
		for agent in range(self.n_nodes-1):
			A[agent, agent+1] = 1.0*np.greater(self.p_join, np.random.rand())
		A = (A + np.transpose(A))
		self.Adj = np.dot(S, np.dot(A, la.inv(S)))
		return self.Adj

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
			preMeanSum = alpha*currentD
			preMeanSum = preMeanSum + np.sum(beta*neighborValues[d, :])
			preMean = preMeanSum * mean_norm

			proposalValue = sigma*np.random.randn() + preMean 
			proposal[d] = proposalValue

			postMeanSum = alpha*proposalValue 
			postMeanSum = postMeanSum + np.sum(beta*neighborValues[d, :])
			postMean = postMeanSum*mean_norm
			propReverseE = propReverseE + -1.0*logQ(currentD, postMean, precOvr2, logsqrtprecovr2pi)
			propForwardE = propForwardE + -1.0*logQ(proposalValue, preMean, precOvr2, logsqrtprecovr2pi)

		return (proposal, current, propForwardE, propReverseE)

	def getTargetRatio(self, proposal, current):
		return np.exp(self.logtarget_distrib(proposal) - self.logtarget_distrib(current))

	def MHAccept(self, proposal, current, propReverseE, propForwardE):

		target_ratio = self.getTargetRatio(proposal, current)
		a_rat = target_ratio*np.exp(propReverseE - propForwardE)
		a_prob = np.minimum(1, a_rat)
		accept = False
		if np.random.rand() < a_prob:
			accept = True
		return accept

	def MHUpdate(self, updateIndex):

		proposal, current, pFe, pRe = self.getProposal(updateIndex)
		if self.MHAccept(proposal, current, pRe, pFe):
			self.X[:, updateIndex] = proposal
		self.getNewAdjMatrix()

	def setup_sampling(self, T, nskip):
		self.samples = np.zeros((T, self.D, self.n_nodes))
		self.nskip = nskip
		self.T = T

	def sample(self):

		samples_to_go = self.T
		iteration = 0
		nsamples = 0
		while nsamples < self.T:
			iteration = iteration+1
			for ind in range(self.n_nodes):
				self.MHUpdate(ind)
			if np.mod(iteration, nskip)==0:
				self.samples[nsamples, :] = self.X
				nsamples = nsamples+1
		return self.samples