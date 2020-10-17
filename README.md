# UFLDLTutorial
Unsupervised Feature Learning / Deep Learning Tutorial

- Modules are based on the Stanford UFLDL exercises (http://deeplearning.stanford.edu/tutorial/)
- General matrix structure follows the convention where rows indicate feature dimensions and columns indicate samples
- Generally, cost and gradient computations are separated in two separate functions due to external optimization library requirements
- In code documentation where algorithms are explained all formulas are vectorized (i.e. matrix operations) unless indices are used.
- Three debug levels are worth to notice:
	0: no information is given out
	1: warnings and messages are printed
	>1: in addition to 1, figures are displayed

Modules and directories
- data					: data used to test modules
- common				: common modules for data read/write and visualization
- examples				: examples illustrating networks constructed via combinations of different UL-SL methods
- Linreg				: Linear regression
- Logreg				: Logistic regression
- Softmax				: Softmax regression
- SMNN					: Supervised Multilayer Neural Network
- PCA					: Principal Component Analysis
- ICA					: Independent Component Analysis
- SoftICA				: Independent Component Analysis with soft reconstruction constraint
- SparseAutoencoder		: Sparse Autoencoder (Sigmoid and Linear)
- StackedAutoencoder	: Stacked Autoencoder
- SparseCoding			: Sparse Coding
- CNN					: Convolutional Neural Network

Dependencies
- numpy (linear algebra)
- scipy (for optimization)
- pylab/matplotlib (for visualization)
