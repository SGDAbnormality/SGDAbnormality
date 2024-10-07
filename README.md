# A Large Deviation Theory Analysis on the Implicit Bias of SGD


Stochastic Gradient Descent (SGD) plays a key role in training deep learning models, yet its ability to implicitly regularize and enhance generalization remains an open theoretical question. We apply Large Deviation Theory (LDT) to analyze why SGD selects models with strong generalization properties. We show that the generalization error jointly depends on the level of concentration of its empirical loss around its expected value and the \textit{abnormality} of the random deviations stemming from the stochastic nature of the training data observation process. Our analysis reveals that SGD gradients are inherently biased toward models exhibiting more concentrated losses and less abnormal and smaller random deviations. These theoretical insights are empirically validated using deep convolutional neural networks, confirming that mini-batch training acts as a natural regularizer by preventing convergence to models with high generalization errors. 


