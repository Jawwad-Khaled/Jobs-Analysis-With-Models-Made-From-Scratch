## Linear Regression With Ridge Regularization

Here, we have n data points and p predictors not including the intercept. First, I'll write the loss function for linear regression without ridge regularization.

```math
L=\sum^n_{i=1}(y-(\beta_0+\beta_1x_1+\cdots+\beta_px_p))^2
```

Now, I'll differentiate this with respect to $\beta_0$ and set it equal to 0 because at the minimum, all partial derivatives will be equal to 0.

```math
\frac{\partial{L}}{\partial\beta_0}=-2\sum^n_{i=1}(y-(\beta_0+\beta_1x_1+\cdots+\beta_px_p))=0
```

Now, we divide both sides by $-2$ and also n to get the equation in terms of the means for each predictor and for the response.

```math
\bar{y}-\sum^n_{i=1}\frac{\beta_0}{n}-\beta_1\bar{x_1}-\cdots-\beta_p\bar{x_p}=0
```

Now, solve for $\beta_0$.

```math
\beta_0=\bar{y}-(\beta_1\bar{x_1}+\cdots+\beta_p\bar{x_p})
```

Since, we now have an equation for $\beta_0$, we can write the loss function for linear regression with ridge regularization.

```math
L=\sum^n_{i=1}(y-(\beta_0+\beta_1x_1+\cdots+\beta_px_p))^2+\alpha\sum^p_{i=1}{\beta_j}^2
```

The reason we solved for $\beta_0$ is in order to eliminate $\beta_0$ from the loss function with ridge regularization. Ridge regularization causes the magnitude of the parameters to shrink towards zero but we don't want the intercept to shrink so we eliminate it first.

```math
L=\sum^n_{i=1}((y-\bar{y})-(\beta_1(x_1-\bar{x_1})+\cdots+\beta_p(x_p-\bar{x_p})))^2+\alpha\sum^p_{i=1}{\beta_j}^2
```

Now, let's differentiate the loss with respect to an arbitrary parameter which will be the jth one and see if the result came from some sort of matrix multiplication. Also, for the second term, the sum disappears because all of the parameters except for $\beta_j$ are constants with respect to $\beta_j$.

```math
\frac{\partial{L}}{\partial\beta_j}=-2\sum^n_{i=1}(x_j-\bar{x_j})((y-\bar{y})-(\beta_1(x_1-\bar{x_1})+\cdots+\beta_p(x_p-\bar{x_p})))+2\alpha\beta_j
```

The first part looks like $-2$ multiplied by the jth column of $X-\bar{X}$ which is also multiplied by the vector $(y-\bar{y})-(X-\bar{X})\beta$. And then to that, we add on $2\alpha$ multiplied by $\beta$. This entire thing is then the gradient which we set equal to 0 and then solve for $\beta$.

```math
\nabla{L}=-2(X-\bar{X})^T((y-\bar{y})-(X-\bar{X})\beta)+2\alpha\beta
```
```math
-2(X-\bar{X})^T(y-\bar{y})+2(X-\bar{X})^T(X-\bar{X})\beta+2\alpha\beta=0
```
```math
((X-\bar{X})^T(X-\bar{X})+\alpha{I})\beta=(X-\bar{X})^T(y-\bar{y})

```
```math
\beta=((X-\bar{X})^T(X-\bar{X})+\alpha{I})^{-1}(X-\bar{X})^T(y-\bar{y})
```

These are the solutions for the parameters excluding the intercept as we have removed that before hand. Since, we have removed the intercept, we need to add it back in during prediction using the formula we got for it before.

```math
\beta_0=\bar{y}-\beta^T\bar{X}
```
```math
\hat{y}=X\beta+\bar{y}-\beta^T\bar{X}
```

And with this, we have trained the model by solving for its parameters and we can also make predictions so we are done. Notice that broadcasting does quite a lot of heavy lifting here because a lot of the matrix sizes are such that addition between them is undefined.

## Logistic Regression With Ridge Regularization

Here, we have n data points and p predictors not including the intercept. First I'll write the loss function for binary cross entropy loss with ridge regularization.

```math
z = \beta_0+\beta_1x_1+\cdots+\beta_px_p
```
```math
L=-\frac{1}{n}\sum^n_{i=1}y\log(\sigma(z))+(1-y)\log(1-\sigma(z))+\alpha\sum^p_{i=1}{\beta_j}^2
```

Again, we want to leave $\beta_0$ unpenalized but we can't do the same thing we did for linear regression as we cannot directly solve for $\beta_0$ here. So we'll just treat $\beta_0$ separately. We'll find the gradient of the $\beta$ vector which does not include $\beta_0$ and also the gradient for $\beta_0$ separately.

```math
\frac{\partial{L}}{\partial\beta_0}=-\frac{1}{n}\sum^n_{i=1}y-\sigma(\beta_0+\beta_1x_1+\cdots+\beta_px_p)
```
```math
\frac{\partial{L}}{\partial\beta_j}=-\frac{1}{n}\sum^n_{i=1}x_j(y-\sigma(\beta_0+\beta_1x_1+\cdots+\beta_px_p))+2\alpha\beta_j
```

Now, I'll write these partial derivatives in terms of vectors.

```math
\frac{\partial{L}}{\partial\beta_0}=-\frac{1}{n}1^T(y-\sigma(X\beta+\beta_0))
```
```math
\frac{\partial{L}}{\partial\beta}=-\frac{1}{n}X^T(y-\sigma(X\beta+\beta_0))+2\alpha\beta
```

The partial derivative with respect to $\beta_0$ is the negative of a dot product of the vector $y-\sigma(X\beta+\beta_0)$ with a vector of 1s so the final partial derivative ends up being a scalar. It also is just the negative mean of the vector $y-\sigma(X\beta+\beta_0)$. The partial derivative with respect to $\beta$ is a vector and the scalar $\beta_0$ is added using broadcasting or by filling a vector with the value of $\beta_0$.

I then update the parameters using the Adam optimizer because standard gradient descent is painfully slow. Especially when you need to also do hyperparameter tuning which you do need to do here because $\alpha$ is a hyperparameter.

## Neural Networks

First, I will consider a simple neural network with two hidden layers, p inputs, and with a hidden size h in each hidden layer. The hidden size will be the same for each hidden layer so that hyperparameter tuning can be done more easily. There will be 3 activation vectors which are written below.

```math
A_1=A(XW_1+B_1)
```
```math
A_2=A(A_1W_2+B_2)
```
```math
A_3=A(A_2W_3+B_3)
```

First, I will write down the sizes for each matrix. Note that $A_3$ is the same as the predicted values $\hat{y}$. Every row for the bias vectors are identical so there only really is one unique row. Technically, you could just write all of the shapes of the bias vectors with 1 row and then broadcasting would take care of the rest.
- $X$ -> n x p | $W_1$ -> p x h | $B_1$ -> n x h
- $A_1$ -> n x h | $W_2$ -> h x h | $B_2$ -> n x h
- $A_2$ -> n x h | $W_3$ -> h x 1 | $B_3$ -> n x 1
- $A_3$ -> n x 1

Now, let's assume a general loss function $L$. We will use mean squared error loss for regression and binary cross entropy loss for classification. To calculate the gradients we will use the chain rule.

```math
\frac{\partial{L}}{\partial{B_3}}=\frac{\partial{L}}{\partial{A_3}}\frac{\partial{A_3}}{\partial{B_3}}
```
```math
\frac{\partial{L}}{\partial{W_3}}=\frac{\partial{L}}{\partial{A_3}}\frac{\partial{A_3}}{\partial{W_3}}
```

To calculate these derivatives we will use standard differentiation rules and matching shapes. Here, we will run into elementwise multiplications as well as matrix multiplications. Elementwise multiplications are always used between the derivative of an activation function and a matrix with the same size. Also, the partial derivative of something must have the same shape as that something. With this in mind, the partial derivatives above are the following.

```math
\frac{\partial{L}}{\partial{B_3}}=\frac{\partial{L}}{\partial{A_3}}\odot A^\prime(A_2W_3+B_3)
```
```math
\frac{\partial{L}}{\partial{W_3}}={A_2}^T\left(\frac{\partial{L}}{\partial{A_3}}\odot A^\prime(A_2W_3+B_3)\right)
```

Now, we'll calculate the partial derivatives for $B_2$ and $W_2$.

```math
\frac{\partial{L}}{\partial{B_2}}=\frac{\partial{L}}{\partial{A_3}}\frac{\partial{A_3}}{\partial{A_2}}\frac{\partial{A_2}}{\partial{B_2}}
```
```math
\frac{\partial{L}}{\partial{W_2}}=\frac{\partial{L}}{\partial{A_3}}\frac{\partial{A_3}}{\partial{A_2}}\frac{\partial{A_2}}{\partial{W_2}}
```
```math
\frac{\partial{L}}{\partial{B_2}}=\left(\left(\frac{\partial{L}}{\partial{A_3}}\odot A^\prime(A_2W_3+B_3)\right){W_3}^T\right)\odot A^\prime(A_1W_2+B_2)
```
```math
\frac{\partial{L}}{\partial{W_2}}={A_1}^T\left(\left(\left(\frac{\partial{L}}{\partial{A_3}}\odot A^\prime(A_2W_3+B_3)\right){W_3}^T\right)\odot A^\prime(A_1W_2+B_2)\right)
```

Now, we'll calculate the partial derivatives for $B_1$ and $W_1$

```math
\frac{\partial{L}}{\partial{B_1}}=\frac{\partial{L}}{\partial{A_3}}\frac{\partial{A_3}}{\partial{A_2}}\frac{\partial{A_2}}{\partial{A_1}}\frac{\partial{A_1}}{\partial{B_1}}
```
```math
\frac{\partial{L}}{\partial{W_1}}=\frac{\partial{L}}{\partial{A_3}}\frac{\partial{A_3}}{\partial{A_2}}\frac{\partial{A_2}}{\partial{A_1}}\frac{\partial{A_1}}{\partial{W_1}}
```
```math
\frac{\partial{L}}{\partial{B_1}}=\left(\left(\left(\left(\frac{\partial{L}}{\partial{A_3}}\odot A^\prime(A_2W_3+B_3)\right){W_3}^T\right)\odot A^\prime(A_1W_2+B_2)\right){W_2}^T\right)\odot A^\prime(XW_1+B_1)
```
```math
\frac{\partial{L}}{\partial{W_1}}=X^T\left(\left(\left(\left(\left(\frac{\partial{L}}{\partial{A_3}}\odot A^\prime(A_2W_3+B_3)\right){W_3}^T\right)\odot A^\prime(A_1W_2+B_2)\right){W_2}^T\right)\odot A^\prime(XW_1+B_1)\right)
```

Is there some sort of pattern that can be seen that let's us expand this to any number of hidden layers? There is which is why I was able to code any number of hidden layers. First, you take the partial derivative of the gradient with respect to the bias closest to the loss. For the gradient with respect to the weight matrix closest to the loss, you multiply the bias gradient on the left by the transpose of the activation matrix you multiplied that weight matrix with.

To get to the next bias gradient, you multiply the previous bias gradient by the transpose of the previous weight matrix on the right and then with the entire matrix product so far, multiply elementwise by the derivative of the activation which takes in the current bias and weight matrix along with the activation matrix multiplying that weight matrix. Again, to get the next weight gradient, you multiply on the left by the transpose of the activation matrix you multiplied the next weight with.

With this, you have all of the gradients calculated. Just like for logistic regression, I update the parameters using the Adam optimizer. Using gradient descent here would really suck because a neural network takes so much longer to train than logistic regression.

Implementing the neural network from scratch has caused me quite some struggles so I'll mention them here. First of all, broadcasting happens automatically but you unbroadcasting for gradient calculations does not. So my bias vectors and their gradients ended up being different shapes and I had to figure out how to fix that. I originally wanted to go through 4 values of each of the 4 hyperparameters instead of 3 but I ran into a random memory error like 10 hours in. I didn't really want to deal with that so I just reduced it to the number of values for each hyperparameter to 3. It took a lot less time to train as well. I still don't fully understand the chain rule with matrix derivatives which is why I relied on shape matching which I got from the Stanford CS229 lectures. But I do plan on going through the MIT 18.S096 matrix calculus course on MIT OCW in order to understand it better.

## Decision Trees

For trees, this is the algorithm I carried. First, I looped through each feature and calculated the midpoints. Then I looped over each midpoint. The prediction for the left of the midpoint was the mean of all of the points on the left and the same for the right of the midpoint. I calculated the sum of squared errors for regression and the Gini impurity for classification. This happened for all predictors and for all midpoints within each predictor. Then, the best predictor and midpoint was chosen which is the one with the lowest sum of squared errors or impurity. 

There were some safeguards. If there array y was completely empty, then I would return that there was no valid split. If a predictor only had one unique value which also means it only has one midpoint, then that predictor is skipped. If no sum of squared errors or impurities are found, the predictor is skipped. If the list containing the best midpoint and best sum of squared error or impurity for each predictor turned out empty, then I would return that there was no valid split. Also, if one side of the midpoint was empty, that midpoint would be skipped. I found out most of these through debugging.

The best predictors and midpoints are then used to recursively grow the trees. All the data points on the left side of a midpoint are then directed to one side and the same for the right. Then a new tree is grown from the newly separated data points unless there are two few data points, the tree got too big, or every data point has the same value. Then, a leaf is made where the mean of all of the data points is the prediction.

To get predictions, I iterated over each row of the data matrix as each row represents one data point. I start at the root and and the value of one of the features of each row is checked against the node's feature and threshold. If the value is less than the threshold, the row gets directed to the left branch and if its value is greater than the threshold, the row gets directed to the right branch. If a leaf is reached, the leaf's value is returned and added to the array of predictions.

## Decision Tree Ensembles

Random forests were pretty straightforward. Bootstrapped datasets were used and I added a features variable that is basically just a list of a few randomly chosen features that each tree will consider in order to decorrelate the trees. The features are reassigned every time a new split is required. The number of random features that each split sees is a hyperparameter I tuned with cross validation. Regression random forests used regression trees while classification random forests used classification trees.

Regression gradient boosting wasn't too complicated either. First, every data point gets assigned a prediction equal to the means. The residuals are calculated which can either be negative or positive. A tree is then fit to try to minimize these residuals. The predictions are multiplied by a learning rate to make the trees slow leaners and then more trees are added to fit the previous trees residuals. The number of trees is a hyperparameter I tuned with cross validation.