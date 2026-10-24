## Linear Regression With Ridge Regularization

Here, we have n data points and p predictors not including the intercept. First, I'll write the loss function for linear regression without ridge regularization.

$$

L=\sum^n_{i=1}(y-(\beta_0+\beta_1x_1+\cdots+\beta_px_p))^2

$$
Now, I'll differentiate this with respect to $\beta_0$ and set it equal to 0 because at the minimum, all partial derivatives will be equal to 0.

$$

\frac{\partial{L}}{\partial\beta_0}=-2\sum^n_{i=1}(y-(\beta_0+\beta_1x_1+\cdots+\beta_px_p))=0

$$
Now, we divide both sides by $-2$ and also n to get the equation in terms of the means for each predictor and for the response.
$$

\bar{y}-\sum^n_{i=1}\frac{\beta_0}{n}-\beta_1\bar{x_1}-\cdots-\beta_p\bar{x_p}=0

$$
Now, solve for $\beta_0$.

$$

\beta_0=\bar{y}-(\beta_1\bar{x_1}+\cdots+\beta_p\bar{x_p})

$$
Since, we now have an equation for $\beta_0$, we can write the loss function for linear regression with ridge regularization.

$$

L=\sum^n_{i=1}(y-(\beta_0+\beta_1x_1+\cdots+\beta_px_p))^2+\alpha\sum^p_{i=1}{\beta_j}^2

$$
The reason we solved for $\beta_0$ is in order to eliminate $\beta_0$ from the loss function with ridge regularization. Ridge regularization causes the magnitude of the parameters to shrink towards zero but we don't want the intercept to shrink so we eliminate it first.

$$

L=\sum^n_{i=1}((y-\bar{y})-(\beta_1(x_1-\bar{x_1})+\cdots+\beta_p(x_p-\bar{x_p})))^2+\alpha\sum^p_{i=1}{\beta_j}^2

$$
Now, let's differentiate the loss with respect to an arbitrary parameter which will be the jth one and see if the result came from some sort of matrix multiplication. Also, for the second term, the sum disappears because all of the parameters except for $\beta_j$ are constants with respect to $\beta_j$.

$$

\frac{\partial{L}}{\partial\beta_j}=-2\sum^n_{i=1}(x_j-\bar{x_j})((y-\bar{y})-(\beta_1(x_1-\bar{x_1})+\cdots+\beta_p(x_p-\bar{x_p})))+2\alpha\beta_j

$$
The first part looks like $-2$ multiplied by the jth column of $X-\bar{X}$ which is also multiplied by the vector $(y-\bar{y})-(X-\bar{X})\beta$. And then to that, we add on $2\alpha$ multiplied by $\beta$. This entire thing is then the gradient which we set equal to 0 and then solve for $\beta$.

$$

\nabla{L}=-2(X-\bar{X})^T((y-\bar{y})-(X-\bar{X})\beta)+2\alpha\beta

$$
$$

-2(X-\bar{X})^T(y-\bar{y})+2(X-\bar{X})^T(X-\bar{X})\beta+2\alpha\beta=0

$$
$$

((X-\bar{X})^T(X-\bar{X})+\alpha{I})\beta=(X-\bar{X})^T(y-\bar{y})

$$
$$

\beta=((X-\bar{X})^T(X-\bar{X})+\alpha{I})^{-1}(X-\bar{X})^T(y-\bar{y})

$$
These are the solutions for the parameters excluding the intercept as we have removed that before hand. Since, we have removed the intercept, we need to add it back in during prediction using the formula we got for it before.

$$

\beta_0=\bar{y}-\beta^T\bar{X}

$$
$$

\hat{y}=X\beta+\bar{y}-\beta^T\bar{X}

$$
And with this, we have trained the model by solving for its parameters and we can also make predictions so we are done. Notice that broadcasting does quite a lot of heavy lifting here because a lot of the matrix sizes are such that addition between them is undefined.

## Logistic Regression With Ridge Regularization

Here, we have n data points and p predictors not including the intercept. First I'll write the loss function for binary cross entropy loss with ridge regularization.

$$

z = \beta_0+\beta_1x_1+\cdots+\beta_px_p

$$
$$

L=-\frac{1}{n}\sum^n_{i=1}y\log(\sigma(z))+(1-y)\log(1-\sigma(z))+\alpha\sum^p_{i=1}{\beta_j}^2

$$
Again, we want to leave $\beta_0$ unpenalized but we can't do the same thing we did for linear regression as we cannot directly solve for $\beta_0$ here. So we'll just treat $\beta_0$ separately. We'll find the gradient of the $\beta$ vector which does not include $\beta_0$ and also the gradient for $\beta_0$ separately.

$$

\frac{\partial{L}}{\partial\beta_0}=-\frac{1}{n}\sum^n_{i=1}y-\sigma(\beta_0+\beta_1x_1+\cdots+\beta_px_p)

$$
$$

\frac{\partial{L}}{\partial\beta_j}=-\frac{1}{n}\sum^n_{i=1}x_j(y-\sigma(\beta_0+\beta_1x_1+\cdots+\beta_px_p))+2\alpha\beta_j

$$
Now, I'll write these partial derivatives in terms of vectors.

$$

\frac{\partial{L}}{\partial\beta_0}=-\frac{1}{n}1^T(y-\sigma(X\beta+\beta_0))

$$
$$

\frac{\partial{L}}{\partial\beta}=-\frac{1}{n}X^T(y-\sigma(X\beta+\beta_0))+2\alpha\beta

$$
The partial derivative with respect to $\beta_0$ is the negative of a dot product of the vector $y-\sigma(X\beta+\beta_0)$ with a vector of 1s so the final partial derivative ends up being a scalar. It also is just the negative mean of the vector $y-\sigma(X\beta+\beta_0)$. The partial derivative with respect to $\beta$ is a vector and the scalar $\beta_0$ is added using broadcasting or by filling a vector with the value of $\beta_0$.

I then update the parameters using the Adam optimizer because standard gradient descent is painfully slow. Especially when you need to also do hyperparameter tuning which you do need to do here because $\alpha$ is a hyperparameter.

