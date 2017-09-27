# machine-learning-for-finance
Using Supervised machine learning methods (Decision Tree, Boosting, KNN, ANN, SVM) to trade stocks

I try to predict and trade stock prices using machine learning method. Recent years, famous algorithm trading systems are all using Black-Scholes-Merton methods and Monte-Carlo simulation methods to predict stock prices. The problem is everyone using these methods so that the marginal profits are close to zero. New methods and algorithms are desperately needed to gain advantages. Supervised learning methods are simple, straightforward and powerful. Naturally we could use them to train the historical stock prices and test the future stock prices.

In this project, I use five famous supervised learning methods to train and test AAPL stock. The results are very promising.

In order to run this project, one simply runs the following commands

#execute the simulations

$python supervised_learning.py

#measure the performance and draw the figures

$python performance.py
