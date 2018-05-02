Competition data is rather challenging, so the sooner you get yourself familiar with it - the better. You can start with submitting sample_submission.csv from "Data" page on Kaggle and try submitting different constants.

A good exercise is to reproduce previous_value_benchmark. As the name suggest - in this benchmark for the each shop/item pair our predictions are just monthly sales from the previous month, i.e. October 2015.

The most important step at reproducing this score is correctly aggregating daily data and constructing monthly sales data frame. You need to get lagged values, fill NaNs with zeros and clip the values into [0,20] range. If you do it correctly, you'll get precisely 1.16777 on the public leaderboard.

Generating features like this is a necessary basis for more complex models. Also, if you decide to fit some model, don't forget to clip the target into [0,20] range, it makes a big difference.

You can get a rather good score after creating some lag-based features like in advice from previous week and feeding them into gradient boosted trees model.

Apart from item/shop pair lags you can try adding lagged values of total shop or total item sales (which are essentially mean-encodings). All of that is going to add some new information.

If you successfully made use of previous advises, it's time to move forward and incorporate some new knowledge from week 4. Here are several things you can do:

1. Try to carefully tune hyper parameters of your models, maybe there is a better set of parameters for your model out there. But don't spend too much time on it.

2. Try ensembling. Start with simple averaging of linear model and gradient boosted trees like in programming assignment notebook. And then try to use stacking.

3. Explore new features! There is a lot of useful information in the data: text descriptions, item categories, seasonal trends.
