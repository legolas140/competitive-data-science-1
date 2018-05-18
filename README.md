# Additional Materials and Links


## Week 1

### Recap of main ML algorithms

#### Overview of methods
* [Scikit-Learn (or sklearn) library](http://scikit-learn.org/)
* [Overview of k-NN](http://scikit-learn.org/stable/modules/neighbors.html) (sklearn's documentation)
* [Overview of Linear Models](http://scikit-learn.org/stable/modules/linear_model.html) (sklearn's documentation)
* [Overview of Decision Trees](http://scikit-learn.org/stable/modules/tree.html) (sklearn's documentation)
* Overview of algorithms and parameters in [H2O documentation](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science.html)

#### Additional Tools
* [Vowpal Wabbit](https://github.com/JohnLangford/vowpal_wabbit) repository
* [XGBoost](https://github.com/dmlc/xgboost) repository
* [LightGBM](https://github.com/Microsoft/LightGBM) repository
* [Interactive demo](http://playground.tensorflow.org/) of simple feed-forward Neural Net
* Frameworks for Neural Nets: [Keras](https://keras.io/),[PyTorch](http://pytorch.org/),[TensorFlow](https://www.tensorflow.org/),[MXNet](http://mxnet.io/), [Lasagne](http://lasagne.readthedocs.io/)
* [Example from sklearn with different decision surfaces](http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html)
* [Arbitrary order factorization machines](https://github.com/geffy/tffm)

### Software/Hardware requirements

#### StandCloud Computing:
* [AWS](https://aws.amazon.com/), [Google Cloud](https://cloud.google.com/), [Microsoft Azure](https://azure.microsoft.com/)

#### AWS spot option:
* [Overview of Spot mechanism](http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-spot-instances.html)
* [Spot Setup Guide](http://www.datasciencebowl.com/aws_guide/)

#### Stack and packages:
* [Basic SciPy stack (ipython, numpy, pandas, matplotlib)](https://www.scipy.org/)
* [Jupyter Notebook](http://jupyter.org/)
* [Stand-alone python tSNE package](https://github.com/danielfrg/tsne)
* Libraries to work with sparse CTR-like data: [LibFM](http://www.libfm.org/), [LibFFM](https://www.csie.ntu.edu.tw/~cjlin/libffm/)
* Another tree-based method: RGF ([implemetation](https://github.com/baidu/fast_rgf), [paper](https://arxiv.org/pdf/1109.0887.pdf))
* Python distribution with all-included packages: [Anaconda](https://www.continuum.io/what-is-anaconda)
* [Blog "datas-frame" (contains posts about effective Pandas usage)](https://tomaugspurger.github.io/)

### Feature preprocessing and generation with respect to models

#### Feature preprocessing
* [Preprocessing in Sklearn](http://scikit-learn.org/stable/modules/preprocessing.html)
* [Andrew NG about gradient descent and feature scaling](https://www.coursera.org/learn/machine-learning/lecture/xx3Da/gradient-descent-in-practice-i-feature-scaling)
* [Feature Scaling and the effect of standardization for machine learning algorithms](http://sebastianraschka.com/Articles/2014_about_feature_scaling.html)

#### Feature generation
* [Discover Feature Engineering, How to Engineer Features and How to Get Good at It](https://machinelearningmastery.com/discover-feature-engineering-how-to-engineer-features-and-how-to-get-good-at-it/)
* [Discussion of feature engineering on Quora](https://www.quora.com/What-are-some-best-practices-in-Feature-Engineering)

### Feature extraction from text and images

#### Feature extraction from text

Bag of words
* [Feature extraction from text with Sklearn](http://scikit-learn.org/stable/modules/feature_extraction.html)
* [More examples of using Sklearn](https://andhint.github.io/machine-learning/nlp/Feature-Extraction-From-Text/)

Word2vec
* [Tutorial to Word2vec](https://www.tensorflow.org/tutorials/word2vec)
* [Tutorial to word2vec usage](https://rare-technologies.com/word2vec-tutorial/)
* [Text Classification With Word2Vec](http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/)
* [Introduction to Word Embedding Models with Word2Vec](https://taylorwhitten.github.io/blog/word2vec)

NLP Libraries
* [NLTK](http://www.nltk.org/)
* [TextBlob](https://github.com/sloria/TextBlob)

#### Feature extraction from images

Pretrained models
* [Using pretrained models in Keras](https://keras.io/applications/)
* [Image classification with a pre-trained deep neural network](https://www.kernix.com/blog/image-classification-with-a-pre-trained-deep-neural-network_p11)

Finetuning
* [How to Retrain Inception's Final Layer for New Categories in Tensorflow](https://www.tensorflow.org/tutorials/image_retraining)
* [Fine-tuning Deep Learning Models in Keras](https://flyyufelix.github.io/2016/10/08/fine-tuning-in-keras-part2.html)


## Week 2

### Exploratory data analysis

#### Visualization tools
* [Seaborn](https://seaborn.pydata.org/)
* [Plotly](https://plot.ly/python/)
* [Bokeh](https://github.com/bokeh/bokeh)
* [ggplot](http://ggplot.yhathq.com/)
* [Graph visualization with NetworkX](https://networkx.github.io/)

#### Others
* [Biclustering algorithms for sorting corrplots](http://scikit-learn.org/stable/auto_examples/bicluster/plot_spectral_biclustering.html)

### Validation

* [Validation in Sklearn](http://scikit-learn.org/stable/modules/cross_validation.html)
* [Advices on validation in a competition](http://www.chioka.in/how-to-select-your-final-models-in-a-kaggle-competitio/)

### Data leakages

* [Perfect score script by Oleg Trott](https://www.kaggle.com/olegtrott/the-perfect-score-script) -- used to probe leaderboard
* [Page about data leakages on Kaggle](https://www.kaggle.com/wiki/Leakage)


## Week 3

### Metrics optimization

#### Classification
* [Evaluation Metrics for Classification Problems: Quick Examples + References](http://queirozf.com/entries/evaluation-metrics-for-classification-quick-examples-references)
* [Decision Trees: “Gini” vs. “Entropy” criteria](https://www.garysieling.com/blog/sklearn-gini-vs-entropy-criteria)
* [Understanding ROC curves](http://www.navan.name/roc/)

#### Ranking
* [Learning to Rank using Gradient Descent](http://icml.cc/2015/wp-content/uploads/2015/06/icml_ranking.pdf) -- original paper about pairwise method for AUC optimization
* [Overview of further developments of RankNet](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf)
* [RankLib](https://sourceforge.net/p/lemur/wiki/RankLib/) (implemtations for the 2 papers from above)
* [Learning to Rank Overview](https://wellecks.wordpress.com/2015/01/15/learning-to-rank-overview)

#### Clustering
* [Evaluation metrics for clustering](http://nlp.uned.es/docs/amigo2007a.pdf)


## Week 4

### Hyperparameter tuning

* [Tuning the hyper-parameters of an estimator (sklearn)](http://scikit-learn.org/stable/modules/grid_search.html)
* [Optimizing hyperparameters with hyperopt](http://fastml.com/optimizing-hyperparams-with-hyperopt/)
* [Complete Guide to Parameter Tuning in Gradient Boosting (GBM) in Python](https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/)

### Tips and tricks

* [Far0n's framework for Kaggle competitions "kaggletils"](https://github.com/Far0n/kaggletils)
* [28 Jupyter Notebook tips, tricks and shortcuts](https://www.dataquest.io/blog/jupyter-notebook-tips-tricks-shortcuts/)

### Advanced features II

#### Matrix Factorization:
* [Overview of Matrix Decomposition methods (sklearn)](http://scikit-learn.org/stable/modules/decomposition.html)

#### t-SNE:
* [Multicore t-SNE implementation](https://github.com/DmitryUlyanov/Multicore-TSNE)
* [Comparison of Manifold Learning methods (sklearn)](http://scikit-learn.org/stable/auto_examples/manifold/plot_compare_methods.html)
* [How to Use t-SNE Effectively (distill.pub blog)](https://distill.pub/2016/misread-tsne/)
* [tSNE homepage (Laurens van der Maaten)](https://lvdmaaten.github.io/tsne/)
* [Example: tSNE with different perplexities (sklearn)](http://scikit-learn.org/stable/auto_examples/manifold/plot_t_sne_perplexity.html#sphx-glr-auto-examples-manifold-plot-t-sne-perplexity-py)

#### Interactions:
* [Facebook Research's paper about extracting categorical features from trees](https://research.fb.com/publications/practical-lessons-from-predicting-clicks-on-ads-at-facebook/)
* [Example: Feature transformations with ensembles of trees (sklearn)](http://scikit-learn.org/stable/auto_examples/ensemble/plot_feature_transformation.html)

### Ensembling

* [Kaggle ensembling guide at MLWave.com (overview of approaches)](https://mlwave.com/kaggle-ensembling-guide/)
* [StackNet — a computational, scalable and analytical meta modelling framework (by KazAnova)](https://github.com/kaz-Anova/StackNet)
* [Heamy — a set of useful tools for competitive data science (including ensembling)](https://github.com/rushter/heamy)


## Week 5
