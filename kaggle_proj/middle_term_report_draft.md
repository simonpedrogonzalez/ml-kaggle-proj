
# First attempt

For the first attempt I used my own  ml library made for the class programming assigments, mainly as a test for how good the studied techniques will fair with a real dataset and a more complex task.

## Data transformations
-  I transformed all numberical variables with a quartile partition encoding, that is, each variable was divided into 4 categories.

## Models

The models used where:
- Single ID3 Tree with different depths:
	- I tried from decision stumps up to fully expanded, which in my case was 14 depth.
	- THe one that achieved best test error was depth 4 (best_tree).
	- I tried the trhee metrics.
- Ensembles with AdaBoost and RandomForest
	- I tried different configurations of ensembles with increasing number of learners
	- I tried using different trees in each ensemble. The trees used were
		- Adaboost: stump and best_tree depth.
		- Random forest: best_tree and fully_expanded.
From all the models, I took the one with the best performance on test data, and made the first submission.

# Second attempt

## Data exploration

# Feature distributions

I plotted the distribution of the features in the dataset.
Conclusions:

- Age: skewed to lower values, with large peaks in some specific values that may be due to the way the data was collected, or evidence of some sampling bias or estimation.

- Workclass: almost all the data is in the private sector.

- Fnlwgt: this variable is probably useless for a prediction task, as it represents the weight of the sample in the dataset. It could be used for oversampling, but it is not clear to me how to use it in a meaningful way yet.

- Education and education_num: these two variables represent the same information.

- Marital status and relationship: these two variables might have high correlation.

- Race and native country: very skewed distributions and high correlation. Also, native country has high cardinality. It might be useful to group categories or encode them in a different way.

- Capital loss and gain: very skewed distributions, with a lot of zeros. We could summarize them into capital_diff = capital_gain - capital_loss or similar.

- Target variable: very skewed, with a distribution of 3:1 in favor of the <=50k class.

- There are missing values in the dataset, represented by '?', although kaggle states there are no missing values.

- Hours per week: seems to have a peak in a full-time 40 hours per week job. It might be useful to create intervals.

# Feature correlations

I plotted the correlation matrix of the features in the dataset.

Conclusions:

- Confirmed non-usefulness of fnlwgt for prediction.

- Confirmed high correlation between education and education_num. And between native country and race, and between marital status and relationship, but not enough to discard them at first glance.





