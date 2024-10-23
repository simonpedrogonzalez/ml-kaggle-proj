import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


all_df = pd.read_csv('linear_regression/reports/h2e4_report.csv')

df = all_df[all_df['model'] == 'Batch Gradient Descent']
fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(x='n_iter', y='train_error', data=df, ax=ax, label='Training Cost Function')
# sns.lineplot(x='n_iter', y='test_error', data=df, ax=ax, label='Test Cost Function')
ax.set_xlabel('n')
ax.set_ylabel('Error')
ax.set_title('Train Cost function of Batch Gradient Descent')
ax.legend()
plt.savefig('linear_regression/reports/h2e4a_plot.png')

df = all_df[all_df['model'] == 'Stochastic Gradient Descent']
fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(x='n_iter', y='train_error', data=df, ax=ax, label='Training Cost Function')
# sns.lineplot(x='n_iter', y='test_error', data=df, ax=ax, label='Test Cost Function')
ax.set_xlabel('n')
ax.set_ylabel('Error')
ax.set_title('Train Cost function of Stochastic Gradient Descent')
ax.legend()
plt.savefig('linear_regression/reports/h2e4b_plot.png')
