import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

e2_report = pd.read_csv('decision-tree/reports/h1e2_report.csv')

e3b_report = pd.read_csv('decision-tree/reports/h1e3_report.csv')

e3c_report = pd.read_csv('decision-tree/reports/h1e3c_report.csv')

e3b_report['missing'] = 'category'
e3c_report['missing'] = 'imputed'

e3_report = pd.concat([e3b_report, e3c_report])


sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))
sns.lineplot(x='max_depth', y='train_error', hue='metric', style='missing', data=e3_report)
plt.title('Train Error vs Max Depth')
plt.ylabel('Error')
plt.xlabel('Max Depth')
plt.savefig('decision-tree/reports/h1e3_train_report.png')


sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))
sns.lineplot(x='max_depth', y='test_error', hue='metric', style='missing', data=e3_report)
plt.title('Test Error vs Max Depth')
plt.ylabel('Error')
plt.xlabel('Max Depth')
plt.savefig('decision-tree/reports/h1e3_test_report.png')


e2_report['error_type'] = 'train'
e2_report_train = e2_report.copy()
e2_report_train['error_type'] = 'test'
e2_report_train['error'] = e2_report_train['test_error']
e2_report['error'] = e2_report['train_error']
e2_report = pd.concat([e2_report, e2_report_train])


sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))
sns.lineplot(x='max_depth', y='error', hue='metric', style='error_type', data=e2_report)
plt.title('Error vs Max Depth')
plt.ylabel('Error')
plt.xlabel('Max Depth')
plt.savefig('decision-tree/reports/h1e2_report.png')