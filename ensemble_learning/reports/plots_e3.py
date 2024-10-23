import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df_single_tree = pd.read_csv('ensemble_learning/reports/h2e3_single_tree_report.csv')
df_ensemble = pd.read_csv('ensemble_learning/reports/h2e3_report.csv')

def plot_train_error(df_single_tree, df_ensemble):
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.lineplot(x='n', y='train_error', data=df_ensemble, ax=ax, hue='model')
    ax.set_xlabel('t')
    ax.set_ylabel('Error')
    ax.set_title('Train Error on Credit Default Dataset')

    te = df_single_tree.iloc[0].train_error

    ax.axhline(te, color='black', linestyle='--', label=f'Single Tree Train Error ({round(te, 3)})')

    ax.legend()
    plt.savefig('ensemble_learning/reports/h2e3_train_error.png')

def plot_test_error(df_single_tree, df_ensemble):
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.lineplot(x='n', y='test_error', data=df_ensemble, ax=ax, hue='model')
    ax.set_xlabel('t')
    ax.set_ylabel('Error')
    ax.set_title('Test Error on Credit Default Dataset')

    te = df_single_tree.iloc[0].test_error
    ax.axhline(te, color='black', linestyle='--', label=f'Single Tree Test Error ({round(te, 3)})')

    ax.legend()
    plt.savefig('ensemble_learning/reports/h2e3_test_error.png')

plot_train_error(df_single_tree, df_ensemble)
plot_test_error(df_single_tree, df_ensemble)

