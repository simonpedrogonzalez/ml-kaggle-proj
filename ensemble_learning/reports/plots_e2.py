import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def best_single_tree_of_all():
    df_single_tree = pd.read_csv('decision_tree/reports/h1e3b_report.csv')
    min_error = df_single_tree['test_error'].min()
    row = df_single_tree[df_single_tree['test_error'] == min_error]
    depth = row['max_depth'].values[0]
    metric = row['metric'].values[0]
    return min_error, depth, metric

def best_fully_expanded():
    df_single_tree_max_depth = pd.read_csv('ensemble_learning/reports/h2e2a_3rd_report.csv')
    min_error = df_single_tree_max_depth['test_error'].min()
    return min_error

bfe = best_fully_expanded()
bst = best_single_tree_of_all()

def mark_fully_expanded_single_tree(ax):
    min_error = bfe
    ax.axhline(min_error, color='black', linestyle='--', label=f'Fully expanded ID3 Test Error ({round(min_error, 3)})')

def mark_best_single_tree_of_all(ax):
    min_error, depth, metric = bst
    ax.axhline(min_error, color='red', linestyle='--', label=f'Best Single ID3 run (depth={depth}, metric={metric}) Test Error ({round(min_error, 3)})')

def plot_adaboost_errors_over_n(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(x='n', y='train_error', data=df, ax=ax, label='Train Error')
    sns.lineplot(x='n', y='test_error', data=df, ax=ax, label='Test Error')
    ax.set_xlabel('t')
    ax.set_ylabel('Error')
    ax.set_title('Train and Test Error of AdaBoost')
    mark_fully_expanded_single_tree(ax)
    mark_best_single_tree_of_all(ax)
    ax.legend()
    plt.savefig('ensemble_learning/reports/h2e2a_1st_plot.png')

def plot_single_learner_errors_over_n(df):
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(x='t', y='train_error', data=df, ax=ax, label='Train Error')
    sns.lineplot(x='t', y='test_error', data=df, ax=ax, label='Test Error')
    ax.set_xlabel('t')
    ax.set_ylabel('Error')
    ax.set_title('Unweighted Errors of Each Decision Stump')
    mark_fully_expanded_single_tree(ax)
    mark_best_single_tree_of_all(ax)
    ax.legend()
    plt.savefig('ensemble_learning/reports/h2e2a_2nd_plot.png')

def plot_weighted_single_learner_errors_over_n(df):
    fig, ax = plt.subplots(figsize=(10, 4))
    # sns.lineplot(x='t', y='test_error', data=df, ax=ax, label='Test Error', color='red', alpha=0.5)
    sns.lineplot(x='t', y='train_error', data=df, ax=ax)
    ax.set_xlabel('t')
    ax.set_ylabel('Error')
    ax.set_title('Weighted Train Errors of Each Decision Stump')
    # ax.axhline(0.5, color='black', linestyle='--', label='0.5')
    ax.legend()
    plt.savefig('ensemble_learning/reports/h2e2a_4th_plot.png')

def plot_bagged_trees_errors_over_n(df):
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(x='n', y='train_error', data=df, ax=ax, label='Train Error')
    sns.lineplot(x='n', y='test_error', data=df, ax=ax, label='Test Error')
    ax.set_xlabel('n')
    ax.set_ylabel('Error')
    ax.set_title('Train and Test Error of Bagged Trees')

    mark_fully_expanded_single_tree(ax)
    mark_best_single_tree_of_all(ax)

    ax.legend()

    plt.savefig('ensemble_learning/reports/h2e2b_plot.png')

def plot_bagged_trees_vs_adaboost_test_error(df_adaboost, df_bagged_trees):

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(x='n', y='test_error', data=df_adaboost, ax=ax, label='AdaBoost Test Error')
    sns.lineplot(x='n', y='test_error', data=df_bagged_trees, ax=ax, label='Bagged Trees Test Error')
    ax.set_xlabel('n')
    ax.set_ylabel('Error')
    ax.set_title('AdaBoost vs Bagged Trees Test Error')

    mark_fully_expanded_single_tree(ax)
    mark_best_single_tree_of_all(ax)

    ax.legend()

    plt.savefig('ensemble_learning/reports/h2e2b_vs_a_plot.png')

def plot_random_forest_errors_over_n(df):

    # make one line for each 'n_features' distinct value
    unique_n_features = df['n_features'].unique()
    colors = ['blue', 'green', 'orange']

    fig, ax = plt.subplots(figsize=(10, 10))
    for i, n_features in enumerate(unique_n_features):
        df_n_features = df[df['n_features'] == n_features]
        legend = f"Train Error |A|={n_features}"
        sns.lineplot(x='n', y='train_error', data=df_n_features, ax=ax, label=legend, color=colors[i])
        legend = f"Test Error |A|={n_features}"
        sns.lineplot(x='n', y='test_error', data=df_n_features, ax=ax, color=colors[i], linestyle='--', label=legend)

    ax.set_xlabel('t')
    ax.set_ylabel('Error')
    ax.set_title('Train and Test Error of Random Forest')

    mark_fully_expanded_single_tree(ax)
    mark_best_single_tree_of_all(ax)

    ax.legend()

    plt.savefig('ensemble_learning/reports/h2e2d_plot.png')

def plot_random_forest_vs_bagged_tree_test_error(df_bagged_trees, df_random_forest):

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(x='n', y='test_error', data=df_bagged_trees, ax=ax, label='Bagged Trees Test Error')
    sns.lineplot(x='n', y='test_error', data=df_random_forest, ax=ax, label='Random Forest Test Error')
    ax.set_xlabel('n')
    ax.set_ylabel('Error')
    ax.set_title('Bagged Trees vs Random Forest Test Error')

    mark_fully_expanded_single_tree(ax)
    mark_best_single_tree_of_all(ax)

    ax.legend()

    plt.savefig('ensemble_learning/reports/h2e2d_vs_b_plot.png')


df_adaboost = pd.read_csv('ensemble_learning/reports/h2e2a_1st_report.csv')
df_adaboost_sing_lrn = pd.read_csv('ensemble_learning/reports/h2e2a_2nd_report.csv')
df_bagged_trees = pd.read_csv('ensemble_learning/reports/h2e2b_report.csv')
df_random_forest = pd.read_csv('ensemble_learning/reports/h2e2d_report.csv')
df_adaboost_sing_lrn = pd.read_csv('ensemble_learning/reports/h2e2a_4th_report.csv')

# plot_adaboost_errors_over_n(df_adaboost)
# plot_single_learner_errors_over_n(df_adaboost_sing_lrn)
plot_weighted_single_learner_errors_over_n(df_adaboost_sing_lrn)
# plot_bagged_trees_errors_over_n(df_bagged_trees)
# plot_bagged_trees_vs_adaboost_test_error(df_adaboost, df_bagged_trees)
# plot_random_forest_errors_over_n(df_random_forest)
# plot_random_forest_vs_bagged_tree_test_error(df_bagged_trees, df_random_forest)


