import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json
import os

"""
Using the plotter:
Call it from the command line, and supply it with logdirs to experiments.
Suppose you ran an experiment with name 'test', and you ran 'test' for 10
random seeds. The runner code stored it in the directory structure
    data
    L test_EnvName_DateTime
      L  0
        L log.txt
        L params.json
      L  1
        L log.txt
        L params.json
       .
       .
       .
      L  9
        L log.txt
        L params.json
To plot learning curves from the experiment, averaged over all random
seeds, call
    python plot.py data/test_EnvName_DateTime --value AverageReturn
and voila. To see a different statistics, change what you put in for
the keyword --value. You can also enter /multiple/ values, and it will
make all of them in order.
Suppose you ran two experiments: 'test1' and 'test2'. In 'test2' you tried
a different set of hyperparameters from 'test1', and now you would like
to compare them -- see their learning curves side-by-side. Just call
    python plot.py data/test1 data/test2
and it will plot them both! They will be given titles in the legend according
to their exp_name parameters. If you want to use custom legend titles, use
the --legend flag and then provide a title for each logdir.
"""

def plot_data(data, value="AverageReturn"):
    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)
    # rowdicts = []
    # for l, d in data.groupby("Iteration vel_diffs_mean vel_diffs_std min_dists_mean min_dists_std Train_Return policy_loss_sum n_clusters avgAA avgAB area vel_mean Unit Condition".split()):
    #     d = {"Iteration":l[0], "vel_diffs_mean":l[1], "vel_diffs_std":l[2], "min_dists_mean":l[3], "min_dists_std":l[4], "Train_Return":l[5], "policy_loss_sum":l[6], "n_clusters":l[7], "avgAA":l[8] ,"avgAB":l[9], "area":l[10], "vel_mean":l[11], "Unit":l[12], "Condition": l[13]}
    #     rowdicts.append(d)
    #
    # data = pd.DataFrame.from_dict(rowdicts)
    # data=data.groupby(level=0).mean()
    print("id:", data.index.duplicated())
    print("idc:", data.columns.is_unique)
    print("id:", len(data.index.duplicated()))

    # for i in range(len(data.index.duplicated())):
    #     print(data.index.duplicated()[i])

    sns.set(style="darkgrid", font_scale=1.5)
    print(data)
    # sns.lineplot(data=data, x="Iteration", y="area" )
    g = sns.tsplot(data=data, time="Iteration", value=value, unit="Unit", condition="Condition")
    # g.set_xscale("log")
    plt.legend(loc='best').set_draggable(True)
    plt.show()


def get_datasets(fpath, condition=None):
    unit = 0
    datasets = []
    for root, dir, files in os.walk(fpath):
        if 'progress.csv' in files:
            param_path = open(os.path.join(root,'params.json'))
            params = json.load(param_path)
            exp_name = params['exp_name']

            # exp_name = "Segregation"

            log_path = os.path.join(root,'progress.csv')
            experiment_data = pd.read_table(log_path)

            experiment_data.insert(
                len(experiment_data.columns),
                'Unit',
                unit
                )

            experiment_data.insert(
                len(experiment_data.columns),
                'Condition',
                condition or exp_name
                )

            datasets.append(experiment_data)
            unit += 1
            # print(experiment_data)

    return datasets


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', nargs='*')
    parser.add_argument('--legend', nargs='*')
    parser.add_argument('--value', default='AverageReturn', nargs='*')
    args = parser.parse_args()

    use_legend = False
    if args.legend is not None:
        assert len(args.legend) == len(args.logdir), \
            "Must give a legend title for each set of experiments."
        use_legend = True

    data = []
    if use_legend:
        for logdir, legend_title in zip(args.logdir, args.legend):
            data += get_datasets(logdir, legend_title)
    else:
        for logdir in args.logdir:
            data += get_datasets(logdir)

    if isinstance(args.value, list):
        values = args.value
    else:
        values = [args.value]
    for value in values:
        plot_data(data, value=value)

if __name__ == "__main__":
    main()
