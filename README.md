# Deep Reinforcement Learning Algorithms for single and multi-agent systems

This is a Python package named "deeprlalg" developed by Oyindamola Omotuyi. It provides deep reinforcement learning algorithms for single and multi-agent systems. This README file will provide an overview of the project, instructions for installation, and usage of the package.

## Installation

To install the "deeprlalg" package, you can use pip. Open your terminal or command prompt and run the following command:

```bash
pip install deeprlalg
```

This will install the required dependencies, including Gym, NumPy, and Matplotlib.

## Usage

The "deeprlalg" package contains a command-line utility for plotting learning curves from experiment data. You can use it to visualize the performance of your reinforcement learning experiments. To use the plotter, follow these steps:

1. Open your terminal or command prompt.

2. To plot learning curves for a single experiment, navigate to the directory containing the experiment data and log files (usually stored in a directory structure under "data"). You can specify the path to this directory as the argument for the plotter.

   Example for a single experiment:
   ```bash
   python plot.py /path/to/experiment_directory
   ```

3. To plot learning curves for multiple experiments and compare them, specify the paths to the directories of these experiments. The plotter will distinguish between experiments and provide a legend in the plot.

   Example for multiple experiments:
   ```bash
   python plot.py /path/to/experiment1_directory /path/to/experiment2_directory
   ```

4. By default, the plotter will use "AverageReturn" as the metric for plotting. You can change this by specifying a different metric using the `--value` flag. You can provide multiple values to plot multiple metrics.

   Example for plotting a specific metric:
   ```bash
   python plot.py /path/to/experiment_directory --value MetricName
   ```

   Example for plotting multiple metrics:
   ```bash
   python plot.py /path/to/experiment_directory --value Metric1 Metric2
   ```

5. If you want to customize the legend titles for your experiments, use the `--legend` flag and provide titles for each experiment you are plotting.

   Example with custom legend titles:
   ```bash
   python plot.py /path/to/experiment1_directory /path/to/experiment2_directory --legend "Experiment 1" "Experiment 2"
   ```

## Additional Notes

- Make sure your experiment data is structured in a way that the plotter can locate the log files and parameters. The log files should be in a directory named "progress.csv," and the parameters should be stored in "params.json."

- The plotter will create visualizations of the learning curves, and you can interact with them to analyze the performance of your reinforcement learning experiments.

- The project uses the Seaborn and Matplotlib libraries for data visualization, so make sure you have them installed.

Enjoy using the "deeprlalg" package to visualize and analyze your reinforcement learning experiments!
