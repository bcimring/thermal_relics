# Figure formatting toolkit
## Overview
The goal of the figure formatting toolkit is to supply tools that can allow you to easily create publication quality plots. This includes, offering a good colour scheme, properly formatted axes, good looking data, variable aspect ratios (including the golden ratio), variable spacing between plots, properly exporting figures to pdf, and many other features all with good defaults.
## Setup
To setup, your workstation needs to have `TQT-RAAQS.mplstyle` in the `matplotlib\mpl-data\stylelib` directory. In the case of windows, it is located here: `C:\Users\tqtraaqs\.conda\envs\tqtraaqs\Lib\site-packages\matplotlib\mpl-data\stylelib`

Once this is set up, you can start making use of the toolkit

## Workflow
You start by creating the plot layout. 
```python
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt

from experiment.toolkits.figure_formatting.formatter import *

# using TQT-RAAQS.mplystyle style sheet
with plt.style.context(['TQT-RAAQS']): # alternatively, this could just point to TQT-RAAQS.mplstyle in this directory
    # setting up plots
    fig = plt.figure()
    gs = GridSpec(3, 2, figure=fig)
    fig.add_subplot(gs[0, :])
    fig.add_subplot(gs[1, 0])
    fig.add_subplot(gs[1, 1])
    fig.add_subplot(gs[2, 0])
    fig.add_subplot(gs[2, 1])
    ax = fig.axes # list of axes
```

Then for each axes in `ax`, the following steps would be performed to properly format each one, all inside the `with` statement.
```python
ax1 = ax[0]

# setting default colour/marker palette
set_palette(ax1, '2', plot_type='scatter') 

# putting data in plot (making the data look good)
plot_scatter(ax1, x, y, xerry, yerr, series_params, fit_params)

# formatting plot (labels, limits etc)
format_plot(ax1, plot_params)
```
Lastly, the entire figure will be formatted so that it can be exported to a pdf.
```python
# formatting figure so that the arrangement is good
format_figure(gs, axes_param)
plt.savefig('figure.pdf')
```

## Future Development
If you want additional features to be added, or defaults to be changed, you can do so in the following ways
* Change default values stored in `TQT-RAAQS.mplstyle`. It will be required that each user then update their style sheet, which wouldn't be done with git, so this would require some work.
* Adding more colour palettes to `palette.py`, and adjust `set_palette` accordingly. 
* Adding new plots. Each of these start with `plot_`. So if there is a new type of plot which cannot be done with the toolkit without a lot of changes to defaults, a new plot should be made.
* Change defaults in `format_figure`. When the time comes that a publication is made and the current defaults are not suitable, they should be changed.
* Add more defaults for different journals. For example, if two different journals require two different defaults, the corresponding value for a specified journal should be the default.