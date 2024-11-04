import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import sympy
import string
import numbers

INCHES_PER_CM = 1 / 2.54
PHI = (1 + np.sqrt(5)) / 2


PALETTE1 = {
    'primary': ['c53a3c', '4c7ca4', '5ca05a', '8d5b94', 'd68029', 'dede54', '905b3c', 'e495bf'],
    'secondary': ['852728', '2e4b64', '376036', '503354', '96591c', '9e9e3b', '503221', 'a46b89'],
}

PALETTE2 = {
    'primary': ['FFD3AD', 'EDD2EE', 'FFF1AD', 'DAF3CE', 'F6B6C7', 'D9E3FD', '8d5b94', '8d5b94'],
    'secondary': ['E06900', 'C46BC7', 'E0BB00', '57B52C', 'DA1B4E', '648EF7', '503354', '503354'],
}

PALETTE3 = {
    'primary': ['F6B6C7', 'FFD3AD', 'FFF1AD', 'DAF3CE', 'D9E3FD', 'EDD2EE', '8d5b94', '8d5b94'],
    'secondary': ['DA1B4E', 'E06900', 'E0BB00', '57B52C', '648EF7', 'C46BC7', '503354', '503354'],
}


def set_palette(ax, palette_name, plot_type='scatter'):
    if palette_name == '1':
        palette = PALETTE1
    if palette_name == '2':
        palette = PALETTE2
    if palette_name == '3':
        palette = PALETTE3
    if plot_type == 'scatter':
        ax.set_prop_cycle(
            color=palette['secondary'],
            markerfacecolor=palette['primary'],
            markeredgecolor=palette['secondary'],
            marker=['o', 's', 'v', '>', '<', 'D', 'p', 'd'],
        )
    elif plot_type == 'histogram':
        ax.set_prop_cycle(
            color=palette['primary'], edgecolor=palette['secondary'],
        )

def get_colour(palette_name: str, colour_type: str, index: int):
    '''
    Getter for colours of any palette

    Parameters:
    -----------
    palette_name: str
    colour_type: 'primary' or 'secondary'
    index: int
        Colour index to pick from the palette ordering
    '''
    if palette_name == '1':
        return PALETTE1[colour_type][index]
    if palette_name == '2':
        return PALETTE2[colour_type][index]
    if palette_name == '3':
        return PALETTE3[colour_type][index]


def set_xlim(ax, xlimm, xlimp, eps=0.05, log=False):
    if log==True:
        dy = np.log10(xlimp/xlimm)
        xlimp_ = np.log10(xlimp) + eps*dy
        xlimm_ = np.log10(xlimm) - eps*dy
        ax.set_xlim(10**(xlimm_), 10**(xlimp_))
        ax.set_xscale('log')
    else:
        dx = xlimp-xlimm
        ax.set_xlim(xlimm - eps*dx, xlimp + eps*dx)


def set_ylim(ax, ylimm, ylimp, eps=0.05, log=False):
    if log==True:
        dy = np.log10(ylimp/ylimm)
        ylimp_ = np.log10(ylimp) + eps*dy
        ylimm_ = np.log10(ylimm) - eps*dy
        ax.set_ylim(10**(ylimm_), 10**(ylimp_))
        ax.set_yscale('log')
    else:
        dy = ylimp-ylimm
        ax.set_ylim(ylimm - eps*dy, ylimp + eps*dy)


def align_ticklabels(ax, axis='x', align=True, ticks=None, ticklabels=None, ind=0):

    if ticks is not None:
        if axis=='x':
            ax.set_xticks(ticks)
        else:
            ax.set_yticks(ticks)
    
    if ticklabels is None:
        if axis=='x':
            ticklabels = ax.get_xticklabels()
        else:
            ticklabels = ax.get_yticklabels()

    else:
        if axis=='x':
            ax.set_xticklabels(ticklabels)
        else:
            ax.set_yticklabels(ticklabels)

    if align:
        if axis == 'x':
            ticklabels = ax.get_xticklabels()
            ticklabels[ind].set_ha("left")
            ticklabels[-1-ind].set_ha("right")
        else:
            ticklabels = ax.get_yticklabels()
            ticklabels[ind].set_va("bottom")
            ticklabels[-1-ind].set_va("top")


def get_text_len(fig, txt):
    t = fig.text(1.5, 1.5, txt)
    r = 5 # fix renderer object
    bb = t.get_window_extent(renderer=r)
    return bb.width


def figure(type='one_col', rows=1, ratio=(1 + np.sqrt(5))/2, outline=False):
    figlength_one_col = 3.3858
    if type == 'one_col':
        figlength = 3.3858
    elif type == 'onepfive_col':
        figlength = 5.51
    elif type == 'two_col':
        figlength = 7.0866
    
    fig = plt.figure()
    fig.set_size_inches(figlength, rows*figlength_one_col/ratio, forward=True)

    if outline:
        ax=add_subplot(fig, 1, 1, 0, 1, 0, 1, axis_padding=[0,0,0,0], default_padding=False, outline=0.5)
        ax.set_xticks([])
        ax.set_yticks([])

    return fig


def add_subplot(fig, nrows, ncols, start_row, end_row, start_col, end_col, ylabellen=0, axis_padding=[2,2,2,2], get_axis_loc=False, default_padding=True, outline=None):
    xaxispad = axis_padding[0] + ylabellen
    yaxispad = axis_padding[1]

    if default_padding:
        xaxispad +=  plt.rcParams['axes.labelpad']
        yaxispad +=  plt.rcParams['axes.labelpad'] + plt.rcParams['font.size']
        for param in ['tick.major.pad','tick.minor.pad','tick.labelsize']:
            xaxispad += plt.rcParams['x'+param]
            yaxispad += plt.rcParams['y'+param]

    figlengthpoints = fig.get_figwidth()*72
    figheightpoints = fig.get_figheight()*72

    xplotstart = start_col/ncols
    xplotlen = (end_col - start_col)/ncols
    yplotlen = (end_row - start_row)/nrows
    yplotstart = 1 - start_row/nrows - yplotlen


    xaxisstart = xplotstart + xaxispad/figlengthpoints
    xaxislength = xplotlen - (xaxispad + axis_padding[2])/figlengthpoints
    yaxisstart = yplotstart + yaxispad/figheightpoints
    yaxislength = yplotlen - (yaxispad + axis_padding[3])/figheightpoints

    if get_axis_loc:
        return [xaxisstart, yaxisstart, xaxislength, yaxislength]
    else:
        axes_style = {}
        if outline is not None:
            axes_style['linewidth'] = outline
        ax = plt.axes((xaxisstart, yaxisstart, xaxislength, yaxislength))
        return ax


def format_plot(ax, plot_params):
    '''
    Formats axes

    Parameters:
    -----------
    plot_params: dict with fields
        xtitle: str
        ytitle: str
        ztitle: str
        xvariable: str
        xunits: str
        yvariable: str
        yunits: str
        zvariable: str
        zunits: str
        xticks: list
        xtickdecimals: int
        yticks: list
        ytickdecimals: int
        ztickdecimals: int
        xlim: (min_xlim: float, max_xlim: float)
        ylim: (min_ylim: float, max_ylim: float)
        zlim: (min_zlim: float, max_zlim: float)
        xfontsize: int
        yfontsize: int
        zfontsize: int
        xlabel_pad: float
        ylabel_pad: float
        zlabel_pad: float
        xlim_epsilon: float
        ylim_epsilon: float
        zlim_epsilon: float
        caption_label: [char, 'aps' or 'nature' or 'science']
        legend: bool
        xscale: 'linear' or 'log'
        yscale: 'linear' or 'log'
        grid: bool

    '''
    # plt.tight_layout()

    xlabel = plot_params.get('xtitle') or ''
    ylabel = plot_params.get('ytitle') or ''
    zlabel = plot_params.get('ztitle') or ''
    if 'xvariable' in plot_params.keys():
        xlabel += ' '
        xlabel += plot_params['xvariable']
    if 'xunits' in plot_params.keys():
        xlabel += ' (' + plot_params['xunits'] + ')'
    if 'yvariable' in plot_params.keys():
        ylabel += ' '
        ylabel += plot_params['yvariable']
    if 'yunits' in plot_params.keys():
        ylabel += ' (' + plot_params['yunits'] + ')'
    if 'zvariable' in plot_params.keys():
        zlabel += ' '
        zlabel += plot_params['zvariable']
    if 'zunits' in plot_params.keys():
        zlabel += r' (' + plot_params['zunits'] + ')'

    xticks = plot_params.get('xticks')
    yticks = plot_params.get('yticks')
    zticks = plot_params.get('zticks')
    xtickdecimals = plot_params.get('xtickdecimals')
    ytickdecimals = plot_params.get('ytickdecimals')
    ztickdecimals = plot_params.get('ztickdecimals')
    xlim = plot_params.get('xlim')
    ylim = plot_params.get('ylim')
    zlim = plot_params.get('zlim')
    xfontsize = plot_params.get('xfontsize')
    yfontsize = plot_params.get('yfontsize')
    zfontsize = plot_params.get('zfontsize')
    xlabel_pad = plot_params.get('xlabel_pad')
    ylabel_pad = plot_params.get('ylabel_pad')
    zlabel_pad = plot_params.get('zlabel_pad')
    xticks_pad = plot_params.get('xticks_pad')
    yticks_pad = plot_params.get('yticks_pad')
    zticks_pad = plot_params.get('zticks_pad')
    xlim_epsilon = plot_params.get('xlim_epsilon') if plot_params.get('xlim_epsilon') is not None else 0.05
    ylim_epsilon = plot_params.get('ylim_epsilon') if plot_params.get('ylim_epsilon') is not None else 0.05
    zlim_epsilon = plot_params.get('zlim_epsilon') if plot_params.get('zlim_epsilon') is not None else 0.05
    legend = plot_params.get('legend')
    xscale = plot_params.get('xscale') or 'linear'
    yscale = plot_params.get('yscale') or 'linear'
    zscale = plot_params.get('yscale') or 'linear'
    grid = plot_params.get('grid')
    plot_dim = plot_params.get('plot_dim') or 2

    axis_labels = [xlabel, ylabel, zlabel]
    ticks = [xticks, yticks, zticks]
    tickdecimals = [xtickdecimals, ytickdecimals, ztickdecimals]
    axis_limits = [xlim, ylim]
    axis_label_fontsize = [xfontsize, yfontsize, zfontsize]
    axis_label_pads = [xlabel_pad, ylabel_pad, zlabel_pad]
    axis_ticks_pads = [xticks_pad, yticks_pad, zticks_pad]
    axis_ticks_epsilon = [xlim_epsilon, ylim_epsilon, zlim_epsilon]
    tick_scale = [xscale, yscale, zscale]

    set_labels = [ax.set_xlabel, ax.set_ylabel]
    set_scales = [ax.set_xscale, ax.set_yscale]
    set_ticks_l = [ax.set_xticks, ax.set_yticks]
    get_ticks_l = [ax.get_xticks, ax.get_yticks]
    set_lims = [ax.set_xlim, ax.set_ylim]
    get_lims = [ax.get_xlim, ax.get_ylim]
    directions = ['x', 'y', 'z']
    axes = [ax.xaxis, ax.yaxis]
    if plot_dim == 3:
        if zlim is None:
            axis_limits.append(ax.get_zlim())
        set_labels.append(ax.set_zlabel)
        set_scales.append(ax.set_zscale)
        set_lims.append(ax.set_zlim)
        get_lims.append(ax.get_zlim)
        set_ticks_l.append(ax.set_zticks)
        get_ticks_l.append(ax.get_zticks)
        axes.append(ax.zaxis)

    dim = [i for i in range(1, 1 + plot_dim)]

    if grid:
        ax.grid(True, which=plt.rcParams['axes.grid.which'], axis=plt.rcParams['axes.grid.axis'])
        ax.set_axisbelow(True)

    def get_symmetric_limits(ticks, lim, epsilon):
        i = 0
        j = len(ticks) - 1
        while lim[0] > ticks[i]:
            i += 1
        while lim[1] < ticks[j]:
            j -= 1

        top_tick = ticks[j]
        bottom_tick = ticks[i]
        val_range = (lim[1] - lim[0]) / (2 * 0.05 + 1)
        max_val = (lim[1] + lim[0] + val_range) / 2
        min_val = (lim[1] + lim[0] - val_range) / 2
        alpha = max(max_val + epsilon * val_range - top_tick, bottom_tick - (min_val - epsilon * val_range))
        new_lim = (bottom_tick - alpha, top_tick + alpha)
        return ticks[i : j + 1], new_lim

    for (
        label,
        ticks,
        tickdecimals,
        old_lim,
        fontsize,
        label_pad,
        tick_pad,
        epsilon,
        scale,
        set_label,
        set_scale,
        set_ticks,
        get_ticks,
        set_lim,
        get_lim,
        direction,
        axis,
        d,
    ) in zip(
        axis_labels,
        ticks,
        tickdecimals,
        axis_limits,
        axis_label_fontsize,
        axis_label_pads,
        axis_ticks_pads,
        axis_ticks_epsilon,
        tick_scale,
        set_labels,
        set_scales,
        set_ticks_l,
        get_ticks_l,
        set_lims,
        get_lims,
        directions,
        axes,
        dim,
    ):
        set_scale(scale)
        # setting limits and ticks
        if ticks is not None:
            set_ticks(ticks)
        else:
            ticks = get_ticks()
            # print(ticks)

        if old_lim != 'auto':
            if old_lim is not None:
                lower_lim = old_lim[0]
                upper_lim = old_lim[1]
                set_lim(lower_lim - epsilon * (upper_lim - lower_lim), upper_lim + epsilon * (upper_lim - lower_lim))
            else:
                new_ticks, new_lim = get_symmetric_limits(ticks, get_lim(), epsilon)
                set_ticks(new_ticks)
                set_lim(new_lim)
                # print(get_ticks())
        # print()

        # tick formatting
        set_label(label, fontsize=fontsize, labelpad=label_pad)
        if tick_pad is not None:
            ax.tick_params(axis=direction, pad=tick_pad)
        ax.tick_params(axis=direction, labelsize=fontsize)

        # ticker formatting
        if scale == 'linear':
            if not tickdecimals:
                tickdecimals = max(
                    [
                        abs(decimal.Decimal(str(round(tick, 5)).rstrip('0')).as_tuple().exponent) if tick != 0 else 0
                        for tick in ticks
                    ]
                )
            axis.set_major_formatter(ticker.FormatStrFormatter('%.' + str(tickdecimals) + 'f'))
            axis.offsetText.set_visible(True)

    if plot_dim == 2:
        if legend:
            ax.legend()


def format_figure(gs, axes_params={}):
    '''
    example:
    --------
    PHI = (1+np.sqrt(5))/2
    fig = plt.figure()
    gs = GridSpec(3, 2, figure=fig)
    fig.add_subplot(gs[0, :])
    fig.add_subplot(gs[1, 0])
    fig.add_subplot(gs[1, 1])
    fig.add_subplot(gs[2, 0])
    fig.add_subplot(gs[2, 1])
    format_figure(gs, axes_params={'aspect_ratio':[1.3, PHI, 1, PHI, 1]})
    # fig.axes gets you the list of axes

    Parameters:
    -----------
    gs: GridSpec
    axes_params:
        columns: 1 or 2 (changes the default figure_width)
        figure_width: float
            (default 8.6 cm)
        right_spacing: float
            (default 0.1 cm)
        left_spacing: float
            (default 1.15 cm)
        top_spacing: float
            (default 0.1 cm)
        bottom_spacing: float
            (default 0.8 cm)
        h_plot_dist: float
            (default 0.7 cm)
        v_plot_dist: float
            (default 1.5 cm)
        aspect_ratio: float or 'equal' or (list of float or 'equal')
            (default PHI)
        caption: 'aps' or 'nature' or 'science
        caption_x: float
            (default -0.52 cm)
        caption_y: float
            (default -0.1 cm)
    '''
    columns = axes_params.get('columns', 1)
    figure_width = axes_params.get('figure_width')
    right_spacing = axes_params.get('right_spacing')
    left_spacing = axes_params.get('left_spacing')
    top_spacing = axes_params.get('top_spacing')
    bottom_spacing = axes_params.get('bottom_spacing')
    h_plot_dist = axes_params.get('h_plot_dist')
    v_plot_dist = axes_params.get('v_plot_dist')
    aspect_ratio = axes_params.get('aspect_ratio')
    caption = axes_params.get('caption')
    caption_x = axes_params.get('caption_x')
    caption_y = axes_params.get('caption_y')

    # defaults
    if figure_width is None:
        if columns == 2:
            figure_width = 17.6
        else:
            figure_width = 8.6
    if right_spacing is None:
        right_spacing = 0.1
    if left_spacing is None:
        left_spacing = 1.15
    if top_spacing is None:
        top_spacing = 0.1
    if bottom_spacing is None:
        bottom_spacing = 0.8
    if h_plot_dist is None:
        h_plot_dist = 0.7
    if v_plot_dist is None:
        v_plot_dist = 1.5
    if aspect_ratio is None:
        aspect_ratio = PHI
    if caption_x is None:
        caption_x = -0.52
    if caption_y is None:
        caption_y = -0.1

    fig = gs.figure
    ax = [ax for ax in fig.axes if hasattr(ax, 'get_subplotspec')]
    nrows, ncols = gs.get_geometry()
    if isinstance(aspect_ratio, numbers.Number) or aspect_ratio == 'equal':
        aspect_ratio = [aspect_ratio] * len(ax)

    # setting the aspect ratios for each plot
    axes_labels = np.zeros((nrows, ncols), dtype=int)
    height_ratios = [-1] * nrows
    width_ratios = [-1] * ncols
    for i in range(len(ax)):
        if hasattr(ax[i], 'get_xlim3d'):
            # ax[i].set_box_aspect((4, 4, 3))
            pass
        else:
            axes_ratio = ax[i].get_data_ratio()
            if isinstance(aspect_ratio[i], str):
                ax[i].set_aspect(aspect_ratio[i])
                aspect_ratio[i] = 1 / axes_ratio
            else:
                ax[i].set_aspect(1 / (aspect_ratio[i] * axes_ratio))
        colspan = ax[i].get_subplotspec().colspan
        rowspan = ax[i].get_subplotspec().rowspan
        axes_labels[rowspan.start : rowspan.stop, colspan.start : colspan.stop] = i

    aspect_ratio = np.array(aspect_ratio, dtype=float)

    # making system of equations to solve for height and width ratios
    # requiring that plots on first row add up to the figure width, and similarly for figure height (mostly)
    A = [np.zeros(len(ax) + 1), np.zeros(len(ax) + 1)]
    b = []

    indices = np.array(list(set(axes_labels[0])))
    A[0][indices] = aspect_ratio[indices]
    b.append(figure_width - left_spacing - right_spacing - (len(indices) - 1) * h_plot_dist)

    indices = np.array(list(set(axes_labels[:, 0])))
    A[1][indices] = 1
    A[1][-1] = -1
    b.append(-(len(indices) - 1) * v_plot_dist)

    # finding plot alignment equations
    for i in range(len(ax)):
        colspan = ax[i].get_subplotspec().colspan
        rowspan = ax[i].get_subplotspec().rowspan
        # check on right
        if (colspan.stop != ncols) and (
            (
                rowspan.start == 0
                or axes_labels[rowspan.start - 1, colspan.stop] != axes_labels[rowspan.start, colspan.stop]
            )
            and (
                rowspan.stop == nrows
                or axes_labels[rowspan.stop, colspan.stop] != axes_labels[rowspan.stop - 1, colspan.stop]
            )
        ):
            indices = np.array(list(set(axes_labels[rowspan.start : rowspan.stop, colspan.stop])))
            eqn = np.zeros(len(ax) + 1)
            eqn[indices] = 1
            eqn[i] = -1
            A.append(eqn)
            b.append(-(len(indices) - 1) * v_plot_dist)

        # check on left
        if (colspan.start != 0) and (
            (
                rowspan.start == 0
                or axes_labels[rowspan.start - 1, colspan.start - 1] != axes_labels[rowspan.start, colspan.start - 1]
            )
            and (
                rowspan.stop == nrows
                or axes_labels[rowspan.stop, colspan.start - 1] != axes_labels[rowspan.stop - 1, colspan.start - 1]
            )
        ):
            indices = np.array(list(set(axes_labels[rowspan.start : rowspan.stop, colspan.start - 1])))
            eqn = np.zeros(len(ax) + 1)
            eqn[indices] = 1
            eqn[i] = -1
            A.append(eqn)
            b.append(-(len(indices) - 1) * v_plot_dist)

        # check on bottom
        if (rowspan.stop != nrows) and (
            (
                colspan.start == 0
                or axes_labels[rowspan.stop, colspan.start - 1] != axes_labels[rowspan.stop, colspan.start]
            )
            and (
                colspan.stop == ncols
                or axes_labels[rowspan.stop, colspan.stop] != axes_labels[rowspan.stop, colspan.stop - 1]
            )
        ):
            indices = np.array(list(set(axes_labels[rowspan.stop, colspan.start : colspan.stop])))
            eqn = np.zeros(len(ax) + 1)
            eqn[indices] = aspect_ratio[indices]
            eqn[i] = -aspect_ratio[i]
            A.append(eqn)
            b.append(-(len(indices) - 1) * h_plot_dist)

        # check on top
        if (rowspan.start != 0) and (
            (
                colspan.start == 0
                or axes_labels[rowspan.start - 1, colspan.start - 1] != axes_labels[rowspan.start - 1, colspan.start]
            )
            and (
                colspan.stop == ncols
                or axes_labels[rowspan.start - 1, colspan.stop] != axes_labels[rowspan.start - 1, colspan.stop - 1]
            )
        ):
            indices = np.array(list(set(axes_labels[rowspan.start - 1, colspan.start : colspan.stop])))
            eqn = np.zeros(len(ax) + 1)
            eqn[indices] = aspect_ratio[indices]
            eqn[i] = -aspect_ratio[i]
            A.append(eqn)
            b.append(-(len(indices) - 1) * h_plot_dist)

    A = np.array(A)
    b = np.array(b)
    if b[0] <= 0:
        raise Exception('Inputted horizontal spacing is too large, please reduce them.')

    inds = np.array(sympy.Matrix(A).T.rref()[1])
    A = A[inds]
    b = b[inds]
    x = np.linalg.solve(A, b)

    plot_heights = x[: len(ax)]
    plot_widths = aspect_ratio * plot_heights
    figure_height = x[-1] + top_spacing + bottom_spacing

    height_ratios = np.zeros(nrows)
    width_ratios = np.zeros(ncols)
    # convert plot_heights to height_ratios
    for i in range(len(ax)):
        colspan = ax[i].get_subplotspec().colspan
        rowspan = ax[i].get_subplotspec().rowspan
        if colspan.stop - colspan.start == 1:
            width_ratios[colspan.start] = plot_widths[i]
        if rowspan.stop - rowspan.start == 1:
            height_ratios[rowspan.start] = plot_heights[i]

    # setting height and width ratios to the ones calculated
    gs.set_height_ratios(height_ratios)
    gs.set_width_ratios(width_ratios)

    # setting figure size
    fig.set_size_inches(figure_width * INCHES_PER_CM, figure_height * INCHES_PER_CM)

    # adjusting spacing
    av_axes_width = np.mean(plot_widths)
    av_axes_height = np.mean(plot_heights)
    plt.subplots_adjust(
        left=left_spacing / figure_width,
        bottom=bottom_spacing / figure_height,
        top=1 - top_spacing / figure_height,
        right=1 - right_spacing / figure_width,
        wspace=h_plot_dist / av_axes_width,
        hspace=v_plot_dist / av_axes_height,
    )

    if caption is not None:
        alphabet = string.ascii_lowercase
        for i, ax_ in enumerate(ax):
            if isinstance(caption_y, list):
                caption_y_val = caption_y[i]
            else:
                caption_y_val = caption_y

            if isinstance(caption_x, list):
                caption_x_val = caption_x[i]
            else:
                caption_x_val = caption_x

            x0 = ax_.get_position().x0 * figure_width * INCHES_PER_CM
            y0 = ax_.get_position().y1 * figure_height * INCHES_PER_CM
            text = plt.text(
                x0 + caption_x_val * INCHES_PER_CM,
                y0 + caption_y_val * INCHES_PER_CM,
                '',
                transform=fig.dpi_scale_trans,
                horizontalalignment='center',
                verticalalignment='center',
            )

            if caption is not None:
                if caption == 'aps':
                    text.set_text(f'({alphabet[i].lower()})')
                elif caption == 'nature':
                    text.set_fontweight('bold')
                    text.set_text(f'{alphabet[i].lower()}.')
                elif caption == 'science':
                    text.set_fontweight('bold')
                    text.set_text(f'{alphabet[i].upper()}')

