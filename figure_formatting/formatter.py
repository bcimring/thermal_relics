#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 11:37:02 2020

@author: Joon Kim


@author: Jonathon Kambulow
@author: Barry Cimring
@author: Artem Zhutov
"""

import decimal
import numbers
import string
from typing import Optional, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import sympy
from experiment.toolkits.figure_formatting.palettes import PALETTE1, PALETTE2, PALETTE3
from experiment.utils.data_handler import DataHandler
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde

INCHES_PER_CM = 1 / 2.54
PHI = (1 + np.sqrt(5)) / 2


def get_subplots(plots: Union[np.ndarray, int, tuple] = 1, sharey: bool = False, sharex: bool = False):
    '''
    Generates a plot layout. Offers the flexibility of GridSpec with
    the ease of use of plt.subplots
    
    Used as a replacement for plt.subplots()
    as this function returns a gs which can be given to figure_formatting
    to properly customize the layout. 

    Parameters:
    -----------
    plots: (ndarray of int) or int or (tuple of int)
        ndarray of int: allows for specifying the exact geometry of the plot grid
            The numbers represent the index of the axes in the axes list
            ex.
            np.array([[0,0,1],
                      [2,3,1],
                      [2,4,1]])
        int: 
            makes plots with the specified number of rows
        tuple of int:
            makes plots in a grid with dimensions (rows, columns)
        sharex: bool
            whether for all axes should share x axes. Does not apply when an ndarray is inputted for plots
        sharey: bool

    Returns:
    --------
    fig: Figure
    ax: list of axes
    gs: GridSpec
    '''
    share = [sharex, sharey]
    share_title = ['sharex', 'sharey']

    fig = plt.figure()
    if isinstance(plots, np.ndarray):
        gs = GridSpec(*plots.shape, figure=fig)
        for i in range(np.max(plots) + 1):
            rows, cols = np.nonzero(plots == i)
            fig.add_subplot(gs[np.min(rows) : np.max(rows) + 1, np.min(cols) : np.max(cols) + 1])

    elif isinstance(plots, int):
        gs = GridSpec(plots, 1, figure=fig)
        for i in range(plots):
            kwargs = {}
            for share_, share_title_ in zip(share, share_title):
                if share_ and i != 0:
                    kwargs[share_title_] = fig.axes[0]
            fig.add_subplot(gs[i], **kwargs)

    elif isinstance(plots, tuple):
        gs = GridSpec(*plots, figure=fig)
        for i in range(np.prod(plots)):
            indices = np.unravel_index(i, plots)
            kwargs = {}
            for share_, share_title_ in zip(share, share_title):
                if share_ and i != 0:
                    kwargs[share_title_] = fig.axes[0]
            fig.add_subplot(gs[indices[0], indices[1]], **kwargs)
    else:
        raise Exception("Inputs are invalid")

    return fig, fig.axes, gs


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


def plot_distribution(ax, x, series_params={}):
    '''
    Draws scatter plot on ax object

    Parameters:
    -----------
    x: list or np.ndarray
    series_params: dict with fields
        primary: str
        secondary: str
        label: str
        range: tuple
        density: bool
        density_multiplier: float
    '''
    density_multiplier = series_params.get('density_multiplier', 1)

    # inputted parameters
    cycler = ax._get_lines.prop_cycler
    secondary = next(cycler)['edgecolor']
    if series_params.get('secondary'):
        secondary = series_params.get('secondary')

    hist, bin_edges = np.histogram(
        x,
        bins=series_params.get('bins', 10),
        density=series_params.get('density', True),
        range=series_params.get('range'),
    )
    x = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)]
    width = x[1] - x[0]

    where_nonzero = np.where(np.asarray(hist) != 0)
    bar = ax.bar(
        np.asarray(x)[where_nonzero],
        np.asarray(hist)[where_nonzero] / density_multiplier,
        label=series_params.get('label'),
        width=width,
        edgecolor=secondary,
        lw=series_params.get('edgewidth'),
    )

    if series_params.get('primary'):
        primary = series_params.get('primary')
        for patch in bar.patches:
            patch.set_facecolor(primary)
    if series_params.get('alpha'):
        for patch in bar.patches:
            patch.set_alpha(series_params.get('alpha'))

    return np.asarray(x)[where_nonzero], np.asarray(hist)[where_nonzero]


def plot_distribution_2d(ax, x, y, series_params={'plot_dim': 3}, colorbar_params={}):
    '''
    Draws scatter plot on ax object

    Parameters:
    -----------
    x: list or np.ndarray
    series_params: dict with fields
        primary: str
        secondary: str
        label: str
        range: tuple
        density_multiplier: float
        cmap:
        cmap_values:
    colorbar_params: dict with fields
        figure: Figure
        ticks: list

    '''
    density_multiplier = series_params.get('density_multiplier') or 1
    H, xedges, yedges = np.histogram2d(
        x, y, density=True, range=series_params.get('range'), bins=(series_params.get('bins') or 10)
    )
    H = H / density_multiplier

    if series_params['plot_dim'] == 3:
        xpos = []
        ypos = []
        dz = []
        for i in range(len(H)):
            for j in range(len(H[i])):
                if H[i][j] != 0:
                    xpos.append(xedges[i])
                    ypos.append(yedges[j])
                    dz.append(H[i][j])
        zpos = np.zeros(len(xpos))
        dx = (xedges[1] - xedges[0]) * np.ones(len(xpos))
        dy = (yedges[1] - yedges[0]) * np.ones(len(ypos))

        rgba = None
        if series_params.get('cmap') is not None:
            cmap = cm.get_cmap(series_params.get('cmap'))
            max_height = np.max(dz)
            min_height = np.min(dz)
            rgba = [cmap((k - min_height) / max_height) for k in dz]
            ax.bar3d(xpos, ypos, zpos, dx, dy, dz, label=series_params.get('label'), color=rgba, zsort='average')
        elif series_params.get('cmap_values') is not None:
            cmap = series_params.get('cmap_values')
            max_height = np.max(dz)
            min_height = np.min(dz)
            rgba = [cmap((k - min_height) / max_height) for k in dz]
            ax.bar3d(xpos, ypos, zpos, dx, dy, dz, label=series_params.get('label'), color=rgba, zsort='average')
        else:
            ax.bar3d(
                xpos,
                ypos,
                zpos,
                dx,
                dy,
                dz,
                label=series_params.get('label'),
                color=series_params.get('color'),
                zsort='average',
            )

    else:
        prop_cycle = plt.rcParams['axes.prop_cycle']
        cmaps = prop_cycle.by_key()['color']
        cmap = series_params.get('cmap_values')
        if series_params.get('cmap'):
            cmap = plt.get_cmap(series_params.get('cmap'))

        xy = np.vstack([x, y])
        z = gaussian_kde(xy)(xy)

        # Sort the points by density, so that the densest points are plotted last
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

        sc = ax.scatter(x, y, c=z, s=50, cmap=cmap)
        if colorbar_params.get('figure'):
            fig = colorbar_params.get('figure')

            if colorbar_params.get('outside') is None or colorbar_params.get('outside'):
                cax = ax.inset_axes([1.04, 0, 0.05, 1], transform=ax.transAxes)
                colorbar = fig.colorbar(sc, cax=cax)
                cax.tick_params(length=1)
            else:  # this is buggy
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size="10%", pad="3%")
                colorbar = fig.colorbar(sc, cax=cax)

            colorbar.set_ticks(colorbar_params.get('ticks'))


def plot_bar(ax, x, y, xerr=None, yerr=None, series_params={}):
    '''
    Draws bar chart

    Parameters:
    ----------
    ax: Axes
    x:
    y:
    xerr:
    yerr:
    series_params: dict with fields
        primary
        secondary
        label
        width
        edgewidth
        alpha
    '''
    cycler = ax._get_lines.prop_cycler
    secondary = next(cycler)['edgecolor']
    if series_params.get('secondary'):
        secondary = series_params.get('secondary')

    bar = ax.bar(
        x,
        y,
        xerr=xerr,
        yerr=yerr,
        label=series_params.get('label'),
        width=series_params.get('width', 0.8),
        edgecolor=secondary,
        lw=series_params.get('edgewidth'),
    )

    if series_params.get('primary'):
        primary = series_params.get('primary')
        for patch in bar.patches:
            patch.set_facecolor(primary)
    if series_params.get('alpha'):
        for patch in bar.patches:
            patch.set_alpha(series_params.get('alpha'))


def plot_scatter(ax, x, y, xerr=None, yerr=None, series_params={}, fit_params={}):
    '''
    Draws scatter plot on ax object

    Parameters:
    -----------
    ax: Axes
    x: list or np.ndarray
    y: list or np.ndarray
    xerr: np.ndarray or shape (2, len(x))
    yerr: np.ndarray of shape (2, len(y))
    series_params: dict with fields
        primary: colour
        secondary: colour
        markersize:
        marker:
        markeredgewidth:
        linestyle:
        alpha
        errobarwidth:
        label: str
            label for legend
        zorder: float
            (defaults to 10)
    fit_params: dict with fields
        equation: f(x,*params)
        fit_xlim
        linestyle
        color: 'primary', 'secondary', color, (defaults to primary)
        zorder: (defaults to 0)
    '''

    errorbar = ax.errorbar(x=x, y=y, xerr=xerr, yerr=yerr)

    # set default parameters
    # set error bar colours to markeredgecolor

    default_primary = errorbar.lines[0].get_markerfacecolor()
    default_secondary = errorbar.lines[0].get_markeredgecolor()
    for err in errorbar.lines[2]:
        err.set_color(default_secondary)

    # inputted parameters
    if series_params.get('primary'):
        errorbar.lines[0].set_markerfacecolor(series_params.get('primary'))
    if series_params.get('secondary'):
        errorbar.lines[0].set_markeredgecolor(series_params.get('secondary'))
        for err in errorbar.lines[2]:
            err.set_color(series_params.get('secondary'))
    if series_params.get('markersize'):
        errorbar.lines[0].set_markersize(series_params.get('markersize'))
    if series_params.get('marker'):
        errorbar.lines[0].set_marker(series_params.get('marker'))
    if series_params.get('markeredgewidth'):
        errorbar.lines[0].set_markeredgewidth(series_params.get('markeredgewidth'))
    if series_params.get('linestyle'):
        errorbar.lines[0].set_linestyle(series_params.get('linestyle'))
    if series_params.get('alpha'):
        errorbar.lines[0].set_alpha(series_params.get('alpha'))
    if series_params.get('errorbarwidth'):
        errorbar.lines[2][0].set_linewidth(series_params.get('errorbarwidth'))
    errorbar.set_label(series_params.get('label'))
    errorbar.lines[0].set_zorder(10)
    if series_params.get('zorder'):
        errorbar.lines[0].set_zorder(series_params.get('zorder'))

    # parametrized fit
    if fit_params:
        popt, _ = curve_fit(fit_params['equation'], x, y)
        fit_xlim = fit_params['fit_xlim']
        fit_x = np.linspace(*fit_xlim)
        plot = ax.plot(
            fit_x, fit_params['equation'](fit_x, *popt), color='k', marker='', markerfacecolor='', markeredgecolor=''
        )[0]
        plot.set_linestyle(fit_params.get('linestyle') or '-')

        if fit_params.get('color') is None or fit_params.get('color') == 'primary':
            plot.set_color(default_primary)
        elif fit_params.get('color') == 'secondary':
            plot.set_color(default_secondary)
        else:
            plot.set_color(fit_params.get('color'))

        plot.set_zorder(0)
        if fit_params.get('zorder'):
            plot.set_zorder(fit_params.get('zorder'))


def plot_curve(ax, x, y, series_params={}):
    '''
    Draws line plot on ax object

    Parameters:
    -----------
    ax: Axes
    x: list or np.ndarray
    y: list or np.ndarray
    series_params: dict with fields
        primary:
        marker
        linestyle
        errobarwidth
        label
        zorder (defaults to 10)
    '''
    plots = ax.plot(x, y)

    for plot in plots:
        if series_params.get('primary'):
            plot.set_color(series_params.get('primary'))
        plot.set_marker(series_params.get('marker') or '')
        plot.set_linestyle(series_params.get('linestyle') or '-')
        plot.set_label(series_params.get('label'))
        plot.set_zorder(10)
        if series_params.get('zorder'):
            plot.set_zorder(series_params.get('zorder'))


def plot_surface_2d(ax, z, x=None, y=None, series_params={}, colorbar_params={}):
    '''
    Parameters:
    -----------
    ax: Axes
    x: list or np.ndarray
    series_params: dict with fields
    colorbar_params: dict with fields
        figure: Figure
            Mandatory if a colorbar is to be added, you must input the figure
        outside: bool (not implemented)
            Whether the colorbar should be considered when aligning plots. If false,
            both the colorbar and the plot will contribute to the total length of the
            grid element.
        side: str (only 'right' is implemented)
            'top', 'bottom', 'left', 'right' (right is default)
        label_text: str (not implemented)
            add label to colorbar
        normalized: bool
            Whether the colorbar should be from 0 to 1

    '''
    scaling = 1
    if colorbar_params.get('figure') and colorbar_params.get('normalized'):
        scaling = np.max(x)

    if x is None or y is None:
        im = ax.imshow(z / scaling, origin='lower')
    else:
        im = ax.pcolormesh(x, y, z / scaling, shading='auto')

    if colorbar_params.get('figure'):
        fig = colorbar_params.get('figure')

        if colorbar_params.get('outside') is None or colorbar_params.get('outside'):
            cax = ax.inset_axes([1.04, 0, 0.05, 1], transform=ax.transAxes)
            colorbar = fig.colorbar(im, cax=cax)
            cax.tick_params(length=1)
        else:  # this is buggy
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size="10%", pad="3%")
            colorbar = fig.colorbar(im, cax=cax)

        colorbar.set_ticks(colorbar_params.get('ticks'))


def plot_boxplot(ax, x, series_params={}):
    '''
    Parameters:
    -----------
    ax: Axes
    x:
    series_params: dict with fields
        notch: bool
        zorder
    '''
    data_dict = ax.boxplot(
        x, manage_ticks=False, showcaps=False, notch=series_params.get('notch'), zorder=series_params.get('zorder')
    )
    for line in data_dict['medians']:
        line.set_color('k')


def plot_violinplot(ax, x, series_params={}):
    '''
    Parameters:
    -----------
    ax: Axes
    x:
    series_params: dict with fields
    '''
    data_dict = ax.violinplot(x, showmeans=False, showmedians=False, showextrema=False)
    for pc in data_dict['bodies']:
        pc.set_facecolor('#D43F3A')
        pc.set_edgecolor('black')
        pc.set_alpha(1)

    def adjacent_values(vals, q1, q3):
        upper_adjacent_value = q3 + (q3 - q1) * 1.5
        upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

        lower_adjacent_value = q1 - (q3 - q1) * 1.5
        lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
        return lower_adjacent_value, upper_adjacent_value

    quartile1, medians, quartile3 = np.percentile(x, [25, 50, 75], axis=1)
    whiskers = np.array(
        [adjacent_values(sorted_array, q1, q3) for sorted_array, q1, q3 in zip(x, quartile1, quartile3)]
    )
    whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

    inds = np.arange(1, len(medians) + 1)
    ax.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
    ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
    ax.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)


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
        ax.grid(True, which=mpl.rcParams['axes.grid.which'], axis=mpl.rcParams['axes.grid.axis'])
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
