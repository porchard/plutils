#!/usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from . import stats
import seaborn as sns

@ticker.FuncFormatter
def read_count_formatter(x, pos):
    """
    Tick label formatting function that converts labels to B/M/k (billions, millions, thousands).

    Usage:
    ax.xaxis.set_major_formatter(read_count_formatter)
    """
    if abs(x) >= 1e9:
        return '{}B'.format(x/1e9)
    elif abs(x) >= 1e6:
        return '{}M'.format(x/1e6)
    elif abs(x) >= 1e3:
        return '{}k'.format(x/1e3)
    else:
        return x

@ticker.FuncFormatter
def pos_formatter(x, pos):
    """
    Tick label formatting function that converts 1e9 --> Gb, 1e6 --> Mb, 1e3 --> kb.

    Usage:
    ax.xaxis.set_major_formatter(pos_formatter)
    """
    if abs(x) >= 1e9:
        return '{} Gb'.format(x/1e9)
    elif abs(x) >= 1e6:
        return '{} Mb'.format(x/1e6)
    elif abs(x) >= 1e3:
        return '{} kb'.format(x/1e3)
    else:
        return x


def subplot_layout(n, col_wrap):
    """
    Given the number of subplots and the column wrap, return the number of rows and columns needed in the subplot grid (rows, columns).

    Input:
    n [int]: number of subplots to create
    col_wrap [int]: the maximum number of columns in the grid

    Returns:
    Tuple (nrows, ncols)
    """
    if n < 1:
        raise ValueError('n must be greater than 0')
    if col_wrap < 1:
        raise ValueError('col_wrap must be greater than 0')
    x = divmod(n, col_wrap)
    rows = x[0] + int(x[1] > 0)
    cols = min(n, col_wrap)
    return (rows, cols)


def subplots(n, col_wrap, panel_height=4.8, panel_width=6.4, **kwargs):
    """
    Given the number of subplots and the column wrap, return the number of rows and columns needed in the subplot grid (rows, columns).

    Input:
    n [int]: number of subplots to create
    col_wrap [int]: the maximum number of columns in the grid

    Returns:
    Tuple (fig, axs)
    """
    nrows, ncols = subplot_layout(n, col_wrap)
    if 'figsize' not in kwargs:
        kwargs['figsize'] = (panel_width*ncols, panel_height*nrows)
    fig, axs = plt.subplots(nrows, ncols, **kwargs)
    if not isinstance(axs, mpl.axes.Axes):
        for idx, ax in enumerate(axs.flatten(), 1):
            if idx > n:
                ax.remove()
    return (fig, axs)


def make_colormap_dict(keys, palette='viridis'):
    """
    Given list of items (in order), create a dict of item --> color.

    Input:
    keys: list of items.
    palette: name of matplotlib color palette to use

    Returns: Dict of item --> color (hex)
    """
    assert(isinstance(keys, list))
    assert(isinstance(palette, str))
    cmap = mpl.cm.get_cmap(palette, len(keys))
    return {keys[i]: cmap(i) for i in range(cmap.N)}


def rgb_to_hex(rgb):
    x = None
    if isinstance(rgb, str):
        x = [int(i) for i in rgb.split(',')]
    else:
        x = rgb
    return '#%02x%02x%02x' % tuple(x)


def make_quadrant_labels(x, y, drop_0=True):
    if not isinstance(x, (pd.Series, np.ndarray)):
        raise TypeError('x must be a pandas Series or numpy array')
    if not isinstance(y, (pd.Series, np.ndarray)):
        raise TypeError('y must be a pandas Series or numpy array')
    POSSIBLE_SIGNS = [-1, 0, 1]
    combos = pd.DataFrame([[i, j, sum(np.logical_and(np.sign(x)==i, np.sign(y)==j))] for i in POSSIBLE_SIGNS for j in POSSIBLE_SIGNS], columns=['x_sign', 'y_sign', 'count'])
    combos['fraction'] = combos['count'] / combos['count'].sum()
    combos['label'] = combos[['count', 'fraction']].apply(lambda x: '{} ({}%)'.format(int(x[0]), round(100*x[1], 1)), axis=1)
    if drop_0:
        combos = combos[combos['count'] > 0]
    return combos


def plot_quadrant_labels(labels, ax, loc=0.75):
    X_MIN, X_MAX = ax.get_xlim()
    Y_MIN, Y_MAX = ax.get_ylim()
    LABELS = []
    for _, r in labels.iterrows():
        X_POS = 0
        Y_POS = 0
        LABEL = r['label']
        if r['x_sign'] > 0:
            X_POS = X_MAX * loc
        elif r['x_sign'] < 0:
            X_POS = X_MIN * loc
        if r['y_sign'] > 0:
            Y_POS = Y_MAX * loc
        elif r['y_sign'] < 0:
            Y_POS = Y_MIN * loc
        LABELS.append(ax.text(X_POS, Y_POS, LABEL, ha='center'))
    return LABELS


def qqplot(p, ax=None):
    """
    p: list-like of observed p-values 
    ax: matplotlib Axes (in which case the plot will be made in that Axes) or None (in which case new (fig, ax) will be returned)
    """
    df = pd.DataFrame({'observed_p': p}).sort_values('observed_p')
    df['expected_p'] = stats.expected_pvalues(len(p))
    df['-log10(observed)'] = -1*np.log10(df.observed_p)
    df['-log10(expected)'] = -1*np.log10(df.expected_p)
    LIM = df[['-log10(observed)', '-log10(expected)']].max().max()*1.1
    if ax is None:
        fig, ax = plt.subplots()
        ax.set_xlim(0, LIM)
        ax.set_ylim(0, LIM)
        sns.scatterplot(x='-log10(expected)', y='-log10(observed)', data=df, ax=ax, edgecolor=None)
        ax.plot([0, LIM], [0, LIM], color='black', ls='--')
        return (fig, ax)
    else:
        ax.set_xlim(0, LIM)
        ax.set_ylim(0, LIM)
        sns.scatterplot(x='-log10(expected)', y='-log10(observed)', data=df, ax=ax, edgecolor=None)
        ax.plot([0, LIM], [0, LIM], color='black', ls='--')
        return ax


def manhattan_plot(chrom, pos, stat, chrom_sizes, chrom_order, ax=None):
    """
    chrom is a list-like of chromosomes
    pos is  a list-like of positions
    stat is the statistic for the y-axis
    chrom_sizes is a dictionary of chrom --> length
    chrom_order is a list of chromosomes, in the order they should be plotted
    """
    df = pd.DataFrame({'chrom': chrom, 'pos': pos, 'stat': stat})
    chrom_offsets = {}
    cumulative = 0
    for chrom in chrom_order:
        chrom_offsets[chrom] = cumulative
        cumulative += chrom_sizes[chrom]
    chrom_centers = [chrom_offsets[chrom] + chrom_sizes[chrom]/2 for chrom in chrom_order]
    df['absolute_position'] = df.chrom.map(chrom_offsets) + df.pos
    
    chrom_colors = ['red' if idx % 2 == 0 else 'black' for idx, chrom in enumerate(chrom_order)]
    chrom_colors = dict(zip(chrom_order, chrom_colors))

    fig, ax = plt.subplots(figsize=(10, 3)) if ax is None else (None, ax)
    sns.scatterplot(x='absolute_position', y='stat', hue='chrom', palette=chrom_colors, ax=ax, data=df)
    ax.legend().remove()
    ax.set_xlabel('chrom')
    ax.set_xticks(chrom_centers)
    ax.set_xticklabels(chrom_order)

    if fig is not None:
        return (fig, ax)
    else:
        return ax


def rotate_xticklabels(ax, rot=45, ha='right'):
    for t in ax.get_xticklabels():
        t.set(rotation=rot, ha=ha)
    return True


def use_log_scale(x, oom=2):
    if x.min() > 0 and np.log10(x.max() / x.min()) > oom:
        return True
    else:
        return False


def fix_heatmap_limits(ax):
    """
    Used to fix e.g. this bug: https://github.com/matplotlib/matplotlib/issues/14675
    """
    bottom, top = ax.get_ylim()
    if bottom > top:
        bottom += 0.5
        top -= 0.5
    else:
        bottom -= 0.5
        top += 0.5
    ax.set_ylim(bottom, top)
    return True


def append_n(x):
    """
    e.g., pd.Series(['a', 'b', 'a']) --> pd.Series(['a (n=2)', 'b (n=1)', 'a (n=2)'])
    """
    if not isinstance(x, (pd.Series, np.ndarray, list)):
        raise ValueError('x must be an array/list/Series')
    tmp = list(x)
    vc = pd.Series(tmp).value_counts().to_dict()
    tmp = ['{} (n={:,})'.format(i, vc[i]) for i in tmp]
    if isinstance(x, pd.Series):
        tmp = pd.Series(tmp)
        tmp.index = x.index
    return tmp


def add_legend_from_colors(d, ax, loc='best', marker='o'):
    """
    Given a dictionary from label --> color and a matplotlib ax,
    add a legend to the ax
    """
    legend_elements = [Line2D([0], [0], marker=marker, color=color, label=label) for label, color in d.items()]
    ax.legend(handles=legend_elements, loc=loc)
    return ax