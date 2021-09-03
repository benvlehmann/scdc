"""This module defines plotting styles and routines. Some are out of date
but are retained for possible future use."""

import numpy as np

from matplotlib import pyplot as plt
from matplotlib import rc, rcParams
import seaborn as sns

from .particle import Phonon, Quasiparticle, DarkMatter


rc('text', usetex=True)
rc(
    'text.latex',
    preamble=(
        r'\usepackage{amsmath}\usepackage{siunitx}'
        r'\DeclareSIUnit{\year}{yr}'
        r'\usepackage{bm}'
        r'\usepackage{xcolor}'
        r'\newcommand{\bb}[1]{\bm{\mathrm{#1}}}'
    )
)
rc('font', size=14, family='serif', serif=['Times New Roman'])
rc('xtick.major', size=5, pad=7)
rc('ytick.major', size=5, pad=7)
rc('xtick', labelsize=16)
rc('ytick', labelsize=16)
rcParams['figure.figsize'] = 8, 6
rcParams['axes.labelsize'] = 18
rcParams['axes.titlesize'] = 18
rcParams['legend.loc'] = 'best'
rcParams['legend.fontsize'] = 16
rcParams['legend.frameon'] = False

FIG_SINGLE_SIZE = (8, 6)
FIG_DOUBLE_SIZE = (16, 6)
COLORS = sns.color_palette('colorblind')
COLORS.pop(3)


def latex_exp_format(x, mfmt='%.3f'):
    """Format a number in scientific notation in LaTeX.

    Args:
        x (float): number to format.
        mfmt (str, optional): format string for mantissa. Defaults to '%.3f'.
    Returns:
        str: LaTeX string (no $'s).
    """
    if x == 0:
        return '0'
    exponent = np.floor(np.log10(np.abs(x)))
    mantissa = x / 10.**exponent
    result_str = mfmt % mantissa
    if exponent != 0:
        result_str += r'\times 10^{%d}' % exponent
    return result_str


def tree_plot(particle, origin=(0, 0), **kwargs):
    """Plot all child scattering events.

    Args:
        fig (:obj:`Figure`, optional): matplotlib figure object.
        ax (:obj:`AxesSubplot`, optional): matplotlib axis object.
        origin (:obj:`tuple` of :obj:`float`, optional): starting
            coordinates for the tree.
        dm_color (str, optional): color for DM lines.
        phonon_color (str, optional): color for phonon lines.
        qp_color (str, optional): color for quasiparticle lines.
        min_linewidth (float, optional): smallest linewidth (E = Delta).
        max_linewidth (float, optional): largest linewidth, corresponding
            to the energy of the initial excitation.
        max_linewidth_energy (float, optional): energy for max linewidth.
        final_distance (float, optional): if specified, final (ballistic)
            state lines will be extended to this distance from (0, 0).
        alpha (float, optional): opacity.

    """
    ax = kwargs.get('ax')
    dm_color = kwargs.get('dm_color', 'k')
    phonon_color = kwargs.get('phonon_color', 'b')
    qp_color = kwargs.get('qp_color', 'r')
    max_linewidth = kwargs.get('max_linewidth', 10.)
    min_linewidth = kwargs.get('min_linewidth', 1.)
    max_linewidth_energy = kwargs.get('max_linewidth_energy')
    alpha = kwargs.get('alpha', 1)
    if not max_linewidth_energy:
        kwargs['max_linewidth_energy'] = particle.energy
        max_linewidth_energy = particle.energy
    final_distance = kwargs.get('final_distance')
    if ax is None:
        fig, ax = plt.subplots()
        kwargs['ax'] = ax
    else:
        fig = ax.figure
    o_x, o_y = origin
    # Plot a line from the origin at the appropriate angle
    if isinstance(particle, Quasiparticle):
        color = qp_color
    elif isinstance(particle, Phonon):
        color = phonon_color
    elif isinstance(particle, DarkMatter):
        color = dm_color
    else:
        raise RuntimeError("I do not recognize this particle type")
    if isinstance(particle, DarkMatter):
        # skip the rest and go directly to children
        for child in particle.dest.out:
            tree_plot(child, (o_x, o_y), **kwargs)
        # Do draw a marker for the event itself
        plt.scatter(o_x, o_y, marker='*', color=dm_color, s=100, zorder=100)
        return fig, ax
    energy_fraction = (particle.energy - 1) / (max_linewidth_energy - 1)
    linewidth = energy_fraction * max_linewidth + \
        (1 - energy_fraction) * min_linewidth
    dx = np.sqrt(1 - particle.cos_theta**2)
    # We are free to choose the sign here, for visual purposes:
    dx *= np.random.choice([-1, 1])
    dy = particle.cos_theta
    # Normalize to energy
    dx *= particle.energy
    dy *= particle.energy
    # Adjust the plot window
    """We use the following rule: the arrow coordinates must not come closer
    to the edge than a fraction `edge_fraction` of the axis length. So we first
    check to see whether this condition is satisfied. If not, we enforce it.
    """
    edge_fraction = 0.1
    min_x = np.amin(ax.get_xlim())
    max_x = np.amax(ax.get_xlim())
    min_y = np.amin(ax.get_ylim())
    max_y = np.amax(ax.get_ylim())
    x_length = max_x - min_x
    y_length = max_y - min_y
    if max_x - (o_x + dx) < edge_fraction * x_length:
        max_x = o_x + dx + edge_fraction * (o_x + dx - min_x)
    if max_y - (o_y + dy) < edge_fraction * y_length:
        max_y = o_y + dy + edge_fraction * (o_y + dy - min_y)
    if (o_x + dx) - min_x < edge_fraction * x_length:
        min_x = o_x + dx + edge_fraction * (o_x + dx - max_x)
    if (o_y + dy) - min_y < edge_fraction * y_length:
        min_y = o_y + dy + edge_fraction * (o_y + dy - max_y)
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_aspect(1.)
    ax.arrow(
        o_x, o_y, dx, dy,
        color=color,
        linewidth=linewidth,
        head_width=linewidth/3,
        length_includes_head=True,
        head_starts_at_zero=False,
        alpha=alpha
    )
    if final_distance and particle.dest.final:
        # This is a leaf and we should extend it
        dx *= final_distance
        dy *= final_distance
        ax.plot(
            [o_x, o_x + dx], [o_y, o_y + dy], color='#cccccc', lw=0.5,
            zorder=-100
        )
    else:
        # Recurse to the children
        for child in particle.dest.out:
            tree_plot(child, (o_x+dx, o_y+dy), **kwargs)
    return fig, ax


def angular_distribution(ensemble, ax=None, weight_function=None, pkw={}):
    if not ax:
        _, ax = plt.subplots()
        ax.set_xlabel(r'$\cos\,\theta$')
        ax.set_ylabel('probability density')
        ax.set_xlim(-1, 1)
        label = True
    else:
        label = False
    cases = (
        (ensemble.quasiparticles, 'r', 'QPs'),
        (ensemble.phonons, 'b', 'phonons'),
        (ensemble.leaves, 'k', 'all')
    )
    for subset, color, name in cases:
        mask = np.isfinite(subset.cos_theta)
        data = subset.cos_theta[mask]
        if weight_function is not None:
            weights = weight_function(subset)[mask]
        else:
            weights = None
        if not data.size:
            continue
        plot_kwargs = dict(
            density=True, color=color, bins=np.linspace(-1, 1, 30),
            weights=weights, lw=3, alpha=0.1
        )
        plot_kwargs.update(pkw)
        ax.hist(data, **plot_kwargs)
        del plot_kwargs['alpha']
        ax.hist(data, histtype='step', label=name, **plot_kwargs)
        if label:
            ax.legend(loc='best')
    return ax


def highest_energy_distribution(ensemble, **kwargs):
    highest_energy_leaves = ensemble.select(
        lambda c, p: p.energy == np.amax([lf.energy for lf in c.leaves])
    )
    return angular_distribution(highest_energy_leaves, **kwargs)


def differential_distribution(ensemble, n_x, n_y, **kwargs):
    fig, axes = plt.subplots(n_x, n_y)
    fig.subplots_adjust(hspace=0.3)
    fig.set_size_inches(9, 9)
    all_axes = axes.reshape(-1)
    energy_edges = np.linspace(
        np.amin(ensemble.leaves.energy),
        np.amax(ensemble.leaves.energy),
        len(all_axes) + 1
    )
    energy_bins = np.vstack((energy_edges[:-1], energy_edges[1:])).T
    bin_iterator = iter(energy_bins)
    for i, row in enumerate(axes):
        for j, ax in enumerate(row):
            emin, emax = next(bin_iterator)
            bin_ensemble = ensemble.select(
                lambda _, p: emin < p.energy and p.energy <= emax
            )
            angular_distribution(bin_ensemble, ax=ax, **kwargs)
            if i == axes.shape[0] - 1:
                ax.set_xlabel(r'$\cos\,\theta$')
            if j == 0:
                ax.set_ylabel('probability density')
            ax.set_xlim(-1, 1)
            ax.set_title('$%.2f < E/\\Delta < %.2f$' % (emin, emax))
    return fig


def chain_plot(states, nrbins=None, ncbins=None):
    fig, axes = plt.subplots(2, 2)
    fig.set_size_inches(10, 10)

    eps = 10e-5
    rkw = {}
    if nrbins is not None:
        rkw['bins'] = np.linspace(1-eps, 1+eps, nrbins)

    ax = axes[0, 0]
    ax.set_xlim(1-eps, 1+eps)
    ax.set_ylim(1-eps, 1+eps)
    ax.hist2d(states[:, 0], states[:, 2], cmap='Blues', **rkw)
    ax.scatter(states[0, 0], states[0, 2], s=20, c='r')
    ax.set_aspect(1)
    ax.axhline(1, c='r')
    ax.axvline(1, c='r')
    ax.set_xlabel(r'$r_3/k_F$')
    ax.set_ylabel(r'$r_4/k_F$')
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    ax = axes[1, 0]
    ax.set_xlim(1-eps, 1+eps)
    ax.hist(states[:, 0], histtype='step', density=True, **rkw)
    ax.hist(states[:, 2], histtype='step', density=True, **rkw)
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    ax.set_aspect((x_max-x_min)/(y_max-y_min))
    ax.set_xlabel(r'$r_{QP}$')
    ax.set_ylabel('prob. density')
    ax.axvline(1, c='r')
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    ckw = {}
    if ncbins is not None:
        ckw['bins'] = np.linspace(-1, 1, ncbins)

    ax = axes[0, 1]
    ax.hist2d(states[:, 1], states[:, 3], cmap='Blues', **ckw)
    ax.scatter(states[0, 1], states[0, 3], s=20, c='r')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect(1)
    ax.set_xlabel(r'$\cos\theta_3$')
    ax.set_ylabel(r'$\cos\theta_4$')
    ax.axhline(0, c='r')
    ax.axvline(0, c='r')

    ax = axes[1, 1]
    ax.hist(states[:, 1], histtype='step', density=True, **ckw)
    ax.hist(states[:, 3], histtype='step', density=True, **ckw)
    ax.set_xlim(-1, 1)
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    ax.set_aspect((x_max-x_min)/(y_max-y_min))
    ax.set_xlabel(r'$\cos\theta$')
    ax.set_ylabel('prob. density')
    ax.axhline(0.5, c='r')
    ax.axvline(0, c='r')

    return fig, axes
