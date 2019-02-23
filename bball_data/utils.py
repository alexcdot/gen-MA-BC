import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation
from skimage.transform import resize

import bball_data.cfg as cfg


SCALE = cfg.SCALE
MACRO_SIZE = cfg.MACRO_SIZE*SCALE
CMAP = cfg.CMAP_OFFENSE
DEF_COLOR = 'b'


def normalize(x):
    dim = x.shape[-1]
    return np.divide(x-cfg.SHIFT[:dim], cfg.NORMALIZE[:dim])


def unnormalize(x):
    dim = x.shape[-1]
    return np.multiply(x, cfg.NORMALIZE[:dim]) + cfg.SHIFT[:dim]


def _set_figax():
    fig = plt.figure(figsize=(5,5))
    img = plt.imread(cfg.DATAPATH+'court.png')
    img = resize(img,(SCALE*cfg.WIDTH,SCALE*cfg.LENGTH,3))

    ax = fig.add_subplot(111)
    ax.imshow(img)

    # show just the left half-court
    ax.set_xlim([-50,550])
    ax.set_ylim([-50,550])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    return fig, ax


def plot_sequence(seq, macro_goals=None, colormap=CMAP, burn_in=0, save_path='', save_name=''):
    n_players = int(len(seq[0])/2)
    n_offense = len(cfg.CMAP_OFFENSE)
    # Include defense in plots
    if n_players > n_offense:
        # We only trained the offense, so highlight the offensive players
        if macro_goals.shape[1] <= n_offense:
            colormap = cfg.CMAP_OFFENSE_FOCUSED_PLAYERS
        # We trained both offense and defense, so color them the same
        else:
            colormap = cfg.CMAP_ALL_PLAYERS

    while len(colormap) < n_players:
        colormap += DEF_COLOR

    fig, ax = _set_figax()

    for k in range(n_players):
        x = seq[:,(2*k)]
        y = seq[:,(2*k+1)]
        color = colormap[k]

        ax.plot(SCALE*x, SCALE*y, color=color, linewidth=3, alpha=0.7)
        ax.plot(SCALE*x, SCALE*y, 'o', color=color, markersize=8, alpha=0.5)

        if macro_goals is not None:
            for t in range(len(seq)):
                # We also check that k does not exceed the number of trained
                # macro goals. For the untrained defense, we don't have
                # macro goals for them.
                if t >= burn_in and k < macro_goals.shape[1]:
                    m_x = int(macro_goals[t,k]/cfg.N_MACRO_Y)
                    m_y = macro_goals[t,k] - cfg.N_MACRO_Y*m_x
                    ax.add_patch(patches.Rectangle(
                        (m_x*MACRO_SIZE, m_y*MACRO_SIZE), MACRO_SIZE, MACRO_SIZE, alpha=0.02, color=color, linewidth=2)) 

    # starting positions
    x = seq[0,::2]
    y = seq[0,1::2]
    ax.plot(SCALE*x, SCALE*y, 'o', color='black', markersize=12)

    # burn-ins
    if burn_in > 0:
        x = seq[:burn_in,::2]
        y = seq[:burn_in,1::2]
        ax.plot(SCALE*x, SCALE*y, color='0.01', linewidth=8, alpha=0.5)

    plt.tight_layout(pad=0)

    if len(save_name) > 0:
        plt.savefig(save_path+save_name+'.png')
    else:
        plt.show()

def plot_sample_seqs(sample_seqs, sample_macro_goals=None, colormap=CMAP, burn_in=0, save_path='', save_name=''):
    n_players = int(sample_seqs.shape[2]/2)
    n_offense = len(cfg.CMAP_OFFENSE)
    # Include defense in plots
    if n_players > n_offense:
        # We only trained the offense, so highlight the offensive players
        if len(sample_macro_goals) > 0 and sample_macro_goals.shape[2] <= n_offense:
            colormap = cfg.CMAP_OFFENSE_FOCUSED_PLAYERS
        # We trained both offense and defense, so color them the same
        else:
            colormap = cfg.CMAP_ALL_PLAYERS

    while len(colormap) < n_players:
        colormap += DEF_COLOR

    fig, ax = _set_figax()

    for i, seq in enumerate(sample_seqs):
        if len(sample_macro_goals) > 0:
            macro_goals = sample_macro_goals[i]
        for k in range(n_players):
            x = seq[:,(2*k)]
            y = seq[:,(2*k+1)]
            color = colormap[k]
            dilution_factor = np.log(len(sample_seqs) + 1)

            ax.plot(SCALE*x, SCALE*y, color=color, linewidth=3, alpha=0.5 / dilution_factor)
            # ax.plot(SCALE*x, SCALE*y, 'o', color=color, markersize=8, alpha=0.5)

            if len(sample_macro_goals) > 0:
                for t in range(len(seq)):
                    # We also check that k does not exceed the number of trained
                    # macro goals. For the untrained defense, we don't have
                    # macro goals for them.
                    if t >= burn_in and k < macro_goals.shape[1]:
                        m_x = int(macro_goals[t,k]/cfg.N_MACRO_Y)
                        m_y = macro_goals[t,k] - cfg.N_MACRO_Y*m_x
                        ax.add_patch(patches.Rectangle(
                            (m_x*MACRO_SIZE, m_y*MACRO_SIZE), MACRO_SIZE, MACRO_SIZE,
                             alpha=0.02 / dilution_factor, color=color, linewidth=2))

        # only plot these once
        if i == 0:
            # starting positions
            print(seq.shape, seq[::2].shape)
            x = seq[0, ::2]
            y = seq[0, 1::2]
            ax.plot(SCALE*x, SCALE*y, 'o', color='black', markersize=12)

            # burn-ins
            if burn_in > 0:
                x = seq[:burn_in,::2]
                y = seq[:burn_in,1::2]
                ax.plot(SCALE*x, SCALE*y, color='0.01', linewidth=8, alpha=0.5)

    plt.tight_layout(pad=0)

    if len(save_name) > 0:
        plt.savefig(save_path+save_name+'.png')
    else:
        plt.show()


def animate_sequence(seq, macro_goals=None, colormap=CMAP, burn_in=0, save_path='', save_name=''):
    n_players = int(len(seq[0])/2)
    n_offense = len(cfg.CMAP_OFFENSE)
    seq_len = len(seq)

    if n_players > n_offense:
        colormap = cfg.CMAP_ALL_PLAYERS

    while len(colormap) < n_players:
        colormap += DEF_COLOR

    fig, ax = _set_figax()

    trajectories = [ax.plot([],[])[0] for _ in range(n_players)]
    locations = [ax.plot([],[])[0] for _ in range(n_players)]
    burn_ins = [ax.plot([],[])[0] for _ in range(n_players)]

    macros = []
    if macro_goals is not None:
        from matplotlib.patches import Rectangle
        macros = [Rectangle(xy=(0, 0), width=MACRO_SIZE, height=MACRO_SIZE, alpha=0) for k in range(macro_goals.shape[1])]                                  
    
    def init():
        for k in range(n_players):
            traj = trajectories[k]
            loc = locations[k]
            burn = burn_ins[k]
            color = colormap[k % n_players]

            traj.set_data([],[])
            traj.set_color(color)
            traj.set_linewidth(3)
            traj.set_alpha(0.7)

            loc.set_data([],[])
            loc.set_color(color)
            loc.set_marker('o')
            loc.set_markersize(12)

            burn.set_data([],[])
            burn.set_color('0.01')
            burn.set_linewidth(6)
            burn.set_alpha(0.5)

            if k < len(macros):
                m = macros[k]
                ax.add_patch(m)
                m.set_color(color)

        return trajectories+locations+burn_ins+macros

    def animate(t):
        if t >= seq_len:
            t = seq_len-1

        for p in range(n_players):
            trajectories[p].set_data(SCALE*seq[:t+1,2*p], SCALE*seq[:t+1,2*p+1])
            locations[p].set_data(SCALE*seq[t,2*p], SCALE*seq[t,2*p+1])
            burn_ins[p].set_data(SCALE*seq[:min(t, burn_in),2*p], SCALE*seq[:min(t, burn_in),2*p+1])

        # start showing macro-goals after burn-in period
        if t >= burn_in:
            for j,m in enumerate(macros):
                m_x = int(macro_goals[t,j]/cfg.N_MACRO_Y)
                m_y = macro_goals[t,j] - cfg.N_MACRO_Y*m_x
                m.set_xy([m_x*MACRO_SIZE, m_y*MACRO_SIZE])
                m.set_alpha(0.5)

        return trajectories+locations+burn_ins+macros

    plt.tight_layout(pad=0)
    
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=72, interval=100, blit=True)

    if len(save_name) > 0:
        anim.save(save_path+save_name+'.mp4', fps=7, extra_args=['-vcodec', 'libx264'])
    else:
        plt.show()