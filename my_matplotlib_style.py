import matplotlib as mpl
import matplotlib.pyplot as plt
from colorsys import hsv_to_rgb


def set_my_style():
    # lines
    mpl.rcParams['lines.linewidth'] = 2
    mpl.rcParams['lines.color'] = 'r'

    # axis
    mpl.rcParams['axes.titlesize'] = 26
    mpl.rcParams['axes.grid'] = True


def sciy():
    plt.gca().ticklabel_format(style='sci', scilimits=(0, 0), axis='y')


def scix():
    plt.gca().ticklabel_format(style='sci', scilimits=(0, 0), axis='x')


def colorprog(i_prog, Nplots, v1=.9, v2=1., cm='hsv'):
    if hasattr(Nplots, '__len__'):
        Nplots = len(Nplots)
    if cm == 'hsv':
        return hsv_to_rgb(float(i_prog) / float(Nplots), v1, v2)
    elif cm == 'rainbow':
        return [plt.cm.rainbow(k) for k in np.linspace(0, 1, Nplots)][i_prog]
    else:
        raise ValueError('What?!')
