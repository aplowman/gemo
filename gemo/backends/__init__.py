'`gemo.backends.__init__.py'

from gemo.backends.plotly import make_figure as make_figure_plotly
# from gemo.backends.matplotlib import make_figure as make_figure_mpl

make_figure_func = {
    'plotly': make_figure_plotly,
    # 'matplotlib': make_figure_mpl,
}
