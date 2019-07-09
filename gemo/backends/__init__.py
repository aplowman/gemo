'`gemo.backends.__init__.py'

from gemo.backends.plotly import make_figure as make_figure_plotly
# from gemo.backends.matplotlib import make_figure as make_figure_mpl

make_figure_func = {
    'plotly': make_figure_plotly,
    # 'matplotlib': make_figure_mpl,
}

# Might want to add a support-matrix here that states which backends support
# which "styles" (e.g. `fill_color`, `outline_color` etc).
