import sys; sys.path.append("./python-putils")
from putils.plotutils import *

ui1 = PlotUI_Sliders(
    4,
    (SliderParam('a', 0, 1, 0),
     SliderParam('b', 0, 1, 0)),
    ()
)

ui2 = PlotUI_Sliders(
    2,
    (SliderParam('a', 0, 1, 0),
     SliderParam('b', 0, 1, 0)),
    (SliderParam('a', 0, 1, 0),)
)

ui3 = PlotUI_Sliders(
    2,
    (SliderParam('a', 0, 1, 0),
     SliderParam('b', 0, 1, 0)),
    (SliderParam('a', 0, 1, 0),
     SliderParam('b', 0, 1, 0))
)

plt.show()
