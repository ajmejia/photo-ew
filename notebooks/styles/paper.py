from matplotlib import rc
from cycler import cycler

clist  = "#114477 #117755 #E8601C #771111 #771144 #4477AA #44AA88 #F1932D #AA4477 #774411 #777711 #AA4455".split()
font   = {"family":"sans-serif", "size":11.0}
text   = {"usetex":True, "latex.preamble":r"\usepackage[spanish]{babel},\usepackage{amsmath},\usepackage[helvet]{sfmath},\usepackage{helvet},\renewcommand{\familydefault}{\sfdefault}", "hinting":"native"}

rc("figure", figsize=(3.3,3.3))
rc("font", **font)
rc("text", **text)
rc("axes", linewidth=0.3, labelsize="small", titlesize="medium")
rc("grid", linewidth=0.5)
rc("xtick.major", size=3.0, width=0.3, pad=2)
rc("xtick", labelsize="x-small")
rc("ytick.major", size=3.0, width=0.3, pad=2)
rc("ytick", labelsize="x-small")
rc("lines", linewidth=1.0, markeredgewidth=0.0, markersize=7)
rc("patch", linewidth=1.0)
rc("hatch", linewidth=0.5)
rc("legend", numpoints=1, fontsize="xx-small", frameon=False)
rc("savefig", format="pdf", dpi=100, bbox="tight")

# font   = {"family":"sans-serif", "size":9.0, "weight":700}
# text   = {"usetex":True, "latex.preamble":r"\usepackage{amsmath},\usepackage[helvet]{sfmath},\usepackage{helvet},\renewcommand{\familydefault}{\sfdefault},\boldmath", "hinting":"native"}
#
# rc("figure", figsize=(3.3, 3.3))
# rc("font", **font)
# rc("text", **text)
# rc("axes", linewidth=0.5, labelsize="medium", titlesize="medium", prop_cycle=cycler("color", clist))
# rc("xtick.major", size=3.0, width=0.3, pad=2)
# rc("xtick", labelsize="x-small")
# rc("ytick.major", size=3.0, width=0.3, pad=2)
# rc("ytick", labelsize="x-small")
# rc("lines", linewidth=1.0, markeredgewidth=0.0, markersize=7)
# rc("patch", linewidth=1.0)
# rc("legend", numpoints=1, fontsize="xx-small", frameon=False)
# rc("savefig", format="pdf", dpi=92, bbox="tight")
