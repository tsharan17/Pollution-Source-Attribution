"""Embed results/area_sim.json into the 3D viewer -> results/area_viewer.html"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")

if __name__ == "__main__":
    tpl = open(os.path.join(HERE, "area_viewer_template.html"), encoding="utf-8").read()
    data = open(os.path.join(RES, "area_sim.json"), encoding="utf-8").read()
    out = tpl.replace("/*__DATA__*/null", data)
    path = os.path.join(RES, "area_viewer.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(out)
    print(f"wrote {path} ({len(out) / 1e6:.1f} MB)")
