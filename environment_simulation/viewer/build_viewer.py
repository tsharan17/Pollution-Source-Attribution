"""Embed results/area_sim.json into the 3D viewer -> results/area_viewer.html"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")

if __name__ == "__main__":
    tpl = open(os.path.join(HERE, "area_viewer_template.html"), encoding="utf-8").read()
    import json
    d = json.load(open(os.path.join(RES, "area_sim.json"), encoding="utf-8"))
    sel = os.path.join(RES, "pm_cal_selection.json")
    d["pm_cal_selection"] = json.load(open(sel)) if os.path.exists(sel) else None
    # build sheet for the Build tab, regenerated from the source files on every viewer build
    sys.path.insert(0, os.path.dirname(HERE))
    import build_spec
    b = build_spec.build()
    with open(os.path.join(RES, "build_params.json"), "w", encoding="utf-8") as f:
        json.dump(b, f, ensure_ascii=False, indent=1)
    with open(os.path.join(os.path.dirname(HERE), "..", "BUILD_GUIDE.md"), "w", encoding="utf-8") as f:
        f.write(build_spec.to_markdown(b))
    d["build"] = b
    data = json.dumps(d, separators=(",", ":"))
    out = tpl.replace("/*__DATA__*/null", data)
    path = os.path.join(RES, "area_viewer.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(out)
    print(f"wrote {path} ({len(out) / 1e6:.1f} MB)")
