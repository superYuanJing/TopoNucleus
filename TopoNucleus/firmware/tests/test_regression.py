"""回归测试：采集->分析->进化的可复现流水线（固定随机种子）。"""
import glob, json, os, subprocess, sys
_TOP = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # 仓库根: TopoNucleus/
SURF = {"bare": ["iron", "tungsten", "fe-si", "permalloy"],
         "anodized": ["iron", "fe-si", "permalloy"],
         "coated": ["iron", "fe-si", "permalloy"],
         "copper-plated": ["iron", "fe-si"]}
def run(cmd):
    r = subprocess.run([sys.executable, "-m", "firmware." + cmd[0].replace(".py", "")] + cmd[1:],
                        cwd=_TOP, capture_output=True, text=True, check=False)
    assert r.returncode == 0, f"{cmd}\\n{r.stdout}\\n{r.stderr}"
def clean():
    for d in ("data/raw", "data/processed", "firmware/data/processed", "hardware/seed_jig"):
        for f in glob.glob(os.path.join(_TOP, d, "*"), recursive=False):
            if os.path.isfile(f): os.remove(f)
def main():
    clean()
    for ring, pres, mat, surf in [(3.0, 18, "iron", "bare"), (3.0, 21, "iron", "bare"),
                                  (3.0, 24, "iron", "bare"), (3.0, 27, "iron", "bare"),
                                  (4.0, 35, "fe-si", "anodized")]:
        run(["daq/daq.py", "--seed", "trefoil", "--material", mat,
             "--surface", surf, "--ring", str(ring), "--pressure", str(pres),
             "--power", "100", "--tau-r-ms", "8.0"])
    csvs = sorted(glob.glob(os.path.join(_TOP, "data/raw/*.csv")))
    assert len(csvs) == 5
    for c in csvs:
        run(["analysis/analyzer.py", c, "--tau-r-ms", "8.0"])
    feats = sorted(glob.glob(os.path.join(_TOP, "data/processed/*_features.json")))
    assert len(feats) == 5
    for f in feats:
        d = json.load(open(f)); assert "features" in d and "metadata" in d
        m = d["metadata"]
        for k in ["ring_diameter_cm", "wire_diameter_mm", "twist", "pressure_mtorr",
                  "discharge_power_w", "seed_material", "seed_surface", "symmetry_break"]:
            assert k in m and isinstance(m[k], (int, float, str)), (f, k)
    run(["firmware/evolution/evolution_engine.py", "--population-size", "10",
         "--generations", "6", "--random-rounds", "5", "--seed", "42"])
    gens = sorted(glob.glob(os.path.join(_TOP, "hardware/seed_jig/generation_*_seeds.json")))
    assert len(gens) == 6
    last = json.load(open(gens[-1])); assert len(last["seeds"]) == 10
    for g in last["seeds"]:
        assert g["surface"] in SURF
        if g["surface"] != "bare":
            assert g["material"] in SURF[g["surface"]]
        assert 1.0 <= g["ring_diameter_cm"] <= 5.0
        assert 0.5 <= g["wire_diameter_mm"] <= 2.0
    print("OK regression: 5 raw + 5 features + 6 generations, BO produced legal seeds")
if __name__ == "__main__":
    main()
