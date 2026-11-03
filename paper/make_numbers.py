"""Regenerate paper/numbers.tex. Run from the repo root: .venv/bin/python paper/make_numbers.py

Turn the SS64 routed-eval JSONs into paper/numbers.tex macros + a diagnostic dump."""

import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = f"{ROOT}/results/live_learning/confirm/"
missing = []  # input files not found; nothing is written if any


def f3(x):
    return (
        f"{x:.3f}".rstrip("0")
        if len(f"{x:.3f}".rstrip("0").split(".")[1]) >= 2
        else f"{x:.2f}"
    )


def ci(c):
    return ",".join(
        ("." + f"{v:.3f}".split(".")[1]).rstrip("0") if 0 < v < 1 else str(int(v))
        for v in c
    )


out = [
    "% Generated from results/live_learning/confirm/*_n1000_routed_eval.json (prereg SS64)."
]
diag = {}
for tag, fn in [
    ("cf", "cf_n1000_routed_eval.json"),
    ("zs", "zsre_n1000_routed_eval.json"),
]:
    try:
        d = json.load(open(R + fn))
    except FileNotFoundError as e:
        missing.append(e.filename)
        continue
    m = d["metrics"]

    def put(name, key):
        out.append(f"\\newcommand{{\\{tag}{name}}}{{{f3(m[key]['rate'])}}}")
        pct = f"{100 * m[key]['rate']:.1f}".removesuffix(".0")
        out.append(f"\\newcommand{{\\{tag}{name}Pct}}{{{pct}}}")
        out.append(f"\\newcommand{{\\{tag}{name}CI}}{{{ci(m[key]['ci95'])}}}")

    put("Eff", "probe_efficacy")
    put("Para", "paraphrase_efficacy")
    put("NoLeak", "distractor_no_leak")
    put("Gone", "revoke_token_gone_rate")
    put("Return", "revoke_return_match_rate")
    put("Retain", "retain_rate")
    for name, key in [
        ("Route", "probe_route_accuracy"),
        ("ParaRoute", "paraphrase_route_accuracy"),
    ]:
        out.append(f"\\newcommand{{\\{tag}{name}}}{{{f3(m[key]['rate'])}}}")
    out.append(
        f"\\newcommand{{\\{tag}Abst}}{{{f3(m['distractor_abstention_rate']['rate'])}}}"
    )
    out.append(
        f"\\newcommand{{\\{tag}AbstPct}}{{{100 * m['distractor_abstention_rate']['rate']:.1f}}}"
    )
    out.append(
        f"\\newcommand{{\\{tag}RoutePct}}{{{100 * m['probe_route_accuracy']['rate']:.1f}}}"
    )
    out.append(
        f"\\newcommand{{\\{tag}RevMs}}{{{d['revoke']['mean_revoke_seconds'] * 1000:.1f}}}"
    )
    lo, hi = m["probe_efficacy"]["ci95"]
    r = m["probe_efficacy"]["rate"]
    out.append(f"\\newcommand{{\\{tag}EffErr}}{{{max(r - lo, hi - r):.3f}}}")
    ex = d["examples"]
    n = len(ex)
    pr = [e["probe"] for e in ex]
    pa = [e["paraphrase"] for e in ex]
    dg = {
        "probe_abstain": sum(p["abstained"] for p in pr) / n,
        "probe_misroute": sum(
            (not p["abstained"]) and not p["route_correct"] for p in pr
        )
        / n,
        "probe_hit_given_correct": sum(
            p["answer_hit"] for p in pr if p["route_correct"]
        )
        / max(1, sum(p["route_correct"] for p in pr)),
        "probe_hit_given_misroute": sum(
            p["answer_hit"]
            for p in pr
            if (not p["abstained"]) and not p["route_correct"]
        )
        / max(1, sum((not p["abstained"]) and not p["route_correct"] for p in pr)),
        "para_abstain": sum(p["abstained"] for p in pa) / n,
        "para_misroute": sum(
            (not p["abstained"]) and not p["route_correct"] for p in pa
        )
        / n,
        "para_hit_given_correct": sum(p["answer_hit"] for p in pa if p["route_correct"])
        / max(1, sum(p["route_correct"] for p in pa)),
        "dist_routed": sum(not e["distractor"]["abstained"] for e in ex) / n,
        "base_hit_rate_revoke_sample": sum(
            r["return_check"]["base_answer_hit"] for r in d["revoke"]["removed"]
        )
        / len(d["revoke"]["removed"]),
        "retain_route_acc": m["retain_route_accuracy"]["rate"],
        "metrics": {k: (v["rate"], v["ci95"]) for k, v in m.items()},
    }
    diag[tag] = dg


# Router v4 re-evaluation (SS67/68): prefix "v", dataset tag "Cf"/"Zs".
def emit(name, metric, fixed=False):
    """value, percentage and CI macros for a {'rate','ci95'} dict or a plain number.
    fixed: always three decimals and a one-decimal percentage."""
    rate = metric["rate"] if isinstance(metric, dict) else metric
    value = f"{rate:.3f}" if fixed else f3(rate)
    pct = f"{100 * rate:.1f}" if fixed else f"{100 * rate:.1f}".removesuffix(".0")
    out.append(f"\\newcommand{{\\{name}}}{{{value}}}")
    out.append(f"\\newcommand{{\\{name}Pct}}{{{pct}}}")
    if isinstance(metric, dict) and metric.get("ci95"):
        out.append(f"\\newcommand{{\\{name}CI}}{{{ci(metric['ci95'])}}}")
        if fixed:
            lo, hi = metric["ci95"]
            out.append(
                f"\\newcommand{{\\{name}CIPct}}{{{100 * lo:.1f} to {100 * hi:.1f}}}"
            )


def emit_diff(name, d):
    out.append(f"\\newcommand{{\\{name}}}{{{100 * d['diff']:.1f}}}")
    lo, hi = d["ci95"]
    out.append(f"\\newcommand{{\\{name}CI}}{{{100 * lo:.1f} to {100 * hi:.1f}}}")


for tag, fn in [("Cf", "cf_n1000_v4_eval.json"), ("Zs", "zsre_n1000_v4_eval.json")]:
    try:
        d = json.load(open(R + fn))
    except FileNotFoundError as e:
        missing.append(e.filename)
        continue
    sv = d["serving"]
    emit(f"v{tag}Eff", sv["probe"]["adapter"])
    emit(f"v{tag}EffRet", sv["probe"]["retrieval"])
    emit(f"v{tag}EffBoth", sv["probe"]["both"])
    emit(f"v{tag}Route", sv["probe"]["route_correct"])
    emit(f"v{tag}Para", sv["paraphrase"]["adapter"])
    emit(f"v{tag}ParaRet", sv["paraphrase"]["retrieval"])
    emit(f"v{tag}ParaBoth", sv["paraphrase"]["both"])
    emit(f"v{tag}ParaRouted", sv["paraphrase"]["routed"])
    emit(f"v{tag}ParaRoute", sv["paraphrase"]["route_correct"])
    emit_diff(f"v{tag}ParaDiff", sv["paraphrase"]["adapter_minus_retrieval"])
    emit_diff(f"v{tag}EffDiff", sv["probe"]["adapter_minus_retrieval"])
    emit(f"v{tag}NoLeak", sv["distractor"]["no_leak"])
    emit(f"v{tag}TierFrac", d["tiered"]["adapter_fraction"])
    emit(f"v{tag}TierEff", d["tiered"]["probe"])
    emit(f"v{tag}TierPara", d["tiered"]["paraphrase"])
    rm = d["removal"]
    emit(f"v{tag}Gone", rm["gone"])
    emit(f"v{tag}Return", rm["return"])
    emit(f"v{tag}Retain", rm["retention"])
    if "by_content" in rm:
        emit(f"v{tag}ReturnContent", rm["by_content"]["return"])
        out.append(
            f"\\newcommand{{\\v{tag}ContentExtra}}{{{rm['by_content']['extra_entries_removed']}}}"
        )
    names = {
        "E2_neighborhood": "Etwo",
        "E3_other_relation": "Ethree",
        "loc_questions": "Loc",
    }
    for key, short in names.items():
        if key in d["locality"]:
            for m, v in d["locality"][key].items():
                emit(
                    f"v{tag}{short}{ {'routed': 'Routed', 'same_as_base': 'Same', 'leak': 'Leak'}[m] }",
                    v,
                )

try:
    c = json.load(open(R + "canary_eval.json"))
    emit("chkInMean", c["benign_in_sample"]["mean"])
    emit("chkInMax", c["benign_in_sample"]["max"])
    emit("chkHeldMean", c["benign_heldout"]["mean"])
    emit("chkHeldMax", c["benign_heldout"]["max"])
    emit("chkKappa", c["kappa_heldout_p99"])
    emit("chkWrongRoute", c["wrong_route_retained"])
except FileNotFoundError as e:
    missing.append(e.filename)

# SS71 serving-arm locality under v4, SS74 operating points and trade-off curve.
try:
    sl = json.load(open(R + "serving_locality.json"))
    for pol, short in [
        ("adapter", "Adapter"),
        ("note", "Note"),
        ("both", "Both"),
        ("tiered", "Tiered"),
    ]:
        emit(f"vCfEthreeLeak{short}", sl["E3_other_relation"][pol]["leak"])
except FileNotFoundError as e:
    missing.append(e.filename)
try:
    rv = json.load(open(R + "relevance_eval.json"))
    for prefix, point in [("y", "youden"), ("s", "cf_fpr_0.05")]:
        out.append(
            f"\\newcommand{{\\{prefix}Theta}}{{{rv['operating_points'][point]['theta']:.3f}}}"
        )
        t = rv["test"][point]
        for tag, ds in [("Cf", "counterfact"), ("Zs", "zsre")]:
            for kind, k in [("Eff", "probe"), ("Para", "paraphrase")]:
                r = t[ds][k]
                emit(f"{prefix}{tag}{kind}", r["adapter"])
                emit(f"{prefix}{tag}{kind}Ret", r["retrieval"])
                emit(
                    f"{prefix}{tag}{kind}Both",
                    {"rate": r["both"], "ci95": r["both_ci"]},
                )
                emit(f"{prefix}{tag}{kind}Tier", r["tiered"])
                emit(f"{prefix}{tag}{kind}Route", r["route_correct"])
        for loc, short in [
            ("E3_other_relation", "Ethree"),
            ("E2_neighborhood", "Etwo"),
        ]:
            emit(f"{prefix}Cf{short}Routed", t[loc]["routed"])
            for pol, ps in [
                ("adapter", "Adapter"),
                ("note", "Note"),
                ("both", "Both"),
                ("tiered", "Tiered"),
            ]:
                emit(
                    f"{prefix}Cf{short}Leak{ps}",
                    {"rate": t[loc][pol]["leak"], "ci95": t[loc][pol]["leak_ci"]},
                )
    for sc in ("s_rel", "s_judge", "s_type"):
        name = {"s_rel": "Rel", "s_judge": "Judge", "s_type": "Type"}[sc]
        for part, ps in [("counterfact", "Cf"), ("pooled", "Pool")]:
            out.append(
                f"\\newcommand{{\\auc{name}{ps}}}{{{rv['dev_auc'][sc][part]:.3f}}}"
            )
    with open(ROOT / "paper/figures/tradeoff.dat", "w") as fh:
        fh.write(
            "theta para_both para_adapter leak_both leak_note leak_adapter routed\n"
        )
        for c in rv["curve"]:
            fh.write(
                f"{c['theta'] or 0.6308} {c['cf_para_both']} {c['cf_para_adapter']} "
                f"{c['e3_leak_both']} {c['e3_leak_note']} {c['e3_leak_adapter']} {c['e3_routed']}\n"
            )
except FileNotFoundError as e:
    missing.append(e.filename)

# SS75 router v7 (prefix "w") and SS76 trained relation encoder (prefix "x").
for pre, tag, fn in [
    ("w", "Cf", "cf_n1000_v7_eval.json"),
    ("w", "Zs", "zsre_n1000_v7_eval.json"),
    ("x", "Cf", "cf_n1000_v8_eval.json"),
    ("x", "Zs", "zsre_n1000_v8_eval.json"),
    ("u", "Cf", "cf_n1000_v9_eval.json"),
    ("u", "Zs", "zsre_n1000_v9_eval.json"),
]:
    try:
        d = json.load(open(R + fn))
    except FileNotFoundError as e:
        missing.append(e.filename)
        continue
    sv = d["serving"]
    for kind, k in [("Eff", "probe"), ("Para", "paraphrase")]:
        emit(f"{pre}{tag}{kind}", sv[k]["adapter"], fixed=True)
        emit(f"{pre}{tag}{kind}Ret", sv[k]["retrieval"], fixed=True)
        emit(f"{pre}{tag}{kind}Both", sv[k]["both"], fixed=True)
        emit(f"{pre}{tag}{kind}Route", sv[k]["route_correct"], fixed=True)
    emit(f"{pre}{tag}ParaTier", d["tiered"]["paraphrase"], fixed=True)
    emit(f"{pre}{tag}TierFrac", d["tiered"]["adapter_fraction"], fixed=True)
    emit(f"{pre}{tag}Return", d["removal"]["return"], fixed=True)
    emit(f"{pre}{tag}Retain", d["removal"]["retention"], fixed=True)
    emit(f"{pre}{tag}NoLeak", sv["distractor"]["no_leak"], fixed=True)
    for loc, short in [
        ("E3_other_relation", "Ethree"),
        ("E2_neighborhood", "Etwo"),
        ("loc_questions", "Loc"),
    ]:
        if loc in d["locality"]:
            emit(f"{pre}{tag}{short}Routed", d["locality"][loc]["routed"], fixed=True)
            for pol, ps in [
                ("adapter", "Adapter"),
                ("note", "Note"),
                ("both", "Both"),
                ("tiered", "Tiered"),
            ]:
                emit(
                    f"{pre}{tag}{short}Leak{ps}",
                    d["locality"][loc][pol]["leak"],
                    fixed=True,
                )
try:
    sel = json.load(open(R + "router_v7_select.json"))
    out.append(f"\\newcommand{{\\wTheta}}{{{sel['theta']:.3f}}}")
except FileNotFoundError as e:
    missing.append(e.filename)
try:
    sel = json.load(open(R + "router_v8_select.json"))
    out.append(f"\\newcommand{{\\xTheta}}{{{sel['theta']:.3f}}}")
    dv = sel["configs"][sel["chosen"]]["dev"]
    emit("xDevTpr", dv["tpr_at_theta"], fixed=True)
    sp = json.load(open(R + "relenc_split.json"))
    for grp, g in [("seen", "Seen"), ("heldout", "Held")]:
        for key, k in [
            ("default_route_correct", "DefRoute"),
            ("trained_route_correct", "Route"),
            ("trained_both", "Both"),
        ]:
            emit(f"xPara{g}{k}", sp["paraphrase"][grp][key], fixed=True)
        out.append(f"\\newcommand{{\\xPara{g}N}}{{{sp['paraphrase'][grp]['n']}}}")
    for grp, g in [
        ("both_seen", "Seen"),
        ("stored_heldout", "StoredHeld"),
        ("e3_heldout", "QueryHeld"),
        ("both_heldout", "BothHeld"),
    ]:
        emit(f"xEthree{g}Def", sp["E3"][grp]["default_leak_both"], fixed=True)
        emit(f"xEthree{g}Leak", sp["E3"][grp]["trained_leak_both"], fixed=True)
        out.append(f"\\newcommand{{\\xEthree{g}N}}{{{sp['E3'][grp]['n']}}}")
except FileNotFoundError as e:
    missing.append(e.filename)
# SS77 Wikidata-trained encoders: B (prefix "u") end to end, A routing only (prefix "a").
try:
    sel = json.load(open(R + "router_v9_select.json"))
    out.append(f"\\newcommand{{\\uTheta}}{{{sel['theta']:.3f}}}")
    emit("uDevTpr", sel["configs"]["wikiB"]["dev"]["tpr_at_theta"], fixed=True)
    emit("aDevTpr", sel["configs"]["wikiA"]["dev"]["tpr_at_theta"], fixed=True)
    emit(
        "aCfParaRoute",
        sel["configs"]["wikiA"]["test"]["counterfact"]["paraphrase"]["route_correct"],
        fixed=True,
    )
    emit(
        "aCfEthreeRouted",
        sel["configs"]["wikiA"]["test"]["counterfact"]["other_relation_E3"]["routed"],
        fixed=True,
    )
    v7sel = json.load(open(R + "router_v7_select.json"))
    emit(
        "mpCfParaRoute",
        v7sel["configs"]["mpnet"]["test"]["counterfact"]["paraphrase"]["route_correct"],
        fixed=True,
    )
    emit(
        "mpCfEthreeRouted",
        v7sel["configs"]["mpnet"]["test"]["counterfact"]["other_relation_E3"]["routed"],
        fixed=True,
    )
    sp = json.load(open(R + "relenc_split_v9.json"))
    emit(
        "uParaHeldRoute",
        sp["paraphrase"]["heldout"]["trained_route_correct"],
        fixed=True,
    )
    emit(
        "uParaSeenRoute", sp["paraphrase"]["seen"]["trained_route_correct"], fixed=True
    )
    groups = [("stored_heldout", 248), ("e3_heldout", 211), ("both_heldout", 100)]

    def pooled(key, split):
        num = sum(split["E3"][g][key] * split["E3"][g]["n"] for g, _ in groups)
        return num / sum(split["E3"][g]["n"] for g, _ in groups)

    emit("uHeldLeak", pooled("trained_leak_both", sp), fixed=True)
    emit(
        "xHeldLeak",
        pooled("trained_leak_both", json.load(open(R + "relenc_split.json"))),
        fixed=True,
    )
    emit(
        "aParaHeldRoute", sp["routing_only"]["wikiA"]["paraphrase_heldout"], fixed=True
    )
    emit("aParaSeenRoute", sp["routing_only"]["wikiA"]["paraphrase_seen"], fixed=True)
    emit(
        "mpParaHeldRoute", sp["routing_only"]["mpnet"]["paraphrase_heldout"], fixed=True
    )
    emit("mpParaSeenRoute", sp["routing_only"]["mpnet"]["paraphrase_seen"], fixed=True)
except FileNotFoundError as e:
    missing.append(e.filename)

# SS79 template-disjoint split (prefix "ts"): means over the three seeds.
try:
    ts = json.load(open(R + "relenc_template_split.json"))
    for g in "ABC":
        for enc, e in [("trained", "Trained"), ("frozen_mpnet", "Frozen")]:
            row = ts["groups"][g][enc]
            emit(f"tsPara{g}{e}", sum(row["route_correct"]) / 3, fixed=True)
            emit(f"tsEthree{g}{e}", sum(row["e3_fpr"]) / 3, fixed=True)
    emit("tsRatio", ts["P1"]["ratio_B_A"])
    emit_diff("tsInflDiff", {"diff": ts["P3"]["I_diff"], "ci95": ts["P3"]["I_diff_ci95"]})
except FileNotFoundError as e:
    missing.append(e.filename)

# SS81 adapter size (prefix "as"): LoRA+ at rank 16 under the held-out check.
try:
    lp = json.load(open(R + "adapter_size_sweep.json"))["variants"]["r16_qv_all"]
    emit("asLpAbove", lp["above_kappa"])
    emit("asLpHeldMean", lp["d_heldout_mean"])
except FileNotFoundError as e:
    missing.append(e.filename)

# SS85 natural other-relation queries (prefix "nq"), automatic leak rule.
try:
    nq = json.load(open(R + "natural_e3.json"))
    d, fz = nq["summary"]["default_ss76"], nq["summary"]["frozen_mpnet"]
    out.append(f"\\newcommand{{\\nqN}}{{{nq['n_queries']}}}")
    emit("nqNearRouted", d["near"]["routed"], fixed=True)
    emit("nqUniformRouted", d["uniform"]["routed"], fixed=True)
    emit("nqLeak", d["pooled"]["leak_target_both"], fixed=True)
    emit("nqNearLeak", d["near"]["leak_target_both"], fixed=True)
    emit("nqFrozenNearRouted", fz["near"]["routed"], fixed=True)
except FileNotFoundError as e:
    missing.append(e.filename)

# SS85 with the model labels (SS90, SS91): share of sampled automatic leaks that are
# correct answers, and the corrected leak rates of the default router.
try:
    sc = json.load(open(R + "natural_e3_scored.json"))["routers"]["default_ss76"]
    out.append(f"\\newcommand{{\\nqLabelled}}{{{sc['sampled_auto_leaks']}}}")
    emit("nqLegitShare", sc["legit_share"], fixed=True)
    emit("nqLeakCorr", sc["leak_both_corrected"], fixed=True)
    emit("nqNearLeakCorr", sc["near_leak_both_corrected"], fixed=True)
except FileNotFoundError as e:
    missing.append(e.filename)

# SS87 rank-2 LoRA+ in int8 on fresh adapter facts (prefix "fr": "Ref" rank 16, "Rt" rank 2).
try:
    fr = json.load(open(R + "rank2_fresh.json"))
    out.append(f"\\newcommand{{\\frN}}{{{fr['n_adapter']}}}")
    for arm, a in [("ref16", "Ref"), ("r2_lp_int8", "Rt")]:
        sm = fr["summary"][arm]
        emit(f"fr{a}EffNote", sm["pooled"]["probe_note"], fixed=True)
        emit(f"fr{a}ParaNote", sm["pooled"]["paraphrase_note"], fixed=True)
        emit(f"fr{a}D", sm["pooled"]["d_heldout_mean"], fixed=True)
        emit(f"fr{a}Wrong", sm["wrong_route_retained"], fixed=True)
except FileNotFoundError as e:
    missing.append(e.filename)

# SS92 representation finetuning (prefix "rf"), stage 2 in int8, against SS83's rank-2
# LoRA+ adapter on the same N=50 set.
try:
    rf = json.load(open(R + "reft_screen.json"))["variants"]
    two = rf["loreft_r2_id"]["int8"]
    lp2 = json.load(open(R + "adapter_size_sweep.json"))["variants"]["r2_qv_all"]
    above = [v["int8"]["above_kappa"] for v in rf.values()]
    emit("rfTwoPara", two["paraphrase"])
    emit("rfTwoD", two["d_heldout_mean"])
    emit("rfAboveMin", min(above))
    emit("rfAboveMax", max(above))
    emit("rfInMax", max(v["int8"]["d_in_sample_max"] for v in rf.values()))
    emit("asRtwoPara", lp2["paraphrase"])
    emit("asRtwoD", lp2["d_heldout_mean"])
except FileNotFoundError as e:
    missing.append(e.filename)

# SS100 wrong routes on 1000 questions (prefix "wr").
try:
    w = json.load(open(R + "wrong_route.json"))
    emit("wrAdapter", w["retained"]["adapter_r16"], fixed=True)
    emit("wrAdapterTwo", w["retained"]["adapter_r2"], fixed=True)
    emit("wrNote", w["retained"]["note"], fixed=True)
    emit("wrBoth", w["retained"]["both_r16"], fixed=True)
    emit("wrBothTwo", w["retained"]["both_r2"], fixed=True)
    emit_diff("wrRankDiff", w["r2_minus_r16_adapter"])
    out.append(
        f"\\newcommand{{\\wrRankLow}}{{{100 * w['r2_minus_r16_adapter']['lower_95_one_sided']:.1f}}}"
    )
except FileNotFoundError as e:
    missing.append(e.filename)

# SS101 MMLU and ARC with the store active (prefix "gen").
try:
    g = json.load(open(R + "general_eval.json"))["readings"]
    emit("genBase", g["pooled"]["accuracy"]["base"], fixed=True)
    emit("genMmlu", g["mmlu"]["accuracy"]["base"], fixed=True)
    emit("genArc", g["arc"]["accuracy"]["base"], fixed=True)
    emit("genUpper", g["pooled"]["routed_counterfact"]["ci95"][1], fixed=True)
    emit("genForcedAdapter", g["pooled"]["accuracy"]["forced_adapter_r16"], fixed=True)
    emit("genForcedBoth", g["pooled"]["accuracy"]["forced_both_r2"], fixed=True)
    emit_diff("genForcedAdapterDiff", g["pooled"]["minus_base"]["forced_adapter_r16"])
    emit_diff("genForcedBothDiff", g["pooled"]["minus_base"]["forced_both_r2"])
except FileNotFoundError as e:
    missing.append(e.filename)

# SS98 the curve over base models (prefix "sc"; the arm in the macro name).
try:
    sc = json.load(open(R + "scale/summary.json"))
    names = {
        "q05": "Half",
        "q15": "OneFive",
        "q15q": "OneFiveQ",
        "q3q": "Three",
        "q7q": "Seven",
        "smol3q": "Smol",
    }
    for arm, name in names.items():
        row = sc["arms"][arm]
        emit(f"sc{name}Need", row["pooled_needs_adapter"], fixed=True)
        out.append(
            f"\\newcommand{{\\sc{name}Mem}}{{{row['memory_mb'] / 1024:.1f}}}"
        )
        for tag, ds in (("Cf", "counterfact"), ("Zs", "zsre")):
            d = row["datasets"][ds]
            emit(f"sc{name}Need{tag}", d["needs_adapter"], fixed=True)
            emit(f"sc{name}Probe{tag}", d["tiered_probe_word"], fixed=True)
            emit(f"sc{name}Para{tag}", d["tiered_paraphrase_word"], fixed=True)
            emit(f"sc{name}Note{tag}", d["note_paraphrase_word"], fixed=True)
            emit(f"sc{name}Above{tag}", d["above_kappa_share"], fixed=True)
            out.append(
                f"\\newcommand{{\\sc{name}Sec{tag}}}{{{d['train_seconds_mean']:.1f}}}"
            )
        out.append(
            f"\\newcommand{{\\sc{name}KB}}{{{row['datasets']['zsre']['adapter_bytes'] / 1000:.0f}}}"
        )
    out.append(f"\\newcommand{{\\scGap}}{{{sc['quantization_gap']:.3f}}}")
except FileNotFoundError as e:
    missing.append(e.filename)

# SS105 aliases and pronouns, routing after the fix of entry 119 (prefix "al").
try:
    al = json.load(open(R + "alias_eval.json"))
    for tag, ds in (("Cf", "counterfact"), ("Zs", "zsre")):
        d = al["datasets"][ds]
        emit(f"al{tag}Cover", al["alias_coverage"][ds]["rate"], fixed=True)
        emit(f"al{tag}ParaPlain", d["A2"]["plain"], fixed=True)
        emit(f"al{tag}ParaNamed", d["A2"]["named"], fixed=True)
        emit(f"al{tag}Pron", d["C1_probe"]["named"], fixed=True)
        emit(f"al{tag}PronPara", d["C1_paraphrase"]["named"], fixed=True)
        emit(f"al{tag}Later", d["C2"]["routed"], fixed=True)
    served = json.load(open(R + "alias_eval_served.json"))["datasets"]
    for tag, ds in (("Cf", "counterfact"), ("Zs", "zsre")):
        emit(f"al{tag}AnsAlias", served[ds]["served_both"]["A1"]["word"], fixed=True)
        emit(f"al{tag}AnsPron", served[ds]["served_both"]["C1_probe"]["word"], fixed=True)
    false = al["datasets"]["counterfact"]["false"]
    emit("alNatural", false["F4_natural"]["named"], fixed=True)
    emit("alOther", false["F1_other_relation"]["named"], fixed=True)
except FileNotFoundError as e:
    missing.append(e.filename)

# SS104 asymmetric slots (prefix "sl").
try:
    sl = json.load(open(R + "slot_eval_asym.json"))["datasets"]
    fresh = json.load(open(R + "slot_fresh.json"))
    emit("slZsReturn", sl["zsre"]["removal"]["return"]["rate"])
    out.append(f"\\newcommand{{\\slZsMerged}}{{{sl['zsre']['actions']['duplicate']}}}")
    out.append(f"\\newcommand{{\\slZsOld}}{{{sl['zsre']['removal']['old']:.2f}}}")
    emit("slZsPara", sl["zsre"]["serving"]["paraphrase_both"]["new"], fixed=True)
    emit("slCfPara", sl["counterfact"]["serving"]["paraphrase_both"]["new"], fixed=True)
    emit("slFreshPer", fresh["per_entry"]["residual_share"])
    emit("slFreshAsym", fresh["asymmetric"]["residual_share"])
    emit("slFreshUpper", fresh["asymmetric"]["upper_95_one_sided"])
except FileNotFoundError as e:
    missing.append(e.filename)

# SS110 hard negatives from a confusion matrix (prefix "hn").
try:
    hn = json.load(open(R + "hardneg_eval.json"))["readings"]
    for name, arm in (("Base", "baseline"), ("A", "hardA"), ("B", "hardB")):
        emit(f"hn{name}Near", hn[arm]["natural_near"]["rate"], fixed=True)
        emit(f"hn{name}Other", hn[arm]["e3"]["rate"], fixed=True)
        emit(f"hn{name}Para", hn[arm]["cf_paraphrase"]["rate"], fixed=True)
except FileNotFoundError as e:
    missing.append(e.filename)

# SS112 shard adapters.
try:
    s = json.load(open(R + "shard_eval.json"))
    ref = s["reference"]
    emit("shRefPara", ref["paraphrase_note"], fixed=True)
    emit("shRefLeak", ref["e3_leak"], fixed=True)
    for tag, arm in [("EightTwo", "8/2/relation"), ("RandomSixteen", "16/16/random")]:
        a = s["arms"][arm]
        emit(f"sh{tag}Para", a["paraphrase_note"], fixed=True)
        emit(f"sh{tag}Leak", a["e3_leak"], fixed=True)
        out.append(f"\\newcommand{{\\sh{tag}KB}}{{{a['bytes_per_fact'] / 1000:.0f}}}")
    emit_diff("shEightTwoLeakDiff", s["arms"]["8/2/relation"]["e3_leak_minus_ref"])
    for tag, arm in [("Eight", "8/16/relation"), ("ThirtyTwo", "32/16/relation")]:
        out.append(f"\\newcommand{{\\shRemove{tag}}}{{{s['arms'][arm]['removal']['seconds']:.0f}}}")
except FileNotFoundError as e:
    missing.append(e.filename)

# SS113 soft prompt.
try:
    s = json.load(open(R + "soft_prompt.json"))
    sf, rf = s["summary"]["soft"], s["summary"]["r2_lp_int8"]
    out.append(f"\\newcommand{{\\spKB}}{{{s['bytes_per_fact'] / 1000:.0f}}}")
    emit("spProbe", sf["pooled"]["probe"], fixed=True)
    emit("spPara", sf["pooled"]["paraphrase"], fixed=True)
    emit("spRefPara", rf["pooled"]["paraphrase"], fixed=True)
    emit("spParaNote", sf["pooled"]["paraphrase_note"], fixed=True)
    emit("spRefParaNote", rf["pooled"]["paraphrase_note"], fixed=True)
except FileNotFoundError as e:
    missing.append(e.filename)

# SS126 the adapter alone on rewrites (prefix "ao").
try:
    s = json.load(open(R + "adapter_only.json"))["summary"]
    for tag, arm in (("Ref", "A0"), ("Rew", "A1")):
        emit(f"ao{tag}CfPara", s[arm]["cf"]["paraphrase"], fixed=True)
        emit(f"ao{tag}ETh", s[arm]["e3_leak"], fixed=True)
        emit(f"ao{tag}Route", s[arm]["wrong_route_word"], fixed=True)
    emit("aoRefCfNote", s["A0"]["cf"]["paraphrase_note"], fixed=True)
except FileNotFoundError as e:
    missing.append(e.filename)

# SS106 three tiers, SS107 thresholds per relation, SS97 the score audit.
try:
    t = json.load(open(R + "tau_calib.json"))["readings"]
    emit("tcNearGlobal", t["natural_near_routed"]["global"]["rate"], fixed=True)
    emit("tcNearRel", t["natural_near_routed"]["R"]["rate"], fixed=True)
    emit_diff("tcNearDiff", t["natural_near_routed"]["R"]["minus_global"])
    emit_diff("tcParaCfDiff", t["counterfact_paraphrase_route_correct"]["R"]["minus_global"])
    emit_diff("tcParaZsDiff", t["zsre_paraphrase_route_correct"]["R"]["minus_global"])
    a = json.load(open(R + "score_audit.json"))["datasets"]
    both = [
        a[ds]["arms"][arm]["substring"] - a[ds]["arms"][arm]["word"]
        for ds in a
        for arm in ("probe_both", "paraphrase_both")
    ]
    out.append(f"\\newcommand{{\\saBothDropPct}}{{{100 * max(both):.1f}}}")
    emit("saNoteZsSub", a["zsre"]["arms"]["probe_retrieval"]["substring"], fixed=True)
    emit("saNoteZsWord", a["zsre"]["arms"]["probe_retrieval"]["word"], fixed=True)
except FileNotFoundError as e:
    missing.append(e.filename)

if missing:
    sys.exit("not writing paper/numbers.tex; missing inputs:\n" + "\n".join(missing))
(ROOT / "paper/numbers.tex").write_text("\n".join(out) + "\n")
print(json.dumps(diag, indent=1))
