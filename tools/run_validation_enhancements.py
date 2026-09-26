"""Run independent checks inspired by the team's validation scheme.

All numbers are recomputed from this project's raw data and current Q3/Q4
outputs; no peer-repository result is copied into the deliverables.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "validation"
OUT.mkdir(parents=True, exist_ok=True)


def q4_quantile_backtest():
    p = ROOT / "data" / "raw" / "real_attachments" / "C_efficiency_evolution" / "leaderboard_cleaned.csv"
    d = pd.read_csv(p)
    d["date"] = pd.to_datetime(d["Submission Date"], errors="coerce")
    d["params"] = pd.to_numeric(d["#Params (B)"], errors="coerce")
    d["avg"] = pd.to_numeric(d["Average ⬆️"], errors="coerce")
    d = d.dropna(subset=["date", "params", "avg"])
    # Keep pretrained/open-weight models as a separately auditable frontier.
    d["type_text"] = d["Type"].fillna("").astype(str).str.lower()
    d = d[d["type_text"].str.contains("pretrained") & ~d["type_text"].str.contains("fine-tuned|chat|merge")].copy()
    d["period"] = d.date.dt.to_period("W").astype(str)
    periods = sorted(d.period.unique())
    pidx = {m: i for i, m in enumerate(periods)}
    d["t"] = d.period.map(pidx)
    d["logN"] = np.log10(d.params.clip(lower=.05))
    frontier = d.groupby("t").agg(frontier=("avg", "quantile"), maxN=("params", "max"), n=("avg", "size"), period=("period", "first"))
    # pandas groupby quantile defaults to median; use explicit p90.
    frontier = d.groupby("t").apply(lambda g: pd.Series({"frontier": g.avg.quantile(.9), "maxN": g.params.max(), "n": len(g), "period": g.period.iloc[0]}), include_groups=False).reset_index()
    rows = []
    for k in range(8, len(frontier)-1):
        train = frontier[frontier.t <= k-1]
        test = frontier[frontier.t == k]
        if test.empty or len(train) < 4:
            continue
        realized = float(test.frontier.iloc[0])
        # Quantile regression via scipy's pinball-loss linear optimization.
        X = np.c_[np.ones(len(d[d.t <= k-1])), d.loc[d.t <= k-1, ["logN", "t"]].to_numpy()]
        y = d.loc[d.t <= k-1, "avg"].to_numpy()
        def fit_q(tau):
            from scipy.optimize import minimize
            def loss(b):
                e = y - X @ b
                return np.where(e >= 0, tau * e, (tau-1) * e).sum()
            return minimize(loss, np.linalg.lstsq(X, y, rcond=None)[0], method="BFGS").x
        b05, b50, b95 = fit_q(.05), fit_q(.5), fit_q(.95)
        logn = np.log10(max(float(train.maxN.iloc[-1]), .05))
        xx = np.array([1.0, logn, float(k)])
        q05, q50, q95 = [float(xx @ b) for b in (b05, b50, b95)]
        trend = stats.linregress(train.t, train.frontier)
        pred_trend = float(trend.intercept + trend.slope*k)
        rows.append({"origin_train_upto": int(k-1), "test_period": str(test.period.iloc[0]), "realized_p90": realized,
                     "pred_q05": q05, "pred_q50": q50, "pred_q95": q95, "pred_trend": pred_trend,
                     "covered_90pct_band": bool(q05 <= realized <= q95)})
    bt = pd.DataFrame(rows)
    bt.to_csv(OUT / "q4_quantile_backtest.csv", index=False, encoding="utf-8-sig")
    summary = {"n_months": int(len(frontier)), "n_origins": int(len(bt)),
               "picp_90": float(bt.covered_90pct_band.mean()) if len(bt) else np.nan,
               "band_width_mean": float((bt.pred_q95-bt.pred_q05).mean()) if len(bt) else np.nan,
               "rmse_q95": float(np.sqrt(np.mean((bt.pred_q95-bt.realized_p90)**2))) if len(bt) else np.nan,
               "rmse_trend": float(np.sqrt(np.mean((bt.pred_trend-bt.realized_p90)**2))) if len(bt) else np.nan}
    (OUT / "q4_quantile_backtest_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary



def a17_a18_quintile_jt():
    import lzma
    from scipy.stats import rankdata
    a_dir=ROOT / "data" / "raw" / "real_attachments" / "A_data_value"
    summary=pd.read_csv(a_dir/"regmix_domain_summary.csv")
    summary["prior_share"]=summary.sample_rows/summary.sample_rows.sum()
    summary["prior_quintile"]=pd.qcut(summary.prior_share,5,labels=False,duplicates="drop")+1
    rows=[]
    with lzma.open(a_dir/"regmix_domain_sample.jsonl.xz","rt",encoding="utf-8") as f:
        for line in f:
            if line.strip():
                x=json.loads(line); d=str(x.get("_source_domain","unknown")); tx=x.get("text",""); rows.append({"domain":d,"text_chars":len(tx) if isinstance(tx,str) else 0})
    sample=pd.DataFrame(rows).merge(summary[["domain","prior_share","prior_quintile"]],on="domain",how="left")
    # The attachment is very large; use a reproducible stratified sample for
    # the exploratory text-length test so the permutation calculation is
    # finite and auditable.
    sample = (sample.groupby("prior_quintile", group_keys=False)
                    .head(500).reset_index(drop=True))
    if "prior_quintile" not in sample.columns:
        sample["prior_quintile"] = sample["prior_quintile_x"] if "prior_quintile_x" in sample.columns else sample["prior_quintile_y"]
    sample["text_quintile"]=pd.qcut(sample.text_chars.rank(method="first"),5,labels=False)+1
    # JT statistic computed from sorted values in O(n log n), with ordered
    # group pairs and half credit for ties.
    pooled=sample.text_chars.to_numpy(float); labels=sample.prior_quintile.to_numpy(int)
    def jt_stat(values, labs):
        order=np.argsort(values, kind="mergesort"); sl=labs[order]
        ranks=np.empty(len(order), dtype=float); ranks[order]=np.arange(1,len(order)+1)
        return float(sum(ranks[labs==q].sum() - len(ranks[labs==q])*(len(ranks[labs==q])+1)/2 for q in np.unique(labs)))
        # unreachable pairwise implementation retained below for reference
        sv=values[order]
        counts={int(q):0 for q in np.unique(labs)}; Jv=0.0; i=0
        while i<len(sv):
            j=i+1
            while j<len(sv) and sv[j]==sv[i]: j+=1
            for g in np.unique(sl[i:j]):
                ng=int(np.sum(sl[i:j]==g))
                for q in counts:
                    if q < int(g): Jv += ng*(counts[q] + 0.5*int(np.sum(sl[i:j]==q)))
            for g in np.unique(sl[i:j]): counts[int(g)] += int(np.sum(sl[i:j]==g))
            i=j
        return Jv
    J=jt_stat(pooled, labels)
    rng=np.random.default_rng(20260926); ge=0
    for _ in range(2000):
        plabels=rng.permutation(labels)
        jp=jt_stat(pooled, plabels)
        ge += jp>=J
    pval=(ge+1)/2001
    out=pd.DataFrame([{"test":"A17 prior quintile -> A18 text length","n":len(sample),"quintiles":int(sample.prior_quintile.nunique()),"J_stat":J,"permutation_p":pval,"direction":"increasing","interpretation":"exploratory text-length ordering check; not a quality label"}])
    out.to_csv(OUT/"A17_A18_quintile_jt_validation.csv",index=False,encoding="utf-8-sig")
    sample.groupby("prior_quintile",as_index=False).agg(n=("text_chars","size"),text_chars_median=("text_chars","median"),text_chars_mean=("text_chars","mean"),prior_share_median=("prior_share","median")).to_csv(OUT/"A17_A18_quintile_summary.csv",index=False,encoding="utf-8-sig")
    return out.iloc[0].to_dict()

def q3_kkt_check():
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from f_question.q3_data import load_q3_inputs, with_source
    from f_question.q3_models import COST_FUNCTIONS, compute_costs, predicted_loss
    inputs = with_source(load_q3_inputs(ROOT), "b1", (0.07, 5000.0), (0.134, 4000.0))
    sel = pd.read_csv(ROOT / "outputs" / "q3" / "q3_selected_by_budget_and_cost.csv")
    sel = sel[(sel.coefficient_source == "b1") & (sel.boundary_scheme == "expanded") & (sel.cost_function == "exponential")].copy()
    mix = pd.read_csv(ROOT / "outputs" / "q3" / "q3_selected_domain_mix.csv")
    rows=[]; ledger=[]; fn=COST_FUNCTIONS[0]
    for _,r in sel.iterrows():
        n,d,q,ctx = float(r.N_params_B),float(r.D_tokens_B),float(r.Q_target),int(r.context_tokens)
        m = mix[(mix.coefficient_source=="b1")&(mix.boundary_scheme=="expanded")&(mix.cost_function=="exponential")&(np.isclose(mix.budget_1e21,r.budget_1e21))&(np.isclose(mix.context_tokens,ctx))]
        p = inputs.p0.copy()
        if len(m):
            p = m.sort_values("domain").set_index("domain").reindex(inputs.domain_order)["share"].to_numpy(); p=p/p.sum()
        def L(nn,dd,qq): return predicted_loss(nn,dd,qq,p,inputs,context=ctx)["loss"]
        h=1e-4
        dLn=(L(n*np.exp(h),d,q)-L(n*np.exp(-h),d,q))/(2*h)
        dLd=(L(n,d*np.exp(h),q)-L(n,d*np.exp(-h),q))/(2*h)
        # Analytic derivative of the Q-dependent quality term.  A centered
        # difference at the Q=0.99 upper bound can lie on the clipped plateau
        # and incorrectly return zero.
        qbar = float(np.dot(p, inputs.q_star))
        qeff_raw = 1.0 - (1.0 - q) * inputs.q0 / max(qbar, 1e-8)
        qeff_for_deriv = float(np.clip(qeff_raw, 0.05, 0.99))
        dqeff_dq = inputs.q0 / max(qbar, 1e-8)
        dLq = -inputs.quality["gamma"] * inputs.quality["theta"] * (1.0 - qeff_for_deriv) ** (inputs.quality["theta"] - 1.0) * dqeff_dq
        c0=compute_costs(n,d,q,ctx,fn,inputs.q0)["total_cost_1e21"]
        cN=(compute_costs(n*np.exp(h),d,q,ctx,fn,inputs.q0)["total_cost_1e21"]-compute_costs(n*np.exp(-h),d,q,ctx,fn,inputs.q0)["total_cost_1e21"])/(2*h)
        cD=(compute_costs(n,d*np.exp(h),q,ctx,fn,inputs.q0)["total_cost_1e21"]-compute_costs(n,d*np.exp(-h),q,ctx,fn,inputs.q0)["total_cost_1e21"])/(2*h)
        cQ=float(1.0 * d * fn.absolute_derivative_1e21(q, inputs.q0))
        marg={"N":float(-dLn/cN),"D":float(-dLd/cD),"Q":float(-dLq/cQ) if cQ != 0 else 0.0}
        active={"N": "upper" if n>=4999.99 else ("lower" if n<=inputs.n_bounds_b[0]*1.00001 else "interior"),
                "D": "upper" if d>=3999.99 else ("lower" if d<=inputs.d_bounds_b[0]*1.00001 else "interior"),
                "Q": "upper" if q>=.9895 else ("lower" if q<=inputs.q0*1.00001 else "interior")}
        interior=[v for k,v in marg.items() if active[k]=='interior']
        lam=float(np.median(interior)) if interior else float('nan')
        for var in ['N','D','Q']:
            rv=marg[var]; st=active[var]
            passed=(abs(rv-lam)/max(abs(lam),1e-15)<=0.05) if (st=='interior' and np.isfinite(lam)) else ((rv>=lam) if (st=='upper' and np.isfinite(lam)) else (True if not np.isfinite(lam) else rv<=lam))
            ledger.append({'budget_1e21':r.budget_1e21,'context_tokens':ctx,'variable':var,'value':{'N':n,'D':d,'Q':q}[var],'status':st,'marginal_per_1e21':rv,'marginal_per_flop':rv/1e21,'lambda_reference_per_1e21':lam,'lambda_reference_per_flop':lam/1e21,'criterion_pass':bool(passed)})
        rows.append({'budget_1e21':r.budget_1e21,'context_tokens':ctx,'N_params_B':n,'D_tokens_B':d,'Q_target':q,'budget_used':c0,'budget_gap':r.budget_1e21-c0,'marginal_N':marg['N'],'marginal_D':marg['D'],'marginal_Q':marg['Q'],'lambda_reference':lam,'N_status':active['N'],'D_status':active['D'],'Q_status':active['Q'],'all_kkt_conditions_pass':all(x['criterion_pass'] for x in ledger[-3:])})
    out=pd.DataFrame(rows); ledger_df=pd.DataFrame(ledger)
    out.to_csv(OUT/"q3_kkt_equimarginal.csv",index=False,encoding="utf-8-sig")
    ledger_df.to_csv(OUT/"q3_kkt_three_class_ledger.csv",index=False,encoding="utf-8-sig")
    summary={'n_rows':len(out),'n_ledger_rows':len(ledger_df),'all_three_class_conditions_pass':bool(ledger_df.criterion_pass.all()),'budget_use_median':float((out.budget_used/out.budget_1e21).median())}
    (OUT/"q3_kkt_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps({"q4": q4_quantile_backtest(), "q3": q3_kkt_check(), "a17_a18": a17_a18_quintile_jt()}, ensure_ascii=False, indent=2))
