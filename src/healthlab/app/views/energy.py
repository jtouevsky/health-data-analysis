"""Energy Balance (optional module): empirical TDEE vs Apple's estimate."""

import plotly.graph_objects as go
import streamlit as st

from healthlab.analysis.energy import (ESTIMATORS, EnergyParams, expected_trajectory, has_energy_data,
                                       implied_daily_discrepancy, summarize, window_tdee)
from healthlab.app.common import (SERIES, MUTED, cfg, optional_module_notice, require_data, style,
                                  to_display)

st.title("Energy Balance")
daily, catalog = require_data()
c = cfg()
ec, tc = c.section("energy"), c.section("trends")
intake = c.section("nutrition").get("energy_metric", "intake_kcal")

if not has_energy_data(daily, intake):
    optional_module_notice("Energy balance", "well-logged food intake (dietary energy) and body-weight readings "
                           "in the same period")
    st.stop()
if not {"active_kcal", "basal_kcal"} <= set(daily.columns):
    st.warning("Apple active/basal energy is missing, so the Apple comparison will be empty.")

st.markdown(f"""
Over any window of *W* days, energy balance says **mean intake − TDEE = Δweight per day × energy density**.
So **TDEE = mean intake − (Δweight per day × {ec['kcal_per_kg']:,} kcal/kg)**. Only **well-logged** days count toward
intake. A window needs ≥ {ec['min_logged_days']} logged days and ≥ {ec['min_weigh_ins']} weigh-ins.
""")

c1, c2, c3, c4 = st.columns(4)
window = c1.select_slider("Window (days)", [14, 21, 28, 42, 56], value=int(ec["window_days"]))
estimators = list(ESTIMATORS)
est = c2.selectbox("Weight-change estimator", estimators, index=estimators.index(ec.get("estimator", "ols")),
                   format_func=lambda k: ESTIMATORS[k])
kpk = c3.number_input("Energy density (kcal/kg)", 5000, 9500, int(ec["kcal_per_kg"]), step=100,
                      help="7700 kcal/kg is the classic adipose-tissue figure. Early in a diet, glycogen and water "
                           "losses make the effective value lower.")
under = c4.slider("Assumed intake under-reporting", 0, 40, 0, step=5, format="%d%%",
                  help="Logged intake is scaled up by 1 / (1 − u).") / 100

p = EnergyParams.from_config(ec, tc, window_days=window, estimator=est, kcal_per_kg=float(kpk), underreport=under)


@st.cache_data(show_spinner="Fitting windows…")
def compute(df, p):
    return window_tdee(df, p, intake=intake)


win = compute(daily, p)
if win.empty:
    st.info("No window meets the coverage requirements. Try a longer window.")
    st.stop()
s = summarize(win, p)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Empirical TDEE", f"{s['tdee_mean']:,.0f} kcal/d", help=f"Mean over {s['n_windows']} non-overlapping windows")
k2.metric("Apple active + basal", f"{s['apple_mean']:,.0f} kcal/d")
if s.get("n_gap"):
    ci = s.get("gap_ci")
    k3.metric("Gap (empirical − Apple)", f"{s['gap_mean']:+,.0f} kcal/d",
              help=f"95% CI of the mean gap: [{ci[0]:+,.0f}, {ci[1]:+,.0f}]" if ci else None)
    k4.metric("Ratio empirical / Apple", f"{s['ratio']:.2f}")

fig = go.Figure()
fig.add_scatter(x=win.index, y=win["tdee_hi"], line=dict(width=0), showlegend=False, hoverinfo="skip")
fig.add_scatter(x=win.index, y=win["tdee_lo"], fill="tonexty", line=dict(width=0), name="Empirical 95% CI",
                fillcolor="rgba(42,120,214,0.18)", hoverinfo="skip")
fig.add_scatter(x=win.index, y=win["tdee"], name="Empirical TDEE", line=dict(color=SERIES[0], width=2))
fig.add_scatter(x=win.index, y=win["apple_tdee"], name="Apple active + basal", line=dict(color=SERIES[1], width=2))
nw = s["windows"]
fig.add_scatter(x=nw.index, y=nw["tdee"], mode="markers", name="Non-overlapping windows",
                marker=dict(size=9, color=SERIES[0], line=dict(width=2, color="white")))
fig.update_yaxes(title="kcal / day")
st.plotly_chart(style(fig, title=f"TDEE over rolling {window}-day windows (window end date)"), width="stretch")
st.caption("Adjacent rolling windows share most of their days, so the line is smooth by construction. "
           "Summary statistics use only the non-overlapping windows (dots).")

# -- Gap stability ---------------------------------------------------------------------------
st.subheader("How stable is the gap to Apple's estimate?")
if s.get("n_gap", 0) >= 2:
    fig = go.Figure()
    fig.add_bar(x=nw.index, y=nw["gap"], marker_color=[SERIES[0] if g >= 0 else SERIES[7] for g in nw["gap"].fillna(0)],
                name="Gap", hovertemplate="%{x|%d %b %Y}: %{y:+.0f} kcal/d<extra></extra>")
    fig.add_hline(y=s["gap_mean"], line_dash="dash", line_color=MUTED)
    fig.update_yaxes(title="empirical − Apple (kcal/d)")
    st.plotly_chart(style(fig, title="Gap per non-overlapping window", hovermode="closest"), width="stretch")
    msg = f"Gap SD across windows: **{s['gap_sd']:.0f} kcal/d** (n = {s['n_gap']} windows)."
    if "gap_drift_per_90d" in s:
        msg += (f" Linear drift: **{s['gap_drift_per_90d']:+.0f} kcal/d per 90 days** (p = {s['gap_drift_p']:.2g}). "
                + ("No evidence that the gap changes over time." if s["gap_drift_p"] > 0.05
                   else "The gap changes over time, so a single correction factor for Apple's estimate would be misleading."))
    st.markdown(msg)
else:
    st.info("Need at least two non-overlapping windows with Apple data.")

# -- Trajectory ------------------------------------------------------------------------------
st.subheader("Expected vs actual weight")
traj = expected_trajectory(daily, p, logging_error_sd=float(ec["logging_error_sd"]),
                           apple_bias_sd=float(ec["apple_bias_sd"]), n_sims=int(ec.get("trajectory_sims", 400)),
                           intake=intake)
if not traj.empty:
    conv = lambda v: to_display(v, "weight_kg", catalog)[0]
    unit = to_display(1.0, "weight_kg", catalog)[1]
    fig = go.Figure()
    fig.add_scatter(x=traj.index, y=conv(traj["hi"]), line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=traj.index, y=conv(traj["lo"]), fill="tonexty", line=dict(width=0),
                    fillcolor="rgba(235,104,52,0.18)", name="Expected, 90% band", hoverinfo="skip")
    fig.add_scatter(x=traj.index, y=conv(traj["median"]), name="Expected from logged intake − Apple expenditure",
                    line=dict(color=SERIES[1], width=2, dash="dash"))
    fig.add_scatter(x=traj.index, y=conv(traj["weight"]), mode="markers", name="Weigh-ins",
                    marker=dict(size=4, color=MUTED))
    fig.add_scatter(x=traj.index, y=conv(traj["trend"]), name="Actual trend (EWMA)", line=dict(color=SERIES[0], width=2))
    fig.update_yaxes(title=unit)
    st.plotly_chart(style(fig, title="If the food log and Apple's expenditure were both right…", height=420),
                    width="stretch")
    disc = implied_daily_discrepancy(traj, p.kcal_per_kg)
    st.markdown(
        f"The expected path simulates daily balance from logged intake (unlogged days imputed from nearby "
        f"logged days) minus Apple's active + basal energy. The band reflects ±{ec['logging_error_sd']:.0%} per-day "
        f"logging error and ±{ec['apple_bias_sd']:.0%} systematic error in Apple's estimate. "
        f"Over the whole period the expected and actual trends diverge by **{disc:+.0f} kcal/day**: "
        + ("the model predicts more weight loss than happened, so either intake is under-logged or Apple "
           "over-estimates expenditure (most likely both)." if disc < 0 else
           "the model predicts more weight gain than happened, so Apple may be under-estimating expenditure, "
           "or logged portions are overestimated."))

with st.expander("Why under-reporting matters, and what this analysis can and cannot tell you", expanded=True):
    st.markdown(f"""
- **Under-reporting is the dominant bias.** Doubly-labelled-water studies find self-reported intake typically
  10–30% below true intake. Cooking oils, drinks, bites and portion underestimates are the usual culprits. Even
  "well-logged" days only pass a completeness screen, not an accuracy check.
- **Direction of the bias:** if you log 80% of what you eat, empirical TDEE comes out **20% of intake too low**.
  A negative gap to Apple can therefore mean Apple overestimates, you under-log, or both. *This data can't
  separate the two.* The under-reporting slider shows how much of the gap a given rate would explain.
- **Energy density isn't constant.** 7,700 kcal/kg fits fat tissue. Early in a deficit, glycogen and water loss
  carry far fewer kcal per kg, so short windows right after a diet change overstate the deficit.
- **Weight noise.** Daily weight swings of ±1 kg (water, sodium, gut content) swamp a 28-day change of 1–2 kg.
  That's why the window needs several weigh-ins and the CI includes slope uncertainty.
- **Apple's numbers are estimates too.** Without a Watch, active energy is inferred from steps and motion only.
  The iPhone basal estimate only accrues while the phone records (see the Correlation Scan artifact note).
""")
