"""Seasonal Comparison: distributions per metric between labelled periods."""

import calendar

import plotly.graph_objects as go
import streamlit as st

from healthlab.analysis.seasonal import CONFOUNDS, compare_periods, label_days, month_periods
from healthlab.app.common import SERIES, analysable_metrics, cfg, label, require_data, style, to_display

st.title("Seasonal Comparison")
daily, catalog = require_data()
c = cfg()
sc = c.section("seasonal")
metrics = analysable_metrics(daily, catalog)

configured = {k: v for k, v in c.periods.items() if v}
modes = (["Configured periods"] if len(configured) >= 2 else []) + ["Calendar months"]
mode = st.radio("Define periods by", modes, horizontal=True)
if len(configured) < 2:
    st.caption("Tip: add each Bulgaria stay as a labelled `time.overrides` entry (it sets the time zone *and* "
               "becomes the `bulgaria_summer` period) and `school_year` ranges under `seasonal.periods` in "
               "`config/healthlab.yaml`. Until then, compare calendar months.")

if mode == "Configured periods":
    try:
        labels = label_days(daily.index, configured)
    except ValueError as e:
        st.error(f"Period configuration error: {e}")
        st.stop()
    names = list(configured)
    c1, c2 = st.columns(2)
    a = c1.selectbox("Period A", names, index=0)
    b = c2.selectbox("Period B", [n for n in names if n != a], index=0)
    with st.expander("Date ranges"):
        for n in (a, b):
            st.markdown(f"**{n}**: " + ", ".join(f"{s} → {e}" for s, e in configured[n]))
else:
    months = st.multiselect("Period A months (B = all other months)", list(range(1, 13)), default=[6, 7, 8],
                            format_func=lambda m: calendar.month_name[m])
    if not months:
        st.stop()
    a, b = "A: " + ", ".join(calendar.month_abbr[m] for m in months), "B: other months"
    labels = month_periods(daily.index, months, a, b)

c3, c4 = st.columns(2)
unit = c3.radio("Unit of analysis", ["day", "week"], horizontal=True,
                help="Weekly means (weeks with ≥ 4 observed days) are fewer but much closer to independent.")
alpha = c4.number_input("FDR level", 0.001, 0.25, 0.05, step=0.01, format="%.3f")


@st.cache_data(show_spinner="Bootstrapping…")
def run(df, ms, labels, a, b, unit):
    return compare_periods(df, list(ms), labels, a, b, unit=unit, reps=int(sc["bootstrap_reps"]),
                           block=int(sc["block_days"]), seed=int(sc["seed"]))


res = run(daily, tuple(metrics), labels, a, b, unit)
if res.empty:
    st.info("Not enough observed days in both periods for any metric.")
    st.stop()

st.markdown(f"**A = {a}** ({int((labels == a).sum())} days) vs **B = {b}** ({int((labels == b).sum())} days). "
            "Median difference is A − B. Sorted by |Cliff's δ|.")
show = res.copy()
for col in ["median_a", "median_b", "diff_median", "ci_lo", "ci_hi", "ci_block_lo", "ci_block_hi"]:
    show[col] = [to_display(v, m, catalog)[0] for v, m in zip(show[col], show["metric"])]
show["metric"] = show["metric"].map(lambda m: label(m, catalog))
show["95% CI (iid)"] = [f"[{lo:.2f}, {hi:.2f}]" for lo, hi in zip(show["ci_lo"], show["ci_hi"])]
show["95% CI (7-day blocks)"] = [f"[{lo:.2f}, {hi:.2f}]" for lo, hi in zip(show["ci_block_lo"], show["ci_block_hi"])]
show["discovery"] = show["q"] < alpha
st.dataframe(
    show[["metric", "n_a", "n_b", "median_a", "median_b", "diff_median", "95% CI (iid)", "95% CI (7-day blocks)",
          "cliffs_delta", "magnitude", "p", "q", "discovery"]],
    hide_index=True, width="stretch",
    column_config={"median_a": st.column_config.NumberColumn("median A", format="%.2f"),
                   "median_b": st.column_config.NumberColumn("median B", format="%.2f"),
                   "diff_median": st.column_config.NumberColumn("Δ median (A−B)", format="%.2f"),
                   "cliffs_delta": st.column_config.NumberColumn("Cliff's δ", format="%.2f"),
                   "p": st.column_config.NumberColumn("MWU p", format="%.2e"),
                   "q": st.column_config.NumberColumn("BH q", format="%.2e")})
st.caption("Cliff's δ = P(A > B) − P(A < B), which equals the rank-biserial correlation of the Mann–Whitney test. "
           "|δ| < 0.15 negligible, < 0.33 small, < 0.47 medium, larger = large. "
           "The block-bootstrap CI resamples 7-day blocks to respect day-to-day dependence, so it's usually "
           "wider and more honest. When unit = day, MWU p treats days as independent and is optimistic.")

# -- Effect-size chart -----------------------------------------------------------------
fig = go.Figure()
fig.add_bar(y=show["metric"], x=res["cliffs_delta"], orientation="h",
            marker_color=[SERIES[0] if d >= 0 else SERIES[7] for d in res["cliffs_delta"]],
            hovertemplate="%{y}: δ = %{x:.2f}<extra></extra>")
fig.add_vline(x=0, line_width=1, line_color="gray")
for x in (-0.147, 0.147):
    fig.add_vline(x=x, line_width=1, line_dash="dot", line_color="gray")
fig.update_layout(xaxis_range=[-1, 1], xaxis_title=f"Cliff's δ (positive = higher in {a})", yaxis_autorange="reversed")
st.plotly_chart(style(fig, height=80 + 32 * len(res), title="Effect sizes (dotted = negligible band)",
                      hovermode="closest"), width="stretch")

# -- Distribution drill-down ---------------------------------------------------------------
m = st.selectbox("Distribution for", res["metric"].tolist(), format_func=lambda x: label(x, catalog))
fig = go.Figure()
for name, col in ((a, SERIES[0]), (b, SERIES[1])):
    v, unit_lbl = to_display(daily.loc[labels == name, m].dropna(), m, catalog)
    fig.add_violin(x=[name] * len(v), y=v, name=name, line_color=col, box_visible=True, meanline_visible=True,
                   points="outliers", spanmode="hard")
st.plotly_chart(style(fig, title=label(m, catalog), hovermode="closest"), width="stretch")

st.markdown(CONFOUNDS)
