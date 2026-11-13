"""Food price outlook for northeast Nigeria: interactive app.

Run with:  streamlit run app/streamlit_app.py
"""
import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import AQUA, BLUE, ORANGE, bar, lines, load, show, style

st.set_page_config(page_title="Food price outlook, northeast Nigeria", layout="wide")
s = load("summary.json")
months = s["months"]
dates = [f"{m}-15" for m in months]
series = {(x["commodity"], x["state"]): x for x in s["series"]}
bands = {(b["category"], b["horizon"]): b for b in s["bands"]}
commodities = sorted({c for c, _ in series})
latest_month = pd.Timestamp(f"{s['latest_month']}-15")


def latest_price(item):
    return next(v for v in reversed(item["prices"]) if v is not None)


def month_name(k):
    return (latest_month + pd.DateOffset(months=k)).strftime("%B %Y")


st.title("What will food cost in northeast Nigeria in three months?")
st.caption("Monthly market prices for 23 foods in Borno and Yobe from World Food Programme monitoring, a three-month outlook with "
           "uncertainty bands tested on real history, and a planner that turns the outlook into a cash transfer value.")

shock = s["shock"]
rice = s["rice_kg_per_day"]
rice_2019 = [v for v, m in zip(rice, months) if m.startswith("2019")]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Price records", f"{s['records']:,}", f"{s['markets']} markets", delta_color="off")
c2.metric("Staple prices, May 2023 to peak", f"+{shock['peak'] / shock['may_2023'] - 1:.0%}")
c3.metric("Since the peak", f"{shock['latest'] / shock['peak'] - 1:.0%}")
c4.metric("Rice per day of minimum wage", f"{rice[-1]:.1f} kg", f"{sum(rice_2019) / len(rice_2019):.1f} kg in 2019", delta_color="off")

tab_outlook, tab_plan, tab_shock, tab_test = st.tabs(["Price outlook", "Cash transfer planner", "The price shock", "Does ML beat no change?"])

with tab_outlook:
    a, b, c = st.columns(3)
    commodity = a.selectbox("Food", commodities, index=commodities.index("Rice (local)"))
    state = b.selectbox("State", ["Borno", "Yobe"])
    span = c.selectbox("History", ["Last 3 years", "Last 5 years", "All"])
    item = series[(commodity, state)]
    start = max(0, len(months) - {"Last 3 years": 36, "Last 5 years": 60, "All": 999}[span])
    now = latest_price(item)
    future = [latest_month + pd.DateOffset(months=k) for k in range(4)]
    low = [now] + [now * math.exp(bands[(item["category"], h)]["low"]) for h in (1, 2, 3)]
    high = [now] + [now * math.exp(bands[(item["category"], h)]["high"]) for h in (1, 2, 3)]
    figure = go.Figure()
    figure.add_scatter(x=dates[start:], y=item["prices"][start:], mode="lines", name="Observed price", line=dict(color=BLUE, width=2))
    figure.add_scatter(x=future, y=high, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
    figure.add_scatter(x=future, y=low, mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(235,104,52,0.2)", name="80% band")
    figure.add_scatter(x=future, y=[now] * 4, mode="lines", name="Outlook (no change)", line=dict(color=ORANGE, width=2, dash="dash"))
    figure.update_yaxes(title=f"Naira per {item['unit']}", rangemode="tozero")
    show(style(figure, 440))
    st.write(f"**{commodity} in {state}:** ₦{now:,.0f} per {item['unit']} in {month_name(0)}. By {month_name(3)} the price is likely to be between "
             f"₦{low[3]:,.0f} and ₦{high[3]:,.0f}, based on how {item['category']} prices actually moved over three months in the backtest.")

with tab_plan:
    st.caption("Set a monthly food basket for one person. The planner prices it today and at the upper band of the outlook, "
               "the amount that would have been enough in about nine months out of ten for each item.")
    a, b = st.columns(2)
    plan_state = a.selectbox("State", ["Borno", "Yobe"], key="plan_state")
    horizon = b.selectbox("Planning horizon (months)", [1, 2, 3], index=2)
    items = [c for c in ["Rice (local)", "Millet", "Sorghum", "Maize flour", "Beans (white)", "Oil (palm)", "Onions", "Fish"] if c in commodities]
    columns = st.columns(4)
    cost_now = cost_high = 0.0
    for i, name in enumerate(items):
        item = series[(name, plan_state)]
        quantity = columns[i % 4].number_input(f"{name} ({item['unit']})", 0, 60, int(s["default_basket"].get(name, 0)), key=f"qty_{name}")
        price = latest_price(item)
        cost_now += quantity * price
        cost_high += quantity * price * math.exp(bands[(item["category"], horizon)]["high"])
    st.divider()
    left, right = st.columns(2)
    left.metric("Basket cost today", f"₦{cost_now:,.0f}")
    right.metric(f"Plan for, in {month_name(horizon)}", f"₦{cost_high:,.0f}", f"+{cost_high / cost_now - 1:.0%} buffer" if cost_now else None, delta_color="off")
    st.write("Items rarely all reach their upper band together, so this is a cautious figure.")

with tab_shock:
    st.caption("In mid 2023 Nigeria removed its petrol subsidy and floated the naira. Staple prices in these markets nearly tripled in fifteen months.")
    base_index = months.index("2019-01")
    left, right = st.columns(2)
    with left:
        st.markdown("**Staple food price index** (January 2019 = 100)")
        show(lines([("Staple index", dates, s["basket"], BLUE, "solid")], "", "Index"))
        st.markdown("**Local rice bought by a day of minimum wage** (kg)")
        show(lines([("Kilograms of rice", dates, rice, BLUE, "solid")], "", "Kilograms"))
    with right:
        st.markdown("**Exchange rate and petrol price** (January 2019 = 100)")
        show(lines([("Naira per US dollar", dates, [v / s["fx"][base_index] * 100 for v in s["fx"]], BLUE, "solid"),
                    ("Petrol price", dates, [v / s["petrol"][base_index] * 100 for v in s["petrol"]], ORANGE, "solid")], "", "Index"))
        st.markdown("**Seasonal pattern in cereal prices** (average monthly change, 2017 to 2022)")
        show(bar(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], s["seasonal"], "", "Change on previous month (%)"))

with tab_test:
    st.caption(f"A gradient boosted model was tested against two simple rules on {s['backtest_forecasts']:,} forecasts it had never seen. "
               "It did not win, so the outlook uses the simple rule. Lower error is better.")
    left, right = st.columns(2)
    with left:
        st.markdown("**Forecast error by horizon** (mean absolute percentage error)")
        horizons = [f"{r['horizon (months)']} month(s)" for r in s["accuracy"]]
        figure = go.Figure([go.Bar(x=horizons, y=[r[key] for r in s["accuracy"]], name=name, marker_color=color)
                            for key, name, color in [("No change MAPE", "No change", BLUE), ("Gradient boosting MAPE", "Gradient boosting", ORANGE), ("Drift MAPE", "Drift", AQUA)]])
        figure.update_yaxes(title="Error (%)")
        show(style(figure, barmode="group"))
        st.markdown("**Do the bands hold up?** Built on the first half of the backtest, checked on the second. Target: 80% inside.")
        coverage = pd.DataFrame(s["coverage"])
        st.dataframe(coverage.style.format({"coverage": "{:.1%}", "below band": "{:.1%}", "above band": "{:.1%}"}), hide_index=True, width="stretch")
    with right:
        st.markdown("**Three-month error by food**")
        rows = s["by_commodity"][::-1]
        figure = go.Figure([go.Bar(y=[r["commodity"] for r in rows], x=[r["No change"] for r in rows], name="No change", orientation="h", marker_color=BLUE),
                            go.Bar(y=[r["commodity"] for r in rows], x=[r["Gradient boosting"] for r in rows], name="Gradient boosting", orientation="h", marker_color=ORANGE)])
        figure.update_xaxes(title="Error (%)")
        show(style(figure, 640, barmode="group"))
