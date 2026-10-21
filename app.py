import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf
from dash import Dash, Input, Output, State, dcc, html, no_update
import plotly.express as px
import plotly.graph_objects as go


TICKERS = ["AAPL", "MSFT", "GOOGL", "JPM",
           "GS", "NVDA", "AMZN", "BLK"]

NAVY = "#091321"
PANEL = "#111f32"
TEXT = "#edf3fc"
MUTED = "#a4b4ca"
TEAL = "#45dfc0"

app = Dash(__name__, title="Portfolio Lab | Jordan")
server = app.server

app.index_string = """
<!DOCTYPE html>
<html>
<head>
{%metas%}
<title>{%title%}</title>
{%favicon%}
{%css%}
<style>
body { margin:0; background:#091321; color:#edf3fc;
       font-family:Arial,sans-serif; }
* { box-sizing:border-box; }
button { background:#45dfc0; color:#091321; border:0;
         padding:13px 20px; border-radius:9px;
         font-weight:bold; cursor:pointer; }
button:hover { background:#85f0da; }
.workspace { display:grid; grid-template-columns:280px 1fr;
             gap:22px; align-items:start; }
.panel { background:#111f32; border:1px solid #22344d;
         border-radius:16px; padding:22px; }
.metrics { display:grid; grid-template-columns:repeat(4,1fr);
           gap:12px; margin-bottom:20px; }
.metric { background:#111f32; border:1px solid #22344d;
          padding:20px; border-radius:12px; }
.chart { background:#111f32; border-radius:14px;
         overflow:hidden; margin-bottom:20px; }
@media(max-width:850px) {
.workspace { grid-template-columns:1fr; }
.metrics { grid-template-columns:repeat(2,1fr); }
}
@media(max-width:420px) {
.metrics { grid-template-columns:1fr; }
}
</style>
</head>
<body>
{%app_entry%}
<footer>{%config%}{%scripts%}{%renderer%}</footer>
</body>
</html>
"""


def chart_style(fig, title):
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font=dict(color=TEXT, family="Arial"),
        margin=dict(l=55, r=25, t=65, b=50),
        legend=dict(orientation="h", y=-0.2),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor="#22344d")
    fig.update_yaxes(gridcolor="#22344d")
    return fig


def metric(label, value):
    return html.Div([
        html.Div(label, style={
            "color": MUTED, "fontSize": "12px"
        }),
        html.Div(value, style={
            "fontSize": "27px", "fontWeight": "bold",
            "marginTop": "12px"
        }),
    ], className="metric")


sliders = []
for ticker in TICKERS:
    sliders.append(html.Div([
        html.Div(ticker, id=f"label-{ticker}", style={
            "fontWeight": "bold", "marginBottom": "12px"
        }),
        dcc.Slider(
            id=f"weight-{ticker}",
            min=0, max=100, step=1, value=25,
            marks={0: "0", 50: "50", 100: "100"},
tooltip={"placement": "bottom", "always_visible": True},
updatemode="mouseup",
    ], style={"marginBottom": "27px"}))


app.layout = html.Div([
    dcc.Store(id="prices"),
    html.Div("JORDAN  /  MARKETS & ANALYTICS", style={
        "color": TEAL, "letterSpacing": "3px",
        "fontSize": "12px", "fontWeight": "bold"
    }),
    html.H1("Portfolio Lab", style={
        "fontSize": "44px", "marginBottom": "8px"
    }),
    html.P("Explore allocation. Understand performance. Measure risk.",
           style={"color": MUTED, "marginBottom": "28px"}),
    html.Div([
        html.Aside([
            html.H3("Portfolio controls"),
            html.Button("Refresh Yahoo data", id="refresh"),
            html.P(id="data-status", style={
                "color": MUTED, "fontSize": "13px",
                "lineHeight": "1.6"
            }),
            html.Hr(style={"borderColor": "#22344d"}),
            html.H4("Allocation sliders"),
            html.P(
                "Relative weights: allocations automatically "
                "normalise to 100%. Set a stock to zero to exclude it.",
                style={"color": MUTED, "fontSize": "13px",
                       "lineHeight": "1.6"},
            ),
            *sliders,
            html.Div(id="allocation", style={
                "color": TEAL, "fontSize": "13px",
                "lineHeight": "1.8"
            }),
        ], className="panel"),
        html.Main([
            html.Div(id="metrics", className="metrics"),
            html.Div(id="analysis-status", style={
                "color": MUTED, "marginBottom": "16px"
            }),
            *[
                html.Div(
                    dcc.Graph(
                        id=chart_id,
                        config={"displaylogo": False},
                    ),
                    className="chart",
                )
                for chart_id in ["performance", "rolling-var", "correlation"]
            ],
            html.Div([
                html.H4("Methodology"),
                html.P(
                    "Two years of adjusted daily prices, using common "
                    "complete trading dates. Hypothetical portfolio "
                    "rebalanced daily to the selected weights. "
                    "Excludes fees, taxes and transaction costs."
                ),
                html.P(
                    "Historical VaR: negative fifth percentile of daily "
                    "returns, floored at zero. Rolling VaR uses 60 "
                    "observations. A 95% one-day VaR of 2% means the "
                    "estimated loss threshold is exceeded on roughly "
                    "5% of days; it is not a maximum possible loss."
                ),
                html.P(
                    "Annualised volatility uses 252 trading days. "
                    "Sharpe ratio assumes a 0% risk-free rate. "
                    "All stocks are USD-denominated. Educational project."
                ),
            ], className="panel", style={
                "color": MUTED, "fontSize": "13px",
                "lineHeight": "1.7"
            }),
        ]),
    ], className="workspace"),
], style={"maxWidth": "1440px", "margin": "auto", "padding": "30px"})


@app.callback(
    Output("prices", "data"),
    Output("data-status", "children"),
    Input("refresh", "n_clicks"),
    State("prices", "data"),
)
def load_data(clicks, previous):
    try:
        raw = yf.download(
            TICKERS,
            period="2y",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
            timeout=15,
        )
        if raw is None or raw.empty:
            raise ValueError("Yahoo returned no prices.")

        close = raw["Close"].reindex(columns=TICKERS)
        missing = close.columns[close.isna().all()].tolist()
        if missing:
            raise ValueError("Missing prices: " + ", ".join(missing))

        close = close.dropna().sort_index()
        if len(close) < 62:
            raise ValueError("Not enough complete trading dates.")

        payload = {
            "dates": [str(date.date()) for date in close.index],
            "values": close.to_numpy().tolist(),
        }
        fetched = datetime.now(timezone.utc).strftime("%H:%M UTC")
        message = (
            f"Yahoo Finance · latest daily bar "
            f"{payload['dates'][-1]} · fetched {fetched}. "
            "Daily bars may include an unfinished session."
        )
        return payload, message

    except Exception as exc:
        prefix = (
            "Refresh failed; previous data retained. "
            if previous else "Data unavailable. Try refreshing. "
        )
        return no_update, prefix + str(exc)


@app.callback(
    Output("metrics", "children"),
    Output("performance", "figure"),
    Output("rolling-var", "figure"),
    Output("correlation", "figure"),
    Output("allocation", "children"),
    Output("analysis-status", "children"),
    Input("prices", "data"),
    *[Input(f"weight-{ticker}", "value") for ticker in TICKERS],
)
def update_dashboard(data, *slider_values):
    empty = chart_style(go.Figure(), "Awaiting portfolio data")
    weights = np.array(slider_values, dtype=float)

    if weights.sum() == 0:
        return (
            [], empty, empty, empty, "",
            "Move at least one allocation slider above zero.",
        )

    weights /= weights.sum()
    allocation = " · ".join(
        f"{ticker} {weight:.1%}"
        for ticker, weight in zip(TICKERS, weights)
        if weight > 0
    )

    if not data:
        return (
            [], empty, empty, empty, allocation,
            "Fetching prices. Charts will appear when data arrives.",
        )

    prices = pd.DataFrame(
        data["values"],
        index=pd.to_datetime(data["dates"]),
        columns=TICKERS,
    )
    returns = prices.pct_change(fill_method=None).dropna()
    portfolio = returns.dot(weights)
    wealth = (1 + portfolio).cumprod()
    volatility = portfolio.std(ddof=1) * np.sqrt(252)
    sharpe = (
        portfolio.mean() / portfolio.std(ddof=1) * np.sqrt(252)
        if portfolio.std(ddof=1) > 0 else np.nan
    )
    var95 = max(0.0, -portfolio.quantile(0.05))
    rolling_var = (-portfolio.rolling(60).quantile(0.05)).clip(lower=0)

    cards = [
        metric("TOTAL RETURN", f"{wealth.iloc[-1] - 1:+.1%}"),
        metric("ANNUALISED VOLATILITY", f"{volatility:.1%}"),
        metric("SHARPE · RF 0%", f"{sharpe:.2f}"),
        metric("95% ONE-DAY VaR", f"{var95:.2%}"),
    ]

    active = weights > 0
    # Include a zero-return starting point in the growth chart.
    growth = prices.loc[:, active].div(
        prices.loc[:, active].iloc[0]
    ) - 1
    performance = go.Figure()
    for ticker in growth.columns:
        performance.add_scatter(
            x=growth.index, y=growth[ticker],
            name=ticker, opacity=0.45,
            line=dict(width=1),
        )

    portfolio_growth = pd.concat([
        pd.Series([0.0], index=[prices.index[0]]),
        wealth - 1,
    ])
    performance.add_scatter(
        x=portfolio_growth.index, y=portfolio_growth,
        name="Portfolio", line=dict(color=TEAL, width=3),
    )
    chart_style(performance, "Cumulative returns")
    performance.update_yaxes(tickformat=".0%")

    risk = go.Figure(go.Scatter(
        x=rolling_var.index, y=rolling_var,
        name="95% one-day VaR",
        line=dict(color="#ffbb73", width=2),
        fill="tozeroy", fillcolor="rgba(255,187,115,0.08)",
    ))
    chart_style(risk, "Rolling risk · 60-day historical VaR")
    risk.update_yaxes(tickformat=".1%", rangemode="tozero")

    correlation = px.imshow(
        returns.loc[:, active].corr(),
        zmin=-1, zmax=1, text_auto=".2f",
        color_continuous_scale="RdBu_r",
        aspect="auto",
    )
    chart_style(correlation, "Daily return correlations")
    correlation.update_layout(hovermode="closest")

    status = (
        f"{len(returns):,} daily return observations · "
        f"{returns.index[0]:%d %b %Y} – "
        f"{returns.index[-1]:%d %b %Y}"
    )
    return cards, performance, risk, correlation, allocation, status

@app.callback(
    [Output(f"label-{ticker}", "children") for ticker in TICKERS],
    [Input(f"weight-{ticker}", "value") for ticker in TICKERS],
)
def show_allocations(*values):
    weights = np.array(
        [0 if value is None else value for value in values],
        dtype=float,
    )
    total = weights.sum()
    percentages = weights / total * 100 if total > 0 else weights

    return [
        f"{ticker} — {percentage:.1f}% allocation"
        for ticker, percentage in zip(TICKERS, percentages)
    ]
if __name__ == "__main__":
    app.run(
        debug=False,
        host="127.0.0.1",
        port=int(os.environ.get("PORT", 8050)),
    )
