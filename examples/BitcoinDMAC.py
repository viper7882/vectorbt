import vectorbt as vbt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytz
import ipywidgets as widgets
from IPython.display import display
import gc

# Enter your parameters here
seed = 42
symbol = 'BTC-USD'
metric = 'total_return'

start_date = datetime(2018, 1, 1, tzinfo=pytz.utc)  # time period for analysis, must be timezone-aware
end_date = datetime(2020, 1, 1, tzinfo=pytz.utc)
time_buffer = timedelta(days=100)  # buffer before to pre-calculate SMA/EMA, best to set to max window
freq = '1D'

vbt.settings.portfolio['init_cash'] = 100.  # 100$
vbt.settings.portfolio['fees'] = 0.0025  # 0.25%
vbt.settings.portfolio['slippage'] = 0.0025  # 0.25%

# Download data with time buffer
cols = ['Open', 'High', 'Low', 'Close', 'Volume']
ohlcv_wbuf = vbt.YFData.download(symbol, start=start_date - time_buffer, end=end_date).get(cols)

ohlcv_wbuf = ohlcv_wbuf.astype(np.float64)

print("INFO: ohlcv_wbuf.shape:")
print(ohlcv_wbuf.shape)
print("")

print("INFO: ohlcv_wbuf.columns:")
print(ohlcv_wbuf.columns)
print("")

# Create a copy of data without time buffer
wobuf_mask = (ohlcv_wbuf.index >= start_date) & (ohlcv_wbuf.index <= end_date)  # mask without buffer

ohlcv = ohlcv_wbuf.loc[wobuf_mask, :]

print("INFO: ohlcv.shape:")
print(ohlcv.shape)
print("")

# Plot the OHLC data
# ohlcv_wbuf.vbt.ohlcv.plot().show()

fast_window = 30
slow_window = 80
# Pre-calculate running windows on data with time buffer
fast_ma = vbt.MA.run(ohlcv_wbuf['Open'], fast_window)
slow_ma = vbt.MA.run(ohlcv_wbuf['Open'], slow_window)

print("INFO: fast_ma.ma.shape:")
print(fast_ma.ma.shape)
print("")

print("INFO: slow_ma.ma.shape:")
print(slow_ma.ma.shape)
print("")

# Remove time buffer
fast_ma = fast_ma[wobuf_mask]
slow_ma = slow_ma[wobuf_mask]

# there should be no nans after removing time buffer
assert(~fast_ma.ma.isnull().any())
assert(~slow_ma.ma.isnull().any())

print("INFO: fast_ma.ma.shape:")
print(fast_ma.ma.shape)
print("")

print("INFO: slow_ma.ma.shape:")
print(slow_ma.ma.shape)
print("")

# Generate crossover signals
dmac_entries = fast_ma.ma_crossed_above(slow_ma)
dmac_exits = fast_ma.ma_crossed_below(slow_ma)

fig = ohlcv['Open'].vbt.plot(trace_kwargs=dict(name='Price'))
fig = fast_ma.ma.vbt.plot(trace_kwargs=dict(name='Fast MA'), fig=fig)
fig = slow_ma.ma.vbt.plot(trace_kwargs=dict(name='Slow MA'), fig=fig)
fig = dmac_entries.vbt.signals.plot_as_entry_markers(ohlcv['Open'], fig=fig)
fig = dmac_exits.vbt.signals.plot_as_exit_markers(ohlcv['Open'], fig=fig)

# fig.show()

# # Signal stats
# print("INFO: dmac_entries.vbt.signals.stats:")
# print(dmac_entries.vbt.signals.stats(settings=dict(other=dmac_exits)))
# print("")

# Plot signals
# fig = dmac_entries.vbt.signals.plot(trace_kwargs=dict(name='Entries'))
# dmac_exits.vbt.signals.plot(trace_kwargs=dict(name='Exits'), fig=fig).show()

# Build partfolio, which internally calculates the equity curve

# Volume is set to np.inf by default to buy/sell everything
# You don't have to pass freq here because our data is already perfectly time-indexed
dmac_pf = vbt.Portfolio.from_signals(ohlcv['Close'], dmac_entries, dmac_exits)

# # Print stats
# print("INFO: dmac_pf.stats():")
# print(dmac_pf.stats())
# print("")

# Plot trades
print("INFO: dmac_pf.trades.records:")
print(dmac_pf.trades.records)
print("")
# dmac_pf.trades.plot().show()

# Now build portfolio for a "Hold" strategy
# Here we buy once at the beginning and sell at the end
hold_entries = pd.Series.vbt.signals.empty_like(dmac_entries)
hold_entries.iloc[0] = True

hold_exits = pd.Series.vbt.signals.empty_like(hold_entries)
hold_exits.iloc[-1] = True

hold_pf = vbt.Portfolio.from_signals(ohlcv['Close'], hold_entries, hold_exits)

# # Equity
# fig = dmac_pf.value().vbt.plot(trace_kwargs=dict(name='Value (DMAC)'))
# hold_pf.value().vbt.plot(trace_kwargs=dict(name='Value (Hold)'), fig=fig).show()

# Interactive window slider to easily compare windows by their performance
min_window = 2
max_window = 100
perf_metrics = ['total_return', 'positions.win_rate', 'positions.expectancy', 'max_drawdown']
perf_metric_names = ['Total return', 'Win rate', 'Expectancy', 'Max drawdown']

windows_slider = widgets.IntRangeSlider(
    value=[fast_window, slow_window],
    min=min_window,
    max=max_window,
    step=1,
    layout=dict(width='500px'),
    continuous_update=True
)
dmac_fig = None
dmac_img = widgets.Image(
    format='png',
    width=vbt.settings['plotting']['layout']['width'],
    height=vbt.settings['plotting']['layout']['height']
)
metrics_html = widgets.HTML()

def on_value_change(value):
    global dmac_fig

    # Calculate portfolio
    fast_window, slow_window = value['new']
    fast_ma = vbt.MA.run(ohlcv_wbuf['Open'], fast_window)
    slow_ma = vbt.MA.run(ohlcv_wbuf['Open'], slow_window)
    fast_ma = fast_ma[wobuf_mask]
    slow_ma = slow_ma[wobuf_mask]
    dmac_entries = fast_ma.ma_crossed_above(slow_ma)
    dmac_exits = fast_ma.ma_crossed_below(slow_ma)
    dmac_pf = vbt.Portfolio.from_signals(ohlcv['Close'], dmac_entries, dmac_exits)

    # Update figure
    if dmac_fig is None:
        dmac_fig = ohlcv['Open'].vbt.plot(trace_kwargs=dict(name='Price'))
        fast_ma.ma.vbt.plot(trace_kwargs=dict(name='Fast MA'), fig=dmac_fig)
        slow_ma.ma.vbt.plot(trace_kwargs=dict(name='Slow MA'), fig=dmac_fig)
        dmac_entries.vbt.signals.plot_as_entry_markers(ohlcv['Open'], fig=dmac_fig)
        dmac_exits.vbt.signals.plot_as_exit_markers(ohlcv['Open'], fig=dmac_fig)
    else:
        with dmac_fig.batch_update():
            dmac_fig.data[1].y = fast_ma.ma
            dmac_fig.data[2].y = slow_ma.ma
            dmac_fig.data[3].x = ohlcv['Open'].index[dmac_entries]
            dmac_fig.data[3].y = ohlcv['Open'][dmac_entries]
            dmac_fig.data[4].x = ohlcv['Open'].index[dmac_exits]
            dmac_fig.data[4].y = ohlcv['Open'][dmac_exits]
    dmac_img.value = dmac_fig.to_image(format="png")

    # Update metrics table
    sr = pd.Series([dmac_pf.deep_getattr(m) for m in perf_metrics],
                   index=perf_metric_names, name='Performance')
    metrics_html.value = sr.to_frame().style.set_properties(**{'text-align': 'right'}).render()


windows_slider.observe(on_value_change, names='value')
on_value_change({'new': windows_slider.value})

dashboard = widgets.VBox([
    widgets.HBox([widgets.Label('Fast and slow window:'), windows_slider]),
    dmac_img,
    metrics_html
])
# display(dashboard)
dashboard.close()  # after using, release memory and notebook metadata

gc.collect()

# ----------------------------------------------------------------------------------------------------------------------
# Multiple Window Combinations
# ----------------------------------------------------------------------------------------------------------------------
# Pre-calculate running windows on data with time buffer
fast_ma_wbuf, slow_ma_wbuf = vbt.MA.run_combs(
    ohlcv_wbuf['Open'], np.arange(min_window, max_window+1),
    r=2, short_names=['fast_ma_wbuf', 'slow_ma_wbuf'])

print("INFO: fast_ma_wbuf.ma.shape:")
print(fast_ma_wbuf.ma.shape)
print("INFO: slow_ma_wbuf.ma.shape:")
print(slow_ma_wbuf.ma.shape)
print("")

print("INFO: fast_ma_wbuf.ma.columns:")
print(fast_ma_wbuf.ma.columns)
print("")

print("INFO: slow_ma_wbuf.ma.columns:")
print(slow_ma_wbuf.ma.columns)
print("")

# Remove time buffer
fast_ma = fast_ma_wbuf[wobuf_mask]
slow_ma = slow_ma_wbuf[wobuf_mask]

print("INFO: fast_ma.ma.shape:")
print(fast_ma.ma.shape)
print("INFO: slow_ma.ma.shape:")
print(slow_ma.ma.shape)
print("")

# We perform the same steps, but now we have 4851 columns instead of 1
# Each column corresponds to a pair of fast and slow windows
# Generate crossover signals
dmac_entries = fast_ma.ma_crossed_above(slow_ma)
dmac_exits = fast_ma.ma_crossed_below(slow_ma)

print("INFO: dmac_entries.columns:")
print(dmac_entries.columns)     # the same for dmac_exits
print("")

# Build portfolio
dmac_pf = vbt.Portfolio.from_signals(ohlcv['Close'], dmac_entries, dmac_exits)
# Calculate performance of each window combination
dmac_perf = dmac_pf.deep_getattr(metric)

print("INFO: dmac_perf.shape:")
print(dmac_perf.shape)
print("")

print("INFO: dmac_perf.index:")
print(dmac_perf.index)
print("")

# Your optimal window combination
dmac_perf.idxmax()

# Convert this array into a matrix of shape (99, 99): 99 fast windows x 99 slow windows
dmac_perf_matrix = dmac_perf.vbt.unstack_to_df(symmetric=True,
                                               index_levels='fast_ma_wbuf_window', column_levels='slow_ma_wbuf_window')

print("INFO: dmac_perf_matrix.shape:")
print(dmac_perf_matrix.shape)
print("")

dmac_perf_matrix.vbt.heatmap(
    xaxis_title='Slow window',
    yaxis_title='Fast window').show()
pass

# Implement an interactive date range slider to easily compare heatmaps over time
def dmac_pf_from_date_range(from_date, to_date):
    # Portfolio from MA crossover, filtered by time range
    range_mask = (ohlcv.index >= from_date) & (ohlcv.index <= to_date)
    range_fast_ma = fast_ma[range_mask]  # use our variables defined above
    range_slow_ma = slow_ma[range_mask]
    dmac_entries = range_fast_ma.ma_crossed_above(range_slow_ma)
    dmac_exits = range_fast_ma.ma_crossed_below(range_slow_ma)
    dmac_pf = vbt.Portfolio.from_signals(ohlcv.loc[range_mask, 'Close'], dmac_entries, dmac_exits)
    return dmac_pf


def rand_pf_from_date_range(from_date, to_date):
    # Portfolio from random strategy, filtered by time range
    range_mask = (ohlcv.index >= from_date) & (ohlcv.index <= to_date)
    range_fast_ma = fast_ma[range_mask]  # use our variables defined above
    range_slow_ma = slow_ma[range_mask]
    dmac_entries = range_fast_ma.ma_crossed_above(range_slow_ma)
    dmac_exits = range_fast_ma.ma_crossed_below(range_slow_ma)
    rand_entries = dmac_entries.vbt.signals.shuffle(seed=seed)  # same number of signals as in dmac
    rand_exits = rand_entries.vbt.signals.generate_random_exits(seed=seed)
    rand_pf = vbt.Portfolio.from_signals(ohlcv.loc[range_mask, 'Close'], rand_entries, rand_exits)
    return rand_pf


def hold_pf_from_date_range(from_date, to_date):
    # Portfolio from holding strategy, filtered by time range
    range_mask = (ohlcv.index >= from_date) & (ohlcv.index <= to_date)
    hold_entries = pd.Series.vbt.signals.empty(range_mask.sum(), index=ohlcv[range_mask].index)
    hold_entries.iloc[0] = True
    hold_exits = pd.Series.vbt.signals.empty_like(hold_entries)
    hold_exits.iloc[-1] = True
    hold_pf = vbt.Portfolio.from_signals(ohlcv.loc[range_mask, 'Close'], hold_entries, hold_exits)
    return hold_pf


# TimeSeries (OHLC)
ts_fig = ohlcv.vbt.ohlcv.plot(
    title=symbol,
    show_volume=False,
    annotations=[dict(
        align='left',
        showarrow=False,
        xref='paper',
        yref='paper',
        x=0.5,
        y=0.9,
        font=dict(size=14),
        bordercolor='black',
        borderwidth=1,
        bgcolor='white'
    )],
    width=700,
    height=250)

# Histogram (DMAC vs Random)
histogram = vbt.plotting.Histogram(
    trace_names=['Random strategy', 'DMAC strategy'],
    title='%s distribution' % metric,
    xaxis_tickformat='%',
    annotations=[dict(
        y=0,
        xref='x',
        yref='paper',
        showarrow=True,
        arrowcolor="black",
        arrowsize=1,
        arrowwidth=1,
        arrowhead=1,
        xanchor='left',
        text='Hold',
        textangle=0,
        font=dict(size=14),
        bordercolor='black',
        borderwidth=1,
        bgcolor='white',
        ax=0,
        ay=-50,
    )],
    width=700,
    height=250
)

# Heatmap (DMAC vs Holding)
heatmap = vbt.plotting.Heatmap(
    x_labels=np.arange(min_window, max_window + 1),
    y_labels=np.arange(min_window, max_window + 1),
    trace_kwargs=dict(
        colorbar=dict(
            tickformat='%',
            ticks="outside"
        ),
        colorscale='RdBu'),
    title='%s by window' % metric,
    width=650,
    height=420
)

dmac_perf_matrix = None
rand_perf_matrix = None
hold_value = None


def update_heatmap_colorscale(perf_matrix):
    # Update heatmap colorscale based on performance matrix
    with heatmap.fig.batch_update():
        heatmap.fig.data[0].zmid = hold_value
        heatmap.fig.data[0].colorbar.tickvals = [
            np.nanmin(perf_matrix),
            hold_value,
            np.nanmax(perf_matrix)
        ]
        heatmap.fig.data[0].colorbar.ticktext = [
            'Min: {:.0%}'.format(np.nanmin(perf_matrix)).ljust(12),
            'Hold: {:.0%}'.format(hold_value).ljust(12),
            'Max: {:.0%}'.format(np.nanmax(perf_matrix)).ljust(12)
        ]


def update_histogram(dmac_perf_matrix, rand_perf_matrix, hold_value):
    # Update histogram figure
    with histogram.fig.batch_update():
        histogram.update(
            np.asarray([
                rand_perf_matrix.values.flatten(),
                dmac_perf_matrix.values.flatten()
            ]).transpose()
        )
        histogram.fig.layout.annotations[0].x = hold_value


def update_figs(from_date, to_date):
    global dmac_perf_matrix, rand_perf_matrix, hold_value  # needed for on_heatmap_change

    # Build portfolios
    dmac_pf = dmac_pf_from_date_range(from_date, to_date)
    rand_pf = rand_pf_from_date_range(from_date, to_date)
    hold_pf = hold_pf_from_date_range(from_date, to_date)

    # Calculate performance
    dmac_perf_matrix = dmac_pf.deep_getattr(metric)
    dmac_perf_matrix = dmac_perf_matrix.vbt.unstack_to_df(
        symmetric=True, index_levels='fast_ma_window', column_levels='slow_ma_window')
    rand_perf_matrix = rand_pf.deep_getattr(metric)
    rand_perf_matrix = rand_perf_matrix.vbt.unstack_to_df(
        symmetric=True, index_levels='fast_ma_window', column_levels='slow_ma_window')
    hold_value = hold_pf.deep_getattr(metric)

    # Update figures
    update_histogram(dmac_perf_matrix, rand_perf_matrix, hold_value)
    with ts_fig.batch_update():
        ts_fig.update_xaxes(range=(from_date, to_date))
        ts_fig.layout.annotations[0].text = 'Hold: %.f%%' % (hold_value * 100)
    with heatmap.fig.batch_update():
        heatmap.update(dmac_perf_matrix)
        update_heatmap_colorscale(dmac_perf_matrix.values)


def on_ts_change(layout, x_range):
    global dmac_perf_matrix, rand_perf_matrix, hold_value  # needed for on_heatmap_change

    if isinstance(x_range[0], str) and isinstance(x_range[1], str):
        update_figs(x_range[0], x_range[1])


ts_fig.layout.on_change(on_ts_change, 'xaxis.range')


def on_heatmap_change(layout, x_range, y_range):
    if dmac_perf_matrix is not None:
        x_mask = (dmac_perf_matrix.columns >= x_range[0]) & (dmac_perf_matrix.columns <= x_range[1])
        y_mask = (dmac_perf_matrix.index >= y_range[0]) & (dmac_perf_matrix.index <= y_range[1])
        if x_mask.any() and y_mask.any():
            # Update widgets
            sub_dmac_perf_matrix = dmac_perf_matrix.loc[y_mask, x_mask]  # y_mask is index, x_mask is columns
            sub_rand_perf_matrix = rand_perf_matrix.loc[y_mask, x_mask]
            update_histogram(sub_dmac_perf_matrix, sub_rand_perf_matrix, hold_value)
            update_heatmap_colorscale(sub_dmac_perf_matrix.values)


heatmap.fig.layout.on_change(on_heatmap_change, 'xaxis.range', 'yaxis.range')

dashboard = widgets.VBox([
    ts_fig,
    histogram.fig,
    heatmap.fig
])
display(dashboard)
dashboard.close()   # after using, release memory and notebook metadata

