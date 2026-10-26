(function () {
  'use strict';

  var data = JSON.parse(document.getElementById('report-data').textContent);
  var LC = window.LightweightCharts;
  if (!LC) {
    document.querySelectorAll('.chart').forEach(function (el) {
      el.textContent = 'Interactive charts need an internet connection to load lightweight-charts.';
    });
    return;
  }

  var colors = { green: '#22c55e', red: '#ef4444', gold: '#f59e0b', purple: '#a855f7', muted: '#64748b' };
  var priceFormat = { type: 'price', precision: data.precision, minMove: Math.pow(10, -data.precision) };

  function chartOptions() {
    return {
      autoSize: true,
      layout: {
        background: { type: 'solid', color: 'transparent' },
        textColor: '#8b95a7',
        fontFamily: 'Inter, system-ui, sans-serif',
        panes: { separatorColor: '#232a3b', separatorHoverColor: 'rgba(96, 165, 250, 0.2)' }
      },
      grid: { vertLines: { color: 'rgba(42, 46, 57, 0.35)' }, horzLines: { color: 'rgba(42, 46, 57, 0.35)' } },
      rightPriceScale: { borderColor: '#232a3b' },
      timeScale: { borderColor: '#232a3b', timeVisible: data.intraday, secondsVisible: false },
      crosshair: { mode: LC.CrosshairMode.Normal }
    };
  }

  function overlay(color, dashed) {
    return {
      color: color,
      lineWidth: 1,
      lineStyle: dashed ? LC.LineStyle.Dashed : LC.LineStyle.Solid,
      priceLineVisible: false,
      lastValueVisible: false,
      crosshairMarkerVisible: false,
      priceFormat: priceFormat
    };
  }

  // Price chart: candles, moving averages, trendlines, signals and trades, volume pane
  var chart = LC.createChart(document.getElementById('price-chart'), chartOptions());
  var candles = chart.addSeries(LC.CandlestickSeries, {
    upColor: colors.green, downColor: colors.red, borderVisible: false,
    wickUpColor: colors.green, wickDownColor: colors.red, priceFormat: priceFormat
  });
  candles.setData(data.candles);

  var styles = {
    'SMA 50': overlay(colors.gold, false),
    'SMA 200': overlay(colors.purple, false),
    'Resistance trendline': overlay('rgba(34, 197, 94, 0.75)', true),
    'Support trendline': overlay('rgba(239, 68, 68, 0.75)', true)
  };
  Object.keys(data.lines).forEach(function (name) {
    chart.addSeries(LC.LineSeries, styles[name]).setData(data.lines[name]);
  });
  LC.createSeriesMarkers(candles, data.markers);

  var volume = chart.addSeries(LC.HistogramSeries, {
    priceFormat: { type: 'volume' }, priceLineVisible: false, lastValueVisible: false
  }, 1);
  volume.setData(data.volume);
  chart.panes()[1].setHeight(90);
  chart.timeScale().fitContent();

  // Hover legend: OHLC plus the signal explanation of the bar under the cursor
  var legend = document.getElementById('legend');
  var indexByTime = new Map(data.candles.map(function (c, i) { return [c.time, i]; }));
  var last = data.candles.length - 1;

  function escapeHtml(text) {
    return String(text).replace(/[&<>"']/g, function (ch) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch];
    });
  }

  function formatTime(seconds) {
    var iso = new Date(seconds * 1000).toISOString();
    return data.intraday ? iso.slice(0, 16).replace('T', ' ') : iso.slice(0, 10);
  }

  function renderLegend(i) {
    var c = data.candles[i];
    var bar = data.bars;
    var cls = { 'Strong Long': 'strong-long', 'Long': 'long' }[bar.decision[i]] || 'neutral';
    var pattern = bar.pattern[i] !== 'None' ? ' &middot; ' + escapeHtml(bar.pattern[i]) : '';
    legend.innerHTML =
      '<b>' + formatTime(c.time) + '</b> &nbsp;O ' + c.open.toFixed(data.precision) +
      ' H ' + c.high.toFixed(data.precision) + ' L ' + c.low.toFixed(data.precision) +
      ' C ' + c.close.toFixed(data.precision) +
      ' &nbsp;<span class="badge ' + cls + '">' + escapeHtml(bar.decision[i]) + ' ' +
      bar.score[i].toFixed(1) + '</span>' + pattern +
      '<span class="why">' + escapeHtml(bar.reasoning[i] || 'No scoring condition met') + '</span>';
  }

  renderLegend(last);
  chart.subscribeCrosshairMove(function (param) {
    var i = param.time === undefined ? undefined : indexByTime.get(param.time);
    renderLegend(i === undefined ? last : i);
  });

  // Equity chart synchronised with the price chart
  var equityElement = document.getElementById('equity-chart');
  if (!equityElement || !data.equity.length) {
    return;
  }
  var equity = LC.createChart(equityElement, chartOptions());
  equity.addSeries(LC.LineSeries, {
    color: colors.green, lineWidth: 2, title: 'Strategy', priceLineVisible: false
  }).setData(data.equity);
  equity.addSeries(LC.LineSeries, {
    color: colors.muted, lineWidth: 1, lineStyle: LC.LineStyle.Dashed, title: 'Buy & hold', priceLineVisible: false
  }).setData(data.buyHold);
  equity.addSeries(LC.AreaSeries, {
    lineColor: colors.red, lineWidth: 1, topColor: 'rgba(239, 68, 68, 0.05)', bottomColor: 'rgba(239, 68, 68, 0.35)',
    invertFilledArea: true, priceFormat: { type: 'percent' }, title: 'Drawdown', priceLineVisible: false
  }, 1).setData(data.drawdown);
  equity.panes()[1].setHeight(80);
  equity.timeScale().fitContent();

  var syncing = false;
  function follow(source, target) {
    source.timeScale().subscribeVisibleLogicalRangeChange(function (range) {
      if (syncing || !range) {
        return;
      }
      syncing = true;
      target.timeScale().setVisibleLogicalRange(range);
      syncing = false;
    });
  }
  follow(chart, equity);
  follow(equity, chart);
})();
