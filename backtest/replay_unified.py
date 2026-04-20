"""
Visual HTML replay for unified ORB+MR portfolio.

Per symbol: runs ORB + MR signals in isolation and renders TradingView chart
with trade boxes, signal cards sidebar, playback controls.

Usage:
    python -m backtest.replay_unified                    # default US30
    python -m backtest.replay_unified --symbol DAX
    python -m backtest.replay_unified --symbol SP500 --days 90
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from strategy.mean_reversion import ibs as mr_ibs

logging.basicConfig(level=logging.WARNING)

# Mirror wf_unified.py / live.unified_signals
INSTRUMENTS = {
    "US30":  {"ticker": "YM=F",   "spread": 2.0, "lot_mult": 5,  "min_sl": 30.0, "comm": 3.0,
              "orb_hour": 13, "mr_check_hour": 20, "use_orb": True,  "use_mr": True},
    "RTY":   {"ticker": "RTY=F",  "spread": 0.5, "lot_mult": 50, "min_sl": 5.0,  "comm": 3.0,
              "orb_hour": 13, "mr_check_hour": 20, "use_orb": True,  "use_mr": True},
    "DAX":   {"ticker": "^GDAXI", "spread": 1.0, "lot_mult": 25, "min_sl": 20.0, "comm": 3.0,
              "orb_hour": 7,  "mr_check_hour": 15, "use_orb": True,  "use_mr": False},
    "SP500": {"ticker": "ES=F",   "spread": 0.5, "lot_mult": 50, "min_sl": 10.0, "comm": 3.0,
              "orb_hour": 13, "mr_check_hour": 20, "use_orb": False, "use_mr": True},
}
ORB_TP_MULT = 2.0
ORB_TIME_STOP = 5
ORB_RISK_PCT = 0.01
MR_SL_ATR_MULT = 1.5
MR_TIME_STOP = 5 * 24
MR_RISK_PCT = 0.01


def fetch(ticker: str) -> dict:
    r = {}
    for tf, interval, period in [("1h", "1h", "730d"), ("daily", "1d", "5y"), ("weekly", "1wk", "10y")]:
        d = yf.download(ticker, period=period, interval=interval, progress=False)
        d.columns = [c[0].lower() for c in d.columns]
        if d.index.tz is None:
            d.index = d.index.tz_localize("UTC")
        r[tf] = d
    r["4h"] = r["1h"].resample("4h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    return r


# ============================================================
# Signal generation (mirrors wf_unified + live.unified_signals)
# ============================================================
def orb_range(df, now_ts, orb_hour):
    start = now_ts.replace(hour=orb_hour, minute=0, second=0, microsecond=0)
    end = start + pd.Timedelta(hours=1)
    s = df.loc[start:end]
    if len(s) < 1:
        return 0, 0, 0, False
    h, l = float(s["high"].max()), float(s["low"].min())
    size = h - l
    return h, l, size, size > 0


def check_orb(df_1h, price, bh, bl, now_ts, cfg):
    hour = now_ts.hour
    orb_h = cfg["orb_hour"]
    if hour < orb_h + 1 or hour >= orb_h + 5:
        return None
    h, l, size, valid = orb_range(df_1h, now_ts, orb_h)
    if not valid or size < cfg["min_sl"]:
        return None
    direction = None
    if bh > h and price > h: direction = "long"
    elif bl < l and price < l: direction = "short"
    else: return None
    daily = df_1h["close"].resample("1D").last().dropna()
    if len(daily) >= 20:
        sma20 = float(daily.tail(20).mean()); last = float(daily.iloc[-1])
        if direction == "long" and last < sma20: return None
        if direction == "short" and last > sma20: return None
    buf = size * 0.1
    if direction == "long":
        sl = l - buf; tp = price + size * ORB_TP_MULT; fill = price + cfg["spread"]
    else:
        sl = h + buf; tp = price - size * ORB_TP_MULT; fill = price - cfg["spread"]
    risk = abs(fill - sl)
    if risk < cfg["min_sl"] * 0.5: return None
    return {"strategy": "ORB", "side": direction, "entry": fill, "sl": sl, "tp": tp,
            "orb_high": h, "orb_low": l, "orb_size": size,
            "reason": f"ORB {direction.upper()} | range {size:.2f}"}


def check_mr(df_daily, price, bh, bl, cfg):
    if len(df_daily) < 10: return None
    prior5_low = float(df_daily["low"].iloc[-6:-1].min())
    if price >= prior5_low: return None
    if mr_ibs(bh, bl, price) >= 0.25: return None
    daily_range = float((df_daily["high"] - df_daily["low"]).tail(14).mean())
    sl_dist = max(daily_range * MR_SL_ATR_MULT, cfg["min_sl"])
    sma5 = float(df_daily["close"].tail(5).mean())
    fill = price + cfg["spread"]; sl = price - sl_dist; tp = sma5
    if tp <= fill: return None
    risk = fill - sl
    if risk <= 0 or (tp - fill) / risk < 0.8: return None
    return {"strategy": "MR", "side": "long", "entry": fill, "sl": sl, "tp": tp,
            "sma5": sma5, "prior5_low": prior5_low,
            "reason": f"MR | close<5dlow IBS={mr_ibs(bh,bl,price):.2f}"}


# ============================================================
# Backtest loop — one symbol only, starting equity $5000
# ============================================================
def run_bt(sym: str, tf: dict):
    cfg = INSTRUMENTS[sym]
    d1h = tf["1h"]; d_daily = tf["daily"]
    warmup = 500
    if len(d1h) <= warmup:
        return [], [], d1h

    tradeable = d1h.index[warmup:]
    equity = config.ACCOUNT_SIZE
    pos = None
    trades = []
    eq_data = []
    daily_orb_taken = set()
    daily_mr_taken = set()

    for i, ts in enumerate(tradeable):
        bar = i
        day = ts.date()
        loc = d1h.index.get_loc(ts)
        w1h = d1h.iloc[max(0, loc - 500):loc + 1]
        price = float(w1h["close"].iloc[-1])
        bh = float(w1h["high"].iloc[-1])
        bl = float(w1h["low"].iloc[-1])

        # Manage open position
        if pos is not None:
            bars_held = bar - pos["bar"]
            close_it, reason, ep = False, "", price

            if pos["side"] == "long" and bl <= pos["sl"]:
                close_it, reason, ep = True, "SL", pos["sl"]
            elif pos["side"] == "short" and bh >= pos["sl"]:
                close_it, reason, ep = True, "SL", pos["sl"]
            if not close_it:
                if pos["side"] == "long" and bh >= pos["tp"]:
                    close_it, reason, ep = True, "TP", pos["tp"]
                elif pos["side"] == "short" and bl <= pos["tp"]:
                    close_it, reason, ep = True, "TP", pos["tp"]
            if pos["side"] == "long" and bl <= pos["sl"] and bh >= pos["tp"]:
                close_it, reason, ep = True, "SL", pos["sl"]
            elif pos["side"] == "short" and bh >= pos["sl"] and bl <= pos["tp"]:
                close_it, reason, ep = True, "SL", pos["sl"]

            tlimit = ORB_TIME_STOP if pos["strategy"] == "ORB" else MR_TIME_STOP
            if not close_it and bars_held >= tlimit:
                close_it, reason, ep = True, "Time", price
            if pos["strategy"] == "ORB" and not close_it and ts.hour >= cfg["orb_hour"] + 7:
                close_it, reason, ep = True, "SessionEnd", price

            if close_it:
                raw = ((ep - pos["entry"]) if pos["side"] == "long" else
                       (pos["entry"] - ep)) * pos["lots"] * cfg["lot_mult"]
                c = cfg["comm"] * pos["lots"]
                pnl = raw - c
                equity += pnl
                trades.append({
                    "entry_time": str(pos["et"])[:19], "exit_time": str(ts)[:19],
                    "entry_price": pos["entry"], "exit_price": ep,
                    "sl": pos["osl"], "tp": pos["otp"],
                    "side": pos["side"], "pnl": round(pnl, 2), "reason": reason,
                    "strategy": pos["strategy"], "rr": round(pos["rr"], 2),
                    "sig_reason": pos["sig_reason"], "bias_info": pos["bias_info"],
                    "lots_orig": pos["lots"], "risk_usd": round(pos["risk_usd"], 2),
                })
                pos = None

        # New ORB entry
        if pos is None and cfg["use_orb"] and day not in daily_orb_taken:
            sig = check_orb(w1h, price, bh, bl, ts, cfg)
            if sig:
                risk = abs(sig["entry"] - sig["sl"])
                if risk >= cfg["min_sl"] * 0.5:
                    rd = equity * ORB_RISK_PCT
                    lots = max(0.01, min(0.50, round(rd / (risk * cfg["lot_mult"]), 2)))
                    rr = abs(sig["tp"] - sig["entry"]) / risk
                    pos = {
                        "strategy": "ORB", "side": sig["side"], "entry": sig["entry"],
                        "sl": sig["sl"], "tp": sig["tp"], "osl": sig["sl"], "otp": sig["tp"],
                        "lots": lots, "bar": bar, "et": ts,
                        "rr": rr, "sig_reason": sig["reason"],
                        "bias_info": f"ORB [{sig['orb_low']:.2f}-{sig['orb_high']:.2f}]",
                        "risk_usd": rd,
                    }
                    daily_orb_taken.add(day)

        # New MR entry (at end-of-day check hour)
        if pos is None and cfg["use_mr"] and ts.hour == cfg["mr_check_hour"] and day not in daily_mr_taken:
            today_start = pd.Timestamp(day).tz_localize("UTC")
            today_data = d1h.loc[today_start:ts]
            if len(today_data) >= 3:
                today_close = float(today_data["close"].iloc[-1])
                today_high = float(today_data["high"].max())
                today_low = float(today_data["low"].min())
                hist = d_daily.loc[:today_start - pd.Timedelta(hours=1)]
                if len(hist) >= 20:
                    sig = check_mr(hist, today_close, today_high, today_low, cfg)
                    if sig:
                        risk = abs(sig["entry"] - sig["sl"])
                        if risk >= cfg["min_sl"] * 0.3:
                            rd = equity * MR_RISK_PCT
                            lots = max(0.01, min(0.50, round(rd / (risk * cfg["lot_mult"]), 2)))
                            rr = (sig["tp"] - sig["entry"]) / risk
                            pos = {
                                "strategy": "MR", "side": "long", "entry": sig["entry"],
                                "sl": sig["sl"], "tp": sig["tp"], "osl": sig["sl"], "otp": sig["tp"],
                                "lots": lots, "bar": bar, "et": ts,
                                "rr": rr, "sig_reason": sig["reason"],
                                "bias_info": f"5d-low {sig['prior5_low']:.2f} | SMA5 {sig['sma5']:.2f}",
                                "risk_usd": rd,
                            }
                            daily_mr_taken.add(day)

        ur = 0
        if pos:
            ur = ((price - pos["entry"]) if pos["side"] == "long" else
                  (pos["entry"] - price)) * pos["lots"] * cfg["lot_mult"]
        eq_data.append({"ts": ts, "eq": round(equity + ur, 2)})

    return trades, eq_data, d1h


# ============================================================
# HTML generation — reuses the pattern from replay.py
# ============================================================
def generate_html(sym: str, all_tf: dict, trades: list, eq_data: list) -> str:
    all_tf_json = json.dumps(all_tf)
    candles_json = json.dumps(all_tf["1H"])
    trades_json = json.dumps(trades)
    equity_json = json.dumps(eq_data)
    total_pnl = sum(t["pnl"] for t in trades)
    wins = sum(1 for t in trades if t["pnl"] > 0)
    orb_count = sum(1 for t in trades if t.get("strategy") == "ORB")
    mr_count = sum(1 for t in trades if t.get("strategy") == "MR")

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>VIPER v3 — {sym} Unified Replay (ORB + MR)</title>
<script src="https://unpkg.com/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.js"></script>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:#0a0a0a; color:#ddd; font-family:'Segoe UI',system-ui,sans-serif; }}
#header {{ padding:12px 20px; display:flex; justify-content:space-between; align-items:center; background:#111; border-bottom:1px solid #222; }}
#header h1 {{ font-size:18px; color:#00d4aa; }}
#stats {{ font-size:14px; color:#888; }}
#stats .win {{ color:#00d4aa; }} #stats .loss {{ color:#ff4757; }}
#stats .orb {{ color:#2962ff; }} #stats .mr {{ color:#f0b90b; }}
#controls {{ padding:8px 20px; background:#111; display:flex; gap:12px; align-items:center; border-bottom:1px solid #222; }}
button {{ background:#222; color:#ddd; border:1px solid #333; padding:6px 16px; cursor:pointer; border-radius:4px; font-size:13px; }}
button:hover {{ background:#333; }} button.active {{ background:#00d4aa; color:#000; }}
.tf-btn.active {{ background:#2962ff; color:#fff; border-color:#2962ff; }}
#main {{ display:flex; width:100%; height:calc(100vh - 100px); }}
#left {{ flex:1 1 70%; display:flex; flex-direction:column; min-width:0; }}
#chart {{ width:100%; flex:1; min-height:0; }}
#equity {{ width:100%; height:120px; border-top:1px solid #222; }}
#sidebar {{ flex:0 0 380px; background:#0d0d0d; border-left:1px solid #222; overflow-y:auto; padding:10px; }}
#sidebar h3 {{ font-size:12px; color:#666; margin-bottom:8px; text-transform:uppercase; letter-spacing:1px; }}
.card {{ background:#141414; border:1px solid #222; border-radius:6px; padding:10px 12px; margin-bottom:8px; font-size:12px; line-height:1.55; cursor:pointer; transition:all 0.15s; font-family:'SF Mono','Consolas',monospace; }}
.card:hover {{ border-color:#444; background:#181818; }}
.card.active {{ border-color:#00d4aa; background:#152520; box-shadow:0 0 0 1px #00d4aa; }}
.card.loss {{ border-left:3px solid #ff4757; }}
.card.win {{ border-left:3px solid #00d4aa; }}
.card .hdr {{ font-weight:600; margin-bottom:6px; font-size:13px; }}
.card .row {{ display:flex; justify-content:space-between; color:#aaa; }}
.card .row code {{ color:#fff; font-size:11px; }}
.card .sep {{ height:1px; background:#222; margin:6px 0; }}
.card .reason {{ color:#888; font-size:11px; margin-top:4px; }}
.card .outcome {{ margin-top:6px; padding-top:6px; border-top:1px dashed #222; font-weight:600; }}
.card .outcome.win {{ color:#00d4aa; }}
.card .outcome.loss {{ color:#ff4757; }}
.strat-tag {{ padding:1px 6px; border-radius:3px; font-size:10px; margin-left:4px; font-weight:700; }}
.strat-tag.orb {{ background:#2962ff; color:#fff; }}
.strat-tag.mr {{ background:#f0b90b; color:#000; }}
#info {{ position:fixed; bottom:130px; left:20px; background:rgba(0,0,0,0.9); padding:12px 16px; border-radius:6px; border:1px solid #333; font-size:13px; display:none; z-index:10; max-width:400px; }}
</style></head><body>
<div id="header">
    <h1>VIPER v3 — {sym} Unified Replay</h1>
    <div id="stats">
        <span>{len(trades)} trades</span> |
        <span class="win">{wins}W</span> / <span class="loss">{len(trades)-wins}L</span> |
        <span class="{'win' if total_pnl>0 else 'loss'}">${total_pnl:,.2f}</span> |
        <span class="orb">{orb_count} ORB</span> · <span class="mr">{mr_count} MR</span>
    </div>
</div>
<div id="controls">
    <button id="playBtn" onclick="togglePlay()">▶ Play</button>
    <button onclick="setSpeed(1)">1x</button>
    <button onclick="setSpeed(5)" class="active">5x</button>
    <button onclick="setSpeed(20)">20x</button>
    <button onclick="setSpeed(50)">50x</button>
    <button onclick="skipTrade(-1)">⏮</button>
    <button onclick="skipTrade(1)">⏭</button>
    <button onclick="showAll()">Show All</button>
    <span style="margin-left:16px;color:#555">|</span>
    <button onclick="switchTF('1H')" class="tf-btn active" id="tf-1H">1H</button>
    <button onclick="switchTF('4H')" class="tf-btn" id="tf-4H">4H</button>
    <button onclick="switchTF('1D')" class="tf-btn" id="tf-1D">1D</button>
    <button onclick="switchTF('1W')" class="tf-btn" id="tf-1W">1W</button>
    <span id="speed" style="margin-left:12px">Speed: 5x</span>
    <span id="progress" style="margin-left:auto;color:#666"></span>
</div>
<div id="main">
  <div id="left">
    <div id="chart"></div>
    <div id="equity"></div>
  </div>
  <div id="sidebar">
    <h3 id="sidebar-title">Signals</h3>
    <div id="cards"></div>
  </div>
</div>
<div id="info"></div>
<script>
const SYM='{sym}';
const allTF={all_tf_json};
let allCandles={candles_json};
const allTrades={trades_json};
const allEquity={equity_json};
let currentTF='1H',currentIdx=0,playing=false,speed=5,timer=null,tradeIdx=-1,allMarkers=[],tradeBoxes=[];
let signals=[];

const chartEl=document.getElementById('chart');
const chart=LightweightCharts.createChart(chartEl,{{
    width:chartEl.clientWidth,height:chartEl.clientHeight,
    layout:{{background:{{color:'#0a0a0a'}},textColor:'#999'}},
    grid:{{vertLines:{{color:'#1a1a1a'}},horzLines:{{color:'#1a1a1a'}}}},
    crosshair:{{mode:0}},rightPriceScale:{{borderColor:'#222'}},
    timeScale:{{borderColor:'#222',timeVisible:true}},
}});
const candleSeries=chart.addCandlestickSeries({{
    upColor:'#00d4aa',downColor:'#ff4757',borderUpColor:'#00d4aa',borderDownColor:'#ff4757',
    wickUpColor:'#00d4aa',wickDownColor:'#ff4757',
}});
const volSeries=chart.addHistogramSeries({{priceFormat:{{type:'volume'}},priceScaleId:'vol'}});
chart.priceScale('vol').applyOptions({{scaleMargins:{{top:0.85,bottom:0}}}});

const eqEl=document.getElementById('equity');
const eqChart=LightweightCharts.createChart(eqEl,{{
    width:eqEl.clientWidth,height:eqEl.clientHeight,
    layout:{{background:{{color:'#0a0a0a'}},textColor:'#999'}},
    grid:{{vertLines:{{color:'#1a1a1a'}},horzLines:{{color:'#1a1a1a'}}}},
    rightPriceScale:{{borderColor:'#222'}},timeScale:{{borderColor:'#222',timeVisible:true}},
}});
const eqLine=eqChart.addLineSeries({{color:'#00d4aa',lineWidth:2}});

function snapTime(ts){{let best=allCandles[0].time,bd=Math.abs(best-ts);
    for(const c of allCandles){{const d=Math.abs(c.time-ts);if(d<bd){{best=c.time;bd=d;}}}}return best;}}

allTrades.forEach(t=>{{
    t.entry_ct=snapTime(Math.floor(new Date(t.entry_time+'Z').getTime()/1000));
    t.exit_ct=snapTime(Math.floor(new Date(t.exit_time+'Z').getTime()/1000));
}});

function buildSignalTrades(){{
    const m=new Map();
    allTrades.forEach(t=>{{
        const key=t.entry_time+t.strategy;
        if(!m.has(key)){{
            m.set(key,{{entry_time:t.entry_time,entry_ct:t.entry_ct,exit_ct:t.exit_ct,
                entry_price:t.entry_price,sl:t.sl,tp:t.tp,side:t.side,strategy:t.strategy,pnl:0,reason:''}});
        }}
        const s=m.get(key);s.pnl+=t.pnl;
        if(t.exit_ct>=s.exit_ct){{s.exit_ct=t.exit_ct;s.reason=t.reason;}}
    }});
    return [...m.values()];
}}
let signalTrades=buildSignalTrades();

function buildSignals(){{
    const g=new Map();
    allTrades.forEach(t=>{{
        const key=t.entry_time+t.strategy;
        if(!g.has(key)){{
            g.set(key,{{entry_time:t.entry_time,entry_ct:t.entry_ct,entry_price:t.entry_price,
                sl:t.sl,tp:t.tp,side:t.side,strategy:t.strategy,rr:t.rr,
                sig_reason:t.sig_reason,bias_info:t.bias_info,lots_orig:t.lots_orig,
                risk_usd:t.risk_usd,pnl:0,exits:[]}});
        }}
        const s=g.get(key);s.pnl+=t.pnl;
        s.exits.push({{reason:t.reason,pnl:t.pnl,exit_time:t.exit_time}});
    }});
    return [...g.values()].sort((a,b)=>b.entry_ct-a.entry_ct);
}}

function renderCards(){{
    signals=buildSignals();
    document.getElementById('sidebar-title').textContent=`Signals (${{signals.length}})`;
    const html=signals.map((s,idx)=>{{
        const emoji=s.side==='long'?'🟢':'🔴';
        const dir=s.side==='long'?'BUY':'SELL';
        const cls=s.pnl>=0?'win':'loss';
        const sign=s.pnl>=0?'+':'';
        const stag=`<span class="strat-tag ${{s.strategy.toLowerCase()}}">${{s.strategy}}</span>`;
        const exits=s.exits.map(e=>`${{e.reason}} ${{e.pnl>=0?'+':''}}$${{e.pnl.toFixed(2)}}`).join(' · ');
        const icon=s.pnl>=0?'✅':'❌';
        return `<div class="card ${{cls}}" data-idx="${{idx}}" data-ct="${{s.entry_ct}}" onclick="jumpToSignal(${{idx}})">
            <div class="hdr">${{emoji}} ${{dir}} ${{SYM}} ${{stag}}</div>
            <div class="row"><span>📍 Entry</span><code>${{s.entry_price.toFixed(2)}}</code></div>
            <div class="row"><span>🛑 SL</span><code>${{s.sl.toFixed(2)}}</code></div>
            <div class="row"><span>🎯 TP</span><code>${{s.tp.toFixed(2)}}</code></div>
            <div class="sep"></div>
            <div class="row"><span>📦 Lot</span><code>${{s.lots_orig.toFixed(2)}}</code></div>
            <div class="row"><span>💰 Risk</span><code>$${{s.risk_usd.toFixed(0)}}</code></div>
            <div class="row"><span>📊 R:R</span><code>1:${{s.rr.toFixed(1)}}</code></div>
            <div class="sep"></div>
            <div style="color:#888;font-size:11px">📊 ${{s.bias_info}}</div>
            <div class="reason">🧠 ${{s.sig_reason}}</div>
            <div style="color:#555;font-size:10px;margin-top:4px">⏰ ${{s.entry_time}} UTC</div>
            <div class="outcome ${{cls}}">${{icon}} ${{sign}}$${{s.pnl.toFixed(2)}} — ${{exits}}</div>
        </div>`;
    }}).join('');
    document.getElementById('cards').innerHTML=html;
}}

function highlightCard(idx,scroll){{
    document.querySelectorAll('.card').forEach(c=>c.classList.remove('active'));
    const card=document.querySelector(`.card[data-idx="${{idx}}"]`);
    if(card){{card.classList.add('active');
        if(scroll)card.scrollIntoView({{behavior:'smooth',block:'center'}});}}
}}

function findSignalByTime(t){{return signals.findIndex(s=>s.entry_ct===t);}}

function jumpToSignal(idx){{
    const s=signals[idx];tradeIdx=allTrades.findIndex(t=>t.entry_time===s.entry_time&&t.strategy===s.strategy);
    for(let i=0;i<allCandles.length;i++){{if(allCandles[i].time>=s.entry_ct){{
        const end=Math.min(i+50,allCandles.length);
        candleSeries.setData(allCandles.slice(0,end).map(c=>({{time:c.time,open:c.open,high:c.high,low:c.low,close:c.close}})));
        volSeries.setData(allCandles.slice(0,end).map(c=>({{time:c.time,value:c.volume,color:c.close>=c.open?'rgba(0,212,170,0.2)':'rgba(255,71,87,0.2)'}})));
        currentIdx=end;allMarkers=[];
        tradeBoxes.forEach(ln=>{{try{{chart.removeSeries(ln)}}catch(e){{}}}});tradeBoxes=[];
        signalTrades.forEach(tr=>{{if(tr.entry_ct<=allCandles[currentIdx-1].time){{
            allMarkers.push({{time:tr.entry_ct,position:tr.side==='long'?'belowBar':'aboveBar',
                color:tr.pnl>0?'#00d4aa':'#ff4757',shape:tr.side==='long'?'arrowUp':'arrowDown',
                text:`[${{tr.strategy}}] $`+tr.pnl.toFixed(2)}});
            if(tr.exit_ct<=allCandles[currentIdx-1].time){{drawBox(tr);}}
        }}}});
        allTrades.forEach(tr=>{{if(tr.exit_ct<=allCandles[currentIdx-1].time){{
            allMarkers.push({{time:tr.exit_ct,position:'inBar',color:tr.pnl>0?'#00d4aa':'#ff4757',
                shape:'circle',text:'✕ '+tr.reason+' $'+tr.pnl.toFixed(2)}});}}}});
        allMarkers.sort((a,b)=>a.time-b.time);candleSeries.setMarkers(allMarkers);
        chart.timeScale().scrollToPosition(-5,false);highlightCard(idx,true);break;}}}}
}}

function addBar(idx){{
    if(idx>=allCandles.length)return;const c=allCandles[idx],t=c.time;
    candleSeries.update({{time:t,open:c.open,high:c.high,low:c.low,close:c.close}});
    volSeries.update({{time:t,value:c.volume,color:c.close>=c.open?'rgba(0,212,170,0.2)':'rgba(255,71,87,0.2)'}});
    if(idx<allEquity.length)eqLine.update({{time:t,value:allEquity[idx].eq}});
    signalTrades.forEach(tr=>{{
        if(tr.entry_ct===t){{
            allMarkers.push({{time:t,position:tr.side==='long'?'belowBar':'aboveBar',
                color:tr.pnl>0?'#00d4aa':'#ff4757',
                shape:tr.side==='long'?'arrowUp':'arrowDown',
                text:`[${{tr.strategy}}] ${{tr.side==='long'?'▲':'▼'}} $`+tr.pnl.toFixed(2)}});
            showInfo(tr,'ENTRY');drawBox(tr);
            const sidx2=findSignalByTime(tr.entry_ct);if(sidx2>=0)highlightCard(sidx2,true);
        }}
    }});
    allTrades.forEach(tr=>{{
        if(tr.exit_ct===t){{
            allMarkers.push({{time:t,position:'inBar',
                color:tr.pnl>0?'#00d4aa':'#ff4757',shape:'circle',
                text:'✕ '+tr.reason+' $'+tr.pnl.toFixed(2)}});
        }}
    }});
    allMarkers.sort((a,b)=>a.time-b.time);candleSeries.setMarkers(allMarkers);
    document.getElementById('progress').textContent=(idx+1)+'/'+allCandles.length;
    chart.timeScale().setVisibleLogicalRange({{from:Math.max(0,idx-180),to:idx+20}});
}}

function drawBox(tr){{
    const t1=tr.entry_ct,t2=tr.exit_ct;
    const tpFill=chart.addBaselineSeries({{
        baseValue:{{type:'price',price:tr.entry_price}},
        topFillColor1:'rgba(0,212,170,0.22)',topFillColor2:'rgba(0,212,170,0.22)',
        bottomFillColor1:'rgba(0,212,170,0.22)',bottomFillColor2:'rgba(0,212,170,0.22)',
        topLineColor:'rgba(0,0,0,0)',bottomLineColor:'rgba(0,0,0,0)',
        lineWidth:0,lastValueVisible:false,priceLineVisible:false}});
    tpFill.setData([{{time:t1,value:tr.tp}},{{time:t2,value:tr.tp}}]);
    const slFill=chart.addBaselineSeries({{
        baseValue:{{type:'price',price:tr.entry_price}},
        topFillColor1:'rgba(255,71,87,0.22)',topFillColor2:'rgba(255,71,87,0.22)',
        bottomFillColor1:'rgba(255,71,87,0.22)',bottomFillColor2:'rgba(255,71,87,0.22)',
        topLineColor:'rgba(0,0,0,0)',bottomLineColor:'rgba(0,0,0,0)',
        lineWidth:0,lastValueVisible:false,priceLineVisible:false}});
    slFill.setData([{{time:t1,value:tr.sl}},{{time:t2,value:tr.sl}}]);
    for(const[val,col] of [[tr.entry_price,'#fff'],[tr.sl,'#ff4757'],[tr.tp,'#00d4aa']]){{
        const ln=chart.addLineSeries({{color:col,lineWidth:1,lineStyle:2,lastValueVisible:false,priceLineVisible:false}});
        ln.setData([{{time:t1,value:val}},{{time:t2,value:val}}]);
        tradeBoxes.push(ln);
    }}
    tradeBoxes.push(tpFill,slFill);
}}

function showInfo(t,label){{const el=document.getElementById('info');
    el.innerHTML=`<b>${{label}}</b> <span style="color:#2962ff">[${{t.strategy}}]</span> <span style="color:${{t.pnl>0?'#00d4aa':'#ff4757'}}">${{t.side.toUpperCase()}}</span><br>
        Entry: ${{t.entry_price.toFixed(2)}} | SL: ${{t.sl.toFixed(2)}} | TP: ${{t.tp.toFixed(2)}}<br>
        <span style="color:${{t.pnl>0?'#00d4aa':'#ff4757'}}">PnL: $${{t.pnl.toFixed(2)}}</span> | ${{t.reason}}`;
    el.style.display='block';setTimeout(()=>el.style.display='none',6000);}}

function togglePlay(){{playing=!playing;document.getElementById('playBtn').textContent=playing?'⏸ Pause':'▶ Play';
    if(playing){{if(currentIdx===0)allMarkers=[];startReplay();}}else clearInterval(timer);}}
function startReplay(){{clearInterval(timer);timer=setInterval(()=>{{
    if(currentIdx>=allCandles.length){{togglePlay();return;}}addBar(currentIdx++);
}},Math.max(10,200/speed));}}
function setSpeed(s){{speed=s;document.getElementById('speed').textContent='Speed: '+s+'x';
    document.querySelectorAll('#controls button').forEach(b=>b.classList.remove('active'));
    event.target.classList.add('active');if(playing)startReplay();}}

function skipTrade(dir){{tradeIdx+=dir;if(tradeIdx<0)tradeIdx=0;
    if(tradeIdx>=allTrades.length)tradeIdx=allTrades.length-1;
    const t=allTrades[tradeIdx];
    for(let i=0;i<allCandles.length;i++){{if(allCandles[i].time>=t.entry_ct){{
        const end=Math.min(i+50,allCandles.length);
        candleSeries.setData(allCandles.slice(0,end).map(c=>({{time:c.time,open:c.open,high:c.high,low:c.low,close:c.close}})));
        currentIdx=end;allMarkers=[];
        signalTrades.forEach(tr=>{{if(tr.entry_ct<=allCandles[currentIdx-1].time){{
            allMarkers.push({{time:tr.entry_ct,position:tr.side==='long'?'belowBar':'aboveBar',
                color:tr.pnl>0?'#00d4aa':'#ff4757',shape:tr.side==='long'?'arrowUp':'arrowDown',
                text:`[${{tr.strategy}}] $`+tr.pnl.toFixed(2)}});}}
        }});allMarkers.sort((a,b)=>a.time-b.time);candleSeries.setMarkers(allMarkers);
        showInfo(t,'TRADE #'+(tradeIdx+1));chart.timeScale().scrollToPosition(-5,false);break;}}}}}}

function switchTF(tf){{currentTF=tf;allCandles=allTF[tf];
    document.querySelectorAll('.tf-btn').forEach(b=>b.classList.remove('active'));
    document.getElementById('tf-'+tf).classList.add('active');
    tradeBoxes.forEach(s=>{{try{{chart.removeSeries(s)}}catch(e){{}}}});tradeBoxes=[];
    candleSeries.setData(allCandles.map(c=>({{time:c.time,open:c.open,high:c.high,low:c.low,close:c.close}})));
    volSeries.setData(allCandles.map(c=>({{time:c.time,value:c.volume,color:c.close>=c.open?'rgba(0,212,170,0.2)':'rgba(255,71,87,0.2)'}})));
    allTrades.forEach(t=>{{t.entry_ct=snapTime(Math.floor(new Date(t.entry_time+'Z').getTime()/1000));
        t.exit_ct=snapTime(Math.floor(new Date(t.exit_time+'Z').getTime()/1000));}});
    signalTrades=buildSignalTrades();
    signalTrades.forEach(t=>drawBox(t));
    allMarkers=[];signalTrades.forEach(t=>{{allMarkers.push({{time:t.entry_ct,position:t.side==='long'?'belowBar':'aboveBar',
        color:t.pnl>0?'#00d4aa':'#ff4757',shape:t.side==='long'?'arrowUp':'arrowDown',
        text:`[${{t.strategy}}] $`+t.pnl.toFixed(2)}});}});
    allTrades.forEach(t=>{{allMarkers.push({{time:t.exit_ct,position:'inBar',color:t.pnl>0?'#00d4aa':'#ff4757',
        shape:'circle',text:'✕ '+t.reason}});}});
    allMarkers.sort((a,b)=>a.time-b.time);candleSeries.setMarkers(allMarkers);
    chart.timeScale().fitContent();currentIdx=allCandles.length;renderCards();}}

function showAll(){{playing=false;clearInterval(timer);document.getElementById('playBtn').textContent='▶ Play';
    tradeBoxes.forEach(s=>{{try{{chart.removeSeries(s)}}catch(e){{}}}});tradeBoxes=[];
    candleSeries.setData(allCandles.map(c=>({{time:c.time,open:c.open,high:c.high,low:c.low,close:c.close}})));
    volSeries.setData(allCandles.map(c=>({{time:c.time,value:c.volume,color:c.close>=c.open?'rgba(0,212,170,0.2)':'rgba(255,71,87,0.2)'}})));
    eqLine.setData(allEquity.map(e=>({{time:e.time,value:e.eq}})));
    currentIdx=allCandles.length;allMarkers=[];
    signalTrades.forEach(t=>{{allMarkers.push({{time:t.entry_ct,position:t.side==='long'?'belowBar':'aboveBar',
        color:t.pnl>0?'#00d4aa':'#ff4757',shape:t.side==='long'?'arrowUp':'arrowDown',
        text:`[${{t.strategy}}] $`+t.pnl.toFixed(2)}});drawBox(t);}});
    allTrades.forEach(t=>{{allMarkers.push({{time:t.exit_ct,position:'inBar',color:t.pnl>0?'#00d4aa':'#ff4757',
        shape:'circle',text:'✕ '+t.reason+' $'+t.pnl.toFixed(2)}});}});
    allMarkers.sort((a,b)=>a.time-b.time);candleSeries.setMarkers(allMarkers);
    chart.timeScale().fitContent();document.getElementById('progress').textContent=allCandles.length+'/'+allCandles.length;}}

window.addEventListener('resize',()=>{{chart.resize(chartEl.clientWidth,chartEl.clientHeight);
    eqChart.resize(eqEl.clientWidth,eqEl.clientHeight);}});
window.addEventListener('keydown',e=>{{if(e.code==='Space'&&e.target.tagName!=='INPUT'){{e.preventDefault();togglePlay();}}}});
currentIdx=0;allMarkers=[];renderCards();
document.getElementById('progress').textContent='Press Play or Show All';
</script></body></html>"""
    return html


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="US30")
    parser.add_argument("--days", type=int, default=0, help="Filter to last N days (0=all)")
    args = parser.parse_args()
    sym = args.symbol

    if sym not in INSTRUMENTS:
        print(f"Available: {', '.join(INSTRUMENTS.keys())}"); return

    print(f"\n  VIPER v3 — Generating {sym} unified replay{' (last '+str(args.days)+'d)' if args.days else ''}...\n")
    cfg = INSTRUMENTS[sym]
    tf = fetch(cfg["ticker"])

    print(f"  Running backtest...", end=" ", flush=True)
    trades, eq_data, d1h = run_bt(sym, tf)

    if args.days > 0:
        cutoff = tf["1h"].index[-1] - pd.Timedelta(days=args.days)
        cutoff_str = str(cutoff)[:19]
        trades = [t for t in trades if t["entry_time"] >= cutoff_str]
        eq_data = [e for e in eq_data if e["ts"] >= cutoff]
        tf = {k: v[v.index >= cutoff] for k, v in tf.items()}

    wins = sum(1 for t in trades if t["pnl"] > 0)
    pnl = sum(t["pnl"] for t in trades)
    orb = sum(1 for t in trades if t.get("strategy") == "ORB")
    mr = sum(1 for t in trades if t.get("strategy") == "MR")
    print(f"{len(trades)}T {wins}W ${pnl:,.2f} ({orb} ORB / {mr} MR)")

    def to_candles(df, n=20000):
        d = df.tail(n).copy()
        d.index = d.index.tz_localize(None) if d.index.tz else d.index
        return [{"time": int(ts.timestamp()),
                 "open": round(float(r["open"]), 6),
                 "high": round(float(r["high"]), 6),
                 "low": round(float(r["low"]), 6),
                 "close": round(float(r["close"]), 6),
                 "volume": round(float(r.get("volume", 0)), 2)}
                for ts, r in d.iterrows()]

    all_tf = {"1H": to_candles(tf["1h"], 20000), "4H": to_candles(tf["4h"], 5000),
              "1D": to_candles(tf["daily"], 1000), "1W": to_candles(tf["weekly"], 300)}

    candle_times = [c["time"] for c in all_tf["1H"]]

    def snap(ts_str):
        t = pd.Timestamp(ts_str)
        if t.tzinfo: t = t.tz_localize(None)
        u = int(t.timestamp())
        return min(candle_times, key=lambda ct: abs(ct - u))

    for t in trades:
        t["entry_ct"] = snap(t["entry_time"])
        t["exit_ct"] = snap(t["exit_time"])

    eq_json = []
    for e in eq_data:
        t = e["ts"]
        if t.tzinfo: t = t.tz_localize(None)
        eq_json.append({"time": int(t.timestamp()), "eq": e["eq"]})

    html = generate_html(sym, all_tf, trades, eq_json)
    suffix = f"_{args.days}d" if args.days else ""
    fname = f"replay_unified_{sym.lower()}{suffix}.html"
    with open(fname, "w") as f:
        f.write(html)

    print(f"\n  Saved: {fname}")
    print(f"  Open: file://{os.path.abspath(fname)}")


if __name__ == "__main__":
    main()
