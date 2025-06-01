/ --- Data Loading ---
loadData:{[filepath]
  ("DS"; enlist ",") 0: filepath
}

/ --- Historical Volatility ---
histVol:{[px;window]
  r:log px % prev px;
  dev:dev r;
  sqrt[252]*dev window _ r
}

/ --- Realized Variance ---
realizedVar:{[r]
  sum r*r
}

/ --- Simple GARCH(1,1) Model ---
garch11:{[r;omega;alpha;beta]
  n:count r;
  h:enlist var r;
  do[n-1; h,: omega + alpha * last h * last r * last r + beta * last h];
  h
}

/ --- ARIMA(1,1,1) Model (Simplified) ---
arima111:{[x;phi;theta;mu]
  d:1 _ x - prev x;
  y:enlist first d;
  do[count d - 1; y,: mu + phi * last y + theta * last d];
  y
}

/ --- Tick-Level Transaction Analysis ---
tickStats:{[tbl]
  select count i as nTicks,
         avg price as avgPx,
         max price as maxPx,
         min price as minPx,
         sum size as totalVol
  by sym from tbl
}

/ --- Example Usage ---
/ px: select price from trade where sym=`AAPL
/ hv: histVol[px;21]
/ rv: realizedVar[hv]
/ garch: garch11[hv;0.00001;0.05;0.9]
/ arima: arima111[px;0.7;0.2;0]
/ stats: tickStats[trade]