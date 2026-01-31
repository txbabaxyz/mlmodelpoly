# Binance Collector

Real-time trading data collector for Binance Futures/Spot with Polymarket prediction market integration.

## Features

### Data Collection
- **Binance WebSocket Streams**: aggTrade, bookTicker, markPrice, forceOrder, depth
- **Multi-market Support**: Futures and Spot markets simultaneously
- **OHLCV Bar Aggregation**: 5s, 15s, 1m timeframes with delta volume
- **Order Book Depth**: Top N levels with configurable update speed (100ms/500ms)

### Trading Features
- **CVD (Cumulative Volume Delta)**: Real-time buy/sell pressure tracking
- **RVOL (Relative Volume)**: Volume compared to rolling average
- **Impulse Detection**: Price momentum spikes
- **Microprice**: Volume-weighted mid price
- **Basis**: Futures vs Spot premium/discount
- **Anchored VWAP**: Session VWAP with deviation tracking
- **Liquidation Tracking**: Large forced liquidations

### Polymarket Integration
- **Real-time Order Book**: UP/DOWN token prices via WebSocket
- **Fair Value Model**: Probability estimation with fast/smooth modes
- **Spike Detection**: Microstructure spike/dip detection
- **Edge Calculation**: Trading edge in basis points

### Advanced Analytics
- **TAAPI Integration**: External technical indicators (RSI, MACD, etc.)
- **Bias Model**: Multi-timeframe directional bias
- **Volatility Estimation**: Fast/slow/blend sigma calculation
- **Quality Mode**: Data quality monitoring (OK/DEGRADED/BAD)

### API & Monitoring
- **HTTP REST API**: Health checks, features, control endpoints
- **TUI Dashboard**: Terminal-based real-time monitoring
- **Prometheus Metrics**: Lag, throughput, error tracking
- **Structured JSON Logging**: Machine-parseable logs

## Requirements

- Python 3.10+
- Network access to Binance and Polymarket APIs
- (Optional) TAAPI.io API key for technical indicators

## Installation

```bash
# Clone or copy the project
cd binance_collector_clean

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
# Edit .env with your settings
```

## Configuration

All settings are configured via environment variables or `.env` file.

### Core Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `SYMBOL` | `BTCUSDT` | Trading pair symbol |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG/INFO/WARNING/ERROR) |
| `HTTP_HOST` | `0.0.0.0` | API server host |
| `HTTP_PORT` | `8000` | API server port |

### WebSocket Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `FUTURES_WS` | `wss://fstream.binance.com` | Futures WebSocket URL |
| `SPOT_WS` | `wss://stream.binance.com:9443` | Spot WebSocket URL |
| `DEPTH_ENABLED` | `true` | Enable order book depth stream |
| `DEPTH_SPEED` | `100ms` | Depth update speed (100ms/500ms/none) |
| `TOPN` | `10` | Order book levels to track |

### Context Bootstrap

| Variable | Default | Description |
|----------|---------|-------------|
| `CONTEXT_ENABLED` | `true` | Enable HTF klines bootstrap |
| `CONTEXT_TFS` | `1m,5m,15m,1h` | Timeframes to bootstrap |
| `CONTEXT_BOOTSTRAP_LIMIT` | `500` | Klines per timeframe |

### TAAPI Integration

| Variable | Default | Description |
|----------|---------|-------------|
| `TAAPI_ENABLED` | `true` | Enable TAAPI indicators |
| `TAAPI_SECRET` | - | TAAPI.io API secret key |
| `TAAPI_EXCHANGE` | `binancefutures` | Exchange for TAAPI |
| `TAAPI_TFS` | `1m,5m,15m,1h` | Timeframes to fetch |

### Polymarket Integration

| Variable | Default | Description |
|----------|---------|-------------|
| `POLYMARKET_ENABLED` | `true` | Enable Polymarket integration |
| `POLYMARKET_WS_URL` | `wss://ws-subscriptions-clob.polymarket.com/ws/market` | WebSocket URL |
| `POLYMARKET_STALE_THRESHOLD_SEC` | `5.0` | Stale data threshold |
| `POLYMARKET_MIN_DEPTH` | `200.0` | Min depth for execution |
| `POLYMARKET_MAX_SPREAD_BPS` | `500.0` | Max spread before veto |
| `POLY_UP_IS_YES` | `true` | UP = YES token mapping |

### Trading Parameters

| Variable | Default | Description |
|----------|---------|-------------|
| `SLICE_USD` | `20.0` | Standard slice size |
| `MAX_SLICES_PER_WINDOW` | `30` | Max slices per 15-min window |
| `MAX_USD_PER_WINDOW` | `300.0` | Max USD per window |
| `COOLDOWN_SEC` | `2.0` | Cooldown between executions |
| `EDGE_BUFFER_BPS` | `25.0` | Required edge buffer |

### Event Recording

| Variable | Default | Description |
|----------|---------|-------------|
| `RECORD_ENABLED` | `false` | Enable event recording |
| `RECORD_DIR` | `data/recordings` | Recording output directory |
| `RECORD_BUFFER_SIZE` | `100` | Events before flush |

## Usage

### Start the Collector

```bash
cd binance_collector_clean
source venv/bin/activate

# Run with default settings
python -m collector

# Or with custom symbol
SYMBOL=ETHUSDT python -m collector
```

### Start TUI Dashboard

In a separate terminal:

```bash
source venv/bin/activate
python scripts/tui_dashboard.py --host localhost --port 8000
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/state` | GET | Current system state |
| `/latest/features` | GET | Latest computed features |
| `/latest/bars` | GET | Latest OHLCV bars |
| `/latest/edge` | GET | Latest edge decision |
| `/control/anchor/reset` | POST | Reset VWAP anchor |

### Example API Usage

```bash
# Health check
curl http://localhost:8000/health

# Get latest features
curl http://localhost:8000/latest/features | jq .

# Get edge decision
curl http://localhost:8000/latest/edge | jq .
```

## Project Structure

```
binance_collector_clean/
├── src/
│   ├── collector/              # Main collector package
│   │   ├── __init__.py
│   │   ├── main.py            # Entry point
│   │   ├── config.py          # Configuration (pydantic-settings)
│   │   ├── ws_client.py       # Binance WebSocket client
│   │   ├── pipeline.py        # Event processing pipeline
│   │   ├── features.py        # Feature computation engine
│   │   ├── bars.py            # OHLCV bar aggregation
│   │   ├── edge_engine.py     # Trading edge calculation
│   │   ├── fair_model.py      # Fair value estimation
│   │   ├── bias_model.py      # Directional bias model
│   │   ├── volatility.py      # Volatility estimation
│   │   ├── accumulate_engine.py # Trade accumulation logic
│   │   ├── decision_logger.py # Structured decision logging
│   │   ├── http_api.py        # FastAPI REST endpoints
│   │   ├── metrics.py         # Prometheus metrics
│   │   ├── polymarket/        # Polymarket integration
│   │   │   ├── ws_client.py   # PM WebSocket client
│   │   │   ├── book_store.py  # Order book storage
│   │   │   ├── market_resolver.py # Token ID resolution
│   │   │   └── normalize_updown.py # UP/DOWN normalization
│   │   ├── taapi/             # TAAPI.io integration
│   │   │   ├── client.py      # Async HTTP client
│   │   │   ├── store.py       # Indicator storage
│   │   │   └── context_engine.py # Context aggregation
│   │   └── utils/             # Utility functions
│   └── strategies/            # Trading strategies
│       └── z_contra_fav_dip_hedge.py
├── scripts/
│   ├── tui_dashboard.py       # Terminal UI dashboard
│   ├── backtest_*.py          # Backtesting scripts
│   └── analyze_*.py           # Analysis tools
├── data/                      # Data storage (gitignored)
│   ├── decisions/             # Decision logs
│   └── recordings/            # Event recordings
├── logs/                      # Log files (gitignored)
├── requirements.txt           # Python dependencies
├── .env.example              # Example configuration
├── .gitignore                # Git ignore rules
└── README.md                 # This file
```

## Troubleshooting

### Connection Issues

**Problem**: WebSocket disconnects frequently

**Solution**:
1. Check network connectivity to Binance
2. Reduce `DEPTH_SPEED` to `500ms` for lower bandwidth
3. Check firewall/proxy settings

### High CPU Usage

**Problem**: Collector using too much CPU

**Solution**:
1. Set `DEPTH_SPEED=500ms` instead of `100ms`
2. Reduce `TOPN` from 10 to 5
3. Disable unused features (`TAAPI_ENABLED=false`)

### Missing Features

**Problem**: Features showing `null` or `0`

**Solution**:
1. Wait for warmup period (1-5 minutes)
2. Check `CONTEXT_ENABLED=true` for HTF data
3. Verify Binance API is accessible

### TAAPI Errors

**Problem**: TAAPI indicators not updating

**Solution**:
1. Verify `TAAPI_SECRET` is valid
2. Check TAAPI.io rate limits
3. Set `TAAPI_ENABLED=false` if not needed

### Polymarket Issues

**Problem**: Polymarket prices not updating

**Solution**:
1. Verify WebSocket URL is accessible
2. Check if market is currently active
3. Verify token IDs are resolving correctly

## Development

### Running Tests

```bash
source venv/bin/activate
pytest tests/
```

### Code Style

```bash
# Format code
black src/ scripts/

# Check types
mypy src/collector/
```

### Adding New Features

1. Add feature computation to `src/collector/features.py`
2. Expose via HTTP API in `src/collector/http_api.py`
3. Add to TUI dashboard if needed

## License

MIT License

## Contributing

1. Fork the repository
2. Create feature branch
3. Submit pull request

## Support

For issues and questions, please open a GitHub issue.
