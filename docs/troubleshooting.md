# Troubleshooting

## ModuleNotFoundError
```bash
pip install -r requirements.txt
```

## Missing model files
Model `.pkl`/`.keras` files are excluded (>100MB). Re-run `src/07_track_a_models.py`.

## Weather API not working
Add to `.streamlit/secrets.toml`: `OPENWEATHER_API_KEY = "your-key"`. App works without it.

## India Map blank
```bash
pip install plotly>=5.15
```

## Sidebar collapse/reopen
App uses button navigation — refresh page (F5) if sidebar buttons stop responding.
