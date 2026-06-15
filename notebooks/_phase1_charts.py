"""Generate Phase 1 historical charts."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import utils, data_loader as dl, financials, charts

cfg = utils.load_config()
yahoo = dl.get_yahoo(cfg, cfg["company"]["ticker"])
facts = dl.get_edgar_facts(cfg, cfg["company"]["cik"])
H = financials.build_historical(cfg, yahoo, facts)
paths = charts.generate_historical_charts(cfg, H, yahoo)
print("Generated charts:")
for p in paths:
    print("  ", Path(p).name)
