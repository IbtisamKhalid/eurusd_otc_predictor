"""Train and save all models from stored candles (chronological fit + calibration)."""
import _common  # noqa: F401
import config
from config import DB_PATH, Settings
from data.database import Database
from features.engineering import build_dataset, trainable
from models.training import MODEL_NAMES, save_model, train_model


def main() -> int:
    s, db = Settings(), Database(DB_PATH)
    candles = db.load_candles(s.venue, s.symbol, s.timeframe_s)
    ds = trainable(build_dataset(candles, s.timeframe_s, s.doji_tolerance)) if len(candles) else candles
    print(f"{len(candles)} candles -> {len(ds)} usable training rows")
    for name in MODEL_NAMES:
        try:
            tm = train_model(ds, name, s.min_train_samples)
        except ValueError as exc:
            print(f"{name}: {exc}")
            return 1
        save_model(tm, config.BASE_DIR / "data_storage" / "models")
        db.save_model_version(tm.version, name, tm.n_train, tm.calibration.status, tm.info)
        print(f"{name}: version={tm.version} fit_rows={tm.n_train} | {tm.calibration.status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
