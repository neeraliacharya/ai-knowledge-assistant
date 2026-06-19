"""
Centralised structured logging for the AI Knowledge Assistant.

Every module imports get_logger(__name__) instead of using print().
Output is newline-delimited JSON so any log aggregator (Datadog, CloudWatch,
Loki) can parse it without extra configuration.

WHY JSON LOGS?
--------------
Plain print() output is unstructured — you can't filter by level, search by
field, or build dashboards from it. JSON logs let you do:
  - grep '"level":"ERROR"' app.log
  - sum latency_ms grouped by path in Datadog
  - alert when faithfulness score drops below 0.6 in RAGAS logs
"""
import json
import logging
import sys
from datetime import datetime, timezone


class _JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        obj: dict = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            obj["exc"] = self.formatException(record.exc_info)
        # Extra fields passed via extra={"key": value} in log calls
        for key, val in vars(record).items():
            if key not in logging.LogRecord.__dict__ and not key.startswith("_"):
                obj[key] = val
        return json.dumps(obj)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_JSONFormatter())
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger
