"""Systematic Portfolio Builder — backend engine.

Universe selection, schema-derived factor catalog, point-in-time screening, and
a vectorized backtest, all reading the investment RDS strictly read-only with
institutional point-in-time correctness (the 6-month annual-data reporting lag +
dated index membership). Persistence (saved strategies / custom factors /
backtest jobs) lives in PRISM's own Postgres — never the read-only RDS.
"""
