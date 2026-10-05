import pandas as pd

for rec, risk in zip(audit_slice.REC.head(8), audit_slice.RISK.head(8)):
    a = alert_txt.get(rec, "")
    r = risk_txt.get(risk, "")
    print(f"{rec} / {risk}")
    print(f"  alert_txt len={len(a):5d}  risk_txt len={len(r):5d}")
    print(f"  risk_txt = {r!r}")
    print()

print("--- risk_txt length distribution across the full false-negative slice ---")
print(pd.Series([len(risk_txt.get(r, "")) for r in audit_slice.RISK]).describe())
