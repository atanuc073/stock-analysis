import json

names = ['AngloGold', 'Newmont', 'Altria', 'Southern Copper', 'EOG', 'Range Resources', 'Cirrus', 'Gilead', 'CF Industries', 'APA', 'Exelixis', 'Match', 'Bristol', 'Evercore', 'Mueller', 'Garmin', 'Popular', 'Incyte', 'Corpay', 'Maplebear', 'Primerica']
with open('D:/MY_WORK/stock_analysis/reports/report_2026-05-13.json', encoding='utf-8') as f:
    data = json.load(f)

res = [s for s in data if any(n.lower() in s.get('name', '').lower() for n in names)]
res.sort(key=lambda x: x.get('composite_score', 0), reverse=True)
for r in res:
    print(f"{r['symbol']:<10} {r.get('composite_score', 0):>5.1f} ({r['verdict']}) - {r.get('name', '')[:30]}")
