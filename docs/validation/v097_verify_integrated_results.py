"""Validate evidence completeness, paired source accounting and conservation.

Standalone check of the committed results; no private files or simulations.
Expected historical numerical failures remain errors in the evidence.
"""
import json
from pathlib import Path

data=json.loads(Path(__file__).with_name('v0.9.7_integrated_results.json').read_text(encoding='utf-8'))
assert not data['production_code_changed']
assert len(data['audit_sources']) == 9
for name,count in [('mihara',48),('onset',30),('benchmark_trace',32)]:
    for snapshot in ('base','current'):
        rows=data[name][snapshot]
        assert len(rows)==count, (name,snapshot,len(rows))
        assert len({r['key'] for r in rows})==count
    before={r['key']:r for r in data[name]['base']}
    after={r['key']:r for r in data[name]['current']}
    assert before.keys()==after.keys()
    if name=='onset':
        assert all(r['status']=='PASS' for r in (*before.values(),*after.values()))
        for key,row in after.items():
            if row['provider'] in ('Elmahdy','Disabled'):
                for field in ('Q_W','condensate_kg_s','gas_out_K','water_out_K'):
                    assert row[field]==before[key][field], (key,field)
    if name=='benchmark_trace':
        assert all(r['status']=='PASS' for r in after.values())
        assert [r['key'] for r in before.values() if r['status']!='PASS']==['D08_cross_small_AUTO']
    if name=='mihara':
        for key,row in after.items():
            assert row['status']==before[key]['status']
            if row['status']=='PASS':
                assert row['provider']=='Elmahdy'
                for field in ('Q_W','condensate_kg_s','gas_out_K','water_out_K'):
                    assert row[field]==before[key][field], (key,field)
        assert sum(r['status']=='PASS' for r in after.values())==24
    for row in after.values():
        if row['status']!='PASS':
            continue
        q=row.get('Q_W',row.get('Q_process_kW',0.)*1000)
        gas=row.get('gas_residual_W',0.)
        water=row.get('water_residual_W',row.get('coolant_residual_W',0.))
        assert max(abs(gas),abs(water)) <= max(1.,.001*abs(q)), (name,row['key'])
        assert abs(row['mass_residual_kg_s'])<=1e-8
        for field,value in row.get('audit',{}).items():
            if field.endswith('_W'):
                assert abs(value)<.01, (row['key'],field,value)
for row in data['calorimetry']:
    assert row['print_min_kW']<row['Q_kW']<row['print_max_kW']
print('PASS: 9 audit hashes; 220 paired executions; conservation, unchanged Elmahdy/dry outputs and historical failures')
