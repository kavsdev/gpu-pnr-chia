#!/bin/bash
python3 -c "
import json
with open('/opt/hammer/lib/python3.10/site-packages/hammer/technology/nangate45/nangate45.tech.json', 'r') as f:
    tech = json.load(f)
for s in tech['stackups']:
    for m in s['metals']:
        m['grid_unit'] = tech['grid_unit']
with open('/opt/hammer/lib/python3.10/site-packages/hammer/technology/nangate45/nangate45.tech.json', 'w') as f:
    json.dump(tech, f, indent=2)
"
hammer-vlsi "$@"
