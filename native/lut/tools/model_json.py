#!/usr/bin/env python3
"""Lossless editable JSON for the format-4 model. Python standard library only."""
import argparse
import json
from pathlib import Path
from table_model import TableModel

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation', choices=['pack', 'unpack'])
    p.add_argument('source', type=Path)
    p.add_argument('destination', type=Path)
    a = p.parse_args()
    if a.operation == 'unpack':
        m = TableModel(a.source)
        data = {'format': 'DOSLUT3', 'version': 4, 'banks': 3,
                'ops': m.ops, 'blocks': m.blocks, 'primary': m.primary,
                'tables': m.tables}
        raw = (json.dumps(data, indent=2) + '\n').encode()
    else:
        data = json.loads(a.source.read_text())
        if (data['format'], data['version'], data['banks']) != ('DOSLUT3', 4, 3):
            raise ValueError('Expected DOSLUT3 format 4, three banks')
        if len(data['primary']) != 768:
            raise ValueError('Expected 768 primary entries')
        m = TableModel.__new__(TableModel)
        for field in ('ops', 'blocks', 'primary', 'tables'):
            setattr(m, field, data[field])
        if any(len(x) != 8 for x in m.ops) or any(len(x) != 9 for x in m.blocks):
            raise ValueError('Invalid operation or block dimensions')
        raw = m.bytes()
    with a.destination.open('xb') as f:
        f.write(raw)

if __name__ == '__main__':
    main()
