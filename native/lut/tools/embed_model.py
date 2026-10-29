#!/usr/bin/env python3
"""Deterministically encode an immutable weight file for C/C++ embedding."""
import argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('model',type=Path);p.add_argument('header',type=Path);a=p.parse_args();symbol='dos_lut_'+a.model.stem.replace('-','_');guard=symbol.upper()+'_H';raw=a.model.read_bytes();a.header.write_text('/* Generated from '+a.model.name+'; do not edit weights by hand. */\n#ifndef '+guard+'\n#define '+guard+'\n#include <stddef.h>\n#include <stdint.h>\nstatic const uint8_t '+symbol+'[] = {\n'+''.join('  '+','.join('0x%02x'%b for b in raw[i:i+16])+',\n' for i in range(0,len(raw),16))+'};\nstatic const size_t '+symbol+'_size = sizeof('+symbol+');\n#endif\n')
