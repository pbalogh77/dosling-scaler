"""Lossless format-4 table editing. These edits are authored rules, not training judgments."""
import struct
from pathlib import Path
class TableModel:
 def __init__(self,path):
  raw=Path(path).read_bytes();assert raw[:8]==b'DOSLUT3\0';version,banks,no,nb,ne=struct.unpack_from('<5I',raw,8);assert version==4 and banks==3;self.version=version;pos=28;self.ops=[tuple(raw[pos+i*8:pos+i*8+8]) for i in range(no)];pos+=no*8;self.blocks=[tuple(raw[pos+i*9:pos+i*9+9]) for i in range(nb)];pos+=nb*9;self.primary=[list(struct.unpack_from('<2H',raw,pos+i*4)) for i in range(768)];pos+=768*4;self.tables=[]
  for _ in range(ne):
   n=struct.unpack_from('<I',raw,pos)[0];pos+=4;self.tables.append([tuple(struct.unpack_from('<3H',raw,pos+i*6)) for i in range(n)]);pos+=n*6
  assert pos==len(raw)
 def bytes(self):
  return b'DOSLUT3\0'+struct.pack('<5I',4,3,len(self.ops),len(self.blocks),len(self.tables))+b''.join(bytes(o) for o in self.ops)+b''.join(bytes(b) for b in self.blocks)+b''.join(struct.pack('<2H',*e) for e in self.primary)+b''.join(struct.pack('<I',len(t))+b''.join(struct.pack('<3H',*r) for r in t) for t in self.tables)
 def recipe(self,ops):
  ops=tuple(ops)
  if ops not in self.blocks:self.blocks.append(ops)
  return self.blocks.index(ops)
 def override(self,prefix,key,recipe):
  if self.primary[prefix][1]==65535:self.primary[prefix][1]=len(self.tables);self.tables.append([])
  t=self.tables[self.primary[prefix][1]];t[:]=[r for r in t if not (r[0]==key and r[1]==65535)];t.insert(0,(key,65535,recipe))
 def write(self,path):
  p=Path(path)
  if p.exists():assert p.read_bytes()==self.bytes(),'Refusing to replace different weights'
  else:p.write_bytes(self.bytes())
