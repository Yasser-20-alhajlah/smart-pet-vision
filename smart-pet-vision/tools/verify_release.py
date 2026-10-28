"""Verify release file hashes without model libraries or private photographs."""
import argparse,hashlib,json
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('manifest',type=Path);p.add_argument('--root',type=Path);a=p.parse_args();root=(a.root or a.manifest.parent).resolve();results=[]
 for line in a.manifest.read_text().splitlines():
  if not line.strip() or line.startswith('#'):continue
  expected,rel=line.split('  ',1);path=(root/rel).resolve()
  if not path.is_relative_to(root):raise ValueError('Unsafe path in manifest: '+rel)
  actual=hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
  results.append({'path':rel,'passed':actual==expected,'actual':actual})
 print(json.dumps({'files':len(results),'passed':sum(x['passed']for x in results),'failures':[x for x in results if not x['passed']]},indent=2))
 if not results or not all(x['passed']for x in results):raise SystemExit(1)
if __name__=='__main__':main()
