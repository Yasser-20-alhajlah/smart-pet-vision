"""Independent stdlib recomputation of exported predictions and 15-bin ECE."""
import argparse,csv,json,math
from pathlib import Path

def recompute(rows,calibrated):
 key='calibrated' if calibrated else 'raw';cm=[[0,0],[0,0]];bins=[[]for _ in range(15)]
 for r in rows:
  p=[float(r['p_cat_'+key]),float(r['p_dog_'+key])];y=int(r['label']);pred=0 if p[0]>=p[1] else 1
  if not all(math.isfinite(x) and 0<=x<=1 for x in p)or abs(sum(p)-1)>1e-6:raise ValueError('Invalid probabilities')
  cm[y][pred]+=1;conf=max(p);idx=max(0,min(14,math.ceil(conf*15)-1));bins[idx].append((conf,int(pred==y)))
 n=len(rows);fs=[]
 for i in range(2):
  tp=cm[i][i];fp=sum(cm[j][i]for j in range(2)if j!=i);fn=sum(cm[i][j]for j in range(2)if j!=i);fs.append(2*tp/(2*tp+fp+fn)if 2*tp+fp+fn else 0)
 ece=sum(abs(sum(x[0]for x in b)-sum(x[1]for x in b))/n for b in bins if b)
 return {'n':n,'confusion_matrix':cm,'accuracy':sum(cm[i][i]for i in range(2))/n,'macro_f1':sum(fs)/2,'ece':ece}

def main():
 p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();reports={}
 for kind in ['scratch','frozen','finetune']:
  rows=list(csv.DictReader((a.directory/f'{kind}_predictions.csv').open()));saved=json.loads((a.directory/f'{kind}_metrics.json').read_text());rs={}
  for calibrated,key in [(False,'raw'),(True,'calibrated_using_saved_temperature')]:
   v=recompute(rows,calibrated);expected=saved[key];ok=v['confusion_matrix']==expected['confusion_matrix'] and all(abs(v[x]-expected[x])<1e-6 for x in ['accuracy','macro_f1','ece']);rs[key]={'recomputed':v,'passed':ok};assert ok,(kind,key,v,expected)
  reports[kind]=rs
 images={k:[r['image']for r in csv.DictReader((a.directory/f'{k}_predictions.csv').open())]for k in reports};assert images['scratch']==images['frozen']==images['finetune'];assert len(set(images['scratch']))==1106
 result={'implementation':'independent Python standard library; no sklearn or project metric calls','models':reports,'sample_order_and_unique_1106':True}
 (a.directory/'independent_recomputation.json').write_text(json.dumps(result,indent=2));print('All 3 models: independent accuracy, macro-F1, confusion matrix and ECE passed')
if __name__=='__main__':main()
