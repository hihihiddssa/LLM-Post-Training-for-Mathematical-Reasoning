"""Download only the two immutable assigned models; no GPU computation."""
from common import WORK,read_json,write_json
from huggingface_hub import snapshot_download
cfg=read_json(WORK/'configs/main.json')
revisions=read_json(WORK/'artifacts/model-revisions.json')
completed={}
for key,model in cfg['models'].items():
 path=snapshot_download(model,revision=revisions[key]['sha'],max_workers=2,allow_patterns=['*.json','*.model','*.bin','*.safetensors','*.txt'])
 completed[key]={'id':model,'revision':revisions[key]['sha'],'snapshot':path}
 write_json(WORK/'artifacts/downloaded-models.json',completed)
 print('Cached:',key,path,flush=True)
