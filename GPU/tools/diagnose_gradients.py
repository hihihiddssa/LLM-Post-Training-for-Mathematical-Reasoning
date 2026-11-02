"""Compare finite gradients across compute dtypes on fixed training items only."""
import sys,json,gc
from common import ROOT,WORK,read_json,write_json
from runtime import libraries,make_adapter
peft,torch,tr=libraries()
sys.path.insert(0,str(ROOT))
from finetune import generate_prompt
cfg=read_json(WORK/'configs/main.json')
rev=read_json(WORK/'artifacts/model-revisions.json')['tinyllama']['sha']
rows=read_json(ROOT/'ft-training_set/math_7k.json')
ids=read_json(WORK/'artifacts/data-manifest.json')['training_indices'][:4]
tokenizer=tr.AutoTokenizer.from_pretrained(cfg['models']['tinyllama'],revision=rev,use_fast=False)
reports=[]
for name,dtype in [('float16',torch.float16),('bfloat16',torch.bfloat16)]:
 torch.cuda.empty_cache();tr.set_seed(42)
 model=tr.AutoModelForCausalLM.from_pretrained(cfg['models']['tinyllama'],revision=rev,torch_dtype=dtype,device_map={'':0},attn_implementation='eager')
 model=make_adapter(model,'lora',cfg)
 model.config.use_cache=False
 model.train()
 for index in ids:
  model.zero_grad(set_to_none=True)
  tokens=tokenizer(generate_prompt(rows[index]),return_tensors='pt',truncation=True,max_length=256).to('cuda')
  with torch.autocast('cuda',dtype=dtype):
   loss=model(**tokens,labels=tokens.input_ids).loss
  loss.backward()
  gradients=[p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
  record={'dtype':name,'training_index':index,'loss':float(loss.detach()) if torch.isfinite(loss) else None,'all_gradients_finite':all(bool(torch.isfinite(g).all()) for g in gradients)}
  reports.append(record);print(record,flush=True)
 del model,tokens,loss,gradients
 gc.collect();torch.cuda.empty_cache()
write_json(WORK/'artifacts/precision-diagnostic.json',reports)
