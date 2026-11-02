"""Submit one bounded dependency chain; retain job IDs, never double-submit."""
import fcntl,json,os,subprocess,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)
(ROOT/'logs').mkdir(exist_ok=True)
lock=(ROOT/'artifacts/submission.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
record=ROOT/'artifacts/pbs-submission.json'
if record.exists():raise SystemExit('Submission record already exists. Inspect stored job IDs before any resubmission.')
data={'state':'submitting','created_at':time.time(),'jobs':{},'config_sha256':hashlib.sha256((ROOT/'configs/main.json').read_bytes()).hexdigest()}
def save():
 temp=record.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2));temp.replace(record)
def submit(key,script,walltime,resources,dependency=None,variables=None):
 args=['qsub','-P','personal-cila001','-q','normal','-N','ass2-'+key[:10],'-l','select=1:'+resources,'-l','walltime='+walltime,'-j','oe','-o','logs/'+key+'-scheduler.log']
 if dependency:args+=['-W','depend=afterok:'+dependency]
 if variables:args+=['-v',variables]
 args+=[script]
 data['pending']={'key':key,'command':args};save()
 response=subprocess.run(args,text=True,capture_output=True)
 if response.returncode:raise RuntimeError(response.stderr)
 job=response.stdout.strip()
 if not job or not job.split('.')[0].isdigit():raise RuntimeError('Ambiguous scheduler response; inspect qstat before retrying')
 data['jobs'][key]={'id':job,'command':args};data.pop('pending',None);save();return job
save()
try:
 prep=submit('prepare','cluster/prepare.pbs','01:00:00','ncpus=4:mem=16gb')
 smoke=submit('smoke','cluster/smoke.pbs','00:45:00','ncpus=16:mem=110gb:ngpus=1',prep)
 for model in ['tinyllama','openllama']:
  for method in ['lora','adapterp']:
   submit(model+'-'+method,'cluster/main.pbs','08:00:00','ncpus=16:mem=110gb:ngpus=1',smoke,'ASS2_MODEL='+model+',ASS2_METHOD='+method)
 data['state']='submitted';save();print(json.dumps(data,indent=2))
except Exception as error:
 data['state']='requires_review';data['error']=str(error);save();raise
