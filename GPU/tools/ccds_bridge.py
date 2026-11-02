"""Interactive CCDS login; password lives only in this process, not in files/chat."""
import datetime
import getpass
import json
import os
from pathlib import Path
import sys
import time

HOST=os.environ.get('ASS2_SSH_HOST','10.96.189.12')
USERNAME=os.environ.get('ASS2_SSH_USER','cila001')
ROOT=Path(__file__).resolve().parents[1]/os.environ.get('ASS2_CONTROL_DIR','cluster-control')
ROOT.mkdir(exist_ok=True)
# Reuse the already installed SSH client library without downloading another copy.
legacy=Path(r'C:\Users\cila\Documents\Codex\2026-10-11\gou\work\ccds-control\deps')
if legacy.is_dir(): sys.path.insert(0,str(legacy))
import paramiko

def save(path,value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    os.replace(temp,path)

def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()

def authorized_credentials():
    """Use only the two user-specified chats after explicit reuse authorization.

    Values are decoded in memory, never printed, returned through a tool, or
    serialized. There is no password candidate guessing or retry loop.
    """
    import io
    import re
    import subprocess
    sys.path.insert(0,r'D:\project\PEFT模型微调\GPU\local-deps')
    import zstandard
    archive=Path(r'C:\Users\cila\.codex\sessions\2026\10\07\rollout-2026-10-07T19-48-27-01a115ed-7627-7483-a475-2e1c02e2081d_01a11631-895f-7310-8635-7143f93cebdf.jsonl.zst')
    def user_text(row):
        p=row.get('payload',{})
        if row.get('type')=='response_item' and p.get('type')=='message' and p.get('role')=='user':
            return '\n'.join(x.get('text','') for x in p.get('content',[]) if x.get('type') in ['input_text','text'])
        return ''
    prior=[]
    with archive.open('rb') as stream:
        for line in io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(stream),encoding='utf-8'):
            row=json.loads(line)
            text=user_text(row)
            if row.get('timestamp')=='2026-10-09T01:53:51.380Z' and '帮我记住密码' in text:
                prior=re.findall(r'@[A-Za-z0-9]{6,}',text)
    if len(prior)!=1:raise RuntimeError('The specifically authorized prior credential is not unambiguous')
    query="import sqlite3; c=sqlite3.connect('file:C:/Users/cila/.codex/state_5.sqlite?mode=ro',uri=True); print(c.execute(\"select rollout_path from threads where id='01a10fc1-1da0-7950-bf12-acc338948bb9'\").fetchone()[0])"
    current_path=subprocess.check_output([r'C:\Users\cila\AppData\Local\Programs\Python\Python312\python.exe','-c',query],text=True,encoding='utf-8').strip()
    current=[]
    for line in Path(current_path).open(encoding='utf-8'):
        text=user_text(json.loads(line))
        match=re.search(r'那就改用nscc集群\s*密码([^\s]+)',text,re.I)
        if match:current.append(match.group(1))
    if not current:raise RuntimeError('The user-provided NSCC credential was not found')
    return prior[0],current[-1]

def main():
    print('AI6130 Assignment 2 - cluster login:',HOST,'Password is never saved.')
    client=paramiko.SSHClient()
    client.load_system_host_keys(os.environ.get('ASS2_KNOWN_HOSTS',str(Path.home()/'.ssh/known_hosts')))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    jump_host=os.environ.get('ASS2_JUMP_HOST')
    if os.environ.get('ASS2_AUTHORIZED_AUTO_LOGIN')=='1':
        jump_password,password=authorized_credentials()
    else:
        jump_password=getpass.getpass('NTU Jump Host password for '+USERNAME+': ') if jump_host else None
        password=getpass.getpass('NSCC/cluster password for '+USERNAME+': ')
    jump=None
    stage='jump' if jump_host else 'cluster'
    try:
        channel=None
        if jump_host:
            jump=paramiko.SSHClient()
            jump.load_host_keys(os.environ['ASS2_JUMP_KNOWN_HOSTS'])
            jump.set_missing_host_key_policy(paramiko.RejectPolicy())
            jump.connect(jump_host,username=USERNAME,password=jump_password,look_for_keys=False,allow_agent=False,
                         timeout=15,banner_timeout=15,auth_timeout=30)
            jump.get_transport().set_keepalive(30)
            print('NTU jump authenticated.',flush=True)
            stage='forwarding'
            channel=jump.get_transport().open_channel('direct-tcpip',(HOST,22),('127.0.0.1',0),timeout=20)
        stage='cluster'
        client.connect(HOST,username=USERNAME,password=password,look_for_keys=False,allow_agent=False,
                       timeout=15,banner_timeout=15,auth_timeout=30,sock=channel)
    except Exception as error:
        save(ROOT/'connection.json',{'state':'failed','stage':stage,'error':type(error).__name__,'updated_at':now()})
        print('Login failed at stage:',stage,'error:',type(error).__name__,flush=True)
        return
    finally:
        password=None
        jump_password=None
    client.get_transport().set_keepalive(30)
    sftp=client.open_sftp()
    save(ROOT/'connection.json',{'state':'connected','updated_at':now(),'host':HOST})
    print('Connected. Leave this window open while Codex inspects and submits the experiments.',flush=True)
    try:
        while not (ROOT/'CLOSE').exists():
            save(ROOT/'heartbeat.json',{'state':'connected' if client.get_transport().is_active() else 'disconnected','updated_at':now()})
            for path in sorted(ROOT.glob('request-*.json')):
                response=ROOT/path.name.replace('request-','response-')
                if response.exists(): continue
                try:
                    request=json.loads(path.read_text(encoding='utf-8'))
                    if request['action']=='exec':
                        _,out,err=client.exec_command(request['command'],timeout=request.get('timeout',90))
                        result={'stdout':out.read().decode('utf-8','replace'),'stderr':err.read().decode('utf-8','replace'),
                                'exit_code':out.channel.recv_exit_status()}
                    elif request['action'] in ['put','get']:
                        for pair in request['files']:
                            if request['action']=='put': sftp.put(pair['local'],pair['remote'])
                            else:
                                Path(pair['local']).parent.mkdir(parents=True,exist_ok=True)
                                sftp.get(pair['remote'],pair['local'])
                        result={'files':len(request['files'])}
                    else: raise ValueError('Unknown action')
                    save(response,{'ok':True,**result,'updated_at':now()})
                except Exception as error:
                    save(response,{'ok':False,'error':type(error).__name__+': '+str(error),'updated_at':now()})
            time.sleep(1)
    finally:
        sftp.close();client.close()
        if jump:jump.close()
        save(ROOT/'connection.json',{'state':'closed','updated_at':now()})

if __name__=='__main__':main()
