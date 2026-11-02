"""Reproducible Trainer driver using instructor prompts and PEFT injection classes."""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from common import WORK, ROOT, digest, file_hash, read_json, write_json
from runtime import libraries, hardware, make_adapter, load_adapter, load_base

def code_hash():
    paths = [WORK/'tools'/name for name in ['common.py','runtime.py','train.py']] + list((ROOT/'peft/src/peft').rglob('*.py')) + [ROOT/'finetune.py']
    return digest({str(p.relative_to(WORK.parent)):file_hash(p) for p in sorted(paths)})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['tinyllama','openllama'],required=True)
    parser.add_argument('--method',choices=['lora','adapterp'],required=True)
    parser.add_argument('--config',default=str(WORK/'configs/main.json'))
    parser.add_argument('--smoke',action='store_true',help='8 train / 2 validation items, one step; never a full experiment')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--smoke-steps',type=int,default=1)
    parser.add_argument('--stop-after-step',type=int,default=0)
    parser.add_argument('--run-tag',default='')
    args = parser.parse_args()
    peft, torch, tr = libraries()
    from datasets import Dataset
    from audit import run as audit
    sys.path.insert(0,str(ROOT))
    from finetune import generate_prompt
    audit()
    cfg = read_json(args.config)
    assert cfg['batch_size'] % cfg['micro_batch_size'] == 0
    assert int(os.environ.get('WORLD_SIZE','1')) == 1, 'This driver requests one GPU per experiment'
    frozen = read_json(WORK/'artifacts/data-manifest.json')
    revisions = read_json(WORK/'artifacts/model-revisions.json')
    revision = revisions[args.model]['sha']
    prefix = 'smoke' if args.smoke else 'main'
    import re
    assert not args.run_tag or re.fullmatch(r'[A-Za-z0-9_-]{1,40}',args.run_tag)
    suffix='-'+args.run_tag if args.run_tag else ''
    run = WORK/'runs'/f'{prefix}-{args.model}-{args.method}-{cfg["epochs"]}ep{suffix}'
    identity = {'config':cfg,'model':args.model,'method':args.method,'revision':revision,
                'data_manifest_sha256':file_hash(WORK/'artifacts/data-manifest.json'),
                'code_sha256':code_hash(),'smoke':args.smoke,'smoke_steps':args.smoke_steps if args.smoke else None}
    run.mkdir(parents=True,exist_ok=True)
    meta = run/'identity.json'
    if meta.exists():
        assert read_json(meta) == identity, 'Run settings/source changed; use a new run directory'
        if (run/'training-complete.json').exists():
            print('Already complete:',run)
            return
        assert args.resume, 'Existing run: explicitly use --resume'
    else:
        assert not any(run.iterdir()), 'Refusing to overwrite existing untracked artifacts'
        write_json(meta,identity)
    checkpoints = []
    for checkpoint in run.glob('checkpoint-*'):
        required = ['adapter_model.bin','trainer_state.json','optimizer.pt','scheduler.pt','rng_state.pth']
        if all((checkpoint/name).exists() for name in required):
            checkpoints.append(checkpoint)
    checkpoint = max(checkpoints,key=lambda p:int(p.name.split('-')[-1])) if checkpoints else None
    if args.resume and not checkpoint and (run/'events.jsonl').exists():
        raise RuntimeError('No complete optimizer checkpoint to resume; preserve this run and inspect it')
    tr.set_seed(cfg['seed'])
    tokenizer, base = load_base(cfg['models'][args.model],revision,cfg.get('compute_dtype','float16'))
    tokenizer.padding_side = 'right'  # Avoid fully masked left-padding queries in FP16 training.
    model = make_adapter(base,args.method,cfg)
    initial_adapter = {k:v.detach().cpu().clone() for k,v in peft.get_peft_model_state_dict(model).items()} if args.smoke else None
    model.config.use_cache = False
    if cfg['gradient_checkpointing']:
        # Original helper enables this only in INT8; enable it for FP16 as well.
        model.base_model.model.enable_input_require_grads()
        model.base_model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    data = read_json(ROOT/'ft-training_set/math_7k.json')
    token_stats = {'items':0,'capped':0}
    def tokenize(item):
        # Exactly the instructor's prompt format and train_on_inputs=True objective.
        result = tokenizer(generate_prompt(item), truncation=True,max_length=cfg['cutoff_len'],padding=False)
        token_stats['items'] += 1
        token_stats['capped'] += len(result['input_ids']) == cfg['cutoff_len']
        if len(result['input_ids']) < cfg['cutoff_len'] and result['input_ids'][-1] != tokenizer.eos_token_id:
            result['input_ids'].append(tokenizer.eos_token_id)
            result['attention_mask'].append(1)
        result['labels'] = result['input_ids'].copy()
        return result
    train_indices = frozen['training_indices'][:8] if args.smoke else frozen['training_indices']
    val_indices = frozen['validation_indices'][:2] if args.smoke else frozen['validation_indices']
    train_data = Dataset.from_list([tokenize(data[i]) for i in train_indices])
    val_data = Dataset.from_list([tokenize(data[i]) for i in val_indices])
    parameters = {'trainable':sum(p.numel() for p in model.parameters() if p.requires_grad),
                  'total':sum(p.numel() for p in model.parameters())}
    parameters['percent'] = 100*parameters['trainable']/parameters['total']
    record = {'identity_sha256':digest(identity),'hardware':hardware(),'parameters':parameters,
              'train_samples':len(train_data),'validation_samples':len(val_data),'tokenization':token_stats,
              'adapter_config':vars(model.peft_config),'resumed_from':str(checkpoint) if checkpoint else None}
    write_json(run/'setup.json',record)
    control_stop_after=args.stop_after_step
    class Events(tr.TrainerCallback):
        def on_log(self,args,state,control,logs=None,**kwargs):
            raw = dict(logs or {})
            nonfinite = [k for k,v in raw.items() if isinstance(v,float) and not math.isfinite(v)]
            for key in nonfinite:raw[key]=None
            with (run/'events.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps({'step':state.global_step,'epoch':state.epoch,**raw,'nonfinite_fields':nonfinite},allow_nan=False)+'\n')
                stream.flush()
        def on_step_end(self,args,state,control,**kwargs):
            if (run/'STOP_AFTER_STEP').exists() or (control_stop_after and state.global_step>=control_stop_after):
                control.should_save = True
                control.should_training_stop = True
    class AdapterTrainer(tr.Trainer):
        def _save(self,output_dir=None,state_dict=None):
            target = Path(output_dir or self.args.output_dir)
            self.model.save_pretrained(str(target))
            tokenizer.save_pretrained(str(target))
            torch.save(self.args,target/'training_args.bin')
        def _load_from_checkpoint(self,resume_from_checkpoint,model=None):
            load_adapter(model or self.model,resume_from_checkpoint)
        def _load_best_model(self):
            load_adapter(self.model,self.state.best_model_checkpoint)
        def _load_rng_state(self,checkpoint):
            # Only locally created, provenance-checked Trainer checkpoints are accepted.
            # New PyTorch defaults to weights_only=True; RNG includes NumPy state.
            import random
            import numpy as np
            state=torch.load(Path(checkpoint)/'rng_state.pth',map_location='cpu',weights_only=False)
            random.setstate(state['python'])
            np.random.set_state(state['numpy'])
            torch.random.set_rng_state(state['cpu'])
            if 'cuda' in state:
                if isinstance(state['cuda'],torch.Tensor):
                    torch.cuda.random.set_rng_state(state['cuda'])
                else:
                    torch.cuda.random.set_rng_state_all(state['cuda'])
    training_args = tr.TrainingArguments(output_dir=str(run),per_device_train_batch_size=cfg['micro_batch_size'],
        per_device_eval_batch_size=cfg['micro_batch_size'],gradient_accumulation_steps=cfg['batch_size']//cfg['micro_batch_size'],
        num_train_epochs=cfg['epochs'],max_steps=args.smoke_steps if args.smoke else -1,learning_rate=cfg['learning_rate'],
        warmup_steps=0 if args.smoke else 100,fp16=cfg.get('compute_dtype','float16')=='float16',bf16=cfg.get('compute_dtype')=='bfloat16',logging_steps=1 if args.smoke else 10,
        optim='adamw_torch',evaluation_strategy='steps',save_strategy='steps',
        eval_steps=1 if args.smoke else cfg['eval_steps'],save_steps=1 if args.smoke else cfg['save_steps'],
        save_total_limit=3,load_best_model_at_end=True,metric_for_best_model='eval_loss',greater_is_better=False,
        seed=cfg['seed'],data_seed=cfg['seed'],report_to=[],remove_unused_columns=False)
    trainer = AdapterTrainer(model=model,args=training_args,train_dataset=train_data,eval_dataset=val_data,
        data_collator=tr.DataCollatorForSeq2Seq(tokenizer,pad_to_multiple_of=8,return_tensors='pt',padding=True),callbacks=[Events()])
    if cfg.get('compute_dtype','float16')=='float16':
        trainer.accelerator.scaler=torch.amp.GradScaler('cuda',init_scale=cfg.get('amp_init_scale',128.))
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    result = trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
    expected_steps = trainer.state.max_steps
    if trainer.state.global_step < expected_steps:
        print('Paused with checkpoint at step',trainer.state.global_step,'of',expected_steps)
        return
    evaluation = trainer.evaluate()
    final = run/'final-adapter'
    model.save_pretrained(str(final))
    tokenizer.save_pretrained(str(final))
    # Validate reloading into a newly injected base without requiring a second GPU copy.
    saved = peft.get_peft_model_state_dict(model)
    disk = torch.load(final/'adapter_model.bin',map_location='cpu',weights_only=True)
    assert set(saved) == set(disk)
    assert all(torch.equal(saved[k].detach().cpu(),disk[k]) for k in saved)
    assert all(torch.isfinite(value).all() for value in disk.values())
    if args.smoke:
        assert any(not torch.equal(initial_adapter[k],disk[k]) for k in disk), 'AMP skipped every optimizer step; smoke did not update adapters'
    history=[]
    for event in trainer.state.log_history:
        event=dict(event)
        nonfinite=[k for k,v in event.items() if isinstance(v,float) and not math.isfinite(v)]
        for key in nonfinite:event[key]=None
        if nonfinite:event['nonfinite_fields']=nonfinite
        history.append(event)
    write_json(run/'training-complete.json',{'completed':True,'smoke':args.smoke,'global_step':trainer.state.global_step,
        'metrics':result.metrics,'validation':evaluation,'seconds_this_session':time.time()-started,
        'best_checkpoint':trainer.state.best_model_checkpoint,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),
        'final_adapter_sha256':file_hash(final/'adapter_model.bin'),'history':history})
    print('Training complete:',run)

if __name__ == '__main__':
    main()
