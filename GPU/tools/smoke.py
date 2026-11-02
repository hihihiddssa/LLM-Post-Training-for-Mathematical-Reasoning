"""Offline random tiny Llama checks actual teacher adapters; NOT benchmark training."""
import argparse
import tempfile
from pathlib import Path
from common import WORK, ROOT, read_json, write_json
from runtime import hardware, libraries, load_adapter, make_adapter

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cpu',action='store_true')
    args=p.parse_args()
    peft,torch,tr=libraries()
    device='cpu' if args.cpu or not torch.cuda.is_available() else 'cuda'
    cfg=read_json(WORK/'configs/main.json')
    reports=[]
    for method in ['lora','adapterp']:
        tr.set_seed(42)
        config=tr.LlamaConfig(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
            num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=128,
            bos_token_id=1,eos_token_id=2,pad_token_id=0,attn_implementation='eager')
        base=tr.LlamaForCausalLM(config).to(device)
        pristine={k:v.detach().cpu().clone() for k,v in base.state_dict().items()}
        model=make_adapter(base,method,cfg)
        params=[p for p in model.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(params,lr=3e-4)
        ids=torch.randint(3,64,(2,16),device=device)
        before={k:v.detach().cpu().clone() for k,v in peft.get_peft_model_state_dict(model).items()}
        model.train()
        for step in range(2):
            optimizer.zero_grad()
            loss=model(input_ids=ids,labels=ids).loss
            assert torch.isfinite(loss)
            loss.backward()
            assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in params)
            optimizer.step()
        after=peft.get_peft_model_state_dict(model)
        assert any(not torch.equal(before[k],v.detach().cpu()) for k,v in after.items())
        # Frozen parameters stay unchanged, including the original Linear weights.
        for key,value in pristine.items():
            assert torch.equal(model.base_model.model.state_dict()[key].cpu(),value), key
        model.eval()
        with torch.no_grad():
            expected=model(input_ids=ids).logits
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            newbase=tr.LlamaForCausalLM(config).to(device)
            newbase.load_state_dict(pristine)
            restored=make_adapter(newbase,method,cfg)
            load_adapter(restored,directory)
            restored.eval()
            with torch.no_grad():
                actual=restored(input_ids=ids).logits
                tokens=restored.generate(input_ids=ids[:1],attention_mask=torch.ones_like(ids[:1]),max_new_tokens=3,do_sample=False,use_cache=True)
            assert torch.allclose(expected,actual,atol=1e-5,rtol=1e-5)
            assert tokens.shape[1]>ids.shape[1]
            # Validate the original PeftModel.from_pretrained reload too.
            legacybase=tr.LlamaForCausalLM(config).to(device)
            legacybase.load_state_dict(pristine)
            legacy=peft.PeftModel.from_pretrained(legacybase,directory).eval()
            with torch.no_grad():
                assert torch.allclose(expected,legacy(input_ids=ids).logits,atol=1e-5,rtol=1e-5)
        reports.append({'method':method,'device':device,'loss':float(loss.detach()),'trainable_params':sum(p.numel() for p in params),
                        'forward_backward':True,'base_unchanged':True,'save_reload_logits_equal':True,'generation':True})
    result={'type':'offline_synthetic_smoke','hardware':hardware(),'checks':reports}
    write_json(WORK/'artifacts/synthetic-smoke.json',result)
    print(result)

if __name__ == '__main__':
    main()
