"""Use the instructor's PEFT implementation, never an accidentally installed fork."""
import importlib.metadata
import os
import sys
from pathlib import Path
from common import WORK, ROOT

sys.path.insert(0, str(ROOT/'peft/src'))
os.environ.setdefault('WANDB_DISABLED','true')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')

# Keep compilation caches away from the full home allocation. The tested cache
# bypass avoids the A100 reduction failure without changing model computations.
os.environ.setdefault('USE_PYTORCH_KERNEL_CACHE','0')
os.environ.setdefault('CUDA_CACHE_DISABLE','1')
for variable,folder in [('CUDA_CACHE_PATH','cuda'),('PYTORCH_KERNEL_CACHE_PATH','pytorch-kernels'),('TRITON_CACHE_DIR','triton')]:
    location=WORK/'cache'/folder
    location.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault(variable,str(location))


def libraries():
    import peft
    import torch
    import transformers
    assert Path(peft.__file__).resolve().is_relative_to(ROOT/'peft/src'), peft.__file__
    assert hasattr(peft,'BottleneckConfig')
    return peft, torch, transformers

def hardware():
    peft, torch, transformers = libraries()
    return {'python':sys.version,'torch':torch.__version__, 'transformers':transformers.__version__,
            'accelerate':importlib.metadata.version('accelerate'), 'peft_file':str(peft.__file__),
            'cuda_available':torch.cuda.is_available(), 'cuda':torch.version.cuda,
            'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'gpu_memory_bytes':torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None}

def make_adapter(model, method, cfg):
    peft, torch, _ = libraries()
    for p in model.parameters():
        p.requires_grad = False
    if method == 'lora':
        config = peft.LoraConfig(r=cfg['lora_r'], lora_alpha=cfg['lora_alpha'], lora_dropout=cfg['lora_dropout'],
                                 target_modules=['q_proj','v_proj'], bias='none', task_type='CAUSAL_LM', merge_weights=False)
    elif method == 'adapterp':
        config = peft.BottleneckConfig(bottleneck_size=cfg['bottleneck_size'], use_adapterp=True,
                                       target_modules=None, non_linearity='tanh', adapter_dropout=0.,
                                       scaling=1., bias='none', task_type='CAUSAL_LM')
    else:
        raise ValueError(method)
    model = peft.get_peft_model(model, config)
    for name,p in model.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()
    return model

def load_adapter(model, directory):
    """Same original adapter layout; strict coverage rather than silent missing weights."""
    peft, torch, _ = libraries()
    weights = torch.load(Path(directory)/'adapter_model.bin',map_location='cpu',weights_only=True)
    expected = peft.get_peft_model_state_dict(model)
    if set(weights) != set(expected):
        raise ValueError(f'Adapter keys differ: missing={set(expected)-set(weights)}, extra={set(weights)-set(expected)}')
    for k in expected:
        assert weights[k].shape == expected[k].shape, k
    peft.set_peft_model_state_dict(model,weights)

def load_base(model_id, revision=None, compute_dtype="float16"):
    _, torch, transformers = libraries()
    if not torch.cuda.is_available():
        raise RuntimeError('Full assigned models require a GPU. Run tools/smoke.py for CPU synthetic checks.')
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, revision=revision, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0
    tokenizer.padding_side = 'left'
    model = transformers.AutoModelForCausalLM.from_pretrained(model_id,revision=revision,torch_dtype=getattr(torch,compute_dtype),
                    device_map={'':0},attn_implementation='eager')
    return tokenizer, model
