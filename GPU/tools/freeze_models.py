"""Resolve immutable public model revisions before model loading (no API key)."""
from common import WORK, ROOT, read_json, write_json

def main():
    from huggingface_hub import HfApi
    path=WORK/'artifacts/model-revisions.json'
    if path.exists():
        print('Preserving frozen revisions:',read_json(path))
        return
    cfg=read_json(WORK/'configs/main.json')
    api=HfApi()
    revisions={key:{'id':model,'sha':api.model_info(model).sha} for key,model in cfg['models'].items()}
    write_json(path,revisions)
    print(revisions)

if __name__ == '__main__':main()
