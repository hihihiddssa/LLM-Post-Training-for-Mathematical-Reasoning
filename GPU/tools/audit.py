"""Freeze the actual datasets and common split; retain original labels unchanged."""
import argparse
import random
import re
import subprocess
from common import WORK, ROOT, file_hash, read_json, write_json

def run():
    datasets = {}
    rows = {}
    for name, file in [('train','ft-training_set/math_7k.json'), ('AQuA','dataset/AQuA/test.json'), ('AddSub','dataset/AddSub/test.json')]:
        path = ROOT/file
        data = read_json(path)
        assert all(isinstance(x.get('instruction'),str) and 'answer' in x for x in data)
        if name == 'train':
            assert all(isinstance(x.get('output'),str) for x in data)
        elif name == 'AQuA':
            assert all(x['answer'] in 'ABCDE' and len(x['answer']) == 1 for x in data)
        rows[name] = data
        datasets[name] = {'path':file, 'count':len(data), 'sha256':file_hash(path),'fields':sorted(data[0])}
    norm = lambda s: re.sub(r'\s+',' ',s).strip().casefold()
    train_questions = {norm(x['instruction']) for x in rows['train']}
    overlap = {task:sum(norm(x['instruction']) in train_questions for x in rows[task]) for task in ['AQuA','AddSub']}
    if any(overlap.values()):
        raise RuntimeError(f'Exact normalized train/test question overlap: {overlap}; investigate before training')
    ids = list(range(len(rows['train'])))
    random.Random(42).shuffle(ids)
    manifest = {'seed':42, 'validation_indices':ids[:120], 'training_indices':ids[120:], 'datasets':datasets,
                'exact_normalized_question_overlap':overlap,
                'upstream_commit':(subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip() if (ROOT/'.git').exists() else (ROOT/'UPSTREAM_COMMIT').read_text().strip()),
                'note':'Evaluation uses instruction/input and answer only; test output/rationale is never a prompt or gold source.'}
    destination = WORK/'artifacts/data-manifest.json'
    if destination.exists():
        assert read_json(destination) == manifest, 'Frozen data changed'
    else:
        write_json(destination,manifest)
    print({key:d['count'] for key,d in datasets.items()}, 'train:',len(ids[120:]),'validation:',120,'overlap:',overlap)

if __name__ == '__main__':
    run()
