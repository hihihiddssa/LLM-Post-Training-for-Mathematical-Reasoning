"""Versioned scoring and durable, identity-checked evaluation records (stdlib only)."""
import hashlib
import json
import os
import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path

WORK = Path(__file__).resolve().parents[1]
ROOT = WORK.parent / 'AI6130_Assignment2'
PROTOCOL = "ass2-v1"

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)

NUMBER = r'[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:\s*/\s*[-+]?\d+(?:\.\d+)?)?'

def numeric(value):
    try:
        value = str(value).strip().replace(',', '').replace(' ', '')
        return Decimal(str(float(Fraction(value)))) if '/' in value else Decimal(value)
    except (InvalidOperation, ValueError, ZeroDivisionError, OverflowError):
        return None

def parse_answer(text, task):
    """Never guess an answer from an option list or the final number in reasoning."""
    text = text.strip()
    if '<think>' in text and '</think>' not in text:
        return None
    text = text.rsplit('</think>', 1)[-1].strip()
    if task == 'AQuA':
        if re.fullmatch(r'\(?[A-E]\)?[.\s]*', text):
            return re.search('[A-E]', text).group()
        signals = re.findall(r'\\boxed\{\s*\(?([A-E])\)?\s*\}', text)
        signals += re.findall(r'(?i:(?:final\s+)?answer\s*(?:is\s*)?[:=]?\s*(?:option\s*)?)\(?([A-E])\)?(?=[\s.,:;!\)]|$)', text)
        return signals[0] if signals and len(set(signals)) == 1 else None
    if task != 'AddSub':
        raise ValueError(task)
    signals = re.findall(r'\\boxed\{\s*(' + NUMBER + r')\s*\}', text)
    signals += re.findall(r'####\s*\$?\s*(' + NUMBER + ')', text)
    signals += re.findall(r'(?i:(?:final\s+)?answer(?:\s+in\s+Arabic\s+numerals)?\s*(?:is|equals)?\s*[:=]?\s*)\$?\s*(' + NUMBER + r')(?![\d/])', text)
    if not signals and re.fullmatch(r'\$?\s*' + NUMBER + r'\s*\.?', text):
        signals = re.findall(NUMBER, text)
    values = [numeric(s) for s in signals]
    if not values or any(v is None or not v.is_finite() for v in values) or len(set(values)) != 1:
        return None
    return str(values[0])

def correct(prediction, gold, task):
    if prediction is None:
        return False
    if task == 'AQuA':
        return prediction == gold
    p, g = numeric(prediction), numeric(gold)
    return p is not None and g is not None and abs(p-g) <= Decimal('0.001')

class Journal:
    """Append one complete record per item; only repair an interrupted final line."""
    def __init__(self, path, identity):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        meta = self.path.with_suffix('.meta.json')
        if meta.exists():
            if read_json(meta) != identity:
                raise ValueError('Run identity differs; choose a new output directory')
        elif self.path.exists() and self.path.stat().st_size:
            raise ValueError('Refusing a journal without provenance')
        else:
            write_json(meta, identity)
        self.rows = {}
        if self.path.exists():
            with self.path.open('r+b') as stream:
                while True:
                    start = stream.tell()
                    line = stream.readline()
                    if not line:
                        break
                    if not line.endswith(b'\n'):
                        stream.truncate(start)
                        break
                    row = json.loads(line)
                    if row['index'] in self.rows:
                        raise ValueError('Duplicate question index')
                    self.rows[row['index']] = row

    def append(self, row):
        if row['index'] in self.rows:
            raise ValueError('Duplicate item')
        with self.path.open('ab') as stream:
            stream.write((json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n').encode())
            stream.flush()
            os.fsync(stream.fileno())
        self.rows[row['index']] = row

def summary(rows, total):
    rows = list(rows)
    n = len(rows)
    return {'completed': n == total, 'evaluated': n, 'total': total,
            'correct': sum(r['correct'] for r in rows),
            'accuracy': 100 * sum(r['correct'] for r in rows)/total if n == total else None,
            'invalid': sum(r['prediction'] is None for r in rows),
            'truncated': sum(r['truncated'] for r in rows)}
