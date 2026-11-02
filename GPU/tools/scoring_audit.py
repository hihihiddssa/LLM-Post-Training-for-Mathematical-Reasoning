"""Post-generation scoring audit. Never changes raw outputs or original v1 journals."""
import re
from common import parse_answer, NUMBER, numeric
VERSION='conclusion-audit-v2'
def audited_answer(text,task):
    original=parse_answer(text,task)
    if original is not None or task!='AddSub':return original
    if '<think>' in text and '</think>' not in text:return None
    text=text.rsplit('</think>',1)[-1].strip()
    # Existing numerical answer markers take precedence, including conflicting ones.
    if re.search(r'\\boxed|####|(?i:answer\s*(?:is|equals)?\s*[:=]?\s*)\$?\s*'+NUMBER,text):return None
    lines=[x.strip() for x in text.splitlines() if x.strip()]
    if not lines:return None
    final=lines[-1]
    if not re.match(r'(?i)^(?:therefore|thus|hence|so)\b[,:]?|^(?:final\s+)?answer\s*:',final):return None
    if not re.search(r'[.!]$',final):return None
    # A single value in a concluding statement, never a reasoning equation or last-number fallback.
    if re.search(r'=|[+*/]|\d\s*-\s*\d',final):return None
    values=re.findall(NUMBER,final)
    if len(values)!=1:return None
    value=numeric(values[0])
    return str(value) if value is not None and value.is_finite() else None

def selftest():
    cases=[('Therefore, there are now 17 crayons.','17'),
      ('Therefore, there are now 17 crayons and 22 pencils.',None),
      ('We calculate 81 - 29 = 52.',None),('Therefore, 81 - 29 = 52.',None),
      ('Answer: Kim now has 17 crayons.','17'),('Therefore, Kim has 17 crayons',None),
      ('Answer: 17. Answer: 22.\nTherefore, 17 crayons.',None),
      ('Thus, the cost is $ 17.25.','17.25'),('There are 17 crayons.',None)]
    for text,expected in cases:assert audited_answer(text,'AddSub')==expected,(text,expected)
if __name__=='__main__':selftest();print('9 synthetic audit cases passed')
