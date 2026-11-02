"""Render the assignment report from verified experiment evidence."""
import sys,json,html
from pathlib import Path
P=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(P/'report/python-deps'))
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image,KeepTogether
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=json.loads((P/'results/final/summary.json').read_text())
lookup={(r['model'],r['method']):r for r in D['results']}
order=[('tinyllama','baseline'),('tinyllama','lora'),('tinyllama','adapterp'),('openllama','baseline'),('openllama','lora'),('openllama','adapterp')]
names={'tinyllama':'TinyLlama','openllama':'OpenLLaMA'}
methods={'baseline':'Base','lora':'LoRA','adapterp':'AdapterP'}
report=P/'report';report.mkdir(exist_ok=True)
fig,ax=plt.subplots(figsize=(6.6,2.15))
for model,method in order:
 if method=='baseline':continue
 r=lookup[model,method];hist=r['training']['history'];points=[x for x in hist if 'eval_loss' in x]
 ax.plot([x['step'] for x in points],[x['eval_loss'] for x in points],label=names[model]+' '+methods[method],linewidth=1.5)
ax.set(xlabel='Optimizer step',ylabel='Validation loss');ax.spines[['top','right']].set_visible(False);ax.legend(fontsize=7,ncol=2);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(report/'validation-loss.png',dpi=180);plt.close(fig)
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyReport',fontName='Times-Roman',fontSize=10.5,leading=13.7,spaceAfter=7))
styles.add(ParagraphStyle(name='HeadReport',fontName='Times-Bold',fontSize=14,leading=17,spaceBefore=7,spaceAfter=8))
styles.add(ParagraphStyle(name='SmallReport',fontName='Times-Roman',fontSize=9,leading=11,spaceAfter=6))
styles.add(ParagraphStyle(name='CoverTitle',fontName='Times-Bold',fontSize=25,leading=31,alignment=TA_CENTER,spaceAfter=22))
styles.add(ParagraphStyle(name='CoverText',fontName='Times-Roman',fontSize=13,leading=20,alignment=TA_CENTER))
story=[];md=['# Parameter-Efficient Fine-Tuning of Large Language Models','','AGA CILA | G2605698H | AI6130 Assignment 2','']
def para(text,style='BodyReport'):
 story.append(Paragraph(text,styles[style]));md.append(text.replace('<b>','').replace('</b>',''));md.append('')
def head(text):para(text,'HeadReport')
def table(rows,widths=None):
 cell=[[Paragraph(html.escape(str(x)),styles['SmallReport']) for x in row] for row in rows]
 t=Table(cell,colWidths=widths,repeatRows=1,hAlign='LEFT')
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#eeeeee')),('LINEABOVE',(0,0),(-1,0),.7,colors.black),('LINEBELOW',(0,0),(-1,0),.4,colors.black),('LINEBELOW',(0,-1),(-1,-1),.7,colors.black),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
 story.append(t);story.append(Spacer(1,6))
 md.extend(['| '+' | '.join(map(str,row))+' |' for row in rows[:1]]+['| '+' | '.join(['---']*len(rows[0]))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows[1:]]+[''])
def page():story.append(PageBreak())
story.append(Spacer(1,43*mm))
para('Parameter-Efficient Fine-Tuning<br/>of Large Language Models','CoverTitle')
para('AI6130 Large Language Models<br/>Assignment 2','CoverText')
story.append(Spacer(1,22*mm))
para('AGA CILA<br/>G2605698H<br/>Nanyang Technological University','CoverText')
story.append(Spacer(1,20*mm))
para('Experiments completed: 1 November 2026<br/>Report prepared: 2 November 2026','CoverText')
page()
head('1. Introduction')
para('This study compares LoRA and AdapterP on TinyLlama_v1.1 and OpenLLaMA 3B v2 for mathematical reasoning. Four adapters were trained on the same math_7k split and evaluated on AQuA and AddSub. Both original models were also evaluated, giving six model configurations and twelve complete benchmark evaluations. The comparison examines answer accuracy, trainable parameters, loss and output behavior.')
head('2. Experimental setup')
para('<b>Data and models.</b> The assigned math_7k.json contains 6,851 examples. A fixed Python seed-42 shuffle assigns 6,731 examples to training and 120 to validation. The test sets contain 254 AQuA and 395 AddSub questions. No exact normalized instruction overlap was found between math_7k and either test set. Test questions and reference answers were kept unchanged. Model IDs are TinyLlama/TinyLlama_v1.1 and openlm-research/open_llama_3b_v2; their Hugging Face revisions and dataset hashes are recorded in the accompanying metadata.')
para('<b>Adapters.</b> The instructor\'s bundled PEFT implementation was used. LoRA targets q_proj and v_proj with rank 8, alpha 16 and dropout 0.05. AdapterP uses BottleneckConfig(use_adapterp=True), equivalent to --adapter_name bottleneck --use_adapterp, with bottleneck width 256, tanh activation and zero adapter dropout. Base weights remain frozen.')
para('<b>Training.</b> All runs use two epochs, AdamW, learning rate 3e-4, 100 warmup steps, effective batch size 4, micro batch size 1, gradient accumulation 4, sequence cutoff 256 and seed 42. Loss covers the prompt and response tokens. Gradient checkpointing is enabled. Validation and checkpointing occur every 200 optimizer steps. Each run completes 3,364 steps; the final adapter is the lowest-validation-loss checkpoint, step 3,200 in all four runs.')
para('<b>Hardware and precision.</b> Each experiment uses one NSCC NVIDIA A100-SXM4-40GB. Base weights use BF16 and trainable adapter weights FP32, without quantization. A training-item diagnostic found non-finite FP16 gradients and finite BF16 gradients, so BF16 was applied consistently. The environment uses Python 3.12, PyTorch 2.8.0+cu126, Transformers 4.38.0, Accelerate 0.28.0 and datasets 2.18.0.')
para('<b>Generation and scoring.</b> All six configurations use the instructor\'s evaluation prompt, greedy decoding, one beam, a 256-token output cap and KV caching. This replaces the example\'s four-beam decoder and disabled cache. Only newly generated tokens are scored. AQuA requires an explicit A-E answer; AddSub uses numeric tolerance 0.001. Missing or ambiguous answers count as incorrect.')
para('<b>Answer extraction audit.</b> The initial scorer accepts explicit answer markers but misses conclusions such as "Therefore, there are now 4 crayons." A separate audit accepts one numeric value in the final line when it is a complete conclusion beginning Therefore, Thus, Hence, So or Answer:. Equations, multiple values and conflicting explicit answers are rejected. This rule was applied uniformly to all stored outputs, without regeneration. Table 1 retains the original scores alongside the audited AddSub scores. AQuA scoring is unchanged.')
page()
head('3. Results')
para('<b>Table 1.</b> Accuracy (%) on complete test sets. AddSub v1 is the original marker-based score; AddSub audit additionally recognizes the final conclusion described in Section 2. AQuA has 254 questions and AddSub has 395 questions per row.','SmallReport')
rows=[['Model','Method','Trainable','AQuA','AddSub v1','AddSub audit']]
for model,method in order:
 r=lookup[model,method];s=r['setup'];a=r['benchmarks']['AQuA']['original_v1'];b=r['benchmarks']['AddSub']
 rows.append([names[model],methods[method],f"{s['parameters']['trainable']/1e6:.3f}M" if s else '0',f"{a['accuracy']:.2f}",f"{b['original_v1']['accuracy']:.2f}",f"{b['conclusion_audit_v2']['accuracy']:.2f}"])
table(rows,[83,57,65,55,74,82])
para('OpenLLaMA + LoRA produces the highest audited AddSub accuracy, 38.23% (151/395), followed by OpenLLaMA + AdapterP at 29.87% (118/395). TinyLlama + LoRA leads AQuA with 11.81% (30/254). Both baselines score below their corresponding fine-tuned models under the shared evaluation protocol.')
para('The extraction audit changes 1,005 AddSub predictions and recovers 294 correct answers across the six configurations. For OpenLLaMA + LoRA, the count rises from 31 to 151 correct answers. The baseline scores do not change. These differences show that explicit answer formatting accounts for a substantial part of the original AddSub error count.')
para('<b>Table 2.</b> Training and resource measurements. Train loss is the logged mean over training; validation loss is measured after reloading the selected checkpoint. Time includes the final validation and save. Memory is peak allocated GPU memory, rather than total device occupancy.','SmallReport')
rows=[['Model / method','Trainable %','Train loss','Val. loss','Minutes','GPU GiB']]
for model,method in [x for x in order if x[1]!='baseline']:
 r=lookup[model,method];t=r['training'];s=r['setup']
 rows.append([names[model]+' / '+methods[method],f"{s['parameters']['percent']:.3f}",f"{t['metrics']['train_loss']:.4f}",f"{t['validation']['eval_loss']:.4f}",f"{t['seconds_this_session']/60:.1f}",f"{t['peak_gpu_bytes']/2**30:.2f}"])
table(rows,[137,65,57,57,48,52])
head('4. Comparison of methods and models')
para('<b>LoRA versus AdapterP.</b> LoRA is higher on both TinyLlama benchmarks: +0.79 percentage points on AQuA and +0.76 points on audited AddSub. For OpenLLaMA, AdapterP is higher on AQuA by 1.97 points, while LoRA is higher on audited AddSub by 8.35 points. The method ranking therefore depends on the benchmark.')
para('<b>Model size.</b> OpenLLaMA exceeds TinyLlama on audited AddSub by 20.00 points with LoRA and 12.41 points with AdapterP. TinyLlama is higher on AQuA with either method. Larger model size gives a clear advantage on AddSub in these runs, while AQuA remains dominated by low answer accuracy and frequent unfinished or invalid responses.')
page()
head('4. Comparison of methods and models (continued)')
para('<b>Parameter efficiency.</b> AdapterP trains 20.48 times as many parameters as LoRA on TinyLlama and 16 times as many on OpenLLaMA. These extra parameters do not deliver a consistent accuracy gain. LoRA has the stronger parameter-to-performance trade-off here: it wins three of the four within-model benchmark comparisons, including the largest AddSub gain. AdapterP is faster in these runs, taking 34.9 versus 41.4 minutes on TinyLlama and 40.9 versus 42.1 minutes on OpenLLaMA.')
para('<b>Loss and accuracy.</b> Validation loss decreases throughout most of training, as Figure 1 shows. TinyLlama AdapterP has lower validation loss than LoRA (0.7265 versus 0.7651) but slightly lower accuracy on both benchmarks. OpenLLaMA LoRA has the lowest validation loss (0.6891) and the best AddSub result, while AdapterP has the better AQuA result. Token prediction loss and final-answer accuracy thus measure different behavior.')
story.append(Image(str(report/'validation-loss.png'),width=166*mm,height=54*mm));md.append('![Validation loss](validation-loss.png)\n')
para('<b>Figure 1.</b> Validation loss at saved evaluation steps. All final adapters were selected at step 3,200.','SmallReport')
para('<b>Table 3.</b> Invalid-answer and generation-cap counts. Each cell reports invalid / cap-hit responses; the categories can overlap. AddSub invalid counts use the conclusion audit.','SmallReport')
rows=[['Model','Method','AQuA invalid / cap','AddSub invalid / cap']]
for model,method in order:
 r=lookup[model,method];a=r['benchmarks']['AQuA']['conclusion_audit_v2'];b=r['benchmarks']['AddSub']['conclusion_audit_v2']
 rows.append([names[model],methods[method],f"{a['invalid']} / {a['truncated']}",f"{b['invalid']} / {b['truncated']}"])
table(rows,[100,65,120,131])
para('<b>Output behavior.</b> Every baseline response reaches the 256-token cap. OpenLLaMA produces whitespace-only output on 126 AQuA questions, and 342 of its 395 AddSub outputs trigger the repetition heuristic. Fine-tuning changes this behavior: AddSub cap hits fall to 15-49 per configuration. The baseline comparison therefore captures improved task-following and stopping behavior as well as answer correctness under this prompt.')
page()
head('5. Error analysis')
para('Saved outputs distinguish arithmetic errors, reasoning errors, format errors and extraction errors. The examples below use the zero-based indices in the prediction journals; each remains traceable to the original question and gold answer.')
para('<b>Arithmetic error: OpenLLaMA + LoRA, AddSub 2.</b> The question gives 12 games played and 4 defeats. The model chooses the correct operation but writes "12 - 4 = 10" and concludes 10 wins. The reference answer is 8. This is an incorrect calculation despite a suitable problem formulation.')
para('<b>Reasoning error: OpenLLaMA + AdapterP, AddSub 0.</b> With 7 crayons initially and 3 removed, the model first computes 7 - 3 = 4 correctly, then adds 7 + 3 and concludes 10. The final answer is wrong because the model introduces an unnecessary second operation and replaces its correct intermediate result.')
para('<b>Extraction error: TinyLlama + AdapterP, AddSub 0.</b> The model concludes "Therefore, there are now 4 crayons in the drawer." The answer is correct, but the original scorer returns no prediction because the output lacks an explicit numeric answer marker. The conclusion audit extracts 4 and marks it correct. This illustrates why a reasoning trace\'s last number and its declared final answer must be handled separately.')
para('<b>Format error: TinyLlama + LoRA, AQuA 2.</b> The response ends with a numeric count of children rather than an A-E option. The AQuA scorer returns an invalid answer. A numeric conclusion does not supply the requested option label.')
para('<b>Repetition and truncation: TinyLlama + LoRA, AQuA 3.</b> The model repeatedly restates the discounted price as 20 and reaches the generation cap without selecting an option. This is a generation failure before final-answer extraction.')
para('<b>Training length.</b> At the 256-token training cutoff, 4,424 of 6,851 tokenized examples reach the cap for TinyLlama and 4,064 for OpenLLaMA. Long examples therefore lose later reasoning or answer tokens. A longer training context and more explicit final-answer supervision are concrete follow-up experiments. The current comparison uses one common seed and one hyperparameter setting per model-method pair; the reported gaps describe these completed runs.')
head('6. Conclusion')
para('LoRA offers the best overall parameter efficiency in this experiment. OpenLLaMA + LoRA is strongest on AddSub after conclusion extraction, reaching 38.23% with 2.662 million trainable parameters. TinyLlama + LoRA has the highest AQuA score at 11.81%. AdapterP improves OpenLLaMA\'s AQuA result but does not consistently benefit from its larger trainable parameter count. Fine-tuning substantially improves response behavior relative to the original models. Reliable comparison also requires recognizing explicit conclusions: the AddSub extraction audit recovers 294 correct answers from unchanged outputs.')
head('AI use declaration')
para('OpenAI Codex was used extensively to implement and debug the experiment workflow, configure cluster jobs, develop answer-extraction rules, validate and summarize logs, and draft and format this report. Training and generation were executed on the NSCC GPUs. Reported measurements are derived from the saved experiment records.','SmallReport')
head('Sources and reproducibility')
para('Instructor repository: github.com/duyngtr16061999/AI6130_Assignment2, commit 21086af375ddb6433e8dd00df0a28eeb5ce65d25. Models: TinyLlama/TinyLlama_v1.1 and openlm-research/open_llama_3b_v2. The accompanying package contains pinned revisions, data hashes and split indices, configuration, raw predictions, original and audited scores, training logs and final adapter weights.','SmallReport')
def footer(canvas,doc):
 if doc.page>1:
  canvas.setFont('Times-Roman',9);canvas.setFillColor(colors.HexColor('#555555'))
  canvas.drawString(22*mm,14*mm,'AI6130 Assignment 2 | AGA CILA | G2605698H')
  canvas.drawRightString(A4[0]-22*mm,14*mm,str(doc.page-1))
output=report/'AI6130-Ass2-G2605698H.pdf'
doc=SimpleDocTemplate(str(output),pagesize=A4,leftMargin=22*mm,rightMargin=22*mm,topMargin=19*mm,bottomMargin=22*mm,title='AI6130 Assignment 2 - Parameter-Efficient Fine-Tuning',author='AGA CILA')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
(report/'AI6130-Ass2-G2605698H.md').write_text('\n'.join(md),encoding='utf-8')
import pymupdf
pdf=pymupdf.open(output)
assert len(pdf)==5,f'Expected cover + 4 pages, got {len(pdf)}'
preview=report/'preview';preview.mkdir(exist_ok=True)
for i,page in enumerate(pdf):page.get_pixmap(matrix=pymupdf.Matrix(1.15,1.15)).save(preview/f'page-{i+1}.png')
print('Created',output.name,'pages',len(pdf))
