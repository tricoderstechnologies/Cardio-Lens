(()=>{
const $=s=>document.querySelector(s),f=(v,d=3)=>v==null?'n/a':(+v).toFixed(d);
const meta=document.querySelector('meta[name=cardiolens-api]'),API=(meta&&meta.content)||'';
function run(D,live){
const N=Object.keys(D.models),sh=n=>n.split(' (')[0],col=[ '#8795A5','#0E7C86','#C2772B'],T=n=>D.models[n].test;
const INS='Insufficient sample size for reliable comparison.';
const cat=p=>p<.35?['Lower model likelihood within this dataset','#2E8B57']:p<=.65?['Uncertain model output','#C9A227']:['Higher model likelihood within this dataset','#B4452F'];
// nav
const burger=$('#burger'),menu=$('#menu');
burger.onclick=()=>{const o=menu.classList.toggle('open');burger.setAttribute('aria-expanded',o)};
menu.querySelectorAll('a').forEach(a=>a.onclick=()=>{menu.classList.remove('open');burger.setAttribute('aria-expanded',false)});
const links=[...menu.querySelectorAll('a')];
const io=new IntersectionObserver(es=>es.forEach(e=>{if(e.isIntersecting)links.forEach(l=>l.classList.toggle('on',l.getAttribute('href')==='#'+e.target.id))}),{rootMargin:'-40% 0px -55% 0px'});
document.querySelectorAll('main section[id]').forEach(s=>io.observe(s));
// dataset
const d=D.dataset,mk=Object.entries(d.missing).map(([k,v])=>`${k}: ${v}`).join(', ');
$('#ds-cards').innerHTML=[[d.n_raw,'original records'],[d.n0,'target 0: no angiographic disease'],[d.n1,'target 1: disease present (num > 0)'],[d.n_features,'predictive features'],[d.rows_missing,'rows with missing values ('+mk+')']].map(([v,l])=>`<div class="card"><div class="big">${v}</div><div class="sm">${l}</div></div>`).join('');
$('#ds-text').textContent=`The original diagnosis column ranges 0 to 4 (${Object.entries(d.num_dist).map(([k,v])=>k+': '+v).join(', ')}). It is converted to a binary target: 0 means no angiographic heart disease, anything greater than 0 means disease present. Records after cleaning: ${d.n_clean}. Missing values are imputed inside the model pipelines.`;
// audit
const S=D.split;$('#split-text').textContent=`Stratified 80/20 split (seed ${S.seed}): ${S.n_train} training and ${S.n_test} held-out test patients (${S.test_positive} positive). Threshold ${S.threshold}. Results are from executed code.`;
$('#model-cards').innerHTML=N.map((n,i)=>`<div class="card" style="border-top:3px solid ${col[i]}"><h3>${sh(n)}</h3><div class="sm">Accuracy</div><div class="big">${f(T(n).accuracy)}</div><div class="sm">Recall ${f(T(n).recall)} &middot; ROC-AUC ${f(T(n).roc_auc)}</div></div>`).join('');
const MET=[['accuracy','Accuracy'],['precision','Precision'],['recall','Recall'],['f1','F1'],['macro_f1','Macro F1'],['roc_auc','ROC-AUC'],['false_positive_rate','False-positive rate'],['false_negative_rate','False-negative rate']];
function tabs(el,items,sel,fn,multi){el.innerHTML=items.map(([k,l],i)=>`<button type="button" data-k="${k}" aria-pressed="${multi?true:i===sel}">${l}</button>`).join('');
 el.querySelectorAll('button').forEach(b=>b.onclick=()=>{if(multi)b.setAttribute('aria-pressed',b.getAttribute('aria-pressed')!=='true');else el.querySelectorAll('button').forEach(x=>x.setAttribute('aria-pressed',x===b));fn(b.dataset.k)})}
function bars(k){$('#metric-bars').innerHTML=N.map((n,i)=>`<div class="row"><span>${sh(n)}</span><div class="track"><div class="fill" style="width:${(T(n)[k]||0)*100}%;background:${col[i]}"></div></div><b>${f(T(n)[k])}</b></div>`).join('')}
tabs($('#metric-tabs'),MET,0,bars);bars('accuracy');
function cm(n){const t=T(n);$('#cm').innerHTML=`<div class="cm"><div class="h"></div><div class="h">Predicted 0</div><div class="h">Predicted 1</div><div class="h">True 0</div><div class="c" style="background:#2E8B5740">${t.tn}</div><div class="c" style="background:#B4452F40">${t.fp}</div><div class="h">True 1</div><div class="c" style="background:#B4452F40">${t.fn}</div><div class="c" style="background:#2E8B5740">${t.tp}</div></div>`;
 $('#cm-text').textContent=`${t.fp} false positives (rate ${f(t.false_positive_rate)}) and ${t.fn} false negatives (rate ${f(t.false_negative_rate)}).`}
tabs($('#cm-tabs'),N.map(n=>[n,sh(n)]),1,cm);cm(N[1]);
const show=new Set(N);function roc(){const x=v=>40+v*260,y=v=>260-v*240;let s=`<line x1="40" y1="260" x2="300" y2="260" stroke="currentColor" opacity=".4"/><line x1="40" y1="20" x2="40" y2="260" stroke="currentColor" opacity=".4"/><line x1="${x(0)}" y1="${y(0)}" x2="${x(1)}" y2="${y(1)}" stroke="currentColor" opacity=".4" stroke-dasharray="3 4"/><text x="170" y="285" text-anchor="middle">False positive rate</text><text transform="translate(12 140) rotate(-90)" text-anchor="middle">True positive rate</text>`;
 N.forEach((n,i)=>{if(show.has(n))s+=`<polyline fill="none" stroke="${col[i]}" stroke-width="2.2" points="${D.roc[n].map(p=>x(p[0])+','+y(p[1])).join(' ')}"/>`});$('#roc').innerHTML=s}
tabs($('#roc-tabs'),N.map(n=>[n,`${sh(n)} (AUC ${f(T(n).roc_auc)})`]),0,k=>{show.has(k)?show.delete(k):show.add(k);roc()},true);roc();
const C=D.models;$('#cv').innerHTML='<tr><th>Model</th><th>Accuracy</th><th>Recall</th><th>F1</th><th>ROC-AUC</th></tr>'+N.map(n=>'<tr><td>'+sh(n)+'</td>'+['accuracy','recall','f1','roc_auc'].map(k=>`<td>${f(C[n].cv[k].mean)} &plusmn; ${f(C[n].cv[k].std)}</td>`).join('')+'</tr>').join('');
const lr=N[1],rf=N[2],err=n=>T(n).fp+T(n).fn;
$('#interp').textContent=`Both trained models clearly beat the baseline (accuracy ${f(T(N[0]).accuracy)}, ROC-AUC ${f(T(N[0]).roc_auc)}). ${sh(rf)} scored higher on this one test split (accuracy ${f(T(rf).accuracy)} vs ${f(T(lr).accuracy)}), but that is a difference of ${Math.abs(err(lr)-err(rf))} patients out of ${S.n_test}, and cross-validation accuracy on the training data favoured ${C[lr].cv.accuracy.mean>C[rf].cv.accuracy.mean?sh(lr):sh(rf)} (${f(C[lr].cv.accuracy.mean)} vs ${f(C[rf].cv.accuracy.mean)}). With a test set this small the models should be considered comparable.`;
// importance
const mx=Math.max(...D.importance.map(r=>r.importance_mean+r.importance_std));
$('#imp').innerHTML=D.importance.slice(0,8).map((r,i)=>`<div class="row"><span>${i+1}. ${r.feature}</span><div class="track"><div class="fill" style="width:${r.importance_mean/mx*100}%;background:${i<5?'var(--acc)':'var(--mut)'}"></div></div><b>${f(r.importance_mean)}</b></div>`).join('');
// explorer
$('#ex-note').textContent=`Precomputed research demonstration: Logistic Regression outputs and ${D.local_method} for the ${D.examples.length} held-out test patients, saved from the executed project. Nothing is computed in your browser and no data is collected.`;
const sel=$('#ex-sel');let best=0;D.examples.forEach((e,i)=>{if(Math.abs(e.proba-.5)<Math.abs(D.examples[best].proba-.5))best=i;});
sel.innerHTML=D.examples.map((e,i)=>`<option value="${i}">row ${e.row_id} (probability ${f(e.proba,2)})</option>`).join('');sel.value=best;
function ex(){const e=D.examples[sel.value],c=cat(e.proba),m=Math.max(...e.contrib.map(x=>Math.abs(x.c)));
 const pred=e.proba>=.5?1:0;
 $('#ex-out').innerHTML=`<div class="grid"><div><div class="sm">Model-predicted probability</div><div class="big">${f(e.proba)}</div></div><div><div class="sm">Category</div><b style="color:${c[1]}">${c[0]}</b></div><div><div class="sm">Dataset label / model class at 0.5</div><b>${e.y_true} / ${pred}</b> ${pred===e.y_true?'(match)':'<span class="warn">(error)</span>'}</div></div>
 <h3 style="margin-top:18px">Feature contributions (log-odds)</h3>`+e.contrib.map(x=>`<div class="row"><span>${x.feature} = ${x.value==null?'missing (imputed)':x.value}</span><div class="track"><span class="mid"></span><div class="fill" style="position:absolute;${x.c>=0?'left:50%':'right:50%'};width:${Math.abs(x.c)/m*50}%;background:${x.c>=0?'var(--bad)':'var(--acc)'}"></div></div><b>${x.c>0?'+':''}${f(x.c,2)}</b></div>`).join('')+'<p class="sm">Red pushes the model output toward disease present; teal pushes it toward no disease. This describes the model, not medical causation.</p>'}
sel.onchange=ex;ex();
// gauge
const rng=$('#p-rng');function g(){const p=+rng.value,c=cat(p);$('#p-mk').style.left=p*100+'%';$('#p-out').innerHTML=`Probability ${f(p,2)}: <span style="color:${c[1]}">${c[0]}</span>`}rng.oninput=g;g();
// subgroups
$('#sg-text').textContent=`Logistic Regression, held-out test set (n=${S.n_test}). Reliable rates need enough cases per class; small subgroups are flagged.`;
$('#sg').innerHTML='<tr><th>Group</th><th>n</th><th>Positives</th><th>Recall</th><th>FNR</th><th>FPR</th><th>ROC-AUC</th><th>Note</th></tr>'+D.subgroups.map(r=>{const c=r.reliable?'':' class="muted"';return `<tr><td>${r.group}</td><td>${r.n}</td><td>${r.positives}</td><td${c}>${f(r.recall)}</td><td${c}>${f(r.false_negative_rate)}</td><td${c}>${f(r.false_positive_rate)}</td><td>${r.roc_auc==null?'n/a':f(r.roc_auc)}</td><td class="warn">${r.note||''}</td></tr>`}).join('');
$('#m-split').textContent=`${S.n_train} training / ${S.n_test} test patients, stratified on the target so both sets keep a similar class balance (${S.train_positive} and ${S.test_positive} positives). Seed ${S.seed}.`;

// error analysis
$('#err').textContent=[lr,rf].map(n=>`${sh(n)}: ${T(n).fp} false positives (a model "disease" call where the dataset says none) and ${T(n).fn} false negatives (a missed disease case).`).join(' ')+' Each error moves a rate by several points at this sample size, and the 0.5 threshold was not tuned.';
// mode badge
$('#mode').textContent=live?'Live API connected: playground uses the real trained models':'Static mode: showing saved results (no backend reachable)';
$('#mode').dataset.live=live;
// playground
const SP=D.schema,fm=$('#pg-form'),ps=$('#pg-pre'),out=$('#pg-out'),btn=$('#pg-run');
fm.innerHTML=SP.map(s=>s.kind==='cat'?`<label>${s.label}<select data-f="${s.name}"><option value="">missing (imputed)</option>${s.options.map(o=>`<option value="${o[0]}">${o[1]}</option>`).join('')}</select></label>`:`<label>${s.label}<input data-f="${s.name}" type="number" min="${s.min}" max="${s.max}" step="${s.step}" placeholder="missing (imputed)"></label>`).join('');
ps.innerHTML='<option value="">Choose a held-out test record to fill the form</option>'+D.examples.map((e,i)=>`<option value="${i}">row ${e.row_id}</option>`).join('');
ps.onchange=()=>{const e=D.examples[ps.value];if(!e)return;fm.querySelectorAll('[data-f]').forEach(el=>{const v=e.features[el.dataset.f];el.value=v==null?'':v})};
btn.disabled=!live;
if(!live)out.innerHTML='<p class="sm">Live inference needs the Python backend. Start it with <code>uvicorn src.api:app</code> and open the site from that server, or set the API address in the page. Until then you can still browse every saved result above.</p>';
btn.onclick=async()=>{const body={};fm.querySelectorAll('[data-f]').forEach(el=>{body[el.dataset.f]=el.value===''?null:Number(el.value)});
 out.innerHTML='<p class="sm">Running the trained models...</p>';
 try{const r=await fetch(API+'api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const j=await r.json();
  if(!r.ok){out.innerHTML='<p class="warn">Input rejected: '+(Array.isArray(j.detail)?j.detail.map(d=>d.loc.slice(-1)[0]+' '+d.msg).join('; '):'invalid request')+'</p>';return}
  const m=j.models,c=j.explanation.contributions.slice().sort((a,b)=>Math.abs(b.contribution)-Math.abs(a.contribution)).slice(0,8),mx=Math.max(...c.map(x=>Math.abs(x.contribution)));
  out.innerHTML='<div class="eyebrow">Live model output (not a saved result)</div><div class="grid">'+[['logistic_regression','Logistic Regression'],['random_forest','Random Forest']].map(([k,l])=>{const q=cat(m[k].probability);return `<div class="card"><div class="sm">${l} probability</div><div class="big">${f(m[k].probability)}</div><b style="color:${q[1]}">${q[0]}</b></div>`}).join('')+'</div><h3 style="margin-top:16px">Logistic Regression: feature contributions (log-odds)</h3>'+c.map(x=>`<div class="row"><span>${x.feature} = ${x.value==null?'missing':x.value}</span><div class="track"><span class="mid"></span><div class="fill" style="position:absolute;${x.contribution>=0?'left:50%':'right:50%'};width:${Math.abs(x.contribution)/mx*50}%;background:${x.contribution>=0?'var(--bad)':'var(--acc)'}"></div></div><b>${x.contribution>0?'+':''}${f(x.contribution,2)}</b></div>`).join('')+(j.imputed_features.length?`<p class="sm">Imputed (left blank): ${j.imputed_features.join(', ')}.</p>`:'')+'<p class="sm">'+j.note+'</p>';
 }catch(e){out.innerHTML='<p class="warn">Could not reach the backend. The saved results above are unaffected.</p>'}};
}
let started=false;const go=(d,l)=>{if(started)return;started=true;run(d,l)};
const ctl=new AbortController(),tm=setTimeout(()=>ctl.abort(),5000);
fetch(API+'api/results',{signal:ctl.signal}).then(r=>{if(!r.ok)throw 0;return r.json()}).then(d=>{clearTimeout(tm);go(d,true)}).catch(()=>{clearTimeout(tm);go(window.CARDIOLENS,false)});
})();
