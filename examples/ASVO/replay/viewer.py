"""Render a replay document as a self-contained HTML application."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def _safe_embedded_json(document: Mapping[str, Any]) -> str:
    return json.dumps(document, ensure_ascii=False).replace("</", "<\\/")


def render_replay_html(
    document: Mapping[str, Any],
    output_path: str | Path,
) -> Path:
    """Write a standalone viewer containing its replay data and return its path."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    title = str(document.get("simulation", {}).get("title", "ASVO Simulation"))
    html = _HTML.replace("__TITLE__", _escape_html(title)).replace(
        "__REPLAY_DATA__", _safe_embedded_json(document)
    )
    output.write_text(html, encoding="utf-8")
    return output


def render_replay_file(
    json_path: str | Path, output_path: str | Path | None = None
) -> Path:
    """Render a replay JSON file, defaulting to a sibling ``.html`` file."""
    source = Path(json_path)
    document = json.loads(source.read_text(encoding="utf-8"))
    destination = Path(output_path) if output_path else source.with_suffix(".html")
    return render_replay_html(document, destination)


def _escape_html(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


_HTML = r'''<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__ — Replay Viewer</title>
<style>
:root{--ink:#1e272d;--muted:#667078;--paper:#f5f1e9;--card:#fffefa;--soft:#f4f1eb;--line:#ddd7cd;--blue:#356b91;--up:#2f725c;--down:#a24e47;--shadow:0 16px 42px rgba(35,42,46,.09)}
*{box-sizing:border-box}body{margin:0;min-height:100vh;color:var(--ink);background:linear-gradient(145deg,#eef1ed,var(--paper));font-family:-apple-system,BlinkMacSystemFont,"Hiragino Sans","Yu Gothic UI",sans-serif}button,input,select{font:inherit}button:focus-visible,input:focus-visible,select:focus-visible,summary:focus-visible{outline:3px solid rgba(53,107,145,.3);outline-offset:2px}
.app{max-width:1440px;margin:auto;padding:28px 26px 44px}.masthead{display:flex;justify-content:space-between;align-items:flex-end;gap:20px;margin-bottom:15px}.eyebrow{margin:0 0 5px;color:var(--blue);font-size:11px;font-weight:700;letter-spacing:.15em;text-transform:uppercase}.masthead h1{margin:0;font-family:Georgia,"Yu Mincho",serif;font-size:clamp(26px,3vw,42px);font-weight:500;line-height:1.05}.timebox{text-align:right}.step-label{font-size:20px;font-weight:800}.clock{margin-top:3px;color:var(--muted);font-size:12px;font-variant-numeric:tabular-nums}
.stage{position:relative;border:1px solid rgba(91,79,65,.18);border-radius:22px;background:rgba(255,254,250,.82);box-shadow:var(--shadow);padding:20px}.stage:before{content:"";position:absolute;inset:0 0 auto;height:4px;border-radius:22px 22px 0 0;background:linear-gradient(90deg,#b86757,#c99b3a,#467f76,#4c7093)}
.step-event,.step-summary{margin:3px 0 16px;padding:12px 14px;border-left:3px solid var(--blue);background:var(--soft);font-size:12px;line-height:1.65;white-space:pre-wrap;overflow-wrap:anywhere}.step-summary{border-left-color:#467f76;background:#eef3ef}
.agents{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}.agent{--agent-accent:#b86757;min-width:0;padding:17px;border:1px solid var(--line);border-radius:15px;background:var(--card);opacity:0;transform:translateY(8px);animation:arrive .35s ease forwards}.agent:nth-child(2){--agent-accent:#b0842d;animation-delay:.06s}.agent:nth-child(3){--agent-accent:#39776b;animation-delay:.12s}@keyframes arrive{to{opacity:1;transform:none}}
.identity{display:flex;align-items:center;gap:10px;padding-bottom:12px;border-bottom:1px solid var(--line)}.avatar{display:grid;place-items:center;width:38px;height:38px;flex:0 0 auto;border-radius:50%;background:var(--agent-accent);color:#fff;font-family:Georgia,serif;font-size:17px}.name{font-size:18px;font-weight:800}.agent-meta{margin-top:2px;color:var(--muted);font-size:10px;font-variant-numeric:tabular-nums}.block{margin-top:15px}.block-title{margin:0 0 7px;color:var(--muted);font-size:10px;font-weight:800;letter-spacing:.08em}.action-text,.observation-text{margin:0;white-space:pre-wrap;overflow-wrap:anywhere}.action-text{padding:11px 12px;border-radius:9px;background:var(--soft);font-family:Georgia,"Yu Mincho",serif;font-size:13px;line-height:1.65}.observation-text{padding-left:10px;border-left:3px solid var(--agent-accent);font-size:11px;line-height:1.6}
.teamwork-head{display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:8px}.teamwork-head .block-title{margin:0}.teamwork-vector{color:var(--muted);font:10px ui-monospace,SFMono-Regular,Menlo,monospace}.teamwork-chips{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:5px}.teamwork-chip{display:flex;align-items:center;gap:5px;min-width:0;padding:5px 7px;border:1px solid var(--line);border-radius:7px;color:#8b8b87;background:#f7f5f0;font-size:9px;line-height:1.25}.teamwork-chip:before{content:"○";flex:0 0 auto}.teamwork-chip.active{border-color:rgba(53,107,145,.38);color:#254f6c;background:#eaf1f5;font-weight:700}.teamwork-chip.active:before{content:"●";color:var(--blue)}
.disclosure{margin-top:8px;border-top:1px solid var(--line)}.disclosure>summary,.candidate-prediction>summary,.retry>summary{cursor:pointer;color:var(--blue);font-size:11px;font-weight:700;list-style-position:inside}.disclosure>summary{padding:9px 0 2px}.decision-body{display:grid;gap:11px;padding:9px 0 3px}.internal-note{margin:0;padding:8px 10px;border-radius:7px;background:#eef2f3;color:var(--muted);font-size:10px;line-height:1.5}.fact{margin:0;font-size:11px;line-height:1.55;white-space:pre-wrap;overflow-wrap:anywhere}.fact-label{display:block;margin-bottom:2px;color:var(--muted);font-size:9px;font-weight:800;letter-spacing:.04em}.current-desires{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:4px 10px;margin:0;padding:0;list-style:none;font-size:10px}.current-desires li{display:flex;justify-content:space-between;gap:8px;border-bottom:1px dotted var(--line);font-variant-numeric:tabular-nums}
.candidate-list{display:grid;gap:8px}.candidate{padding:10px;border-left:3px solid #c8c1b7;background:var(--soft)}.candidate.selected{border-left-color:var(--agent-accent);background:#f1f4f0}.candidate-head{display:flex;justify-content:space-between;gap:8px;margin-bottom:5px;font-size:10px;font-weight:800}.selected-mark{color:var(--up)}.candidate-action{margin:0;font-size:11px;line-height:1.55;white-space:pre-wrap;overflow-wrap:anywhere}.candidate-prediction{margin-top:7px}.candidate-prediction>summary{font-size:10px}.prediction-body{display:grid;gap:6px;padding:7px 0 0}.prediction-list{display:grid;grid-template-columns:minmax(90px,auto) 1fr;gap:3px 8px;margin:0;font-size:10px}.prediction-list dt{color:var(--muted);font-weight:700}.prediction-list dd{margin:0;white-space:pre-wrap;overflow-wrap:anywhere}.raw-prediction{margin:4px 0 0;white-space:pre-wrap;overflow-wrap:anywhere;color:var(--muted);font:inherit}
.desire-hint{margin:-2px 0 7px;color:var(--muted);font-size:9px}.desire-list{border-top:1px solid var(--line)}.desire{border-bottom:1px solid var(--line)}.desire>summary{position:relative;display:block;padding:9px 18px 10px 2px;cursor:pointer;list-style:none;font-size:11px}.desire>summary::-webkit-details-marker{display:none}.desire>summary:after{content:"›";position:absolute;right:3px;top:10px;color:#aaa49b;transform:rotate(90deg)}.desire[open]>summary:after{transform:rotate(-90deg)}.desire.changed>summary{font-weight:700}.desire-top{display:flex;align-items:baseline;justify-content:space-between;gap:10px;margin-bottom:6px}.desire-name{overflow-wrap:anywhere}.desire-values{display:flex;align-items:baseline;gap:7px;color:#4f5960;font-variant-numeric:tabular-nums;white-space:nowrap}.desire-delta{min-width:24px;text-align:right;color:var(--muted)}.desire-delta.up{color:var(--up)}.desire-delta.down{color:var(--down)}.desire-bar{position:relative;height:7px;overflow:visible;border-radius:7px;background:#e7e2da}.desire-fill{height:100%;border-radius:7px;background:var(--agent-accent)}.desire-target{position:absolute;top:-3px;width:2px;height:13px;border-radius:2px;background:var(--ink);transform:translateX(-1px)}
.update-detail{margin:0 0 9px;padding:10px;border-radius:8px;background:var(--soft);font-size:10px}.update-line{display:grid;grid-template-columns:minmax(115px,1fr) auto auto;gap:8px;align-items:baseline;padding:4px 0}.update-label{color:var(--muted)}.update-value{font-weight:700;font-variant-numeric:tabular-nums;white-space:nowrap}.update-change{min-width:42px;text-align:right;font-variant-numeric:tabular-nums}.update-change.up{color:var(--up)}.update-change.down{color:var(--down)}.update-separator{height:8px;margin-left:5px;border-left:1px solid #b8b1a8}.validity{margin:5px 0;color:var(--up);font-weight:700}.validity.invalid{color:var(--down)}.retry{margin-top:7px;padding-top:7px;border-top:1px solid var(--line)}.retry-body{display:grid;gap:7px;padding-top:7px}.attempt{padding-left:8px;border-left:2px solid #c4bdb3}.attempt p{margin:2px 0;line-height:1.5;white-space:pre-wrap}.reflection{color:#5b646b}.target-note,.reason-note{margin:6px 0 0;color:var(--muted);line-height:1.5;white-space:pre-wrap}
.result-summary{padding:4px 2px}.result-title{margin:2px 0 18px;font-family:Georgia,"Yu Mincho",serif;font-size:24px}.result-section{margin-top:14px;padding:16px;border:1px solid var(--line);border-radius:13px;background:var(--card)}.result-section h2{margin:0 0 10px;font-size:15px}.result-section p{margin:0;line-height:1.75;white-space:pre-wrap}.behavior-grid,.shortage-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:11px}.result-agent{padding:12px;border-radius:9px;background:var(--soft)}.result-agent h3{margin:0 0 7px;font-size:13px}.result-agent p{font-size:12px}.shortage-list{display:grid;gap:6px;margin:0;padding:0;list-style:none}.shortage-item{display:grid;grid-template-columns:1fr auto;gap:8px;padding-bottom:5px;border-bottom:1px dotted var(--line);font-size:11px}.shortage-values{font-variant-numeric:tabular-nums;white-space:nowrap}.no-shortage{color:var(--muted);font-size:11px}
.teamwork-table-wrap{overflow-x:auto}.teamwork-table{width:100%;border-collapse:collapse;font-size:10px;font-variant-numeric:tabular-nums}.teamwork-table th,.teamwork-table td{padding:8px 7px;border-bottom:1px solid var(--line);text-align:center;white-space:nowrap}.teamwork-table th:first-child,.teamwork-table td:first-child{text-align:left;font-weight:800}.teamwork-table thead th{color:var(--muted);font-size:9px;line-height:1.35}.teamwork-table tfoot td{border-top:2px solid var(--line);font-weight:800}.teamwork-total-cell span{display:block;color:var(--blue);font-size:9px}.teamwork-note{margin-top:10px!important;color:var(--muted);font-size:10px;line-height:1.55!important}
.controls{display:grid;grid-template-columns:auto auto 1fr auto auto auto;gap:10px;align-items:center;margin-top:18px}.control-button{border:1px solid var(--line);border-radius:9px;padding:9px 13px;background:var(--card);color:var(--ink);cursor:pointer}.control-button:hover{border-color:#9f978c}.control-button:disabled{cursor:default;opacity:.42}.play{min-width:86px;background:var(--ink);border-color:var(--ink);color:#fff}.language{min-width:72px}.range{width:100%;accent-color:var(--blue)}select{border:1px solid var(--line);border-radius:8px;padding:8px;background:var(--card);color:var(--ink)}.hint{margin:10px 1px 0;color:var(--muted);font-size:10px;text-align:right}
[hidden]{display:none!important}@media(max-width:1080px){.agents{grid-template-columns:1fr}.agent{padding:16px 18px}.current-desires{grid-template-columns:repeat(4,minmax(0,1fr))}.behavior-grid,.shortage-grid{grid-template-columns:1fr}.teamwork-table-wrap{width:100%}}@media(max-width:700px){.app{padding:18px 10px 32px}.masthead{align-items:flex-start}.masthead h1{font-size:25px}.timebox{min-width:max-content}.stage{padding:15px 11px}.current-desires{grid-template-columns:repeat(2,minmax(0,1fr))}.controls{grid-template-columns:auto auto 1fr}.controls select{grid-column:3}.language{grid-column:2}.update-line{grid-template-columns:1fr auto}.update-change{grid-column:2}.hint{text-align:left}}@media(max-width:430px){.masthead{display:block}.timebox{margin-top:10px;text-align:left}.desire-top{align-items:flex-start}.desire-values{flex-wrap:wrap;justify-content:flex-end;white-space:normal}.current-desires{grid-template-columns:1fr}.teamwork-chips{grid-template-columns:1fr}.controls{grid-template-columns:1fr 1fr}.range{grid-column:1/-1;grid-row:1}.controls select{grid-column:auto}.language{grid-column:auto}}@media(prefers-reduced-motion:reduce){*,*:before,*:after{animation-duration:.01ms!important;transition-duration:.01ms!important}}
</style>
</head>
<body>
<main class="app">
  <header class="masthead">
    <div><p class="eyebrow">Simulation Replay Viewer</p><h1 id="title"></h1></div>
    <div class="timebox"><div class="step-label" id="stepLabel"></div><div class="clock" id="clock"></div></div>
  </header>
  <section class="stage" aria-live="polite">
    <section class="step-summary" id="stepSummary" hidden><h2 class="block-title">【初期状況】</h2><div id="stepSummaryText"></div></section>
    <section class="step-event" id="stepEvent" hidden><h2 class="block-title">【実際に起きたこと（Event statement）】</h2><div id="stepEventText"></div></section>
    <div class="agents" id="agents"></div>
    <section class="result-summary" id="resultSummary" hidden></section>
    <nav class="controls" aria-label="Replay controls">
      <button class="control-button" id="previous" type="button" aria-label="前のStep">← 前へ</button>
      <button class="control-button play" id="play" type="button">▶ 再生</button>
      <input class="range" id="timeline" type="range" min="0" value="0" aria-label="Step">
      <select id="speed" aria-label="再生速度"><option value="2000">0.5×</option><option value="1200" selected>1×</option><option value="650">2×</option></select>
      <button class="control-button language" id="language" type="button">EN原文</button>
      <button class="control-button" id="next" type="button" aria-label="次のStep">次へ →</button>
    </nav>
    <p class="hint">← → キーでStep移動　Spaceキーで再生／停止</p>
  </section>
</main>
<script id="replay-data" type="application/json">__REPLAY_DATA__</script>
<script>
const replay=JSON.parse(document.getElementById('replay-data').textContent);
const frames=replay.frames||[];
const labels=Object.fromEntries((replay.desires||[]).map(item=>[item.key,item.label_ja]));
const teamworkOrder=replay.teamwork_big_five?.vector_order||['team_leadership','mutual_performance_monitoring','backup_behavior','adaptability','team_orientation'];
const teamworkSource=replay.teamwork_big_five?.source==='action'?'action':'event';
const teamworkLabels={team_leadership:'リーダーシップ',mutual_performance_monitoring:'相互モニタリング',backup_behavior:'バックアップ',adaptability:'適応性',team_orientation:'チーム志向'};
const els={
  title:document.getElementById('title'),step:document.getElementById('stepLabel'),
  clock:document.getElementById('clock'),agents:document.getElementById('agents'),
  stepSummary:document.getElementById('stepSummary'),stepSummaryText:document.getElementById('stepSummaryText'),
  stepEvent:document.getElementById('stepEvent'),stepEventText:document.getElementById('stepEventText'),
  resultSummary:document.getElementById('resultSummary'),
  timeline:document.getElementById('timeline'),play:document.getElementById('play'),
  previous:document.getElementById('previous'),next:document.getElementById('next'),
  speed:document.getElementById('speed'),language:document.getElementById('language')
};
const present=value=>value!==null&&value!==undefined&&value!=='';
const number=value=>Number.isInteger(Number(value))?String(Number(value)):Number(value).toFixed(1);
const difference=(before,after)=>present(before)&&present(after)?Number(after)-Number(before):null;
const deltaText=value=>!present(value)||Number(value)===0?'—':`${Number(value)>0?'+':''}${number(value)}`;
const deltaClass=value=>Number(value)>0?'up':Number(value)<0?'down':'';
const shownText=(record,field)=>showOriginal?record[field]:(record[`${field}_ja`]||record[field]);
let index=0,timer=null,showOriginal=false;
const hasResultSummary=Boolean(replay.result_summary);
const pageCount=frames.length+(hasResultSummary?1:0);
const hasEventTranslations=frames.some(frame=>frame.event_ja||(frame.agents||[]).some(agent=>agent.event_ja));
els.title.textContent=replay.simulation?.title||'ASVO Simulation';
els.timeline.max=Math.max(0,pageCount-1);
els.language.hidden=!hasEventTranslations;

function updateLine(label,value,change=null){
  if(!present(value))return null;
  const row=document.createElement('div');row.className='update-line';
  const labelNode=document.createElement('span');labelNode.className='update-label';labelNode.textContent=label;
  const valueNode=document.createElement('span');valueNode.className='update-value';valueNode.textContent=value;
  row.append(labelNode,valueNode);
  if(present(change)){
    const changeNode=document.createElement('span');changeNode.className=`update-change ${deltaClass(change)}`;changeNode.textContent=deltaText(change);row.appendChild(changeNode);
  }
  return row;
}

function separator(parent){const line=document.createElement('div');line.className='update-separator';parent.appendChild(line)}

function renderRetry(parent,item,evidence){
  const attempts=Array.isArray(evidence?.attempts)?evidence.attempts:[];
  const rejected=attempts.some(attempt=>attempt.reasonableness===false);
  if(!rejected&&!present(item.reflection)&&!item.correction)return;
  const details=document.createElement('details');details.className='retry';
  const summary=document.createElement('summary');summary.textContent='Reflection・修正・再試行を見る';
  const body=document.createElement('div');body.className='retry-body';
  attempts.forEach((attempt,attemptIndex)=>{
    const block=document.createElement('div');block.className='attempt';
    const label=document.createElement('p');label.textContent=`${attemptIndex===0?'初回更新案':`再試行 ${attemptIndex}`}：${present(evidence.value_before_update)?`${number(evidence.value_before_update)} → `:''}${number(attempt.proposed_value)}`;
    block.appendChild(label);
    if(attempt.reasonableness===true){const verdict=document.createElement('p');verdict.className='validity';verdict.textContent='✓ 妥当';block.appendChild(verdict)}
    if(attempt.reasonableness===false){const verdict=document.createElement('p');verdict.className='validity invalid';verdict.textContent='✕ 不合理';block.appendChild(verdict)}
    if(present(attempt.reflection)){const reflection=document.createElement('p');reflection.className='reflection';reflection.textContent=`Reflection：${attempt.reflection}`;block.appendChild(reflection)}
    body.appendChild(block);
  });
  if(item.correction&&present(item.correction.from)&&present(item.correction.to)){
    const correction=document.createElement('p');correction.className='fact';correction.textContent=`修正：${number(item.correction.from)} → ${number(item.correction.to)}`;body.appendChild(correction);
  }
  details.append(summary,body);parent.appendChild(details);
}

function renderDesireDetail(parent,item){
  const evidence=item.update_evidence;
  const before=item.before;
  const finalValue=present(item.after)?item.after:item.value;
  const beforeLine=updateLine('更新前',present(before)?number(before):null);
  if(beforeLine)parent.appendChild(beforeLine);
  if(beforeLine&&(evidence||present(item.after)))separator(parent);

  if(evidence&&present(evidence.value_before_update)&&present(evidence.value_after_update)){
    const from=evidence.value_before_update,to=evidence.value_after_update;
    parent.appendChild(updateLine('行動・観測結果による更新',`${number(from)} → ${number(to)}`,difference(from,to)));separator(parent);
    if(item.reasonableness===true){const validity=document.createElement('p');validity.className='validity';validity.textContent='✓ 更新値の妥当性：妥当';parent.appendChild(validity);separator(parent)}
    if(item.reasonableness===false){const validity=document.createElement('p');validity.className='validity invalid';validity.textContent='✕ 更新値の妥当性：不合理';parent.appendChild(validity);separator(parent)}
  }

  const fluctuation=evidence?.fluctuation;
  if(fluctuation&&present(fluctuation.before)&&present(fluctuation.after)){
    const from=fluctuation.before,to=fluctuation.after;
    parent.appendChild(updateLine('時間経過による変動',`${number(from)} → ${number(to)}`,difference(from,to)));separator(parent);
  }

  const finalLine=updateLine('最終値',present(finalValue)?number(finalValue):null);
  if(finalLine)parent.appendChild(finalLine);
  if(present(item.target)){const target=document.createElement('p');target.className='target-note';target.textContent=`基準値・期待値：${number(item.target)}`;parent.appendChild(target)}
  if(present(item.reason)){const reason=document.createElement('p');reason.className='reason-note';reason.textContent=`記録された更新理由：${item.reason}`;parent.appendChild(reason)}
  renderRetry(parent,item,evidence);
}

function renderDesires(container,desires){
  Object.entries(desires||{}).forEach(([key,item])=>{
    const delta=item.delta;
    const details=document.createElement('details');details.className=`desire${Number(delta)!==0&&present(delta)?' changed':''}`;
    const summary=document.createElement('summary');
    const top=document.createElement('div');top.className='desire-top';
    const name=document.createElement('span');name.className='desire-name';name.textContent=labels[key]||key;
    const values=document.createElement('span');values.className='desire-values';
    const current=document.createElement('span');current.textContent=`現在 ${number(item.value)}`;values.appendChild(current);
    if(present(item.target)){const target=document.createElement('span');target.textContent=`/ 理想 ${number(item.target)}`;values.appendChild(target)}
    if(present(delta)&&Number(delta)!==0){const change=document.createElement('span');change.className=`desire-delta ${deltaClass(delta)}`;change.textContent=deltaText(delta);values.appendChild(change)}
    top.append(name,values);
    const bar=document.createElement('div');bar.className='desire-bar';bar.setAttribute('role','img');bar.setAttribute('aria-label',`${labels[key]||key}：現在${number(item.value)}${present(item.target)?`、理想${number(item.target)}`:''}`);
    const fill=document.createElement('div');fill.className='desire-fill';fill.style.width=`${Math.max(0,Math.min(100,Number(item.value)*10))}%`;bar.appendChild(fill);
    if(present(item.target)){const marker=document.createElement('span');marker.className='desire-target';marker.style.left=`${Math.max(0,Math.min(100,Number(item.target)*10))}%`;bar.appendChild(marker)}
    summary.append(top,bar);
    const body=document.createElement('div');body.className='update-detail';renderDesireDetail(body,item);
    details.append(summary,body);container.appendChild(details);
  });
}

function renderTeamworkBigFive(parent,result){
  if(!result||!Array.isArray(result.vector))return;
  const block=document.createElement('section');block.className='block';
  const head=document.createElement('div');head.className='teamwork-head';
  const title=document.createElement('h2');title.className='block-title';title.textContent=teamworkSource==='action'?'【Teamwork Big Five：Action判定】':'【Teamwork Big Five】';
  const vector=document.createElement('span');vector.className='teamwork-vector';vector.textContent=`[${result.vector.map(value=>Number(value)===1?1:0).join(', ')}]`;
  head.append(title,vector);block.appendChild(head);
  const chips=document.createElement('div');chips.className='teamwork-chips';
  teamworkOrder.forEach((key,categoryIndex)=>{
    const value=result.classifications?.[key]?.value??result.vector?.[categoryIndex]??0;
    const chip=document.createElement('span');chip.className=`teamwork-chip${Number(value)===1?' active':''}`;
    chip.textContent=teamworkLabels[key]||key;chips.appendChild(chip);
  });
  block.appendChild(chips);parent.appendChild(block);
}

function renderAgent(agent,agentIndex){
  const card=document.createElement('article');card.className='agent';card.style.animationDelay=`${agentIndex*.06}s`;
  card.innerHTML='<header class="identity"><div class="avatar"></div><div class="name"></div></header>';
  card.querySelector('.avatar').textContent=(agent.name||'?').slice(0,1).toUpperCase();
  card.querySelector('.name').textContent=agent.name||'';

  if(teamworkSource==='action'&&present(shownText(agent,'action'))){
    const block=document.createElement('section');block.className='block';block.innerHTML='<h2 class="block-title">【Agentが選んだ行動（Action attempt）】</h2><p class="observation-text"></p>';
    block.querySelector('.observation-text').textContent=shownText(agent,'action');card.appendChild(block);
  }

  const event=shownText(agent,'event');
  if(present(event)){
    const block=document.createElement('section');block.className='block';block.innerHTML='<h2 class="block-title">【実際に起きたこと（Event statement）】</h2><p class="observation-text"></p>';
    block.querySelector('.observation-text').textContent=event;card.appendChild(block);
  }
  renderTeamworkBigFive(card,agent.teamwork_big_five);
  if(Object.keys(agent.desires||{}).length){
    const block=document.createElement('section');block.className='block';block.innerHTML='<h2 class="block-title">【欲求状態】</h2><p class="desire-hint">各欲求を開くと、ASVO内部の更新過程を確認できます。</p><div class="desire-list"></div>';
    renderDesires(block.querySelector('.desire-list'),agent.desires);card.appendChild(block);
  }
  return card;
}

function resultSection(title){
  const section=document.createElement('section');section.className='result-section';
  const heading=document.createElement('h2');heading.textContent=title;section.appendChild(heading);
  return section;
}

function teamworkSummaryData(){
  const names=[];const counts={};const teamTotals=teamworkOrder.map(()=>0);let classifiedEvents=0;
  frames.forEach(frame=>(frame.agents||[]).forEach(agent=>{
    const name=agent.name||'';
    if(name&&!counts[name]){names.push(name);counts[name]=teamworkOrder.map(()=>0)}
    const vector=agent.teamwork_big_five?.vector;
    if(!name||!Array.isArray(vector)||vector.length!==teamworkOrder.length)return;
    classifiedEvents+=1;
    vector.forEach((value,categoryIndex)=>{
      if(Number(value)!==1)return;
      counts[name][categoryIndex]+=1;teamTotals[categoryIndex]+=1;
    });
  }));
  return {names,counts,teamTotals,classifiedEvents,totalOccurrences:teamTotals.reduce((sum,value)=>sum+value,0)};
}

function teamworkOccurrenceRates(counts,classifiedEvents){
  if(classifiedEvents===0)return counts.map(()=>0);
  return counts.map(value=>Number(value)*100/classifiedEvents);
}

function renderTeamworkSummary(parent){
  const data=teamworkSummaryData();const rates=teamworkOccurrenceRates(data.teamTotals,data.classifiedEvents);
  const tableWrap=document.createElement('div');tableWrap.className='teamwork-table-wrap';
  const table=document.createElement('table');table.className='teamwork-table';
  const thead=document.createElement('thead');const headRow=document.createElement('tr');
  ['Agent',...teamworkOrder.map(key=>teamworkLabels[key]||key)].forEach(label=>{const th=document.createElement('th');th.scope='col';th.textContent=label;headRow.appendChild(th)});thead.appendChild(headRow);table.appendChild(thead);
  const tbody=document.createElement('tbody');data.names.forEach(name=>{const row=document.createElement('tr');const heading=document.createElement('th');heading.scope='row';heading.textContent=name;row.appendChild(heading);data.counts[name].forEach(count=>{const cell=document.createElement('td');cell.textContent=String(count);row.appendChild(cell)});tbody.appendChild(row)});table.appendChild(tbody);
  const tfoot=document.createElement('tfoot');const totalRow=document.createElement('tr');const totalLabel=document.createElement('td');totalLabel.textContent='チーム合計';totalRow.appendChild(totalLabel);
  data.teamTotals.forEach((count,i)=>{const cell=document.createElement('td');cell.className='teamwork-total-cell';cell.textContent=String(count);const ratio=document.createElement('span');ratio.textContent=`${rates[i].toFixed(1)}%`;cell.appendChild(ratio);totalRow.appendChild(cell)});tfoot.appendChild(totalRow);table.appendChild(tfoot);
  tableWrap.appendChild(table);const note=document.createElement('p');note.className='teamwork-note';const sourceLabel=teamworkSource==='action'?'Action attempts':'Event statements';note.textContent=`分類対象 ${data.classifiedEvents} ${sourceLabel}。各割合は、その要素の該当件数を分類対象行動数で割った出現率です。1つの判定対象が複数要素に該当できるため、5要素の割合の合計は100%にならない場合があります。`;tableWrap.appendChild(note);parent.appendChild(tableWrap);
}

function renderResultSummary(summary){
  els.resultSummary.innerHTML='';
  const title=document.createElement('h1');title.className='result-title';title.textContent='【シミュレーション結果サマリー】';
  els.resultSummary.appendChild(title);

  const collaboration=resultSection('① 全体の協働');
  if(present(summary.collaboration_ja)){
    const text=document.createElement('p');text.textContent=summary.collaboration_ja;collaboration.appendChild(text);
  }
  els.resultSummary.appendChild(collaboration);

  const behavior=resultSection('② 各人の振る舞い');
  const behaviorGrid=document.createElement('div');behaviorGrid.className='behavior-grid';
  (summary.agents||[]).forEach(agent=>{
    if(!present(agent.behavior_ja))return;
    const card=document.createElement('article');card.className='result-agent';
    const name=document.createElement('h3');name.textContent=agent.name||'';
    const text=document.createElement('p');text.textContent=agent.behavior_ja;
    card.append(name,text);behaviorGrid.appendChild(card);
  });
  behavior.appendChild(behaviorGrid);els.resultSummary.appendChild(behavior);

  const teamwork=resultSection('③ チームワークBig Five');
  renderTeamworkSummary(teamwork);els.resultSummary.appendChild(teamwork);

  const shortage=resultSection('④ 各人で不足している欲求');
  const shortageGrid=document.createElement('div');shortageGrid.className='shortage-grid';
  (summary.agents||[]).forEach(agent=>{
    const card=document.createElement('article');card.className='result-agent';
    const name=document.createElement('h3');name.textContent=agent.name||'';card.appendChild(name);
    const items=agent.insufficient_desires||[];
    if(items.length){
      const list=document.createElement('ul');list.className='shortage-list';
      items.forEach(item=>{
        const row=document.createElement('li');row.className='shortage-item';
        const label=document.createElement('span');label.textContent=item.label_ja||labels[item.key]||item.key;
        const values=document.createElement('span');values.className='shortage-values';values.textContent=`現在 ${number(item.current)} / 理想 ${number(item.expected)}（不足 ${number(item.gap)}）`;
        row.append(label,values);list.appendChild(row);
      });
      card.appendChild(list);
    }else{
      const none=document.createElement('p');none.className='no-shortage';none.textContent='不足している欲求はありません。';card.appendChild(none);
    }
    shortageGrid.appendChild(card);
  });
  shortage.appendChild(shortageGrid);els.resultSummary.appendChild(shortage);
}

function render(){
  if(hasResultSummary&&index===frames.length){
    els.step.textContent='RESULT SUMMARY';els.clock.textContent='';els.clock.hidden=true;
    els.stepSummary.hidden=true;els.stepEvent.hidden=true;els.agents.hidden=true;els.resultSummary.hidden=false;
    els.language.hidden=true;renderResultSummary(replay.result_summary);
    els.timeline.value=index;els.previous.disabled=index===0;els.next.disabled=true;
    return;
  }
  const frame=frames[index];if(!frame)return;
  els.clock.hidden=false;els.agents.hidden=false;els.resultSummary.hidden=true;
  els.language.hidden=!hasEventTranslations;
  els.step.textContent=`STEP ${frame.step} / ${frames.at(-1).step}`;
  els.clock.textContent=frame.time||'';
  const stepZeroSummary=Number(frame.step)===0&&present(frame.summary_ja)?frame.summary_ja:'';
  els.stepSummary.hidden=!present(stepZeroSummary);
  els.stepSummaryText.textContent=els.stepSummary.hidden?'':stepZeroSummary;
  const hasAgentEvents=(frame.agents||[]).some(agent=>present(agent.event));
  const stepEvent=shownText(frame,'event');
  els.stepEvent.hidden=hasAgentEvents||!present(stepEvent);
  els.stepEventText.textContent=els.stepEvent.hidden?'':stepEvent;
  els.timeline.value=index;els.previous.disabled=index===0;els.next.disabled=index===pageCount-1;
  els.agents.innerHTML='';(frame.agents||[]).forEach((agent,agentIndex)=>els.agents.appendChild(renderAgent(agent,agentIndex)));
}
function move(amount){index=Math.max(0,Math.min(pageCount-1,index+amount));render();if(index===pageCount-1&&timer)stop()}
function stop(){clearInterval(timer);timer=null;els.play.textContent='▶ 再生'}
function play(){if(timer){stop();return}if(index===pageCount-1)index=0;els.play.textContent='Ⅱ 停止';render();timer=setInterval(()=>move(1),Number(els.speed.value))}
els.previous.onclick=()=>move(-1);els.next.onclick=()=>move(1);els.play.onclick=play;
els.language.onclick=()=>{showOriginal=!showOriginal;els.language.textContent=showOriginal?'日本語':'EN原文';render()};
els.timeline.oninput=event=>{index=Number(event.target.value);render()};els.speed.onchange=()=>{if(timer){stop();play()}};
document.addEventListener('keydown',event=>{if(event.key==='ArrowLeft')move(-1);if(event.key==='ArrowRight')move(1);if(event.code==='Space'&&!/INPUT|SELECT/.test(event.target.tagName)){event.preventDefault();play()}});
render();
</script>
</body>
</html>'''
