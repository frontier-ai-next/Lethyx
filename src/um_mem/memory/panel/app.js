const $ = id => document.getElementById(id);
const state = {snapshot:null, records:[], browseRecords:[], block:'', cursor:'', scope:{scope:'workspace',scopeId:''}, view:'memory', mode:'graph',
  selected:'', detail:null, draft:null, query:'', events:[], positions:{},
  scopeLimited:false,
  transform:{x:0,y:0,k:1}, source:null, refreshing:false, pending:false, detailRequest:0,
  searchRequest:0, toastTimer:0, selectedTopic:'', fitNeeded:true, stopped:false};
const statusNames = {candidate:'Не проверено',accepted:'Подтверждено',active:'Действует',
  retracted:'Отозвано',superseded:'Заменено',deprecated:'Устарело',expired:'Срок истёк',quarantined:'На проверке'};
const kindNames = {observation:'Наблюдение',project_fact:'Факт о проекте',procedure:'Процедура',
  rule:'Правило',lesson:'Вывод',fact:'Факт',memory:'Запись'};
const operationNames = {remember:'Добавлена запись',amend:'Изменена запись',forget:'Запись отозвана',
  create_block:'Создана тема',amend_block:'Изменена тема',move_block:'Перемещена тема',
  archive_block:'Тема архивирована',delete_block:'Тема удалена',set_active_mask:'Выбран рабочий контекст',
  amend_active_mask:'Изменён рабочий контекст',transition_active_mask:'Изменено состояние контекста',
  recall:'Поиск в памяти',build_context:'Сборка контекста',resolve_active_mask:'Разрешение контекста',
  indexing:'Индексы обновлены'};
function element(tag, className='', text='') {
  const node=document.createElement(tag); if(className) node.className=className;
  node.textContent=text; return node;
}
function button(text, className, action) {
  const node=element('button',className,text); node.type='button'; node.addEventListener('click',action); return node;
}
function badge(status) {return element('span',`badge ${status}`,statusNames[status]||status||'Без статуса');}
function date(value) {const d=new Date(value); return isNaN(d)?'Без даты':d.toLocaleString('ru',{day:'numeric',month:'short',hour:'2-digit',minute:'2-digit'});}
function flatten(nodes, out=[]) {for(const n of nodes||[]){out.push(n);flatten(n.children,out);}return out;}
function topic(id) {return flatten(state.snapshot?.tree).find(n=>n.block_id===id);}
function scopeLabel(scope=state.scope) {
  if(scope.scope!=='project')return 'Общая память';
  const id=scope.scopeId||'';
  return `Проект ${id.length>24?id.slice(0,10)+'…'+id.slice(-8):id}`;
}
function scopeQuery() {
  const params=new URLSearchParams({scope:state.scope.scope});
  if(state.scope.scope==='project')params.set('scope_id',state.scope.scopeId);
  return params.toString();
}
function recordsUrl(block,cursor) {
  const params=new URLSearchParams({scope:state.scope.scope,block:block||'',cursor:cursor||''});
  if(state.scope.scope==='project')params.set('scope_id',state.scope.scopeId);
  return `api/records?${params}`;
}
function renderScopePicker() {
  if(!state.snapshot)return;
  const select=$('record-scope'),selected=state.scope.scope==='project'?`project:${state.scope.scopeId}`:'workspace';
  const projects=new Map();
  for(const session of state.snapshot.sessions||[])if(session.project_id){
    const entries=projects.get(session.project_id)||[];entries.push(session.session_id||'session');projects.set(session.project_id,entries);
  }
  if(state.scope.scope==='project'&&!projects.has(state.scope.scopeId))projects.set(state.scope.scopeId,[]);
  select.replaceChildren(new Option('Общая память','workspace'));
  for(const [id,sessions] of [...projects.entries()].sort(([a],[b])=>a.localeCompare(b))){
    const provider=sessions.map(id=>id.split(':',1)[0]).filter((v,i,a)=>a.indexOf(v)===i).join(', ');
    const option=new Option(`${scopeLabel({scope:'project',scopeId:id})}${provider?' · '+provider:''}`,`project:${id}`);
    option.title=`Точный ID: ${id}`;select.append(option);
  }
  select.append(new Option('Другой проект…','manual'));
  select.value=selected;
  $('project-scope-form').hidden=select.value!=='manual';
}
async function changeScope(scope,scopeId='') {
  if(!allowLeave()) {renderScopePicker();return;}
  state.scope={scope,scopeId};state.block='';state.records=[];state.browseRecords=[];state.cursor='';state.query='';
  $('query').value='';$('search-info').hidden=true;state.detailRequest++;state.draft=null;state.selected='';state.selectedTopic='';$('detail').hidden=true;
  restoreLayout();await refresh();
}
function toast(text) {clearTimeout(state.toastTimer);$('toast').textContent=text;$('toast').hidden=false;state.toastTimer=setTimeout(()=>$('toast').hidden=true,5500);}
function banner(text='') {$('banner').textContent=text;$('banner').hidden=!text;}
async function api(path,body) {
  const response=await fetch(`/panel/${path}`,{credentials:'same-origin',headers:body?{'Content-Type':'application/json'}:{},
    method:body?'POST':'GET',body:body?JSON.stringify(body):undefined,cache:'no-store'});
  const data=await response.json();
  if(!response.ok||data.ok===false||data.error) {
    const error=new Error(data.error||'Не удалось выполнить запрос');error.data=data;error.status=response.status;throw error;
  }
  return data;
}
function connected(value) {
  $('connection').textContent=value?'Синхронизировано':'Переподключение…';
  $('connection-dot').classList.toggle('online',value);
}
function renderSummary() {
  const s=state.snapshot, nodes=flatten(s.tree).filter(n=>!n.meta);
  renderScopePicker();
  $('workspace').textContent=s.workspace;$('profile').textContent=`Профиль ${s.profile} · общая память`;
  $('backend').textContent=`${s.backend} · локальный сервис`;
  $('stat-topics').textContent=nodes.filter(n=>n.parent_id).length;
  $('stat-records').textContent=nodes.reduce((sum,n)=>sum+Number(n.canonical_record_count||0),0);
  $('stat-sessions').textContent=s.sessions.filter(n=>!n.session_id.startsWith('panel')).length;
  $('stat-health').textContent=s.health.ok?'В порядке':'Нужна проверка';
  const queue=s.health.background_queue;
  $('stat-health-note').textContent=queue?.emergency_read_only?'Временно только чтение':s.health.ok?'Память доступна':'Сервис сообщает об ошибке';
  $('last-sync').textContent=`Синхронизация ${new Date().toLocaleTimeString('ru')}`;
  const list=$('topics');list.replaceChildren();
  for(const n of nodes.filter(n=>n.parent_id)) {
    const b=button('',`topic ${state.block===n.block_id?'selected':''}`,()=>selectBlock(n.block_id));
    b.style.setProperty('--node-color',colorFor(n.block_id));
    b.style.paddingLeft=`${9+Math.max(0,(n.path||[]).length-2)*11}px`;
    b.append(element('span','',n.name),element('small','',String(blockRecordCount(n))));list.append(b);
  }
  $('new-record').disabled=Boolean(queue?.emergency_read_only);
  renderLocation();
}
function renderRecords() {
  if(!state.query)state.browseRecords=[...state.records];
  $('collection-title').textContent=state.query?`Результаты поиска`:state.block?(topic(state.block)?.name||'Выбранная тема'):`Записи · ${scopeLabel()}`;
  $('collection-note').textContent=state.scopeLimited
    ?`Показаны последние 64 записи · ${scopeLabel()}. Выберите тему слева для более узкого списка; поиск охватывает только показанные записи.`
    :state.query?`${state.records.length} результатов`:state.cursor?`${state.records.length} записей загружено · ${scopeLabel()}`:`${state.records.length} записей · ${scopeLabel()}`;
  const list=$('list');list.replaceChildren();
  if(!state.records.length){const e=element('div','empty');e.append(element('strong','',state.query?'Подходящих записей нет':'Здесь пока нет записей'),
    element('span','',state.query?'Память не нашла достаточно подходящих сведений. Уточните вопрос или выберите другую тему.':'Выберите тему или добавьте первую запись.'));list.append(e);}
  for(const r of state.records) {
    const card=button('',`record-card ${r.record_id===state.selected?'selected':''}`,()=>openRecord(r.record_id));
    const top=element('div','card-top'),kind=element('div','record-kind');
    kind.append(element('span','record-icon',r.kind==='procedure'?'↗':'◇'),element('span','',kindNames[r.kind]||r.kind||'Запись'));
    top.append(kind,badge(r.status));
    const bottom=element('div','card-bottom');bottom.append(element('span','',topic(r.logical_block_id)?.name||'Общая память'),
      element('span','',r.score!==undefined?`${r.score_mode==='semantic_cosine'?'Сходство':'Оценка'} ${Number(r.score).toFixed(3)}`:date(r.recorded_at)));
    card.append(top,element('h3','',r.title||'Без названия'),element('p','',r.preview||''),bottom);list.append(card);
  }
  $('load-more').hidden=!state.cursor||(Boolean(state.query)&&state.scope.scope!=='project');renderGraph();
}
function renderEvents() {
  const events=[...state.events].reverse();
  for(const [target,filter] of [['activity',e=>e.kind!=='retrieval'],['context-events',e=>e.kind==='retrieval']]) {
    const container=$(target);container.replaceChildren();const selected=events.filter(filter);
    if(!selected.length) container.append(element('div','empty','Событий пока нет. Новые действия подключённых клиентов появятся здесь.'));
    for(const e of selected) {
      const row=element('div','event'),body=element('div');
      body.append(element('strong','',`${operationNames[e.operation]||e.operation}${e.ok?'':' · не выполнено'}`));
      body.append(element('p','',`${e.session_id||'Сервис'}${e.task_id?' / '+e.task_id:''}`));
      if(e.kind==='retrieval') body.append(element('p','',`${e.count} записей найдено${e.rendered?' · контекст сформирован':''}.`));
      if(e.commit_revision) body.append(element('p','',`Сохранено: ${e.commit_revision} · проекция: ${e.projection_revision||'ожидает подтверждения'}`));
      if(e.reason_code) body.append(element('p','',e.reason_code));
      for(const id of e.record_ids||[])body.append(button(`${id.slice(0,24)} ↗`,'',()=>openRecord(id)));
      row.append(element('span','event-icon',e.ok?'↗':'!'),body,element('time','',date(e.at)));container.append(row);
    }
  }
  const sessions=$('sessions');sessions.replaceChildren();
  for(const s of state.snapshot?.sessions||[]) {
    if(s.session_id.startsWith('panel')) continue;
    sessions.append(element('span','session-chip',`${s.session_id}${s.task_id?' / '+s.task_id:''}`));
  }
}
function mergeEvents(events, cursor) {
  const epoch=cursor.split(':')[0], byId=new Map();
  for(const e of [...events,...state.events])if(e.id.split(':')[0]===epoch)byId.set(e.id,e);
  state.events=[...byId.values()].sort((a,b)=>Number(a.id.split(':')[1])-Number(b.id.split(':')[1])).slice(-256);
}
async function selectBlock(id) {
  if(!allowLeave())return;
  storeLayout();state.detailRequest++;state.draft=null;state.selected='';state.selectedTopic='';$('detail').hidden=true;
  state.records=[];state.cursor='';
  state.block=id;state.query='';$('query').value='';$('search-info').hidden=true;
  restoreLayout();
  state.searchRequest++;setView('memory');await refresh();
}
async function refresh() {
  if(state.refreshing){state.pending=true;return;}
  state.refreshing=true;
  try {
    state.scopeLimited=false;
    const snapshot=await api(`api/snapshot?${scopeQuery()}`);state.snapshot=snapshot;mergeEvents(snapshot.events,snapshot.cursor);
    renderSummary();renderEvents();
    if(state.block) {
      const selected=state.block, count=Math.max(64,state.records.length);let cursor='',records=[];
      do {const page=await api(recordsUrl(selected,cursor));
        state.scopeLimited ||=Boolean(page.scope_limited);
        records.push(...page.records);cursor=page.next_cursor;
      }while(cursor&&records.length<count&&selected===state.block);
      if(state.block===selected){state.browseRecords=records;if(!state.query){state.records=records;state.cursor=cursor;}}
    } else if(state.scope.scope==='project') {
      const page=await api(recordsUrl('', ''));
      state.scopeLimited=Boolean(page.scope_limited);
      state.browseRecords=page.records;if(!state.query){state.records=page.records;state.cursor=page.next_cursor;}
    } else {state.browseRecords=snapshot.records;if(!state.query){state.records=snapshot.records;state.cursor='';}}
    if(state.query)await search(state.query);
    renderRecords();banner();
    if(state.selected) await checkSelected();
    else if(state.selectedTopic)renderBlockDetail(state.selectedTopic);
  } catch(error){banner(error.message);}finally {
    state.refreshing=false;if(state.pending){state.pending=false;void refresh();}
  }
}
async function checkSelected() {
  const id=state.selected;
  try {
    const data=await api(`api/record?id=${encodeURIComponent(id)}`);if(state.selected!==id)return;
    state.detail=data;
    if(state.draft) {if(!data.is_current&&!state.draft.isNew) showConflict();}
    else renderDetail(data);
  }catch(error){if(state.selected===id)banner(error.message);}
}
async function search(query) {
  const request=++state.searchRequest;state.query=query.trim();if(state.scope.scope!=='project')state.cursor='';
  if(!state.query){$('search-info').hidden=true;return refresh();}
  $('search-info').hidden=false;$('search-info').textContent='Ищем по смыслу…';
  if(state.scope.scope==='project'){
    const needle=state.query.toLocaleLowerCase();
    state.records=state.browseRecords.filter(r=>(r.title+' '+(r.preview||'')).toLocaleLowerCase().includes(needle));
    $('search-info').textContent=state.scopeLimited
      ?`Поиск только по показанным последним 64 записям: ${scopeLabel()}. Результаты неполные; выберите тему, чтобы сузить область.`
      :`Поиск по уже загруженным записям: ${scopeLabel()}. Загрузите следующую страницу, чтобы расширить выборку.`;
    setMode('list');renderRecords();return;
  }
  try {
    const arguments_={query:state.query,include_candidates:true};if(state.block)arguments_.logical_block_ids=[state.block];
    const data=await api('api/call',{tool:'recall',arguments:arguments_});
    if(request!==state.searchRequest)return;
    state.records=(data.hits||data.items||[]).map(h=>{
      const m=h.metadata||{};return {record_id:h.record_id||m.record_id||m.block_id||h.source,
        title:h.title||m.title||'Найденная запись',preview:h.text||h.concise_text||h.content,
        status:h.status||m.status,kind:h.kind||m.kind,logical_block_id:h.logical_block_id||m.logical_block_id,
        recorded_at:h.recorded_at||m.recorded_at,score:h.score,score_mode:h.score_mode||m.native_score_mode};
    }).filter(r=>r.record_id);
    // The operator can inspect visible candidates even when agent recall
    // excludes them. This local title/preview match grants no agent authority.
    const found=new Set(state.records.map(r=>r.record_id)),needle=state.query.toLocaleLowerCase();
    const inspected=state.browseRecords.filter(r=>!found.has(r.record_id)&&(r.title+' '+(r.preview||'')).toLocaleLowerCase().includes(needle));
    state.records.push(...inspected);
    $('search-info').textContent=`${data.fallback==='semantic'?'Семантический поиск':data.fallback==='unicode_ngram'?'Поиск по тексту':'Поиск в выбранном хранилище'}${data.semantic_model?' · '+data.semantic_model:''}. Оценка сходства не является вероятностью истинности. Непроверенные записи остаются непроверенными.`;
    if(inspected.length)$('search-info').textContent+=' Совпадения в загруженных названиях и описаниях также включены для просмотра.';
    setMode('list');renderRecords();
  }catch(error){$('search-info').textContent=error.message;}
}
function setView(view) {
  state.view=view;
  for(const name of ['memory','activity','context'])$(`${name}-view`).hidden=name!==view;
  document.querySelectorAll('[data-view]').forEach(b=>b.classList.toggle('active',b.dataset.view===view));
  $('page-title').textContent={memory:topic(state.block)?.name||'Атлас памяти',activity:'История изменений',context:'Сессии и контекст'}[view];
  $('page-subtitle').textContent={memory:'Одно пространство для знаний. Изменения видны всем подключённым агентам.',
    activity:'Действия подключённых клиентов. История версий хранится внутри каждой записи.',
    context:'Сессии, переданные подключёнными агентами, и полученные ими сведения.'}[view];
  $('new-record').hidden=view!=='memory';
  if(view==='memory')renderLocation();
}
function allowLeave() {if(state.draft?.saving){toast('Дождитесь результата сохранения.');return false;}return !state.draft||!state.draft.dirty||window.confirm('Закрыть редактор и отказаться от несохранённых изменений?');}
async function openRecord(id) {
  if(!allowLeave())return;
  const request=++state.detailRequest;state.draft=null;state.selectedTopic='';state.selected=id;$('detail').hidden=false;$('detail-label').textContent='ЗАПИСЬ ПАМЯТИ';
  $('detail-content').replaceChildren(element('p','metadata','Загружаем запись…'));
  try {const data=await api(`api/record?id=${encodeURIComponent(id)}`);
    if(request!==state.detailRequest)return;state.detail=data;renderDetail(data);renderRecords();
  }catch(error){if(request===state.detailRequest)$('detail-content').replaceChildren(element('p','conflict',error.message));}
}
function renderDetail(data) {
  if(state.draft)return;
  const r=data.record, box=$('detail-content');box.replaceChildren();
  box.append(badge(r.status),element('h2','',r.title||'Без названия'),element('p','metadata',
    `${date(r.recorded_at)} · ${kindNames[r.kind]||r.kind}`));
  if(!data.is_current)box.append(element('p','conflict','Это сохранённая версия истории. Она не является текущей записью.'));
  if(r.status==='candidate')box.append(element('p','metadata','Кандидат в память: содержимое ещё не прошло проверку и не становится правилом для агента.'));
  box.append(element('div','record-body',r.content));
  if(data.is_current)box.append(button(r.status==='candidate'?'Редактировать запись':'Предложить уточнение','primary',()=>editRecord(r)));
  box.append(element('h3','','Версии записи'));
  for(const version of [...data.history].reverse()) {
    const b=button(`${statusNames[version.status]||version.status} ${version.is_current?'· текущая':''}`,'history-row',()=>openRecord(version.record_id));
    b.append(element('small','',date(version.recorded_at)));box.append(b);
  }
  if(data.history_truncated)box.append(element('p','metadata','Показаны последние 128 версий.'));
  const details=element('details');details.append(element('summary','','Источники и технические сведения'),element('pre','',JSON.stringify({
    record_id:r.block_id,revision_id:r.revision_id,scope:r.scope,authority:r.authority,evidence_refs:r.evidence_refs,
    origin:r.origin,valid_from:r.valid_from,valid_to:r.valid_to,expires_at:r.expires_at,metadata:r.metadata},null,2)));box.append(details);
}
function editRecord(record=null) {
  if(!allowLeave())return;
  state.selectedTopic='';$('detail-label').textContent='РЕДАКТОР';
  if(!record){state.selected='';state.detail=null;state.detailRequest++;}
  state.draft={isNew:!record,proposal:record&&record.status!=='candidate',dirty:false,memory_id:record?.block_id,expected_revision:record?.revision_id,
    operation_id:`panel:${crypto.randomUUID()}`,conflict:false};
  const box=$('detail-content');box.replaceChildren();$('detail').hidden=false;
  box.append(element('h2','',record?'Изменить запись':'Новая запись'));
  if(record&&record.status!=='candidate')box.append(element('p','metadata','Изменение подтверждённой записи создаёт предложение на проверку. Оно не получает прежнюю степень доверия автоматически.'));
  const conflict=element('div','conflict');conflict.id='edit-conflict';conflict.hidden=true;box.append(conflict);
  const form=element('form','editor');
  function field(label,id,tag,value) {
    const l=element('label','',label);l.htmlFor=id;const input=element(tag);input.id=id;input.value=value||'';input.required=true;
    input.addEventListener('input',()=>{state.draft.dirty=true;});form.append(l,input);return input;
  }
  if(!record) {
    const select=field('Тема','edit-block','select','');
    for(const t of flatten(state.snapshot?.tree).filter(n=>n.writable&&!n.meta)){
      const o=element('option','',t.name);o.value=t.block_id;select.append(o);
    }
    if(state.block)select.value=state.block;
    if(!select.options.length){box.append(element('p','conflict','Сначала создайте тему через агента UM-MEM. В пространстве пока нет доступной для записи темы.'));return;}
  }
  field('Название','edit-title','input',record?.title).maxLength=500;
  field('Содержимое','edit-content','textarea',record?.content).maxLength=100000;
  if(record)field('Почему меняем','edit-reason','input','Уточнение пользователем в панели');
  const actions=element('div','actions'),save=element('button','primary','Сохранить');save.type='submit';save.id='save-record';
  actions.append(save,button('Отмена','secondary',()=>{if(allowLeave()){state.draft=null;if(state.detail)renderDetail(state.detail);else $('detail').hidden=true;}}));form.append(actions);
  form.addEventListener('submit',async event=>{
    event.preventDefault();const draft=state.draft;if(state.stopped||draft.conflict||draft.saving)return;
    const arguments_={title:$('edit-title').value,content:$('edit-content').value,operation_id:draft.operation_id};
    if(draft.isNew)arguments_.requested_block_id=$('edit-block').value;
    else Object.assign(arguments_,{memory_id:draft.memory_id,expected_revision:draft.expected_revision,reason:$('edit-reason').value});
    draft.saving=true;for(const control of form.elements)control.disabled=true;
    try {
      const result=await api('api/call',{tool:draft.isNew?'remember':draft.proposal?'correct':'amend',arguments:arguments_});
      state.draft=null;state.selected='';$('detail').hidden=true;
      toast(result.pending_host_approval?'Предложение сохранено для проверки':'Изменение сохранено в общей памяти');await refresh();
      const id=result.memory_id||result.block_id;if(id)await openRecord(id);
    }catch(error){if(error.status===409||error.data?.reason_code==='canonical_revision_conflict')showConflict();else toast(error.message);}
    finally{draft.saving=false;for(const control of form.elements)control.disabled=false;save.disabled=state.stopped||Boolean(state.draft?.conflict);}
  });box.append(form);$('edit-title').focus();
}
function showConflict() {
  if(!state.draft)return;state.draft.conflict=true;
  const box=$('edit-conflict');if(!box)return;box.hidden=false;box.replaceChildren(element('span','','Запись уже изменена другим клиентом. Ваш текст остаётся в редакторе. Сравните изменения, прежде чем сохранять.'));
  const head=state.detail?.history.find(v=>v.is_current);
  if(head)box.append(button('Сравнить с текущей версией','',async()=>{
    const draft=state.draft;try{const latest=await api(`api/record?id=${encodeURIComponent(head.record_id)}`);
      if(state.draft!==draft)return;
      const comparison=element('details');comparison.open=true;comparison.append(element('summary','','Текущий текст в памяти'),element('pre','',latest.record.content));box.append(comparison);
      if(!latest.is_current){toast('Версия снова изменилась. Обновляем сведения.');await checkSelected();return;}
      box.append(button('Продолжить с этой версией, сохранив мой черновик','',()=>{
        if(state.draft!==draft||draft.saving)return;
        draft.memory_id=latest.record.block_id;draft.expected_revision=latest.record.revision_id;
        draft.proposal=latest.record.status!=='candidate';draft.operation_id=`panel:${crypto.randomUUID()}`;draft.conflict=false;
        state.selected=latest.record.block_id;state.detail=latest;box.hidden=true;$('save-record').disabled=false;
      }));
    }catch(error){toast(error.message);}
  }));
  $('save-record').disabled=true;
}

// Coordinates are a view preference; topology always comes from the canonical catalog.
const ns='http://www.w3.org/2000/svg';
const palette=['#b9a3e6','#83c9b0','#d0a47b','#85add8','#c58ba8','#a6b77e'];
function blockRecordCount(block){return Number(block.canonical_record_count||0)+(block.children||[]).reduce((total,child)=>total+blockRecordCount(child),0);}
function colorFor(id) {let hash=0;for(const c of id||'')hash=(hash*31+c.charCodeAt(0))>>>0;return palette[hash%palette.length];}
function svg(tag,attributes={},text=''){const node=document.createElementNS(ns,tag);for(const [k,v]of Object.entries(attributes))node.setAttribute(k,v);node.textContent=text;return node;}
function layoutKey(){return 'um-mem-atlas:4:'+state.snapshot?.workspace+':'+state.snapshot?.profile+':'+(state.block||'overview');}
function storeLayout(){if(!state.snapshot)return;try{localStorage.setItem(layoutKey(),JSON.stringify({positions:state.positions,transform:state.transform}));}catch{/* Presentation preferences are optional. */}}
function restoreLayout(){state.positions={};state.transform={x:0,y:0,k:1};state.fitNeeded=true;try{const saved=JSON.parse(localStorage.getItem(layoutKey())||'null');if(saved){state.positions=saved.positions||{};state.transform=saved.transform||state.transform;state.fitNeeded=false;}}catch{/* Ignore unavailable or invalid local preferences. */}}
function renderLocation(){
  if(state.view!=='memory')return;
  const current=topic(state.block), path=[],seen=new Set();let node=current;
  while(node&&!seen.has(node.block_id)){seen.add(node.block_id);path.unshift(node);node=topic(node.parent_id);}
  const crumbs=$('breadcrumbs');crumbs.replaceChildren(button('Вся память','',()=>selectBlock('')));
  for(const part of path){crumbs.append(element('span','','/'),button(part.name,'',()=>selectBlock(part.block_id)));}
  $('page-title').textContent=current?.name||'Атлас памяти';
  $('page-subtitle').textContent=current?(current.description||current.synopsis||'Блоки и записи этого пространства.'):'Откройте блок, чтобы увидеть, что внутри.';
}
function openBlock(id){
  if(!allowLeave())return;state.detailRequest++;state.draft=null;state.detail=null;state.selected='';state.selectedTopic=id;
  renderBlockDetail(id);for(const node of $('graph-scene').querySelectorAll('.graph-node'))node.classList.toggle('selected',node.dataset.id===id);updateGraphGeometry();
}
function renderBlockDetail(id){
  const b=topic(id);if(!b){state.selectedTopic='';$('detail').hidden=true;return;}
  $('detail').hidden=false;$('detail-label').textContent='БЛОК ПАМЯТИ';
  const box=$('detail-content');box.replaceChildren();
  const symbol=element('div','block-symbol',b.index_only?'✳':'◇');symbol.style.setProperty('--node-color',colorFor(id));box.append(symbol,element('h2','',b.name));
  const tags=element('div','tags');for(const tag of b.tags||[])tags.append(element('span','',tag));box.append(tags);
  box.append(element('p','record-body',b.description||b.synopsis||'Описание ещё не добавлено.'));
  const counts=element('div','block-stats');
  for(const [value,label]of [[b.children?.length||0,'вложенных блоков'],[blockRecordCount(b),'записей внутри']]){const item=element('div');item.append(element('strong','',String(value)),element('span','',label));counts.append(item);}box.append(counts);
  const enter=button('Открыть пространство','primary block-enter',()=>selectBlock(id));enter.append(element('span','','↗'));box.append(enter);
  if(b.children?.length){box.append(element('h3','','Внутри блока'));for(const child of b.children){const item=button('◇  '+child.name,'block-child',()=>selectBlock(child.block_id));item.append(element('small','','↗'));box.append(item);}}
  const related=(state.snapshot.edges||[]).filter(e=>e.src_id===id||e.dst_id===id);
  if(related.length){box.append(element('h3','','Связанные блоки'));for(const edge of related){const target=topic(edge.src_id===id?edge.dst_id:edge.src_id);if(target)box.append(button(target.name+' ↗','block-child',()=>openBlock(target.block_id)));}}
  if(b.parent_id)box.append(button('← '+(topic(b.parent_id)?.name||'Родительский блок'),'block-child',()=>selectBlock(b.parent_id)));
  box.append(element('p','metadata',b.writable?'Здесь можно хранить записи.':'Этот блок объединяет вложенные пространства.'));
}
let graphNodes=[],graphLinks=[];
function counted(value,forms){const n=value%100;return value+' '+forms[n>=11&&n<=14?2:value%10===1?0:value%10>=2&&value%10<=4?1:2];}
function graphModel(){
  if(state.scope.scope==='project'){
    const records=state.snapshot?.records||[],nodes=[],links=[],visible=new Set(records.map(r=>r.record_id));
    records.forEach((r,index)=>{
      const angle=index/Math.max(records.length,1)*Math.PI*2-Math.PI/2;
      const radius=records.length>1?Math.min(310,105+records.length*3):0;
      nodes.push({id:r.record_id,title:r.title||'Без названия',type:'record',color:colorFor(r.logical_block_id||r.record_id),
        x:550+Math.cos(angle)*radius,y:320+Math.sin(angle)*radius*.62,r:7,subtitle:statusNames[r.status]||r.status||''});
    });
    for(const edge of state.snapshot?.record_edges||[]){
      if(!visible.has(edge.src_id)||!visible.has(edge.dst_id))continue;
      const evidenceCount=edge.evidence_ref_count||edge.evidence_refs?.length||0;
      const shownCount=edge.evidence_refs?.length||0;
      const label=[edge.edge_type,edge.reason,`Подтверждающие источники: ${evidenceCount}`,
        edge.evidence_refs_truncated?`показано ${shownCount}`:''].filter(Boolean).join(' · ');
      links.push({source:edge.src_id,target:edge.dst_id,type:'evidence-related',label,color:colorFor(edge.src_id)});
    }
    return {nodes,links};
  }
  const roots=state.snapshot.tree||[], center=topic(state.block)||(roots.length===1?roots[0]:null),children=center?center.children||[]:roots;
  const nodes=[],links=[],visible=new Set(),visibleRecords=new Set();
  function addBlock(b,x,y,central=false){visible.add(b.block_id);nodes.push({id:b.block_id,title:b.name,type:'block',central,color:colorFor(b.block_id),x,y,r:central?37:27,subtitle:(b.children?.length?counted(b.children.length,['блок','блока','блоков'])+' · ':'')+counted(blockRecordCount(b),['запись','записи','записей'])});}
  if(center)addBlock(center,550,320,true);
  for(const [index,b]of children.entries()){
    const ring=Math.floor(index/8),count=Math.min(8,children.length-ring*8),angle=(index%8)/count*Math.PI*2+(count===2?0:-Math.PI/2+.22);
    const radius=children.length===1?205:250+ring*180;
    addBlock(b,550+Math.cos(angle)*radius,320+Math.sin(angle)*radius*.58);
    if(center)links.push({source:center.block_id,target:b.block_id,type:'tree',color:colorFor(b.block_id)});
  }
  const groups=new Map();
  for(const r of state.records){if(r.logical_block_id!==center?.block_id)continue;if(!groups.has(r.logical_block_id))groups.set(r.logical_block_id,[]);groups.get(r.logical_block_id).push(r);}
  for(const [parent,records]of groups){const owner=nodes.find(n=>n.id===parent),baseAngle=Math.atan2(owner.y-320,owner.x-550);
    records.forEach((r,index)=>{
      const perRing=owner.central?10:6,ring=Math.floor(index/perRing),count=Math.min(perRing,records.length-ring*perRing);
      const angle=owner.central?(index%perRing)/count*Math.PI*2-(count===2?0:Math.PI/2):baseAngle+(index%perRing-(count-1)/2)*Math.PI*1.35/Math.max(count,2);
      const radius=(owner.central?(children.length?345:185):112)+ring*110;
      nodes.push({id:r.record_id,previous:r.revision_of,title:r.title||'Без названия',type:'record',color:owner.color,x:owner.x+Math.cos(angle)*radius,y:owner.y+Math.sin(angle)*radius,r:7,subtitle:statusNames[r.status]||r.status||''});
      visibleRecords.add(r.record_id);
      links.push({source:parent,target:r.record_id,type:'membership',color:owner.color});
    });
  }
  for(const edge of state.snapshot.edges||[]){if(visible.has(edge.src_id)&&visible.has(edge.dst_id))links.push({source:edge.src_id,target:edge.dst_id,type:'related',label:edge.relation||edge.edge_type||'Связь блоков'});}
  for(const edge of state.snapshot.record_edges||[]){
    if(!visibleRecords.has(edge.src_id)||!visibleRecords.has(edge.dst_id))continue;
    const evidenceCount=edge.evidence_ref_count||edge.evidence_refs?.length||0;
    const shownCount=edge.evidence_refs?.length||0;
    const evidence=`Подтверждающие источники: ${evidenceCount}`+
      (edge.evidence_refs_truncated?` · показано ${shownCount}`:'');
    const label=[edge.edge_type,edge.reason,edge.reason_truncated?'описание сокращено':'',evidence]
      .filter(Boolean).join(' · ');
    links.push({source:edge.src_id,target:edge.dst_id,type:'evidence-related',label,color:colorFor(edge.src_id)});
  }
  return {nodes,links};
}
function labelLines(title,max=26){const words=title.split(/\s+/),lines=[''];for(const word of words){let i=lines.length-1;if(lines[i].length+word.length+(lines[i]?1:0)>max&&lines[i]){if(lines.length===2){lines[i]=lines[i].slice(0,max-1)+'…';break;}lines.push('');i++;}lines[i]+=(lines[i]?' ':'')+word;}return lines.map(line=>line.length>max?line.slice(0,max-1)+'…':line);}
function renderGraph(){
  if(!state.snapshot||state.mode!=='graph')return;
  const help=$('graph-wrap').querySelector('.graph-help');
  let scopeNote=help.querySelector('.graph-scope-note');
  if(!scopeNote){scopeNote=element('span','graph-scope-note');help.append(scopeNote);}
  const recordLimit=state.snapshot.record_graph_scope_limit||state.snapshot.recent_limit||64;
  const edgeLimit=state.snapshot.record_graph_edge_limit||128;
  const edgeByteLimit=Math.round((state.snapshot.record_graph_edge_byte_limit||65536)/1024);
  scopeNote.textContent=`${scopeLabel()} · граф записей по последним ${recordLimit}`+
    (state.scope.scope==='workspace'&&state.block?' · связи более старых выбранных записей здесь не загружаются':'')+
    (state.snapshot.record_edges_truncated?` · часть связей скрыта (предел ${edgeLimit} связей / ${edgeByteLimit} КБ)`:'' )+
    (state.snapshot.block_edges_truncated?` · блоковых связей показано не более ${state.snapshot.block_edge_limit}`:'');
  const viewport=$('graph').getBoundingClientRect();if(viewport.width&&viewport.height)$('graph').setAttribute('viewBox','0 0 '+viewport.width+' '+viewport.height);
  const scene=$('graph-scene');scene.replaceChildren();const model=graphModel();graphNodes=model.nodes;graphLinks=model.links;
  $('graph-empty').hidden=Boolean(graphNodes.length);$('graph-empty').textContent=state.scope.scope==='project'?'В выбранном проекте пока нет записей.':`Память пока пуста. Создайте первый блок через агента UM-MEM.`;
  for(const n of graphNodes)if(!state.positions[n.id])state.positions[n.id]=state.positions[n.previous]?{...state.positions[n.previous]}:{x:n.x,y:n.y};
  for(const radius of [180,330,470])scene.append(svg('circle',{class:'orbit-ring',cx:550,cy:320,r:radius}));
  for(const link of graphLinks){const path=svg('path',{class:'graph-link '+link.type,'data-source':link.source,'data-target':link.target});path.style.setProperty('--node-color',link.color||'#b3a2e6');if(link.label)path.append(svg('title',{},link.label));scene.append(path);}
  for(const n of graphNodes){const g=svg('g',{class:'graph-node '+n.type+((state.selected===n.id||state.selectedTopic===n.id)?' selected':''),'data-id':n.id,'data-type':n.type,tabindex:'0',role:'button','aria-label':n.title});g.style.setProperty('--node-color',n.color);g.append(svg('title',{},n.title));
    g.append(svg('rect',{x:-100,y:-n.r-15,width:200,height:n.r*2+90,fill:'transparent'}));
    if(n.type==='block'){g.append(svg('circle',{class:'node-halo',r:n.r+17}),svg('circle',{class:'node-orbit',r:n.r+9}));}
    g.append(svg('circle',{class:'node-core',r:n.r}));
    if(n.type==='block')g.append(svg('text',{class:'node-icon',y:-1},n.central?'✳':'◇'));
    const lines=labelLines(n.title,n.type==='record'?25:28);lines.forEach((line,i)=>g.append(svg('text',{class:'node-title',y:n.r+23+i*16},line)));
    if(n.type==='block')g.append(svg('text',{class:'node-caption',y:n.r+27+lines.length*16},n.subtitle));
    g.addEventListener('keydown',e=>{if(e.key==='Enter'){e.preventDefault();n.type==='record'?openRecord(n.id):selectBlock(n.id);}});
    scene.append(g);
  }
  updateGraphGeometry();applyTransform();if(state.fitNeeded){state.fitNeeded=false;requestAnimationFrame(()=>fitGraph());}
}
function updateGraphGeometry(){
  for(const node of $('graph-scene').querySelectorAll('.graph-node')){const p=state.positions[node.dataset.id];node.setAttribute('transform','translate('+p.x+','+p.y+')');}
  for(const path of $('graph-scene').querySelectorAll('.graph-link')){const a=state.positions[path.dataset.source],b=state.positions[path.dataset.target];if(!a||!b)continue;path.setAttribute('d','M'+a.x+','+a.y+' Q'+((a.x+b.x)/2+18)+','+((a.y+b.y)/2-15)+' '+b.x+','+b.y);path.classList.toggle('lit',[path.dataset.source,path.dataset.target].some(id=>id===state.selected||id===state.selectedTopic));}
}
function applyTransform(){const t=state.transform;$('graph-scene').setAttribute('transform','translate('+t.x+','+t.y+') scale('+t.k+')');$('zoom-level').textContent=Math.round(t.k*100)+'%';}
function fitGraph(){
  if(!graphNodes.length)return;
  const points=graphNodes.map(n=>({...state.positions[n.id],r:n.r}));
  const left=Math.min(...points.map(p=>p.x-95)),right=Math.max(...points.map(p=>p.x+95)),top=Math.min(...points.map(p=>p.y-p.r-30)),bottom=Math.max(...points.map(p=>p.y+p.r+75));
  const {width,height}=$('graph').getBoundingClientRect();if(!width||!height)return;
  $('graph').setAttribute('viewBox','0 0 '+width+' '+height);
  const k=Math.min(1.35,(width-100)/Math.max(right-left,1),(height-100)/Math.max(bottom-top,1));state.transform={k,x:width/2-(left+right)/2*k,y:(height-35)/2-(top+bottom)/2*k};applyTransform();storeLayout();
}
let drag=null,pendingBlockClick=null;
function point(event){const p=$('graph').createSVGPoint();p.x=event.clientX;p.y=event.clientY;return p.matrixTransform($('graph').getScreenCTM().inverse());}
$('graph').addEventListener('pointerdown',e=>{if(e.button!==0)return;const p=point(e),target=e.target.closest('.graph-node');drag={start:p,last:p,id:target?.dataset.id,type:target?.dataset.type,moved:false};$('graph').setPointerCapture(e.pointerId);});
$('graph').addEventListener('pointermove',e=>{if(!drag)return;const p=point(e),dx=p.x-drag.last.x,dy=p.y-drag.last.y;drag.moved ||=Math.hypot(p.x-drag.start.x,p.y-drag.start.y)>4;
  if(drag.moved){if(drag.id){state.positions[drag.id].x+=dx/state.transform.k;state.positions[drag.id].y+=dy/state.transform.k;updateGraphGeometry();}else{state.transform.x+=dx;state.transform.y+=dy;applyTransform();}}drag.last=p;});
$('graph').addEventListener('pointerup',()=>{if(!drag)return;const d=drag;drag=null;
  if(!d.moved&&d.id){
    if(d.type==='record'){if(pendingBlockClick)clearTimeout(pendingBlockClick.timer);pendingBlockClick=null;void openRecord(d.id);}
    else if(pendingBlockClick?.id===d.id){clearTimeout(pendingBlockClick.timer);pendingBlockClick=null;void selectBlock(d.id);}
    else{if(pendingBlockClick)clearTimeout(pendingBlockClick.timer);pendingBlockClick={id:d.id,timer:setTimeout(()=>{pendingBlockClick=null;openBlock(d.id);},300)};}
  }storeLayout();});
$('graph').addEventListener('pointercancel',()=>{drag=null;});
function zoom(factor,p=null){if(!p){const bounds=$('graph').viewBox.baseVal;p={x:bounds.width/2,y:bounds.height/2};}const t=state.transform,k=Math.max(.12,Math.min(4,t.k*factor));t.x=p.x-(p.x-t.x)*k/t.k;t.y=p.y-(p.y-t.y)*k/t.k;t.k=k;applyTransform();storeLayout();}
$('graph').addEventListener('wheel',e=>{e.preventDefault();zoom(Math.exp(-e.deltaY*.0015),point(e));},{passive:false});
$('zoom-in').onclick=()=>zoom(1.2);$('zoom-out').onclick=()=>zoom(1/1.2);$('graph-reset').onclick=()=>fitGraph();
function setMode(mode){state.mode=mode;$('graph-wrap').hidden=mode!=='graph';$('list-area').hidden=mode!=='list';$('graph-mode').classList.toggle('selected',mode==='graph');$('list-mode').classList.toggle('selected',mode==='list');renderGraph();}

document.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>setView(b.dataset.view));
$('list-mode').onclick=()=>setMode('list');$('graph-mode').onclick=()=>{
  if(state.query){state.searchRequest++;state.query='';state.records=[];$('query').value='';$('search-info').hidden=true;void refresh();}
  setMode('graph');
};
$('all-topics').onclick=()=>selectBlock('');$('new-record').onclick=()=>editRecord();
$('close-detail').onclick=()=>{if(allowLeave()){state.detailRequest++;state.draft=null;state.selected='';state.selectedTopic='';$('detail').hidden=true;renderRecords();if(state.stopped)showStopped();}};
$('search-form').onsubmit=e=>{e.preventDefault();void search($('query').value);};
$('record-scope').onchange=()=>{
  const value=$('record-scope').value;
  $('project-scope-form').hidden=value!=='manual';
  if(value==='workspace')void changeScope('workspace');
  else if(value.startsWith('project:'))void changeScope('project',value.slice(8));
};
$('project-scope-form').onsubmit=e=>{e.preventDefault();const id=$('project-scope-id').value;
  if(!id||id!==id.trim()){toast('Укажите точный ID проекта без пробелов по краям.');return;}
  void changeScope('project',id);
};
$('query').addEventListener('input',()=>{if(!$('query').value&&state.query){state.query='';state.searchRequest++;$('search-info').hidden=true;void refresh();}});
$('load-more').onclick=async()=>{const selected=state.block,scopeKey=scopeQuery();$('load-more').disabled=true;try{
  const page=await api(recordsUrl(selected,state.cursor));
  if(selected===state.block&&scopeKey===scopeQuery()){
    const ids=new Set(state.browseRecords.map(r=>r.record_id));state.browseRecords.push(...page.records.filter(r=>!ids.has(r.record_id)));state.cursor=page.next_cursor;
    if(state.query)await search(state.query);else{state.records.push(...page.records.filter(r=>!state.records.some(item=>item.record_id===r.record_id)));renderRecords();}
  }
}catch(error){toast(error.message);}finally{$('load-more').disabled=false;}};
window.addEventListener('beforeunload',e=>{if(state.draft?.dirty){e.preventDefault();e.returnValue='';}});
document.addEventListener('keydown',e=>{if(e.key==='Escape'&&!$('detail').hidden)$('close-detail').click();
  if(e.target.closest('input,textarea,select'))return;
  if(e.key==='/'){e.preventDefault();$('query').focus();}
  if(e.key==='Backspace'&&state.block){e.preventDefault();void selectBlock(topic(state.block)?.parent_id||'');}
});
function showStopped(){state.stopped=true;state.source?.close();connected(false);$('connection').textContent='Панель остановлена';
  if(state.draft){$('stopped').hidden=true;banner('Панель остановлена. Черновик сохранён в этом редакторе — скопируйте его перед закрытием вкладки. Повторный запуск: um-memory dashboard.');
    if($('save-record'))$('save-record').disabled=true;
  }else{$('stopped').hidden=false;banner();}
  $('new-record').disabled=true;$('stop-panel').disabled=true;
}
$('stop-panel').onclick=async()=>{if(state.draft?.saving){toast('Дождитесь результата сохранения.');return;}$('stop-panel').disabled=true;
  try{await api('api/stop',{});showStopped();}catch(error){$('stop-panel').disabled=false;toast(error.message);}};

async function start() {
  try {
    const token=location.hash.slice(1);if(token){history.replaceState(null,'',location.pathname);await api('auth',{token});}
    await refresh();if(!state.snapshot)throw new Error('Не удалось прочитать память. Откройте панель командой um-memory dashboard.');
    restoreLayout();setMode('graph');
    state.source=new EventSource(`/panel/api/events?since=${encodeURIComponent(state.snapshot.cursor)}`);
    state.source.onopen=()=>{connected(true);banner();};state.source.onerror=()=>{if(state.stopped)return;connected(false);banner('Связь потеряна. Черновик остаётся в редакторе. Если панель была остановлена, откройте её снова: um-memory dashboard.');};
    state.source.addEventListener('stopped',()=>showStopped());
    state.source.addEventListener('reset',()=>void refresh());
    let refreshTimer;
    state.source.addEventListener('change',e=>{const event=JSON.parse(e.data);if(!state.events.some(x=>x.id===event.id)){state.events.push(event);state.events=state.events.slice(-256);renderEvents();}
      if(event.kind!=='retrieval'){clearTimeout(refreshTimer);refreshTimer=setTimeout(()=>void refresh(),120);}
    });
  }catch(error){connected(false);banner(error.message);}
}
void start();
