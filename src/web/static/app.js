/* ============================================================
   看图识车 · 前端逻辑
   - 用 marked.js 渲染 Markdown 回复
   - 根据工具日志自动生成结构化"识别卡片"
   - 左/中/右三栏布局，会话计数实时同步
   ============================================================ */

const messagesEl   = document.getElementById('messages');
const toolListEl   = document.getElementById('tool-list');
const toolEmptyEl  = document.getElementById('tool-empty');
const toolCountEl  = document.getElementById('tool-count');
const sessionIdEl  = document.getElementById('session-id-text');
const sessionMsgEl = document.getElementById('session-msg-count');
const sessionToolEl= document.getElementById('session-tool-count');
const statusEl     = document.getElementById('status');
const form         = document.getElementById('composer');
const textInput    = document.getElementById('text');
const fileInput    = document.getElementById('image');
const imagePreview = document.getElementById('image-preview');
const uploadBtn    = document.getElementById('upload-btn');
const clearBtn     = document.getElementById('clear-btn');

let sessionId   = null;
let pendingFile = null;
let toolCount   = 0;
let msgCount    = 0;

/* ----------- 工具函数 ----------- */
function escapeHtml(s) {
  return (s || '').replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[c]));
}

function fmtTime(iso) {
  try {
    return new Date(iso).toLocaleTimeString('zh-CN', { hour12: false });
  } catch { return ''; }
}

function shortSessionId(sid) {
  if (!sid) return '—';
  return sid.length > 14 ? sid.slice(0, 8) + '…' + sid.slice(-4) : sid;
}

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function renderMarkdown(text) {
  if (!text) return '';
  if (window.marked) {
    try { return window.marked.parse(text, { breaks: true, gfm: true }); }
    catch (e) { /* fall through */ }
  }
  return escapeHtml(text);
}

/* ----------- 会话计数 ----------- */
function updateSessionMeta() {
  sessionIdEl.textContent  = shortSessionId(sessionId);
  sessionMsgEl.textContent = msgCount;
  sessionToolEl.textContent= toolCount;
}

/* ----------- 消息气泡 ----------- */
function appendMessage(role, bodyHtml) {
  const welcome = messagesEl.querySelector('.welcome');
  if (welcome) welcome.remove();

  const wrap = document.createElement('div');
  wrap.className = 'bubble ' + role;
  const avatarText = role === 'user' ? '我' : role === 'system' ? '!' : 'AI';
  wrap.innerHTML = `
    <div class="avatar">${avatarText}</div>
    <div class="bubble-body">${bodyHtml}</div>
  `;
  messagesEl.appendChild(wrap);
  scrollToBottom();
  msgCount += 1;
  updateSessionMeta();
  return wrap;
}

function showTyping() {
  return appendMessage('assistant', '<div class="typing"><span></span><span></span><span></span></div>');
}

/* ----------- 结构化卡片：根据工具调用结果自动渲染 ----------- */

// "识别结果"卡片 —— 从 classify_car 工具结果渲染
function renderClassifyCard(result, imageUrl) {
  if (!result || !result.ok) return '';
  const top1Id = result.top1_id;
  const conf = (result.top1_conf || 0) * 100;
  const topK = result.top_k || [];
  const isUnknown = top1Id === 'unknown';

  let imgPart = '';
  if (imageUrl) {
    imgPart = `<img class="tce-img" src="${imageUrl}" alt="识别图" />`;
  }

  let nameHtml;
  if (isUnknown) {
    nameHtml = `<div class="tce-name">未识别为本组支持的车型</div>
                <div class="tce-sub">可能是数据外的车型，或拍摄角度特殊</div>`;
  } else {
    nameHtml = `<div class="tce-name">车型 ID · <code style="background:transparent;color:var(--primary);font-size:13px">${top1Id}</code></div>
                <div class="tce-sub">Top-${topK.length} 候选 · 推理尺寸 ${result.img_size_used || 224}px · ${result.tta || 'flip'}-TTA</div>`;
  }

  const topkHtml = topK.slice(0, 5).map((it, i) => {
    const id = typeof it === 'object' ? it.id : it[0];
    const c  = typeof it === 'object' ? it.confidence : it[1];
    const pct = (c * 100).toFixed(1);
    const width = Math.max(2, c * 100);
    return `<div class="tce-topk-row">
              <span class="tce-topk-id">${escapeHtml(id)}</span>
              <div class="tce-topk-bar"><div class="tce-topk-fill" style="width:${width}%"></div></div>
              <span class="tce-topk-pct">${pct}%</span>
            </div>`;
  }).join('');

  return `
    <div class="tool-card-embed">
      <div class="tce-header">
        <div class="tce-title">🔍 识别结果</div>
        <span class="tce-badge">✓ 已完成</span>
      </div>
      <div class="tce-body">
        <div class="tce-row">
          ${imgPart}
          <div class="tce-info">
            ${nameHtml}
            <div class="tce-conf-bar"><div class="tce-conf-fill" style="width:${Math.max(2, conf)}%"></div></div>
            <div class="tce-sub" style="margin-top:4px">置信度 <strong>${conf.toFixed(2)}%</strong></div>
          </div>
        </div>
        ${topkHtml ? `<div class="tce-topk">${topkHtml}</div>` : ''}
      </div>
    </div>
  `;
}

// "车型资料"卡片 —— 从 query_car_info 渲染
function renderInfoCard(result) {
  if (!result || !result.ok) return '';
  const typeParts = [result.body_type, result.body_type_detail].filter(Boolean).join(' · ');
  const items = [
    ['中文名', result.name_zh],
    ['英文名', result.name_en],
    ['品牌', [result.brand_zh, result.brand_en].filter(Boolean).join(' / ')],
    ['类型', typeParts],
  ].filter(([, v]) => v);
  return `
    <div class="tool-card-embed">
      <div class="tce-header">
        <div class="tce-title">📚 车型资料</div>
        <span class="tce-badge">✓ 已查询</span>
      </div>
      <div class="tce-body">
        <dl class="tce-info-grid">
          ${items.map(([k, v]) => `<dt>${escapeHtml(k)}</dt><dd>${escapeHtml(v)}</dd>`).join('')}
        </dl>
      </div>
    </div>
  `;
}

// "已记录"chip —— 从 add_to_history / mark_favorite / list_history 渲染
function renderHistoryChip(result, name) {
  if (!result || !result.ok) return '';
  let label = '已记录';
  if (name === 'mark_favorite') label = result.favorite ? '⭐ 已收藏' : '已取消收藏';
  if (name === 'add_to_history') label = '📝 已加入浏览记录';
  if (name === 'list_history') return renderHistoryList(result);
  return `<div style="margin-top:10px"><span class="tce-success">✓ ${escapeHtml(label)}${result.history_id ? ' #' + result.history_id : ''}</span></div>`;
}

function renderHistoryList(result) {
  const items = (result.items || []).slice(0, 5);
  if (!items.length) return `<div style="margin-top:10px"><span class="muted small">暂无浏览记录</span></div>`;
  const rows = items.map(it => `
    <div class="tce-topk-row" style="padding:4px 0;border-bottom:1px dashed var(--border)">
      <span class="tce-topk-id">${escapeHtml(it.car_id || '')}</span>
      <span style="flex:1;color:var(--text-2);font-size:12px">${escapeHtml(it.image_id || '')}</span>
      <span class="tce-topk-pct">${(it.confidence*100).toFixed(0)}%</span>
      ${it.favorite ? '<span style="color:var(--warn)">⭐</span>' : ''}
    </div>
  `).join('');
  return `
    <div class="tool-card-embed">
      <div class="tce-header">
        <div class="tce-title">📚 浏览历史（${items.length}）</div>
        <span class="tce-badge">✓ 已查询</span>
      </div>
      <div class="tce-body">${rows}</div>
    </div>
  `;
}

/* ----------- 工具日志 ----------- */
function appendToolLog(entry) {
  toolEmptyEl.classList.add('hidden');
  toolCount += 1;
  toolCountEl.textContent = `${toolCount} 次`;
  updateSessionMeta();

  const li = document.createElement('li');
  li.className = 'tool-card open';
  const ok = entry.result && entry.result.ok;
  const statusClass = ok ? 'ok' : 'fail';
  const statusText  = ok ? '✓ 成功' : '✗ 失败';
  let argsPretty = entry.arguments_raw || '';
  try { argsPretty = JSON.stringify(JSON.parse(argsPretty), null, 2); } catch {}
  const resultPretty = JSON.stringify(entry.result, null, 2);

  li.innerHTML = `
    <div class="tool-head">
      <div class="tool-name">${escapeHtml(entry.name)}</div>
      <div class="tool-status ${statusClass}">${statusText}</div>
    </div>
    <div class="tool-meta">${fmtTime(entry.time)} · 耗时 ${entry.elapsed_ms ?? 0} ms</div>
    <div class="tool-body">
      <details open>
        <summary>返回结果</summary>
        <pre>${escapeHtml(resultPretty)}</pre>
      </details>
      <details>
        <summary>调用参数</summary>
        <pre>${escapeHtml(argsPretty)}</pre>
      </details>
    </div>
  `;
  li.querySelector('.tool-head').addEventListener('click', () => li.classList.toggle('open'));
  toolListEl.prepend(li);
}

/* ----------- 图片预览 ----------- */
function setPendingFile(file) {
  pendingFile = file;
  if (!file) {
    imagePreview.classList.add('hidden');
    imagePreview.innerHTML = '';
    return;
  }
  const url = URL.createObjectURL(file);
  imagePreview.classList.remove('hidden');
  imagePreview.innerHTML = `
    <img src="${url}" />
    <span>${escapeHtml(file.name.length > 18 ? file.name.slice(0, 15) + '…' : file.name)}</span>
    <span class="remove" title="移除">×</span>
  `;
  imagePreview.querySelector('.remove').onclick = () => {
    fileInput.value = '';
    setPendingFile(null);
  };
}

// label[for=image] 会自动把 click 转发到 file input；这里再绑一次兜底（防止浏览器差异）
uploadBtn.addEventListener('click', (e) => {
  // label 时浏览器原生就转发，这里不阻断默认行为
  if (!uploadBtn.htmlFor) fileInput.click();
});
fileInput.addEventListener('change', () => {
  console.log('[upload] file selected:', fileInput.files && fileInput.files[0] && fileInput.files[0].name);
  if (fileInput.files && fileInput.files[0]) setPendingFile(fileInput.files[0]);
});

// 整页拖拽支持：把图拖到对话框区即可上传
const chatEl = document.querySelector('.chat');
['dragenter', 'dragover'].forEach(ev =>
  chatEl.addEventListener(ev, e => { e.preventDefault(); e.dataTransfer.dropEffect = 'copy'; })
);
chatEl.addEventListener('drop', e => {
  e.preventDefault();
  const f = e.dataTransfer.files && e.dataTransfer.files[0];
  if (!f) return;
  if (!f.type.startsWith('image/')) {
    statusEl.textContent = '只支持图片文件';
    return;
  }
  setPendingFile(f);
  statusEl.textContent = `已拖入 ${f.name}，点击发送即可`;
});

/* ----------- 上传 + 发送 ----------- */
async function uploadIfAny() {
  if (!pendingFile) return null;
  const fd = new FormData();
  fd.append('file', pendingFile);
  statusEl.textContent = '上传图片中…';
  const resp = await fetch('/api/upload', { method: 'POST', body: fd });
  statusEl.textContent = '';
  if (!resp.ok) throw new Error('upload failed: ' + resp.status);
  return await resp.json();
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const userText = textInput.value.trim();
  if (!userText) return;
  const sendBtn = document.getElementById('send-btn');
  sendBtn.disabled = true;

  let uploaded = null;
  try { uploaded = await uploadIfAny(); }
  catch (err) {
    sendBtn.disabled = false;
    statusEl.textContent = '上传失败：' + err.message;
    return;
  }

  // 用户消息气泡
  let userHtml = escapeHtml(userText);
  let uploadedImgUrl = null;
  if (uploaded) {
    uploadedImgUrl = '/uploads/' + uploaded.file_id;
    userHtml += `<img src="${uploadedImgUrl}" alt="uploaded" />`;
  }
  appendMessage('user', userHtml);
  fileInput.value = '';
  setPendingFile(null);

  // 打字指示
  const typingEl = showTyping();
  statusEl.textContent = 'Agent 正在思考…';

  // 调 chat
  const fd = new FormData();
  fd.append('text', userText);
  if (sessionId) fd.append('session_id', sessionId);
  if (uploaded) fd.append('file_id', uploaded.file_id);

  try {
    const resp = await fetch('/api/chat', { method: 'POST', body: fd });
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    const data = await resp.json();
    sessionId = data.session_id;
    statusEl.textContent = '';

    // 把识别类工具调用结果渲染成结构化卡片，附加到 assistant 气泡
    let embeds = '';
    for (const log of (data.tool_logs || [])) {
      if (log.name === 'classify_car') {
        embeds += renderClassifyCard(log.result, uploadedImgUrl);
      } else if (log.name === 'query_car_info') {
        embeds += renderInfoCard(log.result);
      } else if (['add_to_history', 'mark_favorite', 'list_history'].includes(log.name)) {
        embeds += renderHistoryChip(log.result, log.name);
      }
    }

    const mdHtml = renderMarkdown(data.answer || '(无回复)');
    const finalHtml = mdHtml + embeds;
    const body = typingEl.querySelector('.bubble-body');
    body.innerHTML = finalHtml;
    body.classList.add('rendered');

    for (const log of (data.tool_logs || [])) appendToolLog(log);
  } catch (err) {
    typingEl.classList.add('system');
    typingEl.querySelector('.bubble-body').innerHTML =
      `请求失败：${escapeHtml(err.message)}。请检查 DEEPSEEK_API_KEY 或网络连接。`;
  } finally {
    sendBtn.disabled = false;
    textInput.value = '';
    updateSessionMeta();
  }
});

/* ----------- 建议词 ----------- */
function wireSuggestions() {
  document.querySelectorAll('.suggest-chip').forEach(btn => {
    btn.addEventListener('click', () => {
      textInput.value = btn.dataset.text || btn.textContent;
      textInput.focus();
    });
  });
}
wireSuggestions();

/* ----------- 新对话 ----------- */
clearBtn.addEventListener('click', () => {
  if (!confirm('开启一个新对话？当前 session 的工具调用日志会清空。')) return;
  sessionId = null;
  toolCount = 0;
  msgCount = 0;
  updateSessionMeta();
  toolListEl.innerHTML = '';
  toolEmptyEl.classList.remove('hidden');
  toolCountEl.textContent = '0 次';
  messagesEl.innerHTML = `
    <div class="welcome">
      <div class="welcome-mark">🔍</div>
      <h2>开始一段新的识车之旅</h2>
      <p>上传一张车辆照片，模型会自动识别并调用相关工具。</p>
      <div class="suggestions">
        <button class="suggest-chip" data-text="这是什么车？介绍一下车型信息。">
          <span class="suggest-emoji">🚗</span>这是什么车
        </button>
        <button class="suggest-chip" data-text="帮我加入浏览记录。">
          <span class="suggest-emoji">📝</span>加入浏览记录
        </button>
        <button class="suggest-chip" data-text="看看我最近看过哪些车。">
          <span class="suggest-emoji">📚</span>查看浏览历史
        </button>
        <button class="suggest-chip" data-text="把它标记为收藏。">
          <span class="suggest-emoji">⭐</span>标记为收藏
        </button>
      </div>
    </div>
  `;
  wireSuggestions();
});

updateSessionMeta();