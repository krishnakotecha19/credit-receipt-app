
    // Receipt image modal
    function showReceiptModal(src) {
        const modal = document.getElementById('receiptModal');
        document.getElementById('receiptModalImg').src = src;
        modal.classList.add('active');
    }

    // Tab switching
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            try {
                document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
                document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
                btn.classList.add('active');
                document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
                
                // Auto-load sync data when Sync tab is first clicked
                if (btn.dataset.tab === 'sync' && !window._syncLoaded) {
                    loadSyncData();
                }
            } catch (err) {
                alert('Tab switch error: ' + err.message);
                console.error(err);
            }
        });
    });

    // Open Sync tab and auto-load
    function openSyncTab() {
        // Switch all tabs off
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
        // Show sync panel
        const syncPanel = document.getElementById('tab-sync');
        if (syncPanel) syncPanel.classList.add('active');
        // Highlight the button
        const syncBtn = document.getElementById('btnSyncSP');
        if (syncBtn) syncBtn.style.outline = '2px solid #3B82F6';
        // Load data
        loadSyncData();
    }

    // SharePoint Sync — load metadata
    function loadSyncData() {
        try {
            const entitySelect = document.querySelector('select[name="entity"]');
            if (!entitySelect) throw new Error("Entity dropdown not found");
            const entity = entitySelect.value.toLowerCase();
            
            const label = document.getElementById('syncEntityLabel');
            if (label) label.textContent = entity.charAt(0).toUpperCase() + entity.slice(1);
            
            const btn = document.getElementById('btnRefreshSync');
            if (btn) {
                btn.disabled = true;
                btn.innerHTML = '<i class="bi bi-arrow-clockwise spin"></i> Loading...';
            }
            
            const msgObj = document.getElementById('syncPlaceholder');
            const resObj = document.getElementById('syncResults');
            if (msgObj) msgObj.style.display = 'none';
            if (resObj) resObj.style.display = 'none';

            fetch('/api/sync/' + encodeURIComponent(entity), {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({})
        })
        .then(r => r.json())
        .then(data => {
            btn.disabled = false;
            btn.innerHTML = '<i class="bi bi-arrow-clockwise"></i> Refresh';
            if (!data.ok) {
                document.getElementById('syncPlaceholder').style.display = 'block';
                document.getElementById('syncPlaceholder').innerHTML =
                    '<i class="bi bi-exclamation-triangle" style="font-size:2rem;color:var(--warning);display:block;margin-bottom:.5rem;"></i>' +
                    '<p style="color:var(--danger);font-size:.85rem;">' + (data.error || 'Sync failed') + '</p>';
                return;
            }
            window._syncLoaded = true;
            const meta = data.metadata;
            const stmtBox = document.getElementById('syncStatements');
            const batchBox = document.getElementById('syncBatches');
            stmtBox.innerHTML = '';
            batchBox.innerHTML = '';

            if (meta.statements.length === 0) {
                stmtBox.innerHTML = '<p style="color:var(--text-muted);font-size:.82rem;">No PDF statements found.</p>';
            } else {
                meta.statements.forEach(s => {
                    const sizeKB = s.size ? (s.size / 1024).toFixed(1) + ' KB' : '';
                    stmtBox.innerHTML +=
                        '<div style="display:flex;align-items:center;gap:10px;padding:8px 12px;border:1px solid var(--border);border-radius:var(--radius-sm);background:var(--surface-alt);font-size:.82rem;">' +
                        '<i class="bi bi-file-earmark-pdf" style="font-size:1.2rem;color:#EF4444;"></i>' +
                        '<div style="flex:1;"><strong>' + s.name + '</strong>' +
                        (sizeKB ? '<span style="color:var(--text-muted);margin-left:8px;">' + sizeKB + '</span>' : '') +
                        '</div>' +
                        '<button class="btn btn-sm btn-outline-primary" onclick="stageFile(\'' + entity + '\',\'' + s.id + '\',null)" style="font-size:.75rem;padding:2px 10px;"><i class="bi bi-download"></i> Stage</button>' +
                        '</div>';
                });
            }

            if (meta.receipt_batches.length === 0) {
                batchBox.innerHTML = '<p style="color:var(--text-muted);font-size:.82rem;">No receipt batches found.</p>';
            } else {
                meta.receipt_batches.forEach(b => {
                    batchBox.innerHTML +=
                        '<div style="display:flex;align-items:center;gap:10px;padding:8px 12px;border:1px solid var(--border);border-radius:var(--radius-sm);background:var(--surface-alt);font-size:.82rem;">' +
                        '<i class="bi bi-folder-fill" style="font-size:1.2rem;color:var(--warning);"></i>' +
                        '<div style="flex:1;"><strong>' + b.name + '</strong></div>' +
                        '<button class="btn btn-sm btn-outline-primary" onclick="stageFile(\'' + entity + '\',null,\'' + b.id + '\')" style="font-size:.75rem;padding:2px 10px;"><i class="bi bi-download"></i> Stage</button>' +
                        '</div>';
                });
            }
            document.getElementById('syncResults').style.display = 'block';
        })
        .catch(err => {
            if (btn) {
                btn.disabled = false;
                btn.innerHTML = '<i class="bi bi-arrow-clockwise"></i> Refresh';
            }
            if (msgObj) {
                msgObj.style.display = 'block';
                msgObj.innerHTML =
                    '<p style="color:var(--danger);font-size:.85rem;">Network error: ' + err + '</p>';
            }
        });
        } catch (err) {
            alert('loadSyncData error: ' + err.message);
            console.error(err);
        }
    }

    // Stage individual statement or batch
    function stageFile(entity, stmtId, batchId) {
        const payload = {};
        if (stmtId) payload.statement_id = stmtId;
        if (batchId) payload.batch_folder_id = batchId;
        fetch('/api/sync/' + encodeURIComponent(entity), {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload)
        })
        .then(r => r.json())
        .then(data => {
            if (data.ok) {
                alert('Staged successfully!\\n' +
                    (data.statement ? 'Statement: ' + data.statement + '\\n' : '') +
                    (data.receipts && data.receipts.length ? 'Receipts: ' + data.receipts.length + ' files' : ''));
            } else {
                alert('Staging failed: ' + (data.error || 'Unknown error'));
            }
        })
        .catch(err => alert('Network error: ' + err));
    }

    // File upload feedback
    document.getElementById('receiptInput').addEventListener('change', function() {
        const n = this.files.length;
        if (n > 0) {
            document.getElementById('receiptBadge').innerHTML =
                '<span class="file-badge"><i class="bi bi-check-circle-fill"></i> ' + n + ' receipt(s) selected</span>';
            document.getElementById('receiptZone').style.borderColor = 'var(--success)';
        }
    });
    document.getElementById('stmtInput').addEventListener('change', function() {
        if (this.files.length > 0) {
            document.getElementById('stmtBadge').innerHTML =
                '<span class="file-badge"><i class="bi bi-check-circle-fill"></i> ' + this.files[0].name + '</span>';
            document.getElementById('stmtZone').style.borderColor = 'var(--success)';
        }
    });

    // Drag and drop
    ['receiptZone', 'stmtZone'].forEach(id => {
        const zone = document.getElementById(id);
        zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('dragover'); });
        zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
        zone.addEventListener('drop', e => {
            e.preventDefault();
            zone.classList.remove('dragover');
            const input = zone.querySelector('input');
            input.files = e.dataTransfer.files;
            input.dispatchEvent(new Event('change'));
        });
    });

    // Processing form — submit and poll for progress
    document.getElementById('processForm').addEventListener('submit', function(e) {
        const btn = document.getElementById('btnProcess');
        btn.disabled = true;
        btn.innerHTML = '<span class="spinner"></span> Processing...';
        document.getElementById('progressSection').style.display = 'block';
        window._progressStartTime = Date.now();
    });

    // GL code auto-fill when cost centre is selected
    document.querySelectorAll('.cc-select').forEach(sel => {
        sel.addEventListener('change', function() {
            const idx = this.dataset.idx;
            const cc = this.value;
            const entity = this.dataset.entity || '{{ entity }}';
            fetch('/api/gl-code?entity=' + encodeURIComponent(entity) + '&cost_centre=' + encodeURIComponent(cc))
                .then(r => r.json())
                .then(data => {
                    const glSpan = document.querySelector('.gl-code-display[data-idx="' + idx + '"]');
                    if (glSpan) glSpan.textContent = data.gl_code || '—';
                    // Also save to backend
                    fetch('/api/update-cc', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({idx: parseInt(idx), cost_centre: cc, gl_code: data.gl_code || ''})
                    });
                });
        });
    });

    // Approve buttons with cost centre
    document.querySelectorAll('.approve-btn').forEach(btn => {
        btn.addEventListener('click', function(e) {
            e.preventDefault();
            const idx = this.dataset.idx;
            const row = this.closest('tr');
            const ccSelect = row ? row.querySelector('.cc-select') : null;
            const cc = ccSelect ? ccSelect.value : '';
            window.location.href = '/approve/' + idx + '?cost_centre=' + encodeURIComponent(cc);
        });
    });

    // Statement transaction reassignment dropdown
    document.querySelectorAll('.stmt-select').forEach(sel => {
        sel.addEventListener('change', function() {
            const matchIdx = parseInt(this.dataset.idx);
            const stmtIdx = parseInt(this.value);
            if (isNaN(stmtIdx)) return;
            fetch('/api/reassign-stmt', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({idx: matchIdx, stmt_idx: stmtIdx})
            })
            .then(r => r.json())
            .then(data => {
                if (!data.ok) { alert(data.error || 'Reassign failed'); return; }
                // Update the amount and date cells in this row
                const amtCell = document.querySelector('.stmt-amount[data-idx="' + matchIdx + '"]');
                const dateCell = document.querySelector('.stmt-date[data-idx="' + matchIdx + '"]');
                if (amtCell && data.stmt_amount != null) amtCell.textContent = data.stmt_amount.toFixed(2);
                if (dateCell) dateCell.textContent = data.stmt_date || '—';
                // Update score badge
                const row = this.closest('tr');
                if (row) {
                    const scoreBadge = row.querySelector('.score-badge');
                    if (scoreBadge) {
                        scoreBadge.textContent = data.match_score;
                        scoreBadge.className = 'score-badge ' + (data.match_score > 90 ? 'high' : data.match_score >= 70 ? 'medium' : 'low');
                    }
                }
            })
            .catch(err => console.error('Reassign error:', err));
        });
    });

    // Poll progress if processing
    {% if processing %}
    window._progressStartTime = window._progressStartTime || Date.now();
    const _STEP_ORDER = ['Receipts', 'Receipts (Online)', 'Statements', 'Matching', 'Done'];
    const _STEP_MAP = {'Receipts': 'Receipts', 'Receipts (Online)': 'Receipts', 'Statements': 'Statements', 'Matching': 'Matching', 'Done': 'Done'};

    function _updateStepDots(currentStep) {
        const mapped = _STEP_MAP[currentStep] || currentStep;
        const order = ['Receipts', 'Statements', 'Matching', 'Done'];
        const ci = order.indexOf(mapped);
        document.querySelectorAll('#progressSteps .progress-step').forEach(el => {
            const si = order.indexOf(el.dataset.step);
            el.classList.remove('active', 'done');
            if (si < ci) el.classList.add('done');
            else if (si === ci) el.classList.add(mapped === 'Done' ? 'done' : 'active');
        });
    }

    function _updateElapsed() {
        const s = Math.floor((Date.now() - window._progressStartTime) / 1000);
        const m = Math.floor(s / 60);
        const sec = String(s % 60).padStart(2, '0');
        const el = document.getElementById('progressElapsed');
        if (el) el.textContent = m + ':' + sec;
    }
    const _elapsedTimer = setInterval(_updateElapsed, 1000);

    (function pollProgress() {
        fetch('/progress')
            .then(r => r.json())
            .then(data => {
                const stepEl = document.getElementById('stepName');
                stepEl.innerHTML = '<span class="detail-spinner"></span> ' + (data.step || 'Starting...');
                document.getElementById('stepPct').textContent = data.pct + '%';

                const bar = document.getElementById('progressBar');
                const track = document.getElementById('progressTrack');
                bar.style.width = data.pct + '%';

                // Toggle indeterminate when at 0%
                if (data.pct === 0) {
                    track.classList.add('indeterminate');
                    bar.style.width = '0%';
                } else {
                    track.classList.remove('indeterminate');
                }

                // Keep shimmer while processing, remove when done
                if (!data.processing) bar.classList.remove('animate');

                document.getElementById('progressDetail').textContent = data.detail || 'Working...';

                // Update step dots
                _updateStepDots(data.step);

                if (data.processing) {
                    setTimeout(pollProgress, 1000);
                } else {
                    clearInterval(_elapsedTimer);
                    stepEl.innerHTML = '<i class="bi bi-check-circle-fill" style="color:var(--success)"></i> Complete!';
                    document.getElementById('stepPct').textContent = '100%';
                    bar.style.width = '100%';
                    track.classList.remove('indeterminate');
                    setTimeout(() => location.reload(), 800);
                }
            })
            .catch(() => setTimeout(pollProgress, 2000));
    })();
    {% endif %}
