// ─────────────────────────────────────────────────
// Stock Selector — Searchable Dropdown (ticker.on style)
// ─────────────────────────────────────────────────
(function () {
    const searchInput = document.getElementById('stockSearch');
    const hiddenInput = document.getElementById('stockInput');
    const dropdown = document.getElementById('dropdownList');
    if (!searchInput || !dropdown) return;

    const items = dropdown.querySelectorAll('.dropdown-item');

    searchInput.addEventListener('focus', () => dropdown.classList.add('open'));

    searchInput.addEventListener('input', () => {
        const q = searchInput.value.toLowerCase();
        items.forEach(item => {
            const ticker = item.dataset.short.toLowerCase();
            const name = item.dataset.name.toLowerCase();
            item.style.display = (ticker.includes(q) || name.includes(q)) ? '' : 'none';
        });
        dropdown.classList.add('open');
        hiddenInput.value = '';
    });

    items.forEach(item => {
        item.addEventListener('click', () => {
            searchInput.value = `${item.dataset.short} — ${item.dataset.name}`;
            hiddenInput.value = item.dataset.ticker;
            dropdown.classList.remove('open');
            searchInput.blur();
        });
    });

    document.addEventListener('click', (e) => {
        if (!e.target.closest('#stockSelector')) dropdown.classList.remove('open');
    });

    // Quick pick buttons
    document.querySelectorAll('.quick-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const ticker = btn.dataset.ticker;
            const match = dropdown.querySelector(`[data-short="${ticker}"]`);
            if (match) {
                searchInput.value = `${match.dataset.short} — ${match.dataset.name}`;
                hiddenInput.value = match.dataset.ticker;
                dropdown.classList.remove('open');
                document.getElementById('analysisForm').requestSubmit();
            }
        });
    });
})();


// ─────────────────────────────────────────────────
// Price Chart (with period buttons)
// ─────────────────────────────────────────────────
if (window.chartData) {
    const ctx = document.getElementById('chart').getContext('2d');
    const allDates = window.chartData.dates;
    const allPrices = window.chartData.prices;

    function createGradient(context) {
        const g = context.createLinearGradient(0, 0, 0, 400);
        g.addColorStop(0, window.trendColor + '30');
        g.addColorStop(0.5, window.trendColor + '10');
        g.addColorStop(1, 'transparent');
        return g;
    }

    let chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: allDates.slice(-30),
            datasets: [{
                label: 'Price',
                data: allPrices.slice(-30),
                borderColor: window.trendColor,
                backgroundColor: createGradient(ctx),
                borderWidth: 2.5, tension: 0.4, fill: true,
                pointRadius: 0, pointHoverRadius: 6,
                pointBackgroundColor: window.trendColor,
                pointBorderColor: '#0f172a', pointBorderWidth: 3,
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: document.body.classList.contains('light-theme') ? 'rgba(255,255,255,0.97)' : 'rgba(15, 23, 42, 0.95)',
                    titleColor: document.body.classList.contains('light-theme') ? '#9ca3af' : '#94a3b8',
                    bodyColor: document.body.classList.contains('light-theme') ? '#1a1a2e' : '#f1f5f9',
                    titleFont: { size: 11, weight: '600', family: 'Inter' },
                    bodyFont: { size: 13, weight: '700', family: 'Inter' },
                    padding: { x: 16, y: 12 }, cornerRadius: 12,
                    borderColor: 'rgba(255,255,255,0.08)', borderWidth: 1,
                    displayColors: false,
                    callbacks: {
                        label: (c) => '\u20B9 ' + c.parsed.y.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
                    }
                }
            },
            scales: {
                y: {
                    grid: { color: document.body.classList.contains('light-theme') ? 'rgba(0,0,0,0.06)' : 'rgba(255,255,255,0.04)', drawBorder: false },
                    border: { display: false },
                    ticks: { color: document.body.classList.contains('light-theme') ? '#9ca3af' : '#64748b', font: { size: 11, family: 'Inter', weight: '500' }, padding: 12,
                        callback: (v) => '\u20B9' + v.toLocaleString('en-IN') }
                },
                x: {
                    grid: { display: false }, border: { display: false },
                    ticks: { color: document.body.classList.contains('light-theme') ? '#9ca3af' : '#64748b', font: { size: 10, family: 'Inter', weight: '500' },
                        padding: 8, maxRotation: 0, autoSkip: true, maxTicksLimit: 7 }
                }
            }
        }
    });

    // Period buttons (support both .period-btn and .per-btn)
    document.querySelectorAll('.period-btn, .per-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.period-btn, .per-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const days = parseInt(btn.dataset.days);
            const slicedDates = days === 0 ? allDates : allDates.slice(-days);
            const slicedPrices = days === 0 ? allPrices : allPrices.slice(-days);
            chart.data.labels = slicedDates;
            chart.data.datasets[0].data = slicedPrices;
            chart.update('none');
        });
    });
}


// ─────────────────────────────────────────────────
// RSI Gauge
// ─────────────────────────────────────────────────
(function () {
    const gaugeCanvas = document.getElementById('rsiGauge');
    if (!gaugeCanvas || !window.rsiValue) return;

    const ctx = gaugeCanvas.getContext('2d');
    const W = gaugeCanvas.width, H = gaugeCanvas.height;
    const cx = W / 2, cy = H - 10, radius = 75, lw = 14;

    // Background arc
    ctx.beginPath(); ctx.arc(cx, cy, radius, Math.PI, 0, false);
    ctx.strokeStyle = 'rgba(255,255,255,0.06)'; ctx.lineWidth = lw; ctx.lineCap = 'round'; ctx.stroke();

    // Colored arc
    const pct = Math.min(Math.max(window.rsiValue / 100, 0), 1);
    const endAngle = Math.PI + (pct * Math.PI);
    const grad = ctx.createLinearGradient(cx - radius, cy, cx + radius, cy);
    grad.addColorStop(0, '#34d399'); grad.addColorStop(0.4, '#818cf8'); grad.addColorStop(1, '#f87171');

    ctx.beginPath(); ctx.arc(cx, cy, radius, Math.PI, endAngle, false);
    ctx.strokeStyle = grad; ctx.lineWidth = lw; ctx.lineCap = 'round'; ctx.stroke();

    // Labels
    ctx.font = '500 9px Inter, sans-serif'; ctx.fillStyle = '#64748b';
    ctx.textAlign = 'left'; ctx.fillText('0', cx - radius - 4, cy + 18);
    ctx.textAlign = 'center'; ctx.fillText('50', cx, cy - radius - 8);
    ctx.textAlign = 'right'; ctx.fillText('100', cx + radius + 6, cy + 18);
})();


// ─────────────────────────────────────────────────
// Day Range Bar
// ─────────────────────────────────────────────────
(function () {
    const fill = document.getElementById('rangeFill');
    const marker = document.getElementById('rangeMarker');
    if (!fill || !marker || !window.dayHigh || !window.dayLow) return;

    const range = window.dayHigh - window.dayLow;
    if (range <= 0) return;
    const pct = Math.min(Math.max(((window.currentPrice - window.dayLow) / range) * 100, 0), 100);
    setTimeout(() => { fill.style.width = pct + '%'; marker.style.left = pct + '%'; }, 400);
})();