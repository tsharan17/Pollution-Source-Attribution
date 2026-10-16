// ---------------------------------------------------------------- STM32 view: the device's own hardware, live
// What is shown and where it comes from:
//   board activity (bus transfers, interrupts, power state, current) - the schedule model's 30 s firmware cycle
//     (stm32_twin/schedule.py) replayed at real time or slow motion; short events are stretched to be visible
//   sensor readings, features, probabilities, labels, corrected PM2.5 - the firmware twin's output for the selected
//     device at the selected moment of the scenario (one value per 30 s window)
//   cycles, memory, currents, battery - operation-count model and spec.py estimates (ranges shown)
const M32 = D.stm32;
let mcuMode = false, mcuArch = 'A4', mcuRate = 0.1, mcuT = 0, mcuLastPanel = 0;
const mScene = new THREE.Scene();
mScene.background = new THREE.Color('#e6e9e5');
const mCam = new THREE.PerspectiveCamera(36, 1, 1, 4000);
mCam.position.set(-40, 230, 250);
const mCtl = new OrbitControls(mCam, renderer.domElement);
mCtl.enabled = false; mCtl.enableDamping = true; mCtl.target.set(0, 0, 12); mCtl.minDistance = 90; mCtl.maxDistance = 700;
mCtl.maxPolarAngle = 1.45;
mScene.add(new THREE.HemisphereLight(0xffffff, 0x8f998f, 1.05));
{ const l = new THREE.DirectionalLight(0xffffff, 1.5); l.position.set(-120, 260, 160); l.castShadow = true;
  Object.assign(l.shadow.camera, {left: -220, right: 220, top: 220, bottom: -220, near: 10, far: 800}); l.shadow.mapSize.set(2048, 2048); mScene.add(l); }
const MV = (x, z, y = 0) => new THREE.Vector3(x, y, z);       // board millimetres: x right, z towards viewer, y up
function mbox(w, d, h, x, z, y0, color, opt = {}) {
  const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), new THREE.MeshStandardMaterial(Object.assign({color, roughness: .7, metalness: 0}, opt)));
  m.position.copy(MV(x, z, y0 + h / 2)); m.castShadow = true; m.receiveShadow = true; mScene.add(m); return m;
}
{ const g = new THREE.Mesh(new THREE.PlaneGeometry(900, 700), new THREE.MeshStandardMaterial({color: '#dfe3df', roughness: 1}));
  g.rotation.x = -Math.PI / 2; g.receiveShadow = true; mScene.add(g); }

// NUCLEO-64 board: 70 x 82.5 mm, ST-LINK section at the back, STM32G474RE (LQFP64) in the middle
mbox(70, 82.5, 1.6, 0, 0, 4, '#f4f4f0');
mbox(70, 1.2, 1.7, 0, -21, 4, '#c9ccc8');                     // break line between ST-LINK and target
mbox(14, 8, 3.2, 0, -38, 5.6, '#a8acaa', {metalness: .6, roughness: .35});   // USB connector
mbox(7, 7, 1.4, -12, -30, 5.6, '#2b2e2d');                    // ST-LINK MCU
const mcuChip = mbox(10, 10, 1.6, 0, 6, 5.6, '#1d2321', {emissive: '#1d2321', emissiveIntensity: .05});
for (let i = 0; i < 16; i++) for (const [dx, dz, w, d] of [[-5.8, -4.5 + i * .6, .9, .25], [5.8, -4.5 + i * .6, .9, .25], [-4.5 + i * .6, -5.8, .25, .9], [-4.5 + i * .6, 5.8, .25, .9]])
  mbox(w, d, .3, dx, 6 + dz, 5.6, '#b9bdbb', {metalness: .7, roughness: .3});
for (const sx of [-31.5, 31.5]) { mbox(5, 48, 8.5, sx, 14, 5.6, '#202322'); }            // morpho headers
for (const sx of [-24, 24]) { mbox(2.6, 40, 8.5, sx, 18, 5.6, '#2b2e2d'); }              // Arduino headers
mbox(6, 6, 2.2, 22, -10, 5.6, '#2a78d6', {roughness: .4});                               // B1 user button
mbox(6, 6, 2.2, 22, 34, 5.6, '#2b2e2d');                                                  // B2 reset
const ld2 = mbox(2, 1.4, 1, -6, -12, 5.6, '#7fd28c', {emissive: '#2bd14b', emissiveIntensity: 0});  // LD2 (green, PA5)
mbox(2, 1.4, 1, 10, -27, 5.6, '#e34948', {emissive: '#e34948', emissiveIntensity: .6});            // power LED

// sensors and peripherals
const pms = new THREE.Group(); mScene.add(pms);
{ const b = mbox(48, 37, 12, -105, -18, 0, '#b8bcbb', {metalness: .55, roughness: .35}); pms.add(b); }
const fan = new THREE.Group(); fan.position.copy(MV(-112, -18, 12.3)); mScene.add(fan);
for (let i = 0; i < 7; i++) { const bl = new THREE.Mesh(new THREE.BoxGeometry(12, .4, 3), new THREE.MeshStandardMaterial({color: '#3a3d3c'}));
  bl.rotation.y = i / 7 * Math.PI * 2; bl.position.set(Math.cos(i / 7 * Math.PI * 2) * 6, 0, -Math.sin(i / 7 * Math.PI * 2) * 6); fan.add(bl); }
{ const ring = new THREE.Mesh(new THREE.TorusGeometry(13, .8, 8, 40), new THREE.MeshStandardMaterial({color: '#7d8380'})); ring.rotation.x = Math.PI / 2; ring.position.copy(MV(-112, -18, 12.4)); mScene.add(ring); }
mbox(22, 15, 1.4, 92, 28, 0, '#5b3a8c');                                                  // BME688 breakout
const heater = mbox(3, 3, 1, 92, 26, 1.4, '#c9ccc8', {emissive: '#000000'});
mbox(26, 36, 1.4, 96, -38, 0, '#2a5fa6'); mbox(25, 25, 4, 96, -40, 1.4, '#c8b88a', {roughness: .5});   // GPS + patch antenna
const gpsLed = mbox(2, 2, 1, 105, -22, 1.4, '#e34948', {emissive: '#e34948', emissiveIntensity: 0});
mbox(26, 34, 1.4, -92, 50, 0, '#2a5fa6'); mbox(15, 12, 2.6, -92, 58, 1.4, '#9aa09d', {metalness: .6});  // microSD module
const sdLed = mbox(2, 2, 1, -82, 38, 1.4, '#f5c542', {emissive: '#f5c542', emissiveIntensity: 0});
const batt = new THREE.Mesh(new THREE.CylinderGeometry(9, 9, 65, 32), new THREE.MeshStandardMaterial({color: '#3f7d4f', roughness: .45}));
batt.rotation.z = Math.PI / 2; batt.position.copy(MV(-5, 105, 9)); batt.castShadow = true; mScene.add(batt);
const battLevel = mbox(60, 3, .6, -5, 116, 0.2, '#1baf7a');
mbox(22, 17, 1.4, 52, 100, 0, '#2a5fa6');                                                 // 5 V boost / charger

// wires: sensor pins to the Nucleo headers, coloured by bus
const WIRES = {
  pms: {col: '#eda100', pts: [[-81, -14], [-60, -12], [-40, -6], [-33.5, -2]], label: 'UART1 (9600 baud)'},
  gps: {col: '#eb6834', pts: [[83, -30], [60, -24], [42, -14], [33.5, -10]], label: 'UART3'},
  i2c: {col: '#2a78d6', pts: [[81, 26], [60, 24], [42, 18], [33.5, 16]], label: 'I2C1 (400 kHz)'},
  spi: {col: '#1baf7a', pts: [[-79, 40], [-60, 34], [-42, 26], [-33.5, 22]], label: 'SPI2 (10 MHz)'},
  pwr: {col: '#e34948', pts: [[28, 98], [20, 80], [6, 55], [0, 41]], label: '5 V / 3.3 V'},
};
const wireCurves = {};
for (const [k, w] of Object.entries(WIRES)) {
  const pts = w.pts.map(([x, z], i) => MV(x, z, 6 + (i === 0 || i === w.pts.length - 1 ? 0 : 8)));
  const curve = new THREE.CatmullRomCurve3(pts);
  for (const off of [-1.2, 1.2]) {
    const c2 = new THREE.CatmullRomCurve3(pts.map(p => p.clone().add(new THREE.Vector3(0, 0, off))));
    const m = new THREE.Mesh(new THREE.TubeGeometry(c2, 48, .55, 6, false), new THREE.MeshStandardMaterial({color: w.col, roughness: .5}));
    m.castShadow = true; mScene.add(m);
  }
  wireCurves[k] = curve;
}
// data packets travelling along the wires
const PK = 60, pkMesh = new THREE.InstancedMesh(new THREE.SphereGeometry(1.9, 10, 8), new THREE.MeshBasicMaterial({color: '#ffffff'}), PK);
const pkState = Array.from({length: PK}, () => ({wire: null, t: 2, dir: 1}));
pkMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage); mScene.add(pkMesh);
let pkNext = 0;
function sendPacket(wire, dir) { const p = pkState[pkNext]; pkNext = (pkNext + 1) % PK; p.wire = wire; p.t = 0; p.dir = dir; pkMesh.setColorAt(pkState.indexOf(p), new THREE.Color(WIRES[wire].col).offsetHSL(0, 0, .2)); }

const mTags = [
  ['STM32G474RE · Cortex-M4F', MV(0, 6, 26)], ['ST-LINK/V3E (debugger)', MV(-12, -33, 22)], ['PMS7003 particle sensor', MV(-105, -18, 34)],
  ['BME688 gas / T / RH', MV(92, 28, 18)], ['GPS', MV(96, -38, 20)], ['microSD logger', MV(-92, 50, 18)],
  ['Li-ion 2000 mAh', MV(-5, 105, 30)], ['5 V boost + charger', MV(52, 100, 16)]].map(([t, p]) => ({el: mkTag(t, 'mcu-tag'), pos: p}));
const wireTags = Object.entries(WIRES).map(([k, w]) => ({el: mkTag(w.label, 'mcu-tag src-tag'), pos: wireCurves[k].getPoint(.5).clone().setY(20)}));

// ---------------------------------------------------------------- data access
const evByArch = {}, segByArch = {};
for (const [a, v] of Object.entries(M32 ? M32.archs : {})) { evByArch[a] = v.events; segByArch[a] = v.segments; }
const BUS_OF = {'PMS7003 UART+DMA': ['pms', 1], 'GPS UART+DMA': ['gps', 1], 'BME688 I2C': ['i2c', 1], 'SD SPI+DMA': ['spi', -1], 'SD busy-wait': ['spi', -1]};
const CPU_LANES = ['PMS ISR + parse', 'GPS ISR + parse', 'I2C ISR', '1 Hz sample task', 'ADC', 'Window: features+MLP', 'FatFs CPU', 'SD busy-wait'];
function stateAt(t) {
  const sg = segByArch[mcuArch];
  let lo = 0, hi = sg.length - 1;
  while (lo < hi) { const m = (lo + hi + 1) >> 1; if (sg[m][0] <= t) lo = m; else hi = m - 1; }
  return sg[lo] ? sg[lo][2] : 1;
}
const STATE_NAME = ['Run', 'Sleep', 'Stop 1'], STATE_COL = ['#eb6834', '#2a78d6', '#1baf7a'];
const iOf = st => { const a = M32.archs[mcuArch]; return [a.i_run, a.i_sleep, a.i_stop][st]; };
const internals = {};
async function getInternals(caseIdx, k) {
  const key = caseIdx + ':' + k;
  if (internals[key]) return internals[key];
  const b64 = D.cases[caseIdx].devices[k].internals;
  const bin = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
  const buf = await new Response(new Blob([bin]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
  const a = new Int16Array(buf);
  return (internals[key] = a);
}
let mcuInt = null, mcuIntKey = '';
function intRow(s) { if (!mcuInt) return null; const r = mcuInt.subarray(s * 30, s * 30 + 30); return Array.from(r, v => v / 1000); }

// ---------------------------------------------------------------- panel
function mcuEnter() {
  layoutDev();
  const caseIdx = D.cases.indexOf(C), key = caseIdx + ':' + devSel;
  if (key !== mcuIntKey) { mcuIntKey = key; mcuInt = null; getInternals(caseIdx, devSel).then(a => { if (mcuIntKey === key) mcuInt = a; mcuLastPanel = 0; }); }
  $('mcuDev').innerHTML = devs.map((v, k) => `<option value="${k}" ${k === devSel ? 'selected' : ''}>${v.d.id}</option>`).join('');
  mcuLastPanel = 0;
}
const fmt = (x, d = 0) => Number.isFinite(x) ? x.toLocaleString(undefined, {maximumFractionDigits: d, minimumFractionDigits: d}) : '–';
const tag = k => `<span class="t-${k}" title="${{SIM: 'Produced by the firmware twin running on simulated sensor signals', MODEL: 'Exact operation counts x Cortex-M4F cycle costs', EST: 'Estimate from datasheet-typical values; range shown; not yet measured'}[k]}">${k}</span>`;
function bars(vals, names, colors, max = 1, fmtv = v => fmt(v, 2)) {
  return vals.map((v, i) => `<div class="mb"><span class="mbn">${names[i]}</span><span class="mbt"><i style="width:${Math.max(0, Math.min(100, 100 * v / max))}%;background:${colors[i] || '#2a78d6'}"></i></span><span class="mbv">${fmtv(v)}</span></div>`).join('');
}
function mcuPanel(s) {
  const dv = C.devices[devSel], A = M32.archs[mcuArch], f = intRow(s);
  const st = stateAt(mcuT), i_now = iOf(st);
  const cyc = M32.cycles[mcuArch === 'A0' ? 'v0' : 'v3'];
  const win = w => cyc[w].wcheck + cyc[w].features + cyc[w].mlp + cyc[w].post;
  const us = win('nom') / M32.spec.f_mhz, uj = us * 1e-6 * A.i_run * 1e-3 * 3.3 * 1e6;
  const sec = Math.floor(mcuT), irq = A.events.filter(e => CPU_LANES.includes(e[0]) && e[1] >= sec && e[1] < sec + 1).length;
  const p = CLS[dv.pred[s]], l10 = CLS[dv.label10 ? dv.label10[s] : dv.pred[s]];
  const sysA0 = M32.levers[0][1], sysMa = mcuArch === 'A0' ? sysA0 : sysA0 - M32.archs.A0.avg_ma[1] + M32.archs.A4.avg_ma[1];
  const usable = 2000 * 0.85, used = sysMa * tSim / 3600, left = Math.max(0, 1 - used / usable);
  let sensors = '<p class="src">Decoding this device…</p>', model = '';
  if (f) {
    const pm25 = dv.pm[s], pm1 = f[8] * (pm25 + 1) - 1, pm10 = (pm25 + 1) / Math.max(f[7], 1e-3) - 1;
    const frac = f.slice(0, 6).map(v => Math.pow(10, v));
    const rg = f.slice(13, 19).map(v => v + f[12]);
    const heatStep = M32.heater[Math.floor(mcuT) % 10];
    sensors = `<div class="mg"><div><b>${fmt(pm1)}</b><span>PM1</span></div><div><b>${fmt(pm25)}</b><span>PM2.5</span></div><div><b>${fmt(pm10)}</b><span>PM10</span></div></div>
      <div class="lbl">PMS7003 size fractions (share of counted particles) ${tag('SIM')}</div>
      ${bars(frac, ['0.3-0.5', '0.5-1', '1-2.5', '2.5-5', '5-10', '>10 µm'], [], 1, v => (100 * v).toFixed(v < .01 ? 2 : 0) + '%')}
      <div class="mg"><div><b>${fmt(f[24] * 50, 1)} °C</b><span>T (reads high: self-heating)</span></div><div><b>${fmt(f[23] * 100)} %</b><span>RH</span></div><div><b>${heatStep} °C</b><span>heater step now</span></div></div>
      <div class="lbl">BME688 gas response ln(R / clean-air R) per heater temperature ${tag('SIM')}</div>
      ${bars(rg.map(v => -v), M32.groups.map(g => g + ' °C'), rg.map(v => v < 0 ? '#eb6834' : '#2a78d6'), 2, v => fmt(-v, 2))}`;
    const probs = f.slice(25, 30);
    model = `<div class="lbl">25 features computed on the MCU ${tag('SIM')}</div>
      <div class="feat">${M32.features.map((n, i) => `<span title="${n}">${n.replace('lfrac_', 'f ').replace('gas_shape_', 'g ')}<b>${fmt(f[i], 2)}</b></span>`).join('')}</div>
      <div class="lbl">int8 MLP output (after smoothing) · gate ${Math.round(D.conf_thr * 100)}% ${tag('SIM')}</div>
      ${bars(probs, CLS.slice(0, 5).map(c => NICE[c]), CLS.slice(0, 5).map(c => COLOR[c]), 1, v => Math.round(100 * v) + '%')}`;
  }
  $('mcuPanel').innerHTML = `
    <div class="mh"><b>MCU now</b><span class="src">${mcuArch === 'A0' ? 'starter firmware (HAL, 1 ms tick)' : 'optimised firmware (tickless, DMA)'}</span></div>
    <div class="mg"><div><b>${M32.spec.f_mhz} MHz</b><span>core clock</span></div>
      <div><b style="color:${STATE_COL[st]}">${STATE_NAME[st]}</b><span>power state ${tag('EST')}</span></div>
      <div><b>${fmt(i_now, i_now < 1 ? 2 : 1)} mA</b><span>current now ${tag('EST')}</span></div>
      <div><b>${fmt(A.avg_ma[1], 2)} mA</b><span>average (${fmt(A.avg_ma[0], 1)}–${fmt(A.avg_ma[2], 1)}) ${tag('EST')}</span></div>
      <div><b>${(100 * A.run).toFixed(3)}%</b><span>time the core runs</span></div>
      <div><b>${(100 * A.stop).toFixed(0)}%</b><span>time in Stop 1</span></div></div>
    <canvas id="mcuCur" height="70"></canvas><canvas id="mcuLA" height="92"></canvas>
    <p class="src">Interrupts this second: ${irq}${A.tick === 'hal1k' ? ' + 1,000 SysTick' : ''}. Board activity replays the firmware's 30 s cycle at ${mcuRate === 1 ? 'real time' : '1/' + Math.round(1 / mcuRate) + ' speed'}; events shorter than a few ms are stretched so you can see them.</p>
    <div class="mh"><b>Sensors</b><span class="src">${devs[devSel].d.id} · ${hhmm(tSim)} · window values</span></div>${sensors}
    <div class="mh"><b>Inference (every 30 s)</b><span class="src">${fmt(win('nom') / 1000, 0)} k cycles ${tag('MODEL')} · ${fmt(us, 0)} µs · ${fmt(uj, 1)} µJ</span></div>${model}
    <div class="mg"><div><b style="color:${COLOR[p] || '#55544f'}">${NICE[p]}</b><span>source now</span></div>
      <div><b style="color:${COLOR[l10] || '#55544f'}">${NICE[l10]}</b><span>10-min source</span></div>
      <div><b>${dv.pm5 && dv.pm5[s] >= 0 ? fmt(dv.pm5[s]) : '–'}</b><span>PM2.5, 5-min corrected (${dv.pm5_lo && dv.pm5_lo[s] >= 0 ? dv.pm5_lo[s] + '–' + dv.pm5_hi[s] : '–'})</span></div></div>
    <div class="mh"><b>Memory ${tag('EST')}</b></div>
    ${bars([M32.memory.flash_total[1] / 1024, M32.memory.ram_total[1] / 1024], ['Flash', 'SRAM'], ['#4a3aa7', '#2a78d6'], 1, v => fmt(v, 0) + ' KB')
      .replace(/width:[0-9.]+%/g, (m, i) => m)}
    <p class="src">of ${M32.memory.flash_cap / 1024} KB flash and ${M32.memory.ram_cap / 1024} KB SRAM (firmware state ${fmt(M32.state_bytes / 1024, 1)} KB, model 1.8 KB)</p>
    <div class="mh"><b>Battery ${tag('EST')}</b></div>
    <div class="mg"><div><b>${Math.round(100 * left)}%</b><span>left after ${fmt(tSim / 3600, 1)} h</span></div><div><b>${fmt(sysMa)} mA</b><span>whole device (Nucleo, PMS on, GPS on)</span></div>
      <div><b>${fmt(usable / sysMa, 1)} h</b><span>on one charge</span></div></div>
    <p class="foot"><b>SIM</b> produced by the firmware twin on simulated sensor signals · <b>MODEL</b> exact operation counts × Cortex-M4F costs · <b>EST</b> datasheet-typical estimate with a range, not yet measured on a board.</p>`;
  // memory bars relative to capacity
  const mb = $('mcuPanel').querySelectorAll('.mb .mbt i');
  if (mb.length >= 2) { const n = mb.length; mb[n - 2].style.width = (100 * M32.memory.flash_total[1] / M32.memory.flash_cap) + '%'; mb[n - 1].style.width = (100 * M32.memory.ram_total[1] / M32.memory.ram_cap) + '%'; }
}
function drawMcuCanvases() {
  const cur = $('mcuCur'), la = $('mcuLA');
  if (!cur || !la) return;
  const sz = cv => { const r = cv.getBoundingClientRect(), dpr = Math.min(devicePixelRatio, 2); cv.width = Math.max(1, r.width * dpr); cv.height = Number(cv.getAttribute('height')) * dpr; const g = cv.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0); return [g, r.width, Number(cv.getAttribute('height'))]; };
  let [g, w, h] = sz(cur);
  const span = 2.0, n = 300, maxI = M32.archs[mcuArch].i_run;
  g.fillStyle = '#eef0ec'; g.fillRect(0, 0, w, h);
  g.beginPath();
  for (let k = 0; k <= n; k++) { let t = mcuT - span + span * k / n; t = ((t % 30) + 30) % 30; const i = iOf(stateAt(t)); const x = w * k / n, y = h - 4 - (h - 14) * i / maxI; k ? g.lineTo(x, y) : g.moveTo(x, y); }
  g.strokeStyle = '#2a78d6'; g.lineWidth = 1.4; g.stroke();
  g.fillStyle = '#56605b'; g.font = '10px IBM Plex Mono, monospace'; g.fillText(`current, last ${span} s of MCU time (${fmt(maxI, 0)} mA top)`, 6, 11);
  [g, w, h] = sz(la);
  const lanes = [['PMS UART', 'pms'], ['GPS UART', 'gps'], ['I2C', 'i2c'], ['SPI / SD', 'spi'], ['CPU', 'cpu']];
  const sec = Math.floor(mcuT), lh = (h - 14) / lanes.length;
  g.fillStyle = '#eef0ec'; g.fillRect(0, 0, w, h);
  g.font = '9.5px IBM Plex Mono, monospace';
  lanes.forEach(([name, key], li) => {
    const y = 12 + li * lh;
    g.fillStyle = '#8b948f'; g.fillText(name, 4, y + lh * .65);
    for (const [lane, a, b] of M32.archs[mcuArch].events) {
      const isCpu = CPU_LANES.includes(lane), bus = BUS_OF[lane];
      if (key === 'cpu' ? !isCpu : !(bus && bus[0] === key)) continue;
      if (b < sec || a >= sec + 1) continue;
      const x0 = 62 + (w - 66) * (Math.max(a, sec) - sec), x1 = 62 + (w - 66) * (Math.min(b, sec + 1) - sec);
      g.fillStyle = key === 'cpu' ? '#eb6834' : WIRES[key].col; g.fillRect(x0, y + 2, Math.max(2, x1 - x0), lh - 5);
    }
  });
  const xc = 62 + (w - 66) * (mcuT - sec);
  g.fillStyle = '#1d2321'; g.fillRect(xc - 1, 10, 2, h - 10);
  g.fillStyle = '#56605b'; g.fillText(`bus activity in second ${sec} of the 30 s cycle`, 6, 9);
}

// ---------------------------------------------------------------- per-frame update
let mcuPrevT = 0;
function mcuFrame(dt, now, s) {
  if (!M32) { renderer.render(mScene, mCam); return; }
  const prev = mcuT;
  mcuT = (mcuT + dt * mcuRate) % 30;
  const wrapped = mcuT < prev;
  // events that started since the last frame become packets
  for (const [lane, a] of M32.archs[mcuArch].events) {
    const bus = BUS_OF[lane];
    if (!bus) continue;
    const started = wrapped ? (a >= prev || a < mcuT) : (a >= prev && a < mcuT);
    if (started) sendPacket(bus[0], bus[1]);
  }
  if (Math.floor(prev) !== Math.floor(mcuT)) sendPacket('pwr', 1);
  // packets advance in wall time so they stay visible
  for (let i = 0; i < PK; i++) {
    const p = pkState[i];
    if (!p.wire || p.t > 1) { tmp.position.set(0, -50, 0); tmp.scale.setScalar(.001); }
    else { p.t += dt / 0.55; const u = p.dir > 0 ? p.t : 1 - p.t; tmp.position.copy(wireCurves[p.wire].getPoint(Math.min(1, Math.max(0, u)))); tmp.scale.setScalar(1); }
    tmp.updateMatrix(); pkMesh.setMatrixAt(i, tmp.matrix);
  }
  pkMesh.instanceMatrix.needsUpdate = true; if (pkMesh.instanceColor) pkMesh.instanceColor.needsUpdate = true;
  // chip, LEDs, fan, heater
  const st = stateAt(mcuT);
  const sysTickFlicker = M32.archs[mcuArch].tick === 'hal1k' && st === 1 && Math.sin(now / 9) > .6;
  mcuChip.material.emissive.set(sysTickFlicker ? STATE_COL[0] : STATE_COL[st]);
  mcuChip.material.emissiveIntensity = st === 0 || sysTickFlicker ? .9 : st === 1 ? .35 : .04;
  const sinceWin = ((mcuT - 29.95) % 30 + 30) % 30;
  ld2.material.emissiveIntensity = sinceWin < 0.4 * mcuRate * 10 ? 1.2 : 0;
  sdLed.material.emissiveIntensity = M32.archs[mcuArch].events.some(e => (e[0] === 'SD SPI+DMA' || e[0] === 'SD busy-wait' || e[0] === 'FatFs CPU') && mcuT >= e[1] && mcuT < e[1] + 0.25 * mcuRate * 10) ? 1.2 : 0;
  gpsLed.material.emissiveIntensity = (mcuT % 1) < 0.12 ? 1 : 0;
  fan.rotation.y += dt * 40;
  const hs = M32.heater[Math.floor(mcuT) % 10], hc = new THREE.Color().setHSL(0.13 - 0.13 * (hs - 150) / 250, 1, .5);
  heater.material.emissive.copy(hc); heater.material.emissiveIntensity = .9;
  // battery level
  const sysA0 = M32.levers[0][1], sysMa = mcuArch === 'A0' ? sysA0 : sysA0 - M32.archs.A0.avg_ma[1] + M32.archs.A4.avg_ma[1];
  const left = Math.max(0, 1 - sysMa * tSim / 3600 / 1700);
  battLevel.scale.x = Math.max(.02, left); battLevel.position.x = -5 - 30 * (1 - left);
  battLevel.material.color.setHSL(0.33 * left, .7, .45);
  mCtl.update();
  renderer.render(mScene, mCam);
  const place2 = (el, pos) => { proj.copy(pos).project(mCam); if (proj.z > 1) { el.style.display = 'none'; return; }
    el.style.display = ''; el.style.left = ((proj.x + 1) / 2 * innerWidth) + 'px'; el.style.top = ((1 - proj.y) / 2 * innerHeight) + 'px'; };
  mTags.forEach(t => place2(t.el, t.pos)); wireTags.forEach(t => place2(t.el, t.pos));
  if (now - mcuLastPanel > 250) { mcuLastPanel = now; mcuPanel(s); }
  drawMcuCanvases();
}
$('mcuArchSel').onchange = e => { mcuArch = e.target.value; mcuLastPanel = 0; };
$('mcuRateSel').onchange = e => { mcuRate = Number(e.target.value); mcuLastPanel = 0; };
$('mcuDev').onchange = e => { devSel = Number(e.target.value); mcuEnter(); };
