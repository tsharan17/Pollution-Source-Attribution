// ---------------------------------------------------------------- 10-minute view, about & accuracy, guided tour
let mode10 = true;
function truth10Of(dv) {                    // the same 10-minute majority rule the firmware uses, applied to the truth
  if (dv._t10) return dv._t10;
  const t = dv.truth, out = new Array(t.length), order = [4, 1, 2, 3, 0];
  for (let w = 0; w < t.length; w++) {
    const cnt = [0, 0, 0, 0, 0]; let ans = 0;
    for (let j = Math.max(0, w - 19); j <= w; j++) if (t[j] <= 4) { cnt[t[j]]++; ans++; }
    if (ans < 6) { out[w] = 8; continue; }
    let best = order[0];
    for (const c of order) if (cnt[c] > cnt[best]) best = c;
    out[w] = 2 * cnt[best] >= ans ? best : 8;
  }
  return (dv._t10 = out);
}
const shownLabel = (dv, s) => (mode10 && dv.label10) ? dv.label10[s] : dv.pred[s];
const shownTruth = (dv, s) => mode10 ? truth10Of(dv)[s] : dv.truth[s];
const pmText = (dv, s) => (dv.pm5 && dv.pm5[s] >= 0) ? `${dv.pm5[s]} µg/m³ (${dv.pm5_lo[s]}–${dv.pm5_hi[s]})` : '–';
$('mode10').onchange = () => { mode10 = $('mode10').checked; drawStrip(); lastStep = -1; lastPanel = -1; devDrawnFor = ''; };

const pc = x => Number.isFinite(x) ? Math.round(100 * x) + '%' : '–';
const f1 = x => Number.isFinite(x) ? x.toFixed(1) : '–';
function openAbout() {
  const A = D.accuracy;
  const row = (name, a) => `<tr><td>${name}</td><td class="n">${pc(a.instant_correct)} / ${pc(a.instant_answered)}</td><td class="n">${pc(a.ten_correct)} / ${pc(a.ten_answered)}</td>
    <td class="n">${f1(a.changes_per_h)} · ${f1(a.changes10_per_h)} · ${f1(a.truth_changes_per_h)}</td><td class="n">${pc(a.pm_raw_med)} → ${pc(a.pm5_med)}</td>
    <td class="n">${pc(a.pm_raw_p90)} → ${pc(a.pm5_p90)}</td><td class="n">${pc(a.band_coverage)}</td></tr>`;
  $('aboutBox').innerHTML = `<h2>About this simulation <button class="x" id="aboutX" aria-label="Close">×</button></h2>
  <p>Every value on screen comes from running the PolSo firmware on simulated air. The simulation models the physics of both sensors and the
  firmware exactly as it would run on the STM32. <b>Nothing here is measured on real hardware yet</b>, so treat these numbers as what the design
  can achieve under the model's assumptions. Expect real accuracy to be lower until the device is checked against reference instruments.</p>
  <h3>Accuracy measured in this simulation</h3>
  <div class="scroll"><table><tr><th>Case</th><th>Source right when answering / share answered (instant)</th><th>Same, 10-min source</th>
  <th>Source changes per hour: instant · 10-min · real air</th><th>PM2.5 median error: raw → corrected 5-min</th><th>90th percentile error</th><th>True PM inside the band (target 68 %)</th></tr>
  ${D.cases.map(c => row(c.name, c.accuracy)).join('')}${row('<b>All cases</b>', A)}</table></div>
  <p class="src">The 10-minute source is scored against the true dominant source over the same 10 minutes. The PM2.5 band is set on held-out data
  so that about 68 % of true values should fall inside it; the last column shows whether that holds here.</p>
  <h3>What is modelled</h3><ul class="tight">
  <li><b>Air:</b> weather, regional haze, plumes from roads, stacks, fires and construction that drift with the wind.</li>
  <li><b>PMS7003:</b> laser counting with real size limits, swelling of particles in humid air (the sensor warms its sample a little),
  counting noise, poor response to coarse dust, unit-to-unit spread, the real 32-byte UART frames.</li>
  <li><b>BME688:</b> metal-oxide chemistry with a 10-step heater scan, strong humidity sensitivity, slow drift and unit spread.</li>
  <li><b>Firmware:</b> the same C code as for the board (checked identical to the simulation on every window). That includes the 60 s windows, 25 features,
  the int8 neural network, the confidence gate, the 10-minute source and the PM2.5 correction.</li></ul>
  <h3>Estimated, not measured</h3><ul class="tight"><li>STM32 currents, cycle counts and memory use, SD-card timing, battery life (STM32 tab: tagged EST / MODEL).</li>
  <li>The sensor-physics parameters, which come from datasheets and published studies. They need re-fitting with the real sensors.</li></ul>
  <h3>Limits of the hardware that no software can remove</h3><ul class="tight">
  <li>The PMS7003 cannot see particles smaller than about 0.3 µm, which is most fresh vehicle soot. Traffic PM2.5 therefore carries the widest band.</li>
  <li>It under-counts coarse dust, so dust storms read low.</li>
  <li>The BME688 is not selective: it says something reacted, not which gas. Sources are told apart only through the combination with particle size.</li>
  <li>Sources the model never saw (cooking, incense, trash burning) are often given the nearest known label.</li></ul>
  <h3>Not modelled</h3><ul class="tight"><li>Rain, real enclosure airflow, wildlife and vandalism, years of ageing, real calibration drift.</li></ul>`;
  $('about').hidden = false;
  $('aboutX').onclick = () => { $('about').hidden = true; };
}
$('aboutBtn').onclick = openAbout;
$('about').onclick = e => { if (e.target.id === 'about') $('about').hidden = true; };

// ---- guided tour (for presenting) ----
function setPlay(b) { playing = b; $('play').textContent = b ? 'Pause' : 'Play'; }
function busiest(caseIdx, cls) {
  const cs = D.cases[caseIdx].devices; let best = 0, bv = -1;
  cs.forEach((dv, k) => { const v = (dv.label10 || dv.pred).filter(x => x === cls).length; if (v > bv) { bv = v; best = k; } });
  return best;
}
const TOUR = [
  {t: 'What you are looking at', p: 'A 2 × 2 km town with 16 PolSo devices. Each one is an STM32 NUCLEO-G474RE with a PMS7003 particle sensor and a BME688 gas sensor. ' +
     'The air, the sensors and the device firmware are all simulated, and the firmware is the same code that will run on the board.',
   go: () => { setView('map'); loadCase(0); tSim = 2 * 3600; setPlay(true); }},
  {t: 'Reading the map', p: 'The glowing ball on each device is what it reports: the main pollution source over the last 10 minutes. The ring on the ground is what is really there. ' +
     '✓ means right, ✗ wrong, ? not sure, ⚠ sensor fault. Smoke and dust plumes are coloured by source.', go: () => {}},
  {t: 'Crop-burning season', p: 'A westerly wind carries stubble-fire smoke over the town. Watch the devices downwind turn orange (biomass burning) while the others stay green (clean background).',
   go: () => { loadCase(2); tSim = 1.6 * 3600; setPlay(true); }},
  {t: 'One device in detail', p: 'The Devices tab shows one device over the whole case: what it reported against reality, how confident it was, and its PM2.5. That last chart has the raw sensor, the corrected 5-minute value with its band, and the true value.',
   go: () => { devSel = busiest(2, 1); setView('dev'); }},
  {t: 'Inside the device', p: 'The STM32 tab is the board itself. Moving dots are data on the wires: particle-sensor frames, GPS, I²C reads of the gas sensor, SD-card writes. ' +
     'The chip colour is its power state. The panel shows live readings, the 25 features, the neural-network output and memory, each tagged as simulated (SIM), modelled (MODEL) or estimated (EST).',
   go: () => { setView('mcu'); mcuArch = 'A4'; $('mcuArchSel').value = 'A4'; mcuRate = 0.1; $('mcuRateSel').value = '0.1'; }},
  {t: 'Firmware design matters', p: () => `With a typical starter firmware the core is woken every millisecond and never reaches deep sleep. It averages ${M32 ? M32.archs.A0.avg_ma[1].toFixed(1) : '8'} mA, against ${M32 ? M32.archs.A4.avg_ma[1].toFixed(1) : '0.9'} mA with the optimised design. Same algorithm, same answers.`,
   go: () => { mcuArch = 'A0'; $('mcuArchSel').value = 'A0'; }},
  {t: 'When things break', p: 'In this case four devices develop faults. P06 loses its particle sensor at 11:00 and reports a sensor fault (⚠) instead of guessing. P10’s fan stalls; that one the firmware cannot detect yet, and it is listed as a known gap.',
   go: () => { setView('map'); loadCase(5); tSim = 3.3 * 3600; setPlay(true); }},
  {t: 'How accurate is it?', p: () => `Across all cases the 10-minute source was right ${pc(D.accuracy.ten_correct)} of the time when it answered (${pc(D.accuracy.ten_answered)} of the time). The corrected 5-minute PM2.5 had a median error of ${pc(D.accuracy.pm5_med)} (raw sensor: ${pc(D.accuracy.pm_raw_med)}). These are simulation results; the next step is checking them against reference instruments.`,
   go: () => { openAbout(); }},
];
let tourI = -1;
function tourShow(i) {
  if (i >= TOUR.length) return tourEnd();
  tourI = Math.max(0, i);
  const st = TOUR[tourI];
  if (tourI !== TOUR.length - 1) $('about').hidden = true;
  st.go();
  $('tour').innerHTML = `<h3>${tourI + 1} / ${TOUR.length} · ${st.t}</h3><p>${typeof st.p === 'function' ? st.p() : st.p}</p>
    <div class="row2"><button class="btn" id="tPrev" ${tourI ? '' : 'disabled'}>Back</button><button class="btn" id="tEnd">End tour</button>
    <button class="btn primary" id="tNext">${tourI === TOUR.length - 1 ? 'Finish' : 'Next'}</button></div>`;
  $('tour').hidden = false;
  $('tPrev').onclick = () => tourShow(tourI - 1);
  $('tNext').onclick = () => tourShow(tourI + 1);
  $('tEnd').onclick = tourEnd;
}
function tourEnd() { $('tour').hidden = true; tourI = -1; }
$('tourBtn').onclick = () => tourShow(0);
addEventListener('keydown', e => {
  if (tourI < 0) return;
  if (e.key === 'ArrowRight') tourShow(tourI + 1);
  else if (e.key === 'ArrowLeft') tourShow(tourI - 1);
  else if (e.key === 'Escape') tourEnd();
});
