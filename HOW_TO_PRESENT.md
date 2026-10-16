# How to start and present the PolSo simulation

## Start it

1. Double-click **`START_SIMULATION.bat`** in this folder. The simulation opens in your browser.
   (Or open `environment_simulation/results/area_viewer.html` directly, or use the shared claude.ai link.)
2. You need internet the first time it loads; it downloads the 3D library from a CDN.
3. Press **Guided tour** (top bar). Use **Next** or the right arrow key to step through, **Esc** to stop.

That is all you need for a presentation. Nothing has to be installed or run in Python.

## The screen in 20 seconds

| Where | What it is |
|---|---|
| Row under the title | Six situations: normal weekday, winter smog, crop burning, dust storm, monsoon fog, industrial upset with sensor faults |
| **3D map / Devices / STM32** | Three views of the same running simulation |
| **Build** | The construction sheet for the real device: parts, pins, settings, firmware constants, steps. Also in `BUILD_GUIDE.md` |
| Bottom bar | Play / pause, speed, time slider; the coloured strip shows when devices were right (green) or wrong (red) |
| **10-min source** tick box | On: devices show the main source over the last 10 minutes (steady, what a user would see). Off: the instant 30-second label |
| **About & accuracy** | The measured accuracy, how the PM2.5 correction was chosen, what is modelled and what is not |

## A 5-minute talk (follows the guided tour)

1. **The setting** (map, Weekday). "Sixteen PolSo devices across a 2 km town. Each is an STM32 NUCLEO-G474RE with a PMS7003 particle
   sensor and a BME688 gas sensor. Air, sensors and firmware are simulated; the firmware is the same C code that will run on the board."
2. **Reading the map.** "The ball on each device is what it reports. The ring on the ground is the truth. Ticks and crosses
   show right and wrong, and the plumes are coloured by source."
3. **Crop burning.** "Smoke drifts east with the wind, and the devices downwind switch to biomass burning."
4. **One device** (Devices tab). "Here is one device over the whole case: its answers against the truth, and its confidence. The PM2.5 chart shows
   the raw sensor, the corrected 5-minute value with its uncertainty band, and the true value."
5. **Inside the device** (STM32 tab). "This is the board. The moving dots are data on each wire, and the chip colour is its power state. On the right
   are the live readings, the 25 features, the neural-network output, memory and battery. Every number is tagged as simulated (SIM),
   modelled (MODEL) or estimated (EST)."
6. **Firmware design.** Switch Firmware to *Starter*. "A typical starter firmware never deep-sleeps: about 8 mA instead of 0.9 mA, for the same answers."
7. **Faults** (Industrial upset, around 11:00). "P06 loses its particle sensor and says *sensor fault* instead of guessing. P10's fan stalls,
   and that one we cannot detect yet: it is on our to-do list."
8. **Honest numbers** (About & accuracy). Read the "All cases" row and the note under it.
9. **Ready to build** (Build tab). "Everything we need to build it is generated from the same files the simulation ran on:
   parts, pins, clock settings, every firmware constant, and the bench tests that check the model against the real board."
   Point at the coloured tags: PROP means a proposal we still check in CubeMX, EST means a number we measure first.

## Questions you will be asked, and honest answers

- **"Is this real data?"** No. It is a physics-based simulation of the sensors plus the real firmware. The numbers show what the
  design can do under the model's assumptions. The next step is to check them against a reference monitor.
- **"How accurate is PM2.5?"** About 25-30 % median error per 5-minute value in these scenarios. The device corrects for humidity and source, which
  cuts the worst errors, but a correction learned in one place only partly carries over to another. Each site needs a short co-location
  with a reference monitor before the PM2.5 numbers can be called accurate.
- **"How accurate is the source?"** In these scenarios the 10-minute source was right about 95 % of the time when the device answered, and it answered about 88 % of
  the time. It says *uncertain* or *mixed* rather than guessing.
- **"Why do the labels change?"** Because the air really does change: plumes swing with the wind. The 10-minute view changes
  about as often as the true 10-minute source does.
- **"What can't it do?"** It cannot see soot particles smaller than 0.3 µm (most fresh traffic soot) or tell individual gases apart, it reads
  coarse dust low, and it may give sources it never saw (cooking, incense) the nearest known label.

## Rebuilding after changes (for the team)

```
cd environment_simulation
python run_all.py          # train + 16 tests          (~12 min)
python run_mcu.py          # STM32 numbers             (~10 min)
python run_area.py         # the 16-device map         (~4 min)
python viewer/build_viewer.py   # also regenerates BUILD_GUIDE.md
python -m unittest discover tests
```
