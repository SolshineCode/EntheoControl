# Show day checklist

One page for a real performance. Details are in the [reference manual](../QUICKSTART.md).

## Pack
- [ ] Laptop **stock charger** (or a higher-watt one), plus an extension lead
- [ ] Projector, its power cable, HDMI cable (active cable for long runs), spare HDMI
- [ ] USB audio interface and cables, if taking a line from the DJ
- [ ] MIDI controller / gamepad, USB webcam and mount (if mapping with AUTO MAP or using the crowd camera)
- [ ] Phone (panel, and hotspot if the venue Wi-Fi isolates devices)
- [ ] USB stick with the show exported as `.entheo` (SHOW > EXPORT, "collect media" ticked)
- [ ] Gaffer tape, something to raise the laptop for airflow

## Power and Windows (before doors)
- [ ] Charger plugged into the wall **directly** (not via a dock or USB-C hub)
- [ ] Power Options: Maximum processor state **99 %** on AC
- [ ] NVIDIA, for `python.exe`: Max Frame Rate **60**, WhisperMode **off**, Threaded optimization **off**, high-performance GPU
- [ ] Notifications off (Focus / Do not disturb), updates paused, other heavy apps closed
- [ ] Laptop raised, vents clear

## Projector
- [ ] Win + P > **Extend**; projector at native resolution, 100 % scaling
- [ ] Auto keystone **off**, keystone 0
- [ ] `python test_pattern.py --monitor 1`: focus sharp, circles round
- [ ] `run_show.ps1` running; picture on the projector, panel open

## Mapping
- [ ] MAPPING: show **ALL** + **TEST GRID**; every surface sits on its object
- [ ] Overlay **OFF** again
- [ ] If AUTO MAP: calibrate **before** the audience arrives (patterns flash on the wall)

## Audio
- [ ] The three meters move with the music (laptop loopback by default; `--audio mic` for the room)
- [ ] TAP the tempo, or set the clock to MIDI / LINK / AUTO

## Safety and load
- [ ] **FLASH GUARD green** (on). If you override it, warn the audience, and make sure there is a photosensitivity notice at the entrance.
- [ ] CONTROLLERS > Health: governor **SUGGEST** (or AUTO if unattended)
- [ ] Power profile **BALANCED** (on battery: set the **Show length**)
- [ ] SAFE scene set (SCENES > SET SAFE…) to a look you are happy to land on
- [ ] PREVIEW **off** unless you need it (saves GPU)
- [ ] CAMERA and ONLINE **off** unless you need them. Do not type API keys over venue Wi-Fi.

## Backups
- [ ] Show saved (**Ctrl+S**), and the `.entheo` copy on the USB stick
- [ ] REC set up if you want a recording (CONTROLLERS > Record), and enough free disk (it stops under about 0.5 GB)

## PANIC muscle memory
- [ ] Hold **PANIC** for half a second, or hold **Esc**, three times now. It lands on the SAFE scene with every trigger off.

## If this happens...
| Problem | Do this |
|---|---|
| Something on the wall looks wrong or too intense | **Hold PANIC** (or hold Esc). Then fix it calmly. |
| Need darkness now | Hold **BLACKOUT**, or pull Master down |
| Picture froze or vanished | Wait a second: the watchdog restarts the show with the same look. Still black? Check Master, BLACKOUT, ALL FX and the tab power buttons. |
| Projector cable knocked out | Plug it back in: rendering pauses and resumes by itself |
| Health meter amber or red, stutter | Press **Apply** on the suggestion, lower the heaviest layer, switch PREVIEW off |
| Battery falling while plugged in | Power profile **BATTERY SAVER**; check the charger is the stock one and plugged in directly |
| Phone panel lost connection | Re-open the same URL; it reconnects. On venue Wi-Fi, switch to your phone's hotspot. |
| Visuals stopped reacting | Check the meters and the laptop's audio output device |
| Laptop crashed completely | Restart, run `run_show.ps1`, then SHOW > **RECOVER** (or open the backup) |
| Quitting at the end | **Shift+Esc** in the projector window |
