+++
title = "Betaflight Install"
description = "Unbox and setup your Aleph Flight Computer."
draft = false
weight = 106
sort_by = "weight"

[extra]
lead = "Install Betaflight onto your Aleph"
toc = true
top = false
order = 6
icon = ""
+++


### Betaflight Installation

Download the patched Betaflight firmware [here](https://storage.googleapis.com/elodin-releases/betaflight/4.5/betaflight_STM32H743_ALEPH_FC.elf)

{% alert(kind="notice") %}
the Betaflight patches needed to support Aleph are trivial and [open source](https://github.com/betaflight/betaflight/compare/4.5-maintenance...elodin-sys:betaflight:4.5/aleph).
{% end %}

Flash the Betaflight firmware using the following command:

```sh
probe-rs run --chip STM32H747IITx betaflight_STM32H743_ALEPH_FC.elf
```

Keep Aleph FC plugged into your computer, and visit [app.betaflight.com](http://app.betaflight.com).

{% alert(kind="notice") %}
The Betaflight web app requires a Chromium-based browser (Chrome, Chromium, Edge) because it uses Web Serial API, which is not well supported on other browsers.
{% end %}

Go to the `Options` tab and enable the `Show all serial devices` option.

<img src="/assets/betaflight-1.jpg" alt="betaflight-1"/>
<br></br>

Click on `Select your device` in the top-right corner, and choose `I can’t find my USB device`.

<img src="/assets/betaflight-2.jpg" alt="betaflight-2"/>
<br></br>

Choose `Aleph FC Debug Probe` from the list of serial port devices and click on `Connect`.

<img src="/assets/betaflight-3.jpg" alt="betaflight-3"/>
<br></br>

In some cases, you may have to click `Connect` again in the top-right corner to connect to the FC.

<img src="/assets/betaflight-4.jpg" alt="betaflight-4"/>
<br></br>

The betaflight configurator should now be fully connected to the FC.

<img src="/assets/betaflight-5.jpg" alt="betaflight-5"/>
<br></br>

### Reference Quadcopter Design

We've tested Aleph FC with Betaflight on a 7" quadcopter design.

<img src="/assets/aleph-drone.jpg" alt="aleph-drone"/>
<br></br>

If you'd like to replicate our quadcopter design, the shopping list includes:
- 4x V2306 V3.0 KV1750 motors [link](https://www.getfpv.com/t-motor-velox-v2306-v3-motor-1500kv-1750kv-1950kv-2550kv.html)
- Generic 1300 mAh 3S Battery [link](https://www.getfpv.com/lumenier-1300mah-3s-35c-lipo-battery-xt60.html)
- Generic 7" Carbon Fiber Frame [link](https://pyrodrone.com/products/hyperlite-floss-3-0-long-range-7-frame)
- Generic 3-blade propellers [link](https://www.getfpv.com/gemfan-hurricane-5536-3-blade-propeller-set-of-4.html)
- RadioMaster Pocket ELRS controller [link](https://www.getfpv.com/radiomaster-pocket-radio-cc2500-elrs-2-4ghz.html)
- RadioMaster ELRS receiver [link](https://www.amazon.com/RadioMaster-Receiver-ExpressLRS-Antenna-Transmitter/dp/B0BZY2M4BS/)
- XT60 pass-through power module [link](https://holybro.com/collections/power-modules-pdbs/products/pm02-v3-12s-power-module)
- Generic 4-in-1 30x30 ESC [link](https://www.getfpv.com/speedybee-60a-3-6s-blheli-s-4-in-1-esc-30x30.html)

<img src="/assets/aleph-drone-pins.jpg" alt="aleph-drone-pins"/>
<br></br>
