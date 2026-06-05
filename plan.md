Implementation Plan: Support AMD GPU Droplet (Device ID 74b5) in AM Driver
This plan outlines the changes required to improve the user-space AM driver to support the virtualized Virtual Function (VF) variant of the Instinct MI300X GPU (Device ID 74b5) running inside a droplet/virtual machine.

Because virtualized/VF devices run under the control of a host hypervisor (Physical Function / PF), they cannot perform host-only actions such as mode1 GPU resets, Platform Security Processor (PSP) firmware loading, System Management Unit (SMU) clock/power configurations, or global GMC/MM hub setup. Attempting these writes on a VF can result in VM faults or hangs.

The proposed modifications will detect VF mode and selectively bypass these PF-only operations, allowing the AM driver to initialize and use the graphics and SDMA queues assigned to the VF.