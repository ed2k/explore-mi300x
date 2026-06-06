SR-IOV VFs block direct MMIO writes to the CP MQD queue management registers (such as 
  regCP_MQD_BASE_ADDR ). When writing to these registers, the writes are silently dropped by the hardware,
  preventing the compute queues from being mapped or activated. Without KFD ( /dev/kfd ) or PSP/KIQ queue
  activation implemented, the bare-metal driver cannot launch queues on a VF
