// dap.c — Aleph Flight Controller debug probe: CMSIS-DAP 2.1.2 (SWD) for the STM32H747 plus
// two CDC-ACM consoles on its UART3 and UART6. Derived from Seward's firmware.
//
// Deliberate limits:
//   - CMSIS-DAP v2 bulk transport only; no HID compatibility interface.
//   - SWD only, bit-banged, 2 MHz top rate.
//   - No JTAG, SWO, DAP UART, timestamp, or atomic-command capability claims;
//     the consoles are separate CDC-ACM functions.

#include "core.h"

#include "arch.h"
#include "boot.h"
#include "bsp.h"
#include "cdc_uart.h"
#include "cmsis_dap.h"
#include "dap_usb.h"
#include "hal.h"
#include "rgb_led.h"
#include "rtt.h"
#include "usb_dfu.h"
#include "usbd.h"

#ifndef PROBE_VERSION
#define PROBE_VERSION "0.0.0-dev"
#endif

extern U32 _estack;
extern U32 _sidata;
extern U32 _sdata;
extern U32 _edata;
extern U32 _sbss;
extern U32 _ebss;

enum {
    DapInterface = 0,
    DfuInterface = 1,
    Console0Interface = 2,  // + data interface 3
    Console1Interface = 4,  // + data interface 5
    UsbInterfaceCount = 6,
    DapEndpointOut = UsbdEpAddr_Out(1),
    DapEndpointIn = UsbdEpAddr_In(1),
    DapEndpointPacketSize = 64,
    Console0NotifyEndpoint = 2,
    Console0DataEndpoint = 3,
    Console1NotifyEndpoint = 4,
    Console1DataEndpoint = 5,
    ActivityUs = 20000,
    HeartbeatOnUs = 100000,
    HeartbeatOffUs = 900000,
    UartIrqPriority = 0,  // USART4/5 have no FIFO; outrank the USB IRQ
    WatchdogMs = 3000,
    DfuDetachDelayUs = 50000, // lets the DETACH status stage finish
    DfuTransferSize = 1024,   // what the ROM reports

    MsOs20VendorCode = 0x20,
    MsOs20DescriptorIndex = 0x0007,

    // DAP_Vendor1: request[1] = 1 drives STM32H747 BOOT0 high, 0 releases it.
    DapVendor_Boot0 = CmsisDapCommand_Vendor0 + 1,
};

static_assert((U32)DapEndpointPacketSize == (U32)CmsisDapPacketSize);

static Usbd_DeviceDescriptor const usb_device_descriptor = {
    .length = sizeof(Usbd_DeviceDescriptor),
    .descriptor_type = UsbdDescriptorType_Device,
    .usb_bcd = 0x0210,
    .device_class = 0xEF,     // miscellaneous: interface association
    .device_subclass = 0x02,
    .device_protocol = 0x01,
    .ep0_max_packet = 64,
    .vendor_id = 0x1209,
    .product_id = 0x0001,
    .device_bcd = 0x0100,
    .manufacturer_string = 1,
    .product_string = 2,
    .serial_string = 3,
    .configuration_count = 1,
};

typedef struct __attribute__((packed)) {
    Usbd_ConfigurationDescriptor configuration;
    Usbd_InterfaceDescriptor dap_interface;
    Usbd_EndpointDescriptor dap_out;
    Usbd_EndpointDescriptor dap_in;
    Usbd_InterfaceDescriptor dfu_interface;
    UsbDfu_FunctionalDescriptor dfu_functional;
    CdcUart_FunctionDescriptor console0;
    CdcUart_FunctionDescriptor console1;
} UsbConfigurationDescriptor;

static_assert(sizeof(UsbConfigurationDescriptor) == 50 + 2 * 66);

static UsbConfigurationDescriptor const usb_configuration_descriptor = {
    .configuration = {
        .length = sizeof(Usbd_ConfigurationDescriptor),
        .descriptor_type = UsbdDescriptorType_Configuration,
        .total_length = sizeof(UsbConfigurationDescriptor),
        .interface_count = UsbInterfaceCount,
        .configuration_value = 1,
        .configuration_string = 0,
        .attributes = 0x80,
        .max_power_2ma = 50,
    },
    .dap_interface = {
        .length = sizeof(Usbd_InterfaceDescriptor),
        .descriptor_type = UsbdDescriptorType_Interface,
        .interface_number = DapInterface,
        .alternate_setting = 0,
        .endpoint_count = 2,
        .interface_class = 0xFF,
        .interface_subclass = 0x00,
        .interface_protocol = 0x00,
        .interface_string = 4,
    },
    .dap_out = {
        .length = sizeof(Usbd_EndpointDescriptor),
        .descriptor_type = UsbdDescriptorType_Endpoint,
        .endpoint_address = DapEndpointOut,
        .attributes = 0x02,
        .max_packet_size = DapEndpointPacketSize,
        .interval = 0,
    },
    .dap_in = {
        .length = sizeof(Usbd_EndpointDescriptor),
        .descriptor_type = UsbdDescriptorType_Endpoint,
        .endpoint_address = DapEndpointIn,
        .attributes = 0x02,
        .max_packet_size = DapEndpointPacketSize,
        .interval = 0,
    },
    .dfu_interface = {
        .length = sizeof(Usbd_InterfaceDescriptor),
        .descriptor_type = UsbdDescriptorType_Interface,
        .interface_number = DfuInterface,
        .alternate_setting = 0,
        .endpoint_count = 0,
        .interface_class = UsbDfu_InterfaceClass,
        .interface_subclass = UsbDfu_InterfaceSubclass,
        .interface_protocol = UsbDfu_Protocol_Runtime,
        .interface_string = 5,
    },
    .dfu_functional = {
        .length = sizeof(UsbDfu_FunctionalDescriptor),
        .descriptor_type = UsbDfu_DescriptorType_Functional,
        .attributes = UsbDfu_Attribute_WillDetach | UsbDfu_Attribute_CanDownload,
        .detach_timeout_ms = 1000,
        .transfer_size = DfuTransferSize,
        .dfu_version = UsbDfu_Version,
    },
    .console0 = CDC_UART_FUNCTION_DESCRIPTOR(Console0Interface, Console0NotifyEndpoint,
                                             Console0DataEndpoint, 6),
    .console1 = CDC_UART_FUNCTION_DESCRIPTOR(Console1Interface, Console1NotifyEndpoint,
                                             Console1DataEndpoint, 7),
};

// Microsoft OS 2.0: WINUSB on both interfaces so libusb (dfu-util) can open
// the DFU one on Windows; Arm's CMSIS-DAP DeviceInterfaceGUIDs on interface 0.
typedef struct __attribute__((packed)) {
    U16 set_header_length;
    U16 set_header_type;
    U32 windows_version;
    U16 set_total_length;

    U16 configuration_header_length;
    U16 configuration_header_type;
    U8 configuration_value;
    U8 configuration_reserved;
    U16 configuration_total_length;

    U16 function_header_length;
    U16 function_header_type;
    U8 first_interface;
    U8 function_reserved;
    U16 function_total_length;

    U16 compatible_id_length;
    U16 compatible_id_type;
    U8 compatible_id[8];
    U8 sub_compatible_id[8];

    U16 property_length;
    U16 property_type;
    U16 property_data_type;
    U16 property_name_length;
    U16 property_name[21];
    U16 property_data_length;
    U16 property_data[40];

    U16 dfu_function_header_length;
    U16 dfu_function_header_type;
    U8 dfu_first_interface;
    U8 dfu_function_reserved;
    U16 dfu_function_total_length;

    U16 dfu_compatible_id_length;
    U16 dfu_compatible_id_type;
    U8 dfu_compatible_id[8];
    U8 dfu_sub_compatible_id[8];
} MsOs20DescriptorSet;

// Set header 10, configuration header 8, DAP function 160, DFU function 28.
static_assert(sizeof(MsOs20DescriptorSet) == 10 + 8 + 160 + 28);

static MsOs20DescriptorSet const ms_os_20_descriptor = {
    .set_header_length = 10,
    .set_header_type = 0,
    .windows_version = 0x06030000,
    .set_total_length = sizeof(MsOs20DescriptorSet),

    .configuration_header_length = 8,
    .configuration_header_type = 1,
    .configuration_value = 0,
    .configuration_total_length = sizeof(MsOs20DescriptorSet) - 10,

    .function_header_length = 8,
    .function_header_type = 2,
    .first_interface = DapInterface,
    .function_total_length = 160,

    .compatible_id_length = 20,
    .compatible_id_type = 3,
    .compatible_id = {'W', 'I', 'N', 'U', 'S', 'B'},

    .property_length = 132,
    .property_type = 4,
    .property_data_type = 7,
    .property_name_length = 42,
    .property_name = {
        'D', 'e', 'v', 'i', 'c', 'e', 'I', 'n', 't', 'e', 'r', 'f', 'a',
        'c', 'e', 'G', 'U', 'I', 'D', 's', 0,
    },
    .property_data_length = 80,
    .property_data = {
        '{', 'C', 'D', 'B', '3', 'B', '5', 'A', 'D', '-', '2', '9', '3',
        'B', '-', '4', '6', '6', '3', '-', 'A', 'A', '3', '6', '-', '1',
        'A', 'A', 'E', '4', '6', '4', '6', '3', '7', '7', '6', '}', 0, 0,
    },

    .dfu_function_header_length = 8,
    .dfu_function_header_type = 2,
    .dfu_first_interface = DfuInterface,
    .dfu_function_total_length = 28,

    .dfu_compatible_id_length = 20,
    .dfu_compatible_id_type = 3,
    .dfu_compatible_id = {'W', 'I', 'N', 'U', 'S', 'B'},
};

// BOS: Microsoft OS 2.0 (WinUSB binding). CDC-ACM binds usbser by class.
typedef struct __attribute__((packed)) {
    U8 length;
    U8 descriptor_type;
    U16 total_length;
    U8 capability_count;

    U8 ms_length;
    U8 ms_descriptor_type;
    U8 ms_capability_type;
    U8 ms_reserved;
    U8 ms_platform_uuid[16];
    U32 ms_windows_version;
    U16 ms_descriptor_set_length;
    U8 ms_vendor_code;
    U8 ms_alternate_enumeration_code;
} UsbBosDescriptor;

static_assert(sizeof(UsbBosDescriptor) == 5 + 28);

static UsbBosDescriptor const usb_bos_descriptor = {
    .length = 5,
    .descriptor_type = UsbdDescriptorType_Bos,
    .total_length = sizeof(UsbBosDescriptor),
    .capability_count = 1,

    .ms_length = 28,
    .ms_descriptor_type = UsbdDescriptorType_DeviceCapability,
    .ms_capability_type = 0x05,
    .ms_platform_uuid = {
        0xDF, 0x60, 0xDD, 0xD8, 0x89, 0x45, 0xC7, 0x4C,
        0x9C, 0xD2, 0x65, 0x9D, 0x9E, 0x64, 0x8A, 0x9F,
    },
    .ms_windows_version = 0x06030000,
    .ms_descriptor_set_length = sizeof(ms_os_20_descriptor),
    .ms_vendor_code = MsOs20VendorCode,
};

static char usb_serial[17];
static char const *usb_strings[] = {
    "Elodin",
    "Aleph FC CMSIS-DAP",
    usb_serial,
    "CMSIS-DAP v2",
    "DFU runtime",
    "Aleph FC UART3",
    "Aleph FC UART6",
};

static Usbd_DescriptorSet const usb_descriptors = {
    .device = UsbdBytes(usb_device_descriptor),
    .configuration = UsbdBytes(usb_configuration_descriptor),
    .bos = UsbdBytes(usb_bos_descriptor),
    .microsoft_os_20 = UsbdBytes(ms_os_20_descriptor),
    .strings = usb_strings,
    .microsoft_descriptor_index = MsOs20DescriptorIndex,
    .language_id = 0x0409,
    .microsoft_vendor_code = MsOs20VendorCode,
    .string_count = ArrayCount(usb_strings),
};

static Bsp_Signal led_ind;
static Bsp_Signal target_boot0;
static RgbLed rgb;
static Bsp_SwdPort swd_bsp;
static Bsp_UsbDevice usb_bsp;
static CmsisDap cmsis_dap;
static CdcUart consoles[2];
static Usbd_Device usb_device;
static UsbDfu_Runtime usb_dfu;
// Twice the advertised count: a host that overruns it still drains.
enum { DapSlotCount = 2 * CmsisDapPacketCount };
static U32 dap_out_slots[USBD_SLOT_WORDS(DapEndpointPacketSize) * DapSlotCount];
static U32 dap_in_slots[USBD_SLOT_WORDS(DapEndpointPacketSize) * DapSlotCount];
static U32 faulting;
static U32 boot_crash_count;
static U32 heartbeat_deadline_us;
static U32 activity_deadline_us;
static U32 console_activity_seen;
static U32 usb_lifecycle_seen;
static Usbd_State usb_state_seen;
static bool heartbeat_on;
static bool activity;

static void board_check(bool ok);

static void serial_from_uid(char out[17]) {
    volatile U32 const *uid = (volatile U32 const *)0x1FFF7590UL;
    U32 hash_a = 2166136261u;
    U32 hash_b = 0x9E3779B9u;
    for (U32 word = 0u; word < 3u; word++) {
        U32 value = uid[word];
        for (U32 byte = 0u; byte < 4u; byte++) {
            U8 octet = (U8)(value >> (byte * 8u));
            hash_a = (hash_a ^ octet) * 16777619u;
            hash_b ^= (U32)octet + 0x9E3779B9u + (hash_b << 6) + (hash_b >> 2);
        }
    }

    static char const hex[] = "0123456789ABCDEF";
    for (U32 i = 0u; i < 8u; i++) {
        U32 shift = (7u - i) * 4u;
        out[i] = hex[(hash_a >> shift) & 0xFu];
        out[i + 8u] = hex[(hash_b >> shift) & 0xFu];
    }
    out[16] = 0;
}

static Usbd_Hardware usb_hardware_from_bsp(Bsp_UsbDevice const *usb) {
    U32 flags = 0u;
    if ((usb->flags & BspUsb_Pma32Bit) != 0u) flags |= UsbdHardware_Pma32Bit;

    return (Usbd_Hardware){
        .reg_base = usb->reg_base,
        .pma_base = usb->pma_base,
        .pma_bytes = usb->pma_bytes,
        .endpoint_count = usb->endpoint_count,
        .irq = usb->irq,
        .usb_hz = usb->usb_hz,
        .flags = flags,
    };
}

// The response slot is reserved before a command runs, so only a lost
// session ends a transfer early.
static bool dap_abort_poll(void *context) {
    (void)context;
    return usbd_lifecycle(&usb_device) != usb_lifecycle_seen ||
           usbd_state(&usb_device) != UsbdState_Configured;
}

static bool usb_control(Usbd_Device *dev, Usbd_SetupPacket const *setup, void *context) {
    if (usb_dfu_runtime_control(dev, setup, context)) return true;
    for (U32 i = 0u; i < ArrayCount(consoles); i++) {
        if (cdc_uart_control(&consoles[i], dev, setup)) return true;
    }
    return false;
}

static bool usb_control_data(Usbd_Device *dev, Usbd_SetupPacket const *setup,
                             U8 const *data, U16 size, void *context) {
    (void)dev;
    (void)context;
    for (U32 i = 0u; i < ArrayCount(consoles); i++) {
        if (cdc_uart_control_data(&consoles[i], setup, data, size)) return true;
    }
    return false;
}

// DAP_TransferAbort takes effect on arrival and is never queued, as in Arm's
// reference implementation.
static bool dap_packet_filter(void *context, U8 const *data, U16 size) {
    (void)context;
    (void)size;
    if (data[0] != CmsisDapCommand_TransferAbort) return true;
    cmsis_dap_abort(&cmsis_dap);
    return false;
}

static void usb_open_dap_endpoints(void) {
    if (!usbd_ep_opened(&usb_device, DapEndpointOut)) {
        board_check(usbd_ep_open(&usb_device, &(Usbd_EpConfig){
            .addr = DapEndpointOut,
            .type = UsbdEpType_Bulk,
            .max_packet = DapEndpointPacketSize,
            .slots = dap_out_slots,
            .slot_count = DapSlotCount,
            .filter = dap_packet_filter,
        }));
    }
    if (!usbd_ep_opened(&usb_device, DapEndpointIn)) {
        board_check(usbd_ep_open(&usb_device, &(Usbd_EpConfig){
            .addr = DapEndpointIn,
            .type = UsbdEpType_Bulk,
            .max_packet = DapEndpointPacketSize,
            .slots = dap_in_slots,
            .slot_count = DapSlotCount,
        }));
    }
}

static void log_hex_field(char const *name, U32 value) {
    rtt_write_string(name);
    rtt_write_hex_u32(value);
}

static void usb_reconcile(void) {
    U32 lock = arch_irq_save();
    Usbd_State state = usbd_state(&usb_device);
    U32 lifecycle = usbd_lifecycle(&usb_device);
    bool reopen = state == UsbdState_Configured &&
                  (!usbd_ep_opened(&usb_device, DapEndpointOut) ||
                   !usbd_ep_opened(&usb_device, DapEndpointIn));
    if (state == UsbdState_Configured) {
        for (U32 i = 0u; i < ArrayCount(consoles); i++) {
            if (!cdc_uart_endpoints_open(&consoles[i])) board_check(cdc_uart_open_endpoints(&consoles[i]));
        }
    }
    bool changed = dap_usb_session_invalidated(usb_state_seen,
                                               usb_lifecycle_seen,
                                               state,
                                               lifecycle,
                                               reopen);
    if (reopen) usb_open_dap_endpoints();
    usb_state_seen = state;
    usb_lifecycle_seen = lifecycle;
    arch_irq_restore(lock);

    if (changed) cmsis_dap_disconnect(&cmsis_dap);
}

static void note_activity(void) {
    activity_deadline_us = hal_now_us() + ActivityUs;
    activity = true;
}

static void dap_service(void) {
    U16 request_size = 0u;
    U8 const *request = 0;
    U8 *response = 0;
    U32 lock = arch_irq_save();
    if (usbd_lifecycle(&usb_device) == usb_lifecycle_seen &&
        usbd_state(&usb_device) == UsbdState_Configured) {
        request = usbd_ep_packet(&usb_device, DapEndpointOut, &request_size);
        response = usbd_ep_slot(&usb_device, DapEndpointIn);
    }
    arch_irq_restore(lock);
    if (!request || !response) return;

    U16 response_size;
    if (request[0] == DapVendor_Boot0) {
        bool force = request_size >= 2u && request[1] != 0u;
        response[0] = DapVendor_Boot0;
        response[1] = bsp_signal_write(&target_boot0, force) ? 0x00u : 0xFFu;
        response_size = 2u;
    } else if (request[0] == CmsisDapCommand_Vendor0) {
        // Last reset cause, consecutive crashes, and the faulting PC/LR.
        response[0] = CmsisDapCommand_Vendor0;
        write_le32(response + 1, boot_last_reset_cause());
        write_le32(response + 5, boot_crash_count);
        write_le32(response + 9, boot_fault_pc());
        write_le32(response + 13, boot_fault_lr());
        response_size = 17u;
    } else {
        response_size = cmsis_dap_execute(&cmsis_dap, request, request_size,
                                          response, DapEndpointPacketSize);
        boot_mark_stable();
    }
    lock = arch_irq_save();
    if (usbd_lifecycle(&usb_device) == usb_lifecycle_seen &&
        usbd_state(&usb_device) == UsbdState_Configured) {
        usbd_ep_release(&usb_device, DapEndpointOut);
        if (response_size != 0u) board_check(usbd_ep_commit(&usb_device, DapEndpointIn, response_size));
    }
    arch_irq_restore(lock);
    note_activity();
}

static void console_service(void) {
    bool configured = usbd_lifecycle(&usb_device) == usb_lifecycle_seen &&
                      usbd_state(&usb_device) == UsbdState_Configured;
    U32 moved = 0u;
    for (U32 i = 0u; i < ArrayCount(consoles); i++) {
        cdc_uart_service(&consoles[i], configured);
        moved += consoles[i].activity;
    }
    if (moved != console_activity_seen) {
        console_activity_seen = moved;
        note_activity();
    }
}

static bool deadline_reached(U32 deadline_us) {
    return (S32)(hal_now_us() - deadline_us) >= 0;
}

// Indicator LED: heartbeat until configured, then steady on, blinking off
// while DAP or console traffic flows. The RGB LED shows state only, so its
// interrupt-masked frames stay rare: amber unconfigured, green configured,
// magenta while BOOT0 is forced, red after a crash reset.
static void led_service(void) {
    U32 now = hal_now_us();
    bool configured = usbd_state(&usb_device) == UsbdState_Configured;
    if (activity && deadline_reached(activity_deadline_us)) activity = false;

    bool ind;
    if (!configured) {
        if ((S32)(now - heartbeat_deadline_us) >= 0) {
            heartbeat_on = !heartbeat_on;
            heartbeat_deadline_us = now + (heartbeat_on ? HeartbeatOnUs : HeartbeatOffUs);
        }
        ind = heartbeat_on;
    } else {
        ind = !(activity && ((now >> 15) & 1u));
    }
    if (ind != led_ind.active) board_check(bsp_signal_write(&led_ind, ind));

    if (boot_crash_count != 0u) {
        (void)rgb_led_set(&rgb, 24, 0, 0);
    } else if (target_boot0.active) {
        (void)rgb_led_set(&rgb, 16, 0, 16);
    } else if (configured) {
        (void)rgb_led_set(&rgb, 0, 16, 0);
    } else {
        (void)rgb_led_set(&rgb, 16, 8, 0);
    }
}

// frame is the stacked exception frame, or 0 from a firmware check. With a
// debugger attached this halts for inspection; otherwise it records the
// fault and resets, and boot_crashes decides what happens next.
static void fault(char const *reason, U32 const *frame) {
    arch_irq_save();
    bsp_fault_safe_outputs();

    U32 pc = frame ? frame[6] : 0u;
    U32 lr = frame ? frame[5] : 0u;
    if (!faulting) {
        faulting = 1u;
        rtt_write_string("\nprobe fault: ");
        rtt_write_string(reason);
        log_hex_field("\n  exception: ", arch_exception_number());
        log_hex_field("\n  pc:        ", pc);
        log_hex_field("\n  lr:        ", lr);
        rtt_write_string("\n");
    }
    if (arch_debugger_attached()) {
        arch_breakpoint();
        while (1) {
        }
    }
    boot_fault_reset(pc, lr);
}

__attribute__((used)) static void fault_from_exception(U32 const *frame) {
    fault(arch_exception_number() == 3u ? "hard fault" : "unexpected interrupt", frame);
}

__attribute__((naked)) void Fault_Handler(void) {
    __asm volatile (
        "mrs r0, msp\n"
        "ldr r1, =fault_from_exception\n"
        "bx r1\n"
    );
}

__attribute__((naked)) void Default_Handler(void) {
    __asm volatile (
        "mrs r0, msp\n"
        "ldr r1, =fault_from_exception\n"
        "bx r1\n"
    );
}

static void board_check(bool ok) {
    if (!ok) fault("board check", 0);
}

static void bootloader_enter(char const *reason) {
    rtt_write_string(reason);
    rtt_write_string(": entering system memory bootloader\n");
    bsp_signal_write(&led_ind, true);
    (void)rgb_led_set(&rgb, 0, 0, 24);
    boot_request_bootloader();
}

static void dfu_service(void) {
    if (!usb_dfu_runtime_detach_pending(&usb_dfu, hal_now_us())) return;
    bootloader_enter("dfu detach");
}

int main(void) {
    // Before any clock or peripheral setup: the ROM expects a reset-state
    // system.
    boot_bootloader_jump_if_pending();
    boot_mark_flash_programmed();
    boot_crash_count = boot_crashes();
    if (boot_crash_count >= BootCrashLimit) {
        boot_mark_stable();
        boot_request_bootloader();
    }

    hal_core_init(&(Hal_CoreConfig){ .clock = HalClock_Hsi16Pll_64MHz });
    hal_watchdog_start(WatchdogMs);
    board_check(bsp_init());

    board_check(bsp_signal_prepare(&led_ind, BspSignal_LedIndicator, "probe-led-indicator"));
    board_check(bsp_signal_prepare(&target_boot0, BspSignal_TargetBoot0, "probe-target-boot0"));
    board_check(bsp_signal_write(&led_ind, false));
    board_check(bsp_signal_write(&target_boot0, false));
    Bsp_RgbLed rgb_bsp = {0};
    board_check(bsp_rgb_led_prepare(&rgb_bsp, "probe-rgb-led"));
    board_check(rgb_led_init(&rgb, &rgb_bsp));

    rtt_init();
    rtt_write_string("Aleph FC probe, CMSIS-DAP 2.1.2, firmware " PROBE_VERSION "\n");
    log_hex_field("reset cause: ", boot_last_reset_cause());
    log_hex_field("  crashes: ", boot_crash_count);
    rtt_write_string("\n");

    serial_from_uid(usb_serial);
    rtt_write_string("serial=");
    rtt_write_string(usb_serial);
    rtt_write_string("\n");

    board_check(bsp_target_swd_prepare(&swd_bsp, "cmsis-dap-swd"));
    board_check(cmsis_dap_init(&cmsis_dap, &swd_bsp, usb_serial, PROBE_VERSION,
                               dap_abort_poll, 0));

    board_check(bsp_usb_device_prepare(&usb_bsp, "cmsis-dap-usb"));
    Usbd_Hardware usb_hardware = usb_hardware_from_bsp(&usb_bsp);
    board_check(usbd_init(&usb_device, &usb_hardware, &usb_descriptors));
    usb_dfu = (UsbDfu_Runtime){
        .interface_number = DfuInterface,
        .detach_delay_us = DfuDetachDelayUs,
    };
    usbd_set_control_handler(&usb_device, usb_control, &usb_dfu);
    usbd_set_control_data_handler(&usb_device, usb_control_data);

    Bsp_Uart uart = {0};
    board_check(bsp_uart_prepare(&uart, BspUart_Console0, "console-uart3"));
    board_check(cdc_uart_init(&consoles[0], &usb_device, uart.base, uart.kernel_hz, Console0Interface,
                              Console0NotifyEndpoint, Console0DataEndpoint));
    board_check(bsp_uart_prepare(&uart, BspUart_Console1, "console-uart6"));
    board_check(cdc_uart_init(&consoles[1], &usb_device, uart.base, uart.kernel_hz, Console1Interface,
                              Console1NotifyEndpoint, Console1DataEndpoint));
    arch_irq_set_priority((Arch_Irq)HalIrq_Usart3_4_5_6Lpuart1, UartIrqPriority);
    arch_irq_enable((Arch_Irq)HalIrq_Usart3_4_5_6Lpuart1);

    board_check(usbd_start(&usb_device));
    rtt_write_string("usb device start\n");

    // No WFI: a Cortex-M0+ in Sleep does not serve debugger memory accesses.
    heartbeat_deadline_us = hal_now_us();
    while (1) {
        hal_watchdog_feed();
        usb_reconcile();
        dap_service();
        console_service();
        dfu_service();
        led_service();
    }
}

void USB_UCPD1_2_IRQHandler(void) {
    usbd_irq(&usb_device);
}

void USART3_4_5_6_LPUART1_IRQHandler(void) {
    for (U32 i = 0u; i < ArrayCount(consoles); i++) cdc_uart_irq(&consoles[i]);
}

extern const UPtr vectors[];

void Reset_Handler(void) {
    arch_startup_state((U32)vectors);

    U32 *src = &_sidata;
    U32 *dst = &_sdata;
    while (dst < &_edata) *dst++ = *src++;

    dst = &_sbss;
    while (dst < &_ebss) *dst++ = 0u;

    main();
    while (1) {
    }
}

enum {
    VectorInitialStack = 0,
    VectorReset = 1,
    VectorNmi = 2,
    VectorHardFault = 3,
    VectorSvCall = 11,
    VectorPendSv = 14,
    VectorSysTick = 15,
    VectorLast = 16 + 30,
    VectorCount = VectorLast + 1,
};

__attribute__((section(".isr_vector"), used))
const UPtr vectors[VectorCount] = {
    [VectorInitialStack] = (UPtr)&_estack,
    [VectorReset] = (UPtr)Reset_Handler,
    [VectorNmi] = (UPtr)Fault_Handler,
    [VectorHardFault] = (UPtr)Fault_Handler,
    [VectorSvCall] = (UPtr)Default_Handler,
    [VectorPendSv] = (UPtr)Default_Handler,
    [VectorSysTick] = (UPtr)SysTick_Handler,

    [16 + 0] = (UPtr)Default_Handler,
    [16 + 1] = (UPtr)Default_Handler,
    [16 + 2] = (UPtr)Default_Handler,
    [16 + 3] = (UPtr)Default_Handler,
    [16 + 4] = (UPtr)Default_Handler,
    [16 + 5] = (UPtr)Default_Handler,
    [16 + 6] = (UPtr)Default_Handler,
    [16 + 7] = (UPtr)Default_Handler,
    [16 + HalIrq_UsbUcpd1_2] = (UPtr)USB_UCPD1_2_IRQHandler,
    [16 + HalIrq_Dma1Channel1] = (UPtr)Default_Handler,
    [16 + HalIrq_Dma1Channel2_3] = (UPtr)Default_Handler,
    [16 + 11] = (UPtr)Default_Handler,
    [16 + 12] = (UPtr)Default_Handler,
    [16 + 13] = (UPtr)Default_Handler,
    [16 + 14] = (UPtr)Default_Handler,
    [16 + 15] = (UPtr)Default_Handler,
    [16 + 16] = (UPtr)Default_Handler,
    [16 + 17] = (UPtr)Default_Handler,
    [16 + 18] = (UPtr)Default_Handler,
    [16 + 19] = (UPtr)Default_Handler,
    [16 + 20] = (UPtr)Default_Handler,
    [16 + 21] = (UPtr)Default_Handler,
    [16 + 22] = (UPtr)Default_Handler,
    [16 + HalIrq_I2c1] = (UPtr)Default_Handler,
    [16 + 24] = (UPtr)Default_Handler,
    [16 + HalIrq_Spi1] = (UPtr)Default_Handler,
    [16 + 26] = (UPtr)Default_Handler,
    [16 + HalIrq_Usart1] = (UPtr)Default_Handler,
    [16 + HalIrq_Usart2Lpuart2] = (UPtr)Default_Handler,
    [16 + HalIrq_Usart3_4_5_6Lpuart1] = (UPtr)USART3_4_5_6_LPUART1_IRQHandler,
    [16 + 30] = (UPtr)Default_Handler,
};
