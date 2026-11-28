#ifndef USB_DFU_H
#define USB_DFU_H

// usb_dfu.h — USB DFU 1.1 run-time interface: how a host asks a running
// application to enter its bootloader. No endpoints, one request that
// matters (DFU_DETACH), and the detach is reported to the main loop after a
// settle delay so the status stage completes before the device drops off.

#include "core.h"

#include "usbd.h"

enum {
    UsbDfu_InterfaceClass = 0xFEu,
    UsbDfu_InterfaceSubclass = 0x01u,
    UsbDfu_Protocol_Runtime = 0x01u,
    UsbDfu_Protocol_DfuMode = 0x02u,
    UsbDfu_DescriptorType_Functional = 0x21u,
    UsbDfu_Version = 0x0110u,

    UsbDfu_Attribute_CanDownload = 1u << 0,
    UsbDfu_Attribute_CanUpload = 1u << 1,
    UsbDfu_Attribute_ManifestationTolerant = 1u << 2,
    UsbDfu_Attribute_WillDetach = 1u << 3,

    UsbDfu_Request_Detach = 0u,
    UsbDfu_Request_Dnload = 1u,
    UsbDfu_Request_Upload = 2u,
    UsbDfu_Request_GetStatus = 3u,
    UsbDfu_Request_ClrStatus = 4u,
    UsbDfu_Request_GetState = 5u,
    UsbDfu_Request_Abort = 6u,

    UsbDfu_State_AppIdle = 0u,
    UsbDfu_State_AppDetach = 1u,
    UsbDfu_Status_Ok = 0u,

    UsbDfu_GetStatusSize = 6u,
};

typedef struct __attribute__((packed)) {
    U8 length;
    U8 descriptor_type;
    U8 attributes;
    U16 detach_timeout_ms;
    U16 transfer_size;
    U16 dfu_version;
} UsbDfu_FunctionalDescriptor;

static_assert(sizeof(UsbDfu_FunctionalDescriptor) == 9);

typedef struct {
    U8 interface_number;
    U32 detach_delay_us;

    volatile bool detach_requested; // written from usbd_irq
    bool detach_armed;
    U32 detach_deadline_us;
} UsbDfu_Runtime;

// Usbd_ControlHandler with the UsbDfu_Runtime as context; false for anything
// not addressed to its interface, so others can chain behind it.
bool usb_dfu_runtime_control(Usbd_Device *dev, Usbd_SetupPacket const *setup, void *context);

// Main-loop poll: true once the delay after a DETACH has elapsed.
bool usb_dfu_runtime_detach_pending(UsbDfu_Runtime *dfu, U32 now_us);

static U8 usb_dfu__state(UsbDfu_Runtime const *dfu) {
    return dfu->detach_requested ? (U8)UsbDfu_State_AppDetach : (U8)UsbDfu_State_AppIdle;
}

bool usb_dfu_runtime_control(Usbd_Device *dev, Usbd_SetupPacket const *setup, void *context) {
    UsbDfu_Runtime *dfu = (UsbDfu_Runtime *)context;
    if (!dev || !setup || !dfu) return false;

    U8 recipient = (U8)(setup->bmRequestType & 0x1Fu);
    U8 type = (U8)(setup->bmRequestType & USBD__REQ_TYPE_MASK);
    bool direction_in = (setup->bmRequestType & 0x80u) != 0u;
    if (type != USBD__REQ_TYPE_CLASS ||
        recipient != USBD__REQ_RECIPIENT_INTERFACE ||
        setup->wIndex != dfu->interface_number) {
        return false;
    }

    switch (setup->bRequest) {
    case UsbDfu_Request_Detach:
        if (direction_in || setup->wLength != 0u) return false;
        dfu->detach_requested = true;
        return usbd_control_ack(dev);

    case UsbDfu_Request_GetStatus: {
        if (!direction_in) return false;
        U8 status[UsbDfu_GetStatusSize] = {
            UsbDfu_Status_Ok,
            0u, 0u, 0u, // bwPollTimeout
            usb_dfu__state(dfu),
            0u,         // iString
        };
        return usbd_control_send(dev, status, sizeof(status));
    }

    case UsbDfu_Request_GetState: {
        if (!direction_in) return false;
        U8 state = usb_dfu__state(dfu);
        return usbd_control_send(dev, &state, sizeof(state));
    }

    default:
        return false;
    }
}

bool usb_dfu_runtime_detach_pending(UsbDfu_Runtime *dfu, U32 now_us) {
    if (!dfu || !dfu->detach_requested) return false;

    if (!dfu->detach_armed) {
        dfu->detach_armed = true;
        dfu->detach_deadline_us = now_us + dfu->detach_delay_us;
        return false;
    }

    return (S32)(now_us - dfu->detach_deadline_us) >= 0;
}

#endif
