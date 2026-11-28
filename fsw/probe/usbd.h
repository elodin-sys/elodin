#ifndef USBD_H
#define USBD_H

#include "core.h"
#include "hal.h"

typedef enum {
    UsbdState_Detached = 0,
    UsbdState_Attached,
    UsbdState_Default,
    UsbdState_Addressed,
    UsbdState_Configured,
    UsbdState_Suspended,
} Usbd_State;

typedef struct {
    U8 bmRequestType;
    U8 bRequest;
    U16 wValue;
    U16 wIndex;
    U16 wLength;
} Usbd_SetupPacket;

typedef struct {
    U8 const *data;
    U16 size;
} Usbd_Bytes;

#define UsbdBytes(value_) ((Usbd_Bytes){ .data = (U8 const *)&(value_), .size = sizeof(value_) })

typedef enum {
    UsbdDescriptorType_Device = 0x01,
    UsbdDescriptorType_Configuration = 0x02,
    UsbdDescriptorType_String = 0x03,
    UsbdDescriptorType_Interface = 0x04,
    UsbdDescriptorType_Endpoint = 0x05,
    UsbdDescriptorType_InterfaceAssociation = 0x0B,
    UsbdDescriptorType_Bos = 0x0F,
    UsbdDescriptorType_DeviceCapability = 0x10,
} Usbd_DescriptorType;

typedef struct __attribute__((packed)) {
    U8 length;
    U8 descriptor_type;
    U16 usb_bcd;
    U8 device_class;
    U8 device_subclass;
    U8 device_protocol;
    U8 ep0_max_packet;
    U16 vendor_id;
    U16 product_id;
    U16 device_bcd;
    U8 manufacturer_string;
    U8 product_string;
    U8 serial_string;
    U8 configuration_count;
} Usbd_DeviceDescriptor;

static_assert(sizeof(Usbd_DeviceDescriptor) == 18);

typedef struct __attribute__((packed)) {
    U8 length;
    U8 descriptor_type;
    U16 total_length;
    U8 interface_count;
    U8 configuration_value;
    U8 configuration_string;
    U8 attributes;
    U8 max_power_2ma;
} Usbd_ConfigurationDescriptor;

static_assert(sizeof(Usbd_ConfigurationDescriptor) == 9);

typedef struct __attribute__((packed)) {
    U8 length;
    U8 descriptor_type;
    U8 interface_number;
    U8 alternate_setting;
    U8 endpoint_count;
    U8 interface_class;
    U8 interface_subclass;
    U8 interface_protocol;
    U8 interface_string;
} Usbd_InterfaceDescriptor;

static_assert(sizeof(Usbd_InterfaceDescriptor) == 9);

typedef struct __attribute__((packed)) {
    U8 length;
    U8 descriptor_type;
    U8 endpoint_address;
    U8 attributes;
    U16 max_packet_size;
    U8 interval;
} Usbd_EndpointDescriptor;

static_assert(sizeof(Usbd_EndpointDescriptor) == 7);

typedef struct __attribute__((packed)) {
    U8 length;
    U8 descriptor_type;
    U8 first_interface;
    U8 interface_count;
    U8 function_class;
    U8 function_subclass;
    U8 function_protocol;
    U8 function_string;
} Usbd_InterfaceAssociationDescriptor;

static_assert(sizeof(Usbd_InterfaceAssociationDescriptor) == 8);

typedef struct {
    Usbd_Bytes device;
    Usbd_Bytes configuration;
    Usbd_Bytes bos;
    Usbd_Bytes microsoft_os_20;
    char const *const *strings;
    U16 microsoft_descriptor_index;
    U16 language_id;
    U8 microsoft_vendor_code;
    U8 string_count;
} Usbd_DescriptorSet;

typedef struct {
    UPtr reg_base;
    UPtr pma_base;
    U32 pma_bytes;
    U8 endpoint_count;
    Hal_Irq irq;
    U32 usb_hz;
    U32 flags;
} Usbd_Hardware;

typedef enum {
    UsbdHardware_NoFlags = 0,
    UsbdHardware_Pma32Bit = 1u << 0,
} Usbd_HardwareFlags;

typedef enum {
    UsbdEpType_Bulk = 0,
    UsbdEpType_Control,
    UsbdEpType_Isochronous,
    UsbdEpType_Interrupt,
} Usbd_EpType;

#define UsbdEpAddr_Out(n_) ((U8)((n_) & 0x0Fu))
#define UsbdEpAddr_In(n_)  ((U8)(0x80u | ((n_) & 0x0Fu)))

// Called from usbd_irq before a received packet is queued; false drops it.
typedef bool (*Usbd_PacketFilter)(void *context, U8 const *data, U16 size);

// Data endpoints queue whole packets in fixed slots, a length word then the
// packet, moved to and from packet memory as words. Zero-length OUT packets
// are dropped.
#define USBD_SLOT_WORDS(max_packet_) (1u + ((max_packet_) + 3u) / 4u)

typedef struct {
    U8 addr;
    Usbd_EpType type;
    U16 max_packet;
    U32 *slots;
    U8 slot_count;
    Usbd_PacketFilter filter;
    void *filter_context;
} Usbd_EpConfig;

typedef struct {
    bool opened;
    U8 addr;
    Usbd_EpType type;
    U16 max_packet;
    U16 pma_offset;
    U32 *slots;
    U16 slot_words;
    U8 slot_count;
    U8 head;    // oldest queued slot
    U8 used;    // queued slots
    Usbd_PacketFilter filter;
    void *filter_context;
} Usbd_Ep;

typedef enum {
    UsbdEp0Stage_Idle = 0,
    UsbdEp0Stage_DataIn,
    UsbdEp0Stage_DataOut,
    UsbdEp0Stage_StatusIn,
    UsbdEp0Stage_StatusOut,
    UsbdEp0Stage_Stall,
} Usbd_Ep0Stage;

typedef struct {
    Usbd_SetupPacket setup;
    Usbd_Ep0Stage stage;

    U8 const *in_data;
    U16 in_size;
    U16 in_at;
    bool in_zlp;

    U16 out_size;
    U16 out_at;

    U8 pending_address;
    U8 scratch[128];
} Usbd_Ep0;

typedef struct Usbd_Device Usbd_Device;

// Class and vendor requests usbd does not answer itself, from usbd_irq. Reply
// with usbd_control_send, usbd_control_ack or (OUT data stage)
// usbd_control_receive and return true, or return false to stall. Interface
// class requests only while configured.
typedef bool (*Usbd_ControlHandler)(Usbd_Device *dev, Usbd_SetupPacket const *setup, void *context);

// Completes a usbd_control_receive data stage: return true to acknowledge the
// status stage, false to stall.
typedef bool (*Usbd_ControlDataHandler)(Usbd_Device *dev, Usbd_SetupPacket const *setup,
                                        U8 const *data, U16 size, void *context);

struct Usbd_Device {
    Usbd_Hardware hw;
    Usbd_DescriptorSet descriptors;
    Usbd_ControlHandler control_handler;
    Usbd_ControlDataHandler control_data_handler;
    void *control_context;
    Usbd_State state;
    Usbd_State pre_suspend_state;
    U32 lifecycle;

    U8 address;
    U8 configuration;
    bool remote_wakeup_enabled;
    bool resume_pending;

    U16 btable_offset;
    U16 next_pma_offset;
    U16 ep0_tx_offset;
    U16 ep0_rx_offset;
    U16 ep0_max_packet;

    Usbd_Ep0 ep0;
    Usbd_Ep ep_in[8];
    Usbd_Ep ep_out[8];

    bool initialized;
    bool attached;
};

bool usbd_init(Usbd_Device *dev, Usbd_Hardware const *hardware, Usbd_DescriptorSet const *descriptors);
bool usbd_start(Usbd_Device *dev);
void usbd_stop(Usbd_Device *dev);
void usbd_irq(Usbd_Device *dev);
Usbd_State usbd_state(Usbd_Device const *dev);
U32 usbd_lifecycle(Usbd_Device const *dev);
void usbd_set_control_handler(Usbd_Device *dev, Usbd_ControlHandler handler, void *context);

// Replies for a Usbd_ControlHandler, valid only inside the handler:
// usbd_control_send answers an IN request from a copy of up to 128 bytes,
// usbd_control_ack completes a no-data OUT request.
bool usbd_control_send(Usbd_Device *dev, void const *data, U16 size);
bool usbd_control_ack(Usbd_Device *dev);
bool usbd_control_receive(Usbd_Device *dev);
void usbd_set_control_data_handler(Usbd_Device *dev, Usbd_ControlDataHandler handler);

bool usbd_controller_reset(Usbd_Device *dev);
bool usbd_attach(Usbd_Device *dev);
void usbd_detach(Usbd_Device *dev);
bool usbd_ep_open(Usbd_Device *dev, Usbd_EpConfig const *config);
bool usbd_ep_opened(Usbd_Device const *dev, U8 addr);
// OUT: the oldest packet in place until released. IN: the next free slot,
// queued by commit. read and write are the copying forms.
U8 const *usbd_ep_packet(Usbd_Device const *dev, U8 addr, U16 *size);
void usbd_ep_release(Usbd_Device *dev, U8 addr);
U8 *usbd_ep_slot(Usbd_Device const *dev, U8 addr);
bool usbd_ep_commit(Usbd_Device *dev, U8 addr, U16 size);
U16 usbd_ep_read_available(Usbd_Device const *dev, U8 addr);
U16 usbd_ep_read(Usbd_Device *dev, U8 addr, void *dst, U16 size);
U16 usbd_ep_write_available(Usbd_Device const *dev, U8 addr);
U16 usbd_ep_write(Usbd_Device *dev, U8 addr, void const *src, U16 size);
bool usbd_ep_stall(Usbd_Device *dev, U8 addr, bool stalled);
bool usbd_ep_is_stalled(Usbd_Device const *dev, U8 addr);
bool usbd_pma_read(Usbd_Device const *dev, U16 pma_offset, void *dst, U16 size);
bool usbd_pma_write(Usbd_Device *dev, U16 pma_offset, void const *src, U16 size);

#ifndef USBD__REG
#define USBD__REG(dev_, offset_) (*(volatile U32 *)((dev_)->hw.reg_base + (offset_)))
#endif
#define USBD__PMA16(dev_, offset_) (*(volatile U16 *)((dev_)->hw.pma_base + (UPtr)(offset_)))
#define USBD__PMA32(dev_, offset_) (*(volatile U32 *)((dev_)->hw.pma_base + (UPtr)(offset_)))

#define USBD__EP_REG(ep_)       ((U32)(ep_) * 4u)
#define USBD__CNTR              0x40u
#define USBD__ISTR              0x44u
#define USBD__DADDR             0x4Cu
#define USBD__BTABLE            0x50u
#define USBD__BCDR              0x58u

#define USBD__EP_CTR_RX         0x8000u
#define USBD__EP_DTOG_RX        0x4000u
#define USBD__EP_STAT_RX_MASK   0x3000u
#define USBD__EP_STAT_RX_DISABLED 0x0000u
#define USBD__EP_STAT_RX_NAK    0x2000u
#define USBD__EP_STAT_RX_VALID  0x3000u
#define USBD__EP_STAT_RX_STALL  0x1000u
#define USBD__EP_SETUP          0x0800u
#define USBD__EP_TYPE_MASK      0x0600u
#define USBD__EP_TYPE_BULK      0x0000u
#define USBD__EP_TYPE_CONTROL   0x0200u
#define USBD__EP_TYPE_ISO       0x0400u
#define USBD__EP_TYPE_INTERRUPT 0x0600u
#define USBD__EP_KIND           0x0100u
#define USBD__EP_CTR_TX         0x0080u
#define USBD__EP_DTOG_TX        0x0040u
#define USBD__EP_STAT_TX_MASK   0x0030u
#define USBD__EP_STAT_TX_DISABLED 0x0000u
#define USBD__EP_STAT_TX_NAK    0x0020u
#define USBD__EP_STAT_TX_VALID  0x0030u
#define USBD__EP_STAT_TX_STALL  0x0010u
#define USBD__EP_ADDR_MASK      0x000Fu
#define USBD__EP_ERR_RX         0x04000000u
#define USBD__EP_ERR_TX         0x02000000u
#define USBD__EP_LSEP           0x01000000u
#define USBD__EP_NAK            0x00800000u
#define USBD__EP_DEVADDR_MASK   0x007F0000u

#define USBD__EP_REG_KEEP_MASK  (USBD__EP_ERR_RX | \
                                  USBD__EP_ERR_TX | \
                                  USBD__EP_LSEP | \
                                  USBD__EP_NAK | \
                                  USBD__EP_DEVADDR_MASK | \
                                  USBD__EP_CTR_RX | \
                                  USBD__EP_CTR_TX | \
                                  USBD__EP_TYPE_MASK | \
                                  USBD__EP_KIND | \
                                  USBD__EP_ADDR_MASK)

#define USBD__CNTR_FRES         0x0001u
#define USBD__CNTR_PDWN         0x0002u
#define USBD__CNTR_SUSPRDY      0x0004u
#define USBD__CNTR_SUSPEN       0x0008u
#define USBD__CNTR_RESETM       0x0400u
#define USBD__CNTR_SUSPM        0x0800u
#define USBD__CNTR_WKUPM        0x1000u
#define USBD__CNTR_ERRM         0x2000u
#define USBD__CNTR_CTRM         0x8000u

#define USBD__ISTR_EP_ID_MASK   0x000Fu
#define USBD__ISTR_DIR          0x0010u
#define USBD__ISTR_RESET        0x0400u
#define USBD__ISTR_SUSP         0x0800u
#define USBD__ISTR_WKUP         0x1000u
#define USBD__ISTR_ERR          0x2000u
#define USBD__ISTR_CTR          0x8000u

#define USBD__DADDR_EF          0x0080u
#define USBD__BCDR_DPPU         0x8000u

#define USBD__REQ_GET_STATUS        0x00u
#define USBD__REQ_CLEAR_FEATURE     0x01u
#define USBD__REQ_SET_FEATURE       0x03u
#define USBD__REQ_SET_ADDRESS       0x05u
#define USBD__REQ_GET_DESCRIPTOR    0x06u
#define USBD__REQ_GET_CONFIGURATION 0x08u
#define USBD__REQ_SET_CONFIGURATION 0x09u
#define USBD__REQ_GET_INTERFACE     0x0Au
#define USBD__REQ_SET_INTERFACE     0x0Bu

#define USBD__REQ_TYPE_STANDARD     0x00u
#define USBD__REQ_TYPE_CLASS        0x20u
#define USBD__REQ_TYPE_VENDOR       0x40u
#define USBD__REQ_TYPE_MASK         0x60u

#define USBD__REQ_RECIPIENT_DEVICE    0x00u
#define USBD__REQ_RECIPIENT_INTERFACE 0x01u
#define USBD__REQ_RECIPIENT_ENDPOINT  0x02u

#define USBD__FEATURE_ENDPOINT_HALT        0x00u
#define USBD__FEATURE_DEVICE_REMOTE_WAKEUP 0x01u
#define USBD__FEATURE_TEST_MODE            0x02u

#define USBD__DESC_DEVICE        ((U8)UsbdDescriptorType_Device)
#define USBD__DESC_CONFIGURATION ((U8)UsbdDescriptorType_Configuration)
#define USBD__DESC_STRING        ((U8)UsbdDescriptorType_String)
#define USBD__DESC_BOS           ((U8)UsbdDescriptorType_Bos)

static bool usbd__ready(Usbd_Device const *dev) {
    return dev && dev->initialized && dev->hw.reg_base && dev->hw.pma_base && dev->hw.endpoint_count;
}

static U16 usbd__string_length(char const *string) {
    U16 result = 0;
    if (!string) return 0;

    while (string[result]) result++;
    return result;
}

static U8 usbd__ep_num(U8 addr) {
    return (U8)(addr & 0x0Fu);
}

static bool usbd__ep_dir_in(U8 addr) {
    return (addr & 0x80u) != 0;
}

static bool usbd__pma_32bit(Usbd_Device const *dev);


// Whole-word packet memory copies for the slots; EP0 keeps the byte forms.
static void usbd__pma_read_words(Usbd_Device const *dev, U16 pma_offset, U32 *dst, U16 words) {
    if (usbd__pma_32bit(dev)) {
        for (U16 i = 0; i < words; i++) dst[i] = USBD__PMA32(dev, (U16)(pma_offset + 4u * i));
        return;
    }
    for (U16 i = 0; i < words; i++) {
        dst[i] = (U32)USBD__PMA16(dev, (U16)(pma_offset + 4u * i)) |
                 ((U32)USBD__PMA16(dev, (U16)(pma_offset + 4u * i + 2u)) << 16);
    }
}

static void usbd__pma_write_words(Usbd_Device *dev, U16 pma_offset, U32 const *src, U16 words) {
    if (usbd__pma_32bit(dev)) {
        for (U16 i = 0; i < words; i++) USBD__PMA32(dev, (U16)(pma_offset + 4u * i)) = src[i];
        return;
    }
    for (U16 i = 0; i < words; i++) {
        USBD__PMA16(dev, (U16)(pma_offset + 4u * i)) = (U16)src[i];
        USBD__PMA16(dev, (U16)(pma_offset + 4u * i + 2u)) = (U16)(src[i] >> 16);
    }
}

static U32 *usbd__slot(Usbd_Ep const *endpoint, U8 index) {
    U8 at = (U8)(endpoint->head + index);
    if (at >= endpoint->slot_count) at = (U8)(at - endpoint->slot_count);
    return endpoint->slots + (U32)at * endpoint->slot_words;
}

static U8 usbd__slot_next(Usbd_Ep const *endpoint, U8 index) {
    return (U8)(index + 1u == endpoint->slot_count ? 0u : index + 1u);
}

static Usbd_Ep *usbd__ep(Usbd_Device *dev, U8 addr) {
    U8 ep = usbd__ep_num(addr);
    if (!dev || ep >= dev->hw.endpoint_count) return 0;
    return usbd__ep_dir_in(addr) ? &dev->ep_in[ep] : &dev->ep_out[ep];
}

static Usbd_Ep const *usbd__ep_const(Usbd_Device const *dev, U8 addr) {
    U8 ep = usbd__ep_num(addr);
    if (!dev || ep >= dev->hw.endpoint_count) return 0;
    return usbd__ep_dir_in(addr) ? &dev->ep_in[ep] : &dev->ep_out[ep];
}

static U16 usbd__ep_type_bits(Usbd_EpType type) {
    if (type == UsbdEpType_Control) return USBD__EP_TYPE_CONTROL;
    if (type == UsbdEpType_Isochronous) return USBD__EP_TYPE_ISO;
    if (type == UsbdEpType_Interrupt) return USBD__EP_TYPE_INTERRUPT;
    return USBD__EP_TYPE_BULK;
}

static U16 usbd__rx_count_bits(U16 size) {
    if (size > 62u) {
        U16 blocks = (U16)((size + 31u) / 32u);
        return (U16)(0x8000u | ((blocks - 1u) << 10));
    }

    U16 blocks = (U16)((size + 1u) / 2u);
    return (U16)(blocks << 10);
}

static U16 usbd__pma_align(U16 offset) {
    return (U16)((offset + 7u) & ~7u);
}

static bool usbd__pma_32bit(Usbd_Device const *dev) {
    return dev && (dev->hw.flags & UsbdHardware_Pma32Bit) != 0;
}

static U16 usbd__btable_entry(Usbd_Device const *dev, U8 ep, U8 field) {
    if (usbd__pma_32bit(dev)) return (U16)(dev->btable_offset + ((U16)ep * 8u) + (((U16)field >> 1u) * 4u));
    return (U16)(dev->btable_offset + ((U16)ep * 8u) + ((U16)field * 2u));
}

static U16 usbd__btable_get(Usbd_Device const *dev, U8 ep, U8 field) {
    if (usbd__pma_32bit(dev)) {
        U32 word = USBD__PMA32(dev, usbd__btable_entry(dev, ep, field));
        return (field & 1u) ? (U16)(word >> 16) : (U16)word;
    }

    return USBD__PMA16(dev, usbd__btable_entry(dev, ep, field));
}

static void usbd__btable_set(Usbd_Device *dev, U8 ep, U8 field, U16 value) {
    if (usbd__pma_32bit(dev)) {
        U16 offset = usbd__btable_entry(dev, ep, field);
        U32 word = USBD__PMA32(dev, offset);
        if (field & 1u) word = (word & 0x0000FFFFu) | ((U32)value << 16);
        else word = (word & 0xFFFF0000u) | (U32)(value & 0xFFFCu);
        USBD__PMA32(dev, offset) = word;
        return;
    }

    USBD__PMA16(dev, usbd__btable_entry(dev, ep, field)) = value;
}

// STAT and DTOG bits toggle when written as one. Zero preserves the opposite
// direction. CTR bits clear when written as zero. Keep that mixed write model
// in one place so EPn OUT and IN can safely share an endpoint register.
static void usbd__ep_write(Usbd_Device *dev, U8 ep, U32 current,
                           U32 clear_ctr, U32 toggle) {
    U32 ctr_mask = USBD__EP_CTR_RX | USBD__EP_CTR_TX;
    U32 ctr = ctr_mask & ~clear_ctr;
    U32 value = (current & USBD__EP_REG_KEEP_MASK & ~ctr_mask) | ctr | toggle;
    USBD__REG(dev, USBD__EP_REG(ep)) = value;
}

static void usbd__ep_set_stat_rx(Usbd_Device *dev, U8 ep, U16 stat) {
    U32 current = USBD__REG(dev, USBD__EP_REG(ep));
    usbd__ep_write(dev, ep, current, 0u,
                   (stat ^ current) & USBD__EP_STAT_RX_MASK);
}

static void usbd__ep_set_stat_tx(Usbd_Device *dev, U8 ep, U16 stat) {
    U32 current = USBD__REG(dev, USBD__EP_REG(ep));
    usbd__ep_write(dev, ep, current, 0u,
                   (stat ^ current) & USBD__EP_STAT_TX_MASK);
}

static void usbd__ep_clear_dtog_rx(Usbd_Device *dev, U8 ep) {
    U32 current = USBD__REG(dev, USBD__EP_REG(ep));
    usbd__ep_write(dev, ep, current, 0u, current & USBD__EP_DTOG_RX);
}

static void usbd__ep_clear_dtog_tx(Usbd_Device *dev, U8 ep) {
    U32 current = USBD__REG(dev, USBD__EP_REG(ep));
    usbd__ep_write(dev, ep, current, 0u, current & USBD__EP_DTOG_TX);
}

static void usbd__ep_clear_ctr_rx(Usbd_Device *dev, U8 ep) {
    U32 current = USBD__REG(dev, USBD__EP_REG(ep));
    usbd__ep_write(dev, ep, current, USBD__EP_CTR_RX, 0u);
}

static void usbd__ep_clear_ctr_tx(Usbd_Device *dev, U8 ep) {
    U32 current = USBD__REG(dev, USBD__EP_REG(ep));
    usbd__ep_write(dev, ep, current, USBD__EP_CTR_TX, 0u);
}

static U16 usbd__ep_rx_count(Usbd_Device const *dev, U8 ep) {
    return (U16)(usbd__btable_get(dev, ep, 3u) & 0x03FFu);
}

static void usbd__set_address_now(Usbd_Device *dev, U8 address) {
    dev->address = (U8)(address & 0x7Fu);
    USBD__REG(dev, USBD__DADDR) = (U16)(USBD__DADDR_EF | dev->address);
    dev->state = dev->address ? UsbdState_Addressed : UsbdState_Default;
}

static void usbd__ep0_reset_transfer(Usbd_Device *dev) {
    dev->ep0.setup = (Usbd_SetupPacket){0};
    dev->ep0.stage = UsbdEp0Stage_Idle;
    dev->ep0.in_data = 0;
    dev->ep0.in_size = 0;
    dev->ep0.in_at = 0;
    dev->ep0.in_zlp = false;
    dev->ep0.out_size = 0;
    dev->ep0.out_at = 0;
    dev->ep0.pending_address = 0;
}

static void usbd__reset_data_endpoints(Usbd_Device *dev) {
    memset(dev->ep_in, 0, sizeof(dev->ep_in));
    memset(dev->ep_out, 0, sizeof(dev->ep_out));

    for (U8 ep = 1u; ep < dev->hw.endpoint_count; ep++) {
        U32 current = USBD__REG(dev, USBD__EP_REG(ep));
        U32 toggles = current & (USBD__EP_DTOG_RX |
                                 USBD__EP_STAT_RX_MASK |
                                 USBD__EP_DTOG_TX |
                                 USBD__EP_STAT_TX_MASK);
        usbd__ep_write(dev, ep, current,
                       USBD__EP_CTR_RX | USBD__EP_CTR_TX,
                       toggles);
        usbd__btable_set(dev, ep, 1u, 0u);
        usbd__btable_set(dev, ep, 3u, 0u);
    }

    if (dev->ep0_rx_offset && dev->ep0_max_packet) {
        dev->next_pma_offset = usbd__pma_align((U16)(dev->ep0_rx_offset + dev->ep0_max_packet));
    }
}

static bool usbd__ep0_rx_valid(Usbd_Device *dev) {
    usbd__btable_set(dev, 0u, 3u, usbd__rx_count_bits(dev->ep0_max_packet));
    usbd__ep_set_stat_rx(dev, 0u, USBD__EP_STAT_RX_VALID);
    return true;
}

static bool usbd__ep0_send_packet(Usbd_Device *dev) {
    U16 remaining = (U16)(dev->ep0.in_size - dev->ep0.in_at);
    U16 packet = Min(remaining, dev->ep0_max_packet);

    if (packet && !usbd_pma_write(dev, dev->ep0_tx_offset, dev->ep0.in_data + dev->ep0.in_at, packet)) return false;
    usbd__btable_set(dev, 0u, 1u, packet);
    dev->ep0.in_at = (U16)(dev->ep0.in_at + packet);
    usbd__ep_set_stat_tx(dev, 0u, USBD__EP_STAT_TX_VALID);
    return true;
}

static bool usbd__ep0_start_in(Usbd_Device *dev, U8 const *data, U16 size, U16 requested) {
    U16 send_size = Min(size, requested);
    if (!data && send_size) return false;

    dev->ep0.in_data = data;
    dev->ep0.in_size = send_size;
    dev->ep0.in_at = 0;
    dev->ep0.in_zlp = send_size != 0u && send_size < requested && (send_size & (dev->ep0_max_packet - 1u)) == 0;
    dev->ep0.stage = UsbdEp0Stage_DataIn;
    return usbd__ep0_send_packet(dev);
}

static bool usbd__ep0_send_buffer(Usbd_Device *dev, U16 size, U16 requested) {
    return usbd__ep0_start_in(dev, dev->ep0.scratch, size, requested);
}

static bool usbd__ep0_send_data(Usbd_Device *dev, U8 const *data, U16 size, U16 requested) {
    return usbd__ep0_start_in(dev, data, size, requested);
}

static bool usbd__ep0_send_string_descriptor(Usbd_Device *dev, U8 index, U16 requested) {
    dev->ep0.scratch[1] = USBD__DESC_STRING;

    if (index == 0u) {
        U16 language_id = dev->descriptors.language_id ? dev->descriptors.language_id : 0x0409u;
        dev->ep0.scratch[0] = 4u;
        write_le16(dev->ep0.scratch + 2, language_id);
        return usbd__ep0_send_buffer(dev, 4u, requested);
    }

    U8 string_index = (U8)(index - 1u);
    if (string_index >= dev->descriptors.string_count || !dev->descriptors.strings) return false;

    char const *string = dev->descriptors.strings[string_index];
    if (!string) return false;

    U16 length = usbd__string_length(string);
    U16 max_length = (U16)((sizeof(dev->ep0.scratch) - 2u) / 2u);
    if (length > max_length) length = max_length;

    dev->ep0.scratch[0] = (U8)(2u + (length * 2u));
    for (U16 i = 0; i < length; i++) {
        dev->ep0.scratch[2u + (i * 2u)] = (U8)string[i];
        dev->ep0.scratch[3u + (i * 2u)] = 0u;
    }

    return usbd__ep0_send_buffer(dev, dev->ep0.scratch[0], requested);
}

static bool usbd__ep0_send_zlp(Usbd_Device *dev) {
    dev->ep0.in_data = 0;
    dev->ep0.in_size = 0;
    dev->ep0.in_at = 0;
    dev->ep0.in_zlp = false;
    dev->ep0.stage = UsbdEp0Stage_StatusIn;
    usbd__btable_set(dev, 0u, 1u, 0u);
    usbd__ep_set_stat_tx(dev, 0u, USBD__EP_STAT_TX_VALID);
    return true;
}

static void usbd__ep0_stall(Usbd_Device *dev) {
    dev->ep0.stage = UsbdEp0Stage_Stall;
    usbd__ep_set_stat_rx(dev, 0u, USBD__EP_STAT_RX_STALL);
    usbd__ep_set_stat_tx(dev, 0u, USBD__EP_STAT_TX_STALL);
}

static bool usbd__descriptor(Usbd_Device const *dev, U8 type, Usbd_Bytes *out) {
    if (!out) return false;

    if (type == USBD__DESC_DEVICE) {
        if (!dev->descriptors.device.data || !dev->descriptors.device.size) return false;
        *out = dev->descriptors.device;
        return true;
    }

    if (type == USBD__DESC_CONFIGURATION) {
        if (!dev->descriptors.configuration.data || !dev->descriptors.configuration.size) return false;
        *out = dev->descriptors.configuration;
        return true;
    }

    if (type == USBD__DESC_BOS) {
        if (!dev->descriptors.bos.data || !dev->descriptors.bos.size) return false;
        *out = dev->descriptors.bos;
        return true;
    }

    return false;
}

static U8 usbd__interface_count(Usbd_Device const *dev) {
    if (!dev || !dev->descriptors.configuration.data || dev->descriptors.configuration.size < sizeof(Usbd_ConfigurationDescriptor)) return 0u;
    return dev->descriptors.configuration.data[4];
}

static bool usbd__interface_valid(Usbd_Device const *dev, U16 index) {
    return index < usbd__interface_count(dev);
}

static bool usbd__handle_standard_setup(Usbd_Device *dev) {
    Usbd_SetupPacket const *setup = &dev->ep0.setup;
    U8 recipient = (U8)(setup->bmRequestType & 0x1Fu);
    U8 type = (U8)(setup->bmRequestType & USBD__REQ_TYPE_MASK);
    U8 direction_in = (U8)(setup->bmRequestType & 0x80u);
    U8 endpoint_addr = (U8)setup->wIndex;
    U8 endpoint_num = usbd__ep_num(endpoint_addr);

    if (type != USBD__REQ_TYPE_STANDARD) return false;

    switch (setup->bRequest) {
    case USBD__REQ_GET_DESCRIPTOR: {
        if (recipient != USBD__REQ_RECIPIENT_DEVICE || !direction_in) return false;
        Usbd_Bytes descriptor = {0};
        U8 descriptor_type = (U8)(setup->wValue >> 8);
        U8 descriptor_index = (U8)setup->wValue;
        if (descriptor_type == USBD__DESC_STRING) {
            return usbd__ep0_send_string_descriptor(dev, descriptor_index, setup->wLength);
        }

        if (!usbd__descriptor(dev, descriptor_type, &descriptor)) return false;
        return usbd__ep0_send_data(dev, descriptor.data, descriptor.size, setup->wLength);
    }

    case USBD__REQ_SET_ADDRESS:
        if (recipient != USBD__REQ_RECIPIENT_DEVICE || direction_in || setup->wIndex || setup->wLength || setup->wValue > 127u) return false;
        dev->ep0.pending_address = (U8)setup->wValue;
        return usbd__ep0_send_zlp(dev);

    case USBD__REQ_GET_CONFIGURATION:
        if (recipient != USBD__REQ_RECIPIENT_DEVICE || !direction_in || setup->wValue || setup->wIndex || setup->wLength != 1u) return false;
        dev->ep0.scratch[0] = dev->configuration;
        return usbd__ep0_send_buffer(dev, 1u, setup->wLength);

    case USBD__REQ_SET_CONFIGURATION:
        if (recipient != USBD__REQ_RECIPIENT_DEVICE || direction_in || setup->wIndex || setup->wLength || setup->wValue > 1u) return false;
        return usbd__ep0_send_zlp(dev);

    case USBD__REQ_GET_STATUS: {
        if (!direction_in || setup->wValue || setup->wLength != 2u) return false;
        if (recipient == USBD__REQ_RECIPIENT_DEVICE && setup->wIndex) return false;
        if (recipient == USBD__REQ_RECIPIENT_INTERFACE && !usbd__interface_valid(dev, setup->wIndex)) return false;
        if (recipient == USBD__REQ_RECIPIENT_ENDPOINT && endpoint_num >= dev->hw.endpoint_count) return false;
        if (recipient > USBD__REQ_RECIPIENT_ENDPOINT) return false;

        U16 status = 0u;
        if (recipient == USBD__REQ_RECIPIENT_DEVICE) {
            if ((dev->descriptors.configuration.data[7] & 0x40u) != 0) status |= 1u;
            if (dev->remote_wakeup_enabled) status |= 2u;
        } else if (recipient == USBD__REQ_RECIPIENT_ENDPOINT && endpoint_num != 0u) {
            Usbd_Ep const *endpoint = usbd__ep_const(dev, endpoint_addr);
            if (!endpoint || !endpoint->opened) return false;
            if (usbd_ep_is_stalled(dev, endpoint_addr)) status |= 1u;
        }
        write_le16(dev->ep0.scratch, status);
        return usbd__ep0_send_buffer(dev, 2u, setup->wLength);
    }

    case USBD__REQ_GET_INTERFACE:
        if (recipient != USBD__REQ_RECIPIENT_INTERFACE || !direction_in || setup->wValue || !usbd__interface_valid(dev, setup->wIndex) || setup->wLength != 1u) return false;
        dev->ep0.scratch[0] = 0u;
        return usbd__ep0_send_buffer(dev, 1u, setup->wLength);

    case USBD__REQ_SET_INTERFACE:
        if (recipient != USBD__REQ_RECIPIENT_INTERFACE || direction_in || setup->wValue != 0u || !usbd__interface_valid(dev, setup->wIndex) || setup->wLength) return false;
        return usbd__ep0_send_zlp(dev);

    case USBD__REQ_CLEAR_FEATURE:
        if (direction_in || setup->wLength) return false;
        if (recipient == USBD__REQ_RECIPIENT_DEVICE && setup->wValue == USBD__FEATURE_DEVICE_REMOTE_WAKEUP && setup->wIndex == 0u) {
            dev->remote_wakeup_enabled = false;
            return usbd__ep0_send_zlp(dev);
        }
        if (recipient != USBD__REQ_RECIPIENT_ENDPOINT || setup->wValue != USBD__FEATURE_ENDPOINT_HALT || endpoint_num == 0u || endpoint_num >= dev->hw.endpoint_count) return false;
        if (!usbd_ep_stall(dev, endpoint_addr, false)) return false;
        return usbd__ep0_send_zlp(dev);

    case USBD__REQ_SET_FEATURE:
        if (direction_in || setup->wLength) return false;
        if (recipient == USBD__REQ_RECIPIENT_DEVICE && setup->wValue == USBD__FEATURE_DEVICE_REMOTE_WAKEUP && setup->wIndex == 0u) {
            if ((dev->descriptors.configuration.data[7] & 0x20u) == 0) return false;
            dev->remote_wakeup_enabled = true;
            return usbd__ep0_send_zlp(dev);
        }
        if (recipient != USBD__REQ_RECIPIENT_ENDPOINT || setup->wValue != USBD__FEATURE_ENDPOINT_HALT || endpoint_num == 0u || endpoint_num >= dev->hw.endpoint_count) return false;
        if (!usbd_ep_stall(dev, endpoint_addr, true)) return false;
        return usbd__ep0_send_zlp(dev);

    default:
        return false;
    }
}

static bool usbd__is_microsoft_setup(Usbd_Device const *dev) {
    Usbd_SetupPacket const *setup = &dev->ep0.setup;
    Usbd_DescriptorSet const *descriptors = &dev->descriptors;
    return setup->bmRequestType == 0xC0u &&
           setup->bRequest == descriptors->microsoft_vendor_code &&
           setup->wValue == 0u &&
           setup->wIndex == descriptors->microsoft_descriptor_index;
}

static bool usbd__forward_setup(Usbd_Device *dev) {
    Usbd_SetupPacket const *setup = &dev->ep0.setup;
    U8 recipient = (U8)(setup->bmRequestType & 0x1Fu);
    U8 type = (U8)(setup->bmRequestType & USBD__REQ_TYPE_MASK);
    bool direction_in = (setup->bmRequestType & 0x80u) != 0u;

    if (!dev->control_handler) return false;
    if (type != USBD__REQ_TYPE_CLASS && type != USBD__REQ_TYPE_VENDOR) return false;
    if (!direction_in && setup->wLength > sizeof(dev->ep0.scratch)) return false;
    if (type == USBD__REQ_TYPE_CLASS && recipient != USBD__REQ_RECIPIENT_DEVICE && dev->state != UsbdState_Configured) return false;
    if (recipient == USBD__REQ_RECIPIENT_INTERFACE && !usbd__interface_valid(dev, setup->wIndex)) return false;

    // A handler that claims the request without replying would leave EP0
    // NAKing forever; stall instead.
    return dev->control_handler(dev, setup, dev->control_context) &&
           dev->ep0.stage != UsbdEp0Stage_Idle;
}

static void usbd__handle_setup(Usbd_Device *dev) {
    U8 raw[8];
    if (!usbd_pma_read(dev, dev->ep0_rx_offset, raw, sizeof(raw))) {
        usbd__ep0_stall(dev);
        return;
    }

    usbd__ep0_reset_transfer(dev);
    usbd__ep_set_stat_rx(dev, 0u, USBD__EP_STAT_RX_NAK);
    usbd__ep_set_stat_tx(dev, 0u, USBD__EP_STAT_TX_NAK);

    dev->ep0.setup.bmRequestType = raw[0];
    dev->ep0.setup.bRequest = raw[1];
    dev->ep0.setup.wValue = read_le16(raw + 2);
    dev->ep0.setup.wIndex = read_le16(raw + 4);
    dev->ep0.setup.wLength = read_le16(raw + 6);

    U8 type = (U8)(dev->ep0.setup.bmRequestType & USBD__REQ_TYPE_MASK);
    bool handled = false;
    if (type == USBD__REQ_TYPE_STANDARD) {
        handled = usbd__handle_standard_setup(dev);
    } else if (usbd__is_microsoft_setup(dev)) {
        handled = usbd__ep0_send_data(dev, dev->descriptors.microsoft_os_20.data,
                                      dev->descriptors.microsoft_os_20.size,
                                      dev->ep0.setup.wLength);
    } else {
        handled = usbd__forward_setup(dev);
    }
    if (!handled) usbd__ep0_stall(dev);
}

static bool usbd__ep0_open(Usbd_Device *dev) {
    dev->ep0_max_packet = dev->descriptors.device.data ? dev->descriptors.device.data[7] : 64u;
    dev->btable_offset = 0u;
    dev->next_pma_offset = usbd__pma_align((U16)(dev->hw.endpoint_count * 8u));
    if (dev->next_pma_offset < 64u) dev->next_pma_offset = 64u;

    dev->ep0_tx_offset = dev->next_pma_offset;
    dev->next_pma_offset = usbd__pma_align((U16)(dev->next_pma_offset + dev->ep0_max_packet));
    dev->ep0_rx_offset = dev->next_pma_offset;
    dev->next_pma_offset = usbd__pma_align((U16)(dev->next_pma_offset + dev->ep0_max_packet));
    if (dev->next_pma_offset > dev->hw.pma_bytes) return false;

    USBD__REG(dev, USBD__BTABLE) = dev->btable_offset;
    usbd__btable_set(dev, 0u, 0u, dev->ep0_tx_offset);
    usbd__btable_set(dev, 0u, 1u, 0u);
    usbd__btable_set(dev, 0u, 2u, dev->ep0_rx_offset);
    usbd__btable_set(dev, 0u, 3u, usbd__rx_count_bits(dev->ep0_max_packet));

    USBD__REG(dev, USBD__EP_REG(0u)) = USBD__EP_TYPE_CONTROL;
    usbd__ep_clear_dtog_rx(dev, 0u);
    usbd__ep_clear_dtog_tx(dev, 0u);
    usbd__ep_set_stat_tx(dev, 0u, USBD__EP_STAT_TX_NAK);
    usbd__ep0_reset_transfer(dev);
    return usbd__ep0_rx_valid(dev);
}

static bool usbd__string_index_valid(Usbd_DescriptorSet const *descriptors, U8 index) {
    if (index == 0u) return true;
    return descriptors->strings && index <= descriptors->string_count && descriptors->strings[index - 1u];
}

static bool usbd__descriptors_valid(Usbd_DescriptorSet const *descriptors) {
    if (!descriptors->device.data || descriptors->device.size != sizeof(Usbd_DeviceDescriptor)) return false;
    if (descriptors->device.data[0] != sizeof(Usbd_DeviceDescriptor)) return false;
    if (descriptors->device.data[1] != USBD__DESC_DEVICE) return false;

    U8 ep0_max_packet = descriptors->device.data[7];
    if (ep0_max_packet != 8u && ep0_max_packet != 16u && ep0_max_packet != 32u && ep0_max_packet != 64u) return false;

    if (!usbd__string_index_valid(descriptors, descriptors->device.data[14])) return false;
    if (!usbd__string_index_valid(descriptors, descriptors->device.data[15])) return false;
    if (!usbd__string_index_valid(descriptors, descriptors->device.data[16])) return false;
    if (descriptors->device.data[17] == 0u) return false;

    if (!descriptors->configuration.data || descriptors->configuration.size < sizeof(Usbd_ConfigurationDescriptor)) return false;
    if (descriptors->configuration.data[0] != sizeof(Usbd_ConfigurationDescriptor)) return false;
    if (descriptors->configuration.data[1] != USBD__DESC_CONFIGURATION) return false;

    U16 total_length = read_le16(descriptors->configuration.data + 2);
    if (total_length < sizeof(Usbd_ConfigurationDescriptor)) return false;
    if (total_length != descriptors->configuration.size) return false;

    if (descriptors->bos.size != 0u) {
        if (!descriptors->bos.data || descriptors->bos.size < 5u) return false;
        if (descriptors->bos.data[0] != 5u || descriptors->bos.data[1] != USBD__DESC_BOS) return false;
        if (read_le16(descriptors->bos.data + 2) != descriptors->bos.size) return false;
    }
    if (!descriptors->microsoft_os_20.data ||
        descriptors->microsoft_os_20.size == 0u ||
        descriptors->microsoft_vendor_code == 0u) return false;
    return true;
}

bool usbd_init(Usbd_Device *dev, Usbd_Hardware const *hardware, Usbd_DescriptorSet const *descriptors) {
    if (!dev || !hardware || !descriptors) return false;
    if (!hardware->reg_base || !hardware->pma_base || !hardware->pma_bytes || !hardware->endpoint_count) return false;
    if (hardware->usb_hz != 48000000u) return false;
    if (!usbd__descriptors_valid(descriptors)) return false;

    memset(dev, 0, sizeof(*dev));
    dev->hw = *hardware;
    dev->descriptors = *descriptors;
    dev->state = UsbdState_Detached;
    dev->pre_suspend_state = UsbdState_Detached;
    dev->initialized = true;
    return true;
}

bool usbd_controller_reset(Usbd_Device *dev) {
    if (!usbd__ready(dev)) return false;

    USBD__REG(dev, USBD__CNTR) = USBD__CNTR_FRES | USBD__CNTR_PDWN;
    USBD__REG(dev, USBD__CNTR) = USBD__CNTR_FRES;
    USBD__REG(dev, USBD__CNTR) = 0u;
    USBD__REG(dev, USBD__ISTR) = 0u;
    USBD__REG(dev, USBD__DADDR) = USBD__DADDR_EF;

    dev->address = 0;
    dev->configuration = 0;
    dev->remote_wakeup_enabled = false;
    usbd__ep0_reset_transfer(dev);
    usbd__reset_data_endpoints(dev);
    dev->state = UsbdState_Default;

    if (!usbd__ep0_open(dev)) return false;

    USBD__REG(dev, USBD__CNTR) = USBD__CNTR_CTRM |
                                   USBD__CNTR_RESETM |
                                   USBD__CNTR_SUSPM |
                                   USBD__CNTR_WKUPM |
                                   USBD__CNTR_ERRM;
    arch_barrier();
    return true;
}

bool usbd_attach(Usbd_Device *dev) {
    if (!usbd__ready(dev)) return false;

    U16 bcdr = USBD__REG(dev, USBD__BCDR);
    USBD__REG(dev, USBD__BCDR) = (U16)(bcdr | USBD__BCDR_DPPU);
    dev->attached = true;
    if (dev->state == UsbdState_Detached) dev->state = UsbdState_Attached;
    arch_barrier();
    return true;
}

void usbd_detach(Usbd_Device *dev) {
    if (!dev || !dev->hw.reg_base) return;

    U16 bcdr = USBD__REG(dev, USBD__BCDR);
    USBD__REG(dev, USBD__BCDR) = (U16)(bcdr & ~USBD__BCDR_DPPU);
    dev->attached = false;
    dev->state = UsbdState_Detached;
    arch_barrier();
}

bool usbd_start(Usbd_Device *dev) {
    if (!usbd_controller_reset(dev)) return false;
    if (arch_irq_valid(dev->hw.irq)) {
        arch_irq_set_priority(dev->hw.irq, 2u);
        arch_irq_enable(dev->hw.irq);
    }
    return usbd_attach(dev);
}

void usbd_stop(Usbd_Device *dev) {
    if (!dev) return;
    if (arch_irq_valid(dev->hw.irq)) arch_irq_disable(dev->hw.irq);
    usbd_detach(dev);
    if (dev->hw.reg_base) USBD__REG(dev, USBD__CNTR) = USBD__CNTR_FRES;
}

// Arm only from NAK with no completion pending, from one register snapshot;
// NAK cannot change underneath the write. The RX descriptor is never
// rewritten after open: a re-arm during a reception would zero its count.
static void usbd__ep_tx_arm(Usbd_Device *dev, U8 ep) {
    Usbd_Ep *endpoint = &dev->ep_in[ep];
    if (endpoint->used == 0u) return;

    U32 reg = USBD__REG(dev, USBD__EP_REG(ep));
    if ((reg & (USBD__EP_CTR_TX | USBD__EP_STAT_TX_MASK)) != USBD__EP_STAT_TX_NAK) return;

    U32 const *slot = usbd__slot(endpoint, 0u);
    U16 size = (U16)slot[0];
    usbd__pma_write_words(dev, endpoint->pma_offset, slot + 1, (U16)((size + 3u) / 4u));
    usbd__btable_set(dev, ep, 1u, size);
    usbd__ep_write(dev, ep, reg, 0u, USBD__EP_STAT_TX_NAK ^ USBD__EP_STAT_TX_VALID);
}

static void usbd__ep_receive(Usbd_Device *dev, Usbd_Ep *endpoint, U16 rx_count) {
    if (rx_count == 0u || rx_count > endpoint->max_packet || endpoint->used == endpoint->slot_count) return;

    U32 *slot = usbd__slot(endpoint, endpoint->used);
    usbd__pma_read_words(dev, endpoint->pma_offset, slot + 1, (U16)((rx_count + 3u) / 4u));
    if (endpoint->filter && !endpoint->filter(endpoint->filter_context, (U8 const *)(slot + 1), rx_count)) return;
    slot[0] = rx_count;
    endpoint->used++;
}

static void usbd__ep_rx_arm(Usbd_Device *dev, U8 ep) {
    Usbd_Ep *endpoint = &dev->ep_out[ep];
    if (endpoint->used == endpoint->slot_count) return;

    U32 reg = USBD__REG(dev, USBD__EP_REG(ep));
    if ((reg & (USBD__EP_CTR_RX | USBD__EP_STAT_RX_MASK)) != USBD__EP_STAT_RX_NAK) return;

    usbd__ep_write(dev, ep, reg, 0u, USBD__EP_STAT_RX_NAK ^ USBD__EP_STAT_RX_VALID);
}

bool usbd_ep_open(Usbd_Device *dev, Usbd_EpConfig const *config) {
    if (!usbd__ready(dev) || !config || !config->slots || config->slot_count == 0u || config->max_packet == 0u) return false;
    if (config->max_packet > 64u || ((UPtr)config->slots & 3u) != 0u) return false;
    if (config->type == UsbdEpType_Control) return false;
    if (config->filter && usbd__ep_dir_in(config->addr)) return false;

    U8 ep = usbd__ep_num(config->addr);
    if (ep == 0u || ep >= dev->hw.endpoint_count) return false;

    bool in = usbd__ep_dir_in(config->addr);
    Usbd_Ep *endpoint = in ? &dev->ep_in[ep] : &dev->ep_out[ep];
    Usbd_Ep *other = in ? &dev->ep_out[ep] : &dev->ep_in[ep];
    if (endpoint->opened) return false;
    if (other->opened && other->type != config->type) return false;

    U16 pma_offset = usbd__pma_align(dev->next_pma_offset);
    U16 next = usbd__pma_align((U16)(pma_offset + config->max_packet));
    if (next > dev->hw.pma_bytes) return false;
    dev->next_pma_offset = next;

    if (!other->opened) {
        USBD__REG(dev, USBD__EP_REG(ep)) = (U16)(usbd__ep_type_bits(config->type) | ep);
    }

    *endpoint = (Usbd_Ep){
        .opened = true,
        .addr = config->addr,
        .type = config->type,
        .max_packet = config->max_packet,
        .pma_offset = pma_offset,
        .slots = config->slots,
        .slot_words = (U16)USBD_SLOT_WORDS(config->max_packet),
        .slot_count = config->slot_count,
        .filter = config->filter,
        .filter_context = config->filter_context,
    };

    if (in) {
        usbd__btable_set(dev, ep, 0u, pma_offset);
        usbd__btable_set(dev, ep, 1u, 0u);
        usbd__ep_clear_dtog_tx(dev, ep);
        usbd__ep_set_stat_tx(dev, ep, USBD__EP_STAT_TX_NAK);
    } else {
        usbd__btable_set(dev, ep, 2u, pma_offset);
        usbd__btable_set(dev, ep, 3u, usbd__rx_count_bits(config->max_packet));
        usbd__ep_clear_dtog_rx(dev, ep);
        usbd__ep_set_stat_rx(dev, ep, USBD__EP_STAT_RX_NAK);
        usbd__ep_rx_arm(dev, ep);
    }

    return true;
}

bool usbd_ep_opened(Usbd_Device const *dev, U8 addr) {
    Usbd_Ep const *endpoint = usbd__ep_const(dev, addr);
    return endpoint && endpoint->opened;
}

U8 const *usbd_ep_packet(Usbd_Device const *dev, U8 addr, U16 *size) {
    Usbd_Ep const *endpoint = usbd__ep_const(dev, addr);
    if (!endpoint || usbd__ep_dir_in(addr) || !endpoint->opened || endpoint->used == 0u) return 0;

    U32 const *slot = usbd__slot(endpoint, 0u);
    if (size) *size = (U16)slot[0];
    return (U8 const *)(slot + 1);
}

void usbd_ep_release(Usbd_Device *dev, U8 addr) {
    Usbd_Ep *endpoint = usbd__ep(dev, addr);
    if (!endpoint || usbd__ep_dir_in(addr) || !endpoint->opened || endpoint->used == 0u) return;

    endpoint->head = usbd__slot_next(endpoint, endpoint->head);
    endpoint->used--;
    usbd__ep_rx_arm(dev, usbd__ep_num(addr));
}

U8 *usbd_ep_slot(Usbd_Device const *dev, U8 addr) {
    Usbd_Ep const *endpoint = usbd__ep_const(dev, addr);
    if (!endpoint || !usbd__ep_dir_in(addr) || !endpoint->opened || endpoint->used == endpoint->slot_count) return 0;
    return (U8 *)(usbd__slot(endpoint, endpoint->used) + 1);
}

bool usbd_ep_commit(Usbd_Device *dev, U8 addr, U16 size) {
    Usbd_Ep *endpoint = usbd__ep(dev, addr);
    if (!endpoint || !usbd__ep_dir_in(addr) || !endpoint->opened || endpoint->used == endpoint->slot_count) return false;
    if (size == 0u || size > endpoint->max_packet) return false;

    usbd__slot(endpoint, endpoint->used)[0] = size;
    endpoint->used++;
    usbd__ep_tx_arm(dev, usbd__ep_num(addr));
    return true;
}

U16 usbd_ep_read_available(Usbd_Device const *dev, U8 addr) {
    U16 size = 0u;
    return usbd_ep_packet(dev, addr, &size) ? size : 0u;
}

U16 usbd_ep_read(Usbd_Device *dev, U8 addr, void *dst, U16 size) {
    U16 packet_size = 0u;
    U8 const *packet = usbd_ep_packet(dev, addr, &packet_size);
    if (!packet || (!dst && size)) return 0;

    U16 result = Min(size, packet_size);
    memcpy(dst, packet, result);
    usbd_ep_release(dev, addr);
    return result;
}

U16 usbd_ep_write_available(Usbd_Device const *dev, U8 addr) {
    Usbd_Ep const *endpoint = usbd__ep_const(dev, addr);
    return usbd_ep_slot(dev, addr) ? endpoint->max_packet : 0u;
}

U16 usbd_ep_write(Usbd_Device *dev, U8 addr, void const *src, U16 size) {
    U8 *slot = usbd_ep_slot(dev, addr);
    if (!slot || !src) return 0;

    memcpy(slot, src, Min(size, usbd__ep_const(dev, addr)->max_packet));
    return usbd_ep_commit(dev, addr, size) ? size : 0u;
}

bool usbd_ep_stall(Usbd_Device *dev, U8 addr, bool stalled) {
    Usbd_Ep *endpoint = usbd__ep(dev, addr);
    if (!usbd__ready(dev) || !endpoint || !endpoint->opened) return false;

    U8 ep = usbd__ep_num(addr);
    if (usbd__ep_dir_in(addr)) {
        usbd__ep_set_stat_tx(dev, ep, stalled ? USBD__EP_STAT_TX_STALL : USBD__EP_STAT_TX_NAK);
        if (!stalled) {
            usbd__ep_clear_dtog_tx(dev, ep);
            usbd__ep_tx_arm(dev, ep);
        }
    } else {
        usbd__ep_set_stat_rx(dev, ep, stalled ? USBD__EP_STAT_RX_STALL : USBD__EP_STAT_RX_NAK);
        if (!stalled) {
            usbd__ep_clear_dtog_rx(dev, ep);
            usbd__ep_rx_arm(dev, ep);
        }
    }
    return true;
}

bool usbd_ep_is_stalled(Usbd_Device const *dev, U8 addr) {
    Usbd_Ep const *endpoint = usbd__ep_const(dev, addr);
    if (!endpoint || !endpoint->opened) return false;

    U32 reg = USBD__REG(dev, USBD__EP_REG(usbd__ep_num(addr)));
    return usbd__ep_dir_in(addr) ? (reg & USBD__EP_STAT_TX_MASK) == USBD__EP_STAT_TX_STALL
                                 : (reg & USBD__EP_STAT_RX_MASK) == USBD__EP_STAT_RX_STALL;
}

bool usbd_pma_read(Usbd_Device const *dev, U16 pma_offset, void *dst, U16 size) {
    if (!dev || !dev->hw.pma_base || !dst) return false;
    if ((U32)pma_offset + size > dev->hw.pma_bytes) return false;

    U8 *out = (U8 *)dst;
    if (usbd__pma_32bit(dev)) {
        for (U16 i = 0; i < size; i += 4u) {
            U32 word = USBD__PMA32(dev, (U16)(pma_offset + i));
            for (U16 byte = 0; byte < 4u && (U16)(i + byte) < size; byte++) {
                out[i + byte] = (U8)(word >> (byte * 8u));
            }
        }
        return true;
    }

    for (U16 i = 0; i < size; i += 2u) {
        U16 word = USBD__PMA16(dev, (U16)(pma_offset + i));
        out[i] = (U8)word;
        if ((U16)(i + 1u) < size) out[i + 1u] = (U8)(word >> 8);
    }
    return true;
}

bool usbd_pma_write(Usbd_Device *dev, U16 pma_offset, void const *src, U16 size) {
    if (!dev || !dev->hw.pma_base || (!src && size)) return false;
    if ((U32)pma_offset + size > dev->hw.pma_bytes) return false;

    U8 const *in = (U8 const *)src;
    if (usbd__pma_32bit(dev)) {
        for (U16 i = 0; i < size; i += 4u) {
            U32 word = 0;
            for (U16 byte = 0; byte < 4u && (U16)(i + byte) < size; byte++) {
                word |= (U32)in[i + byte] << (byte * 8u);
            }
            USBD__PMA32(dev, (U16)(pma_offset + i)) = word;
        }
        return true;
    }

    for (U16 i = 0; i < size; i += 2u) {
        U16 word = in[i];
        if ((U16)(i + 1u) < size) word |= (U16)in[i + 1u] << 8;
        USBD__PMA16(dev, (U16)(pma_offset + i)) = word;
    }
    return true;
}

static void usbd__handle_reset(Usbd_Device *dev) {
    USBD__REG(dev, USBD__CNTR) &= ~(USBD__CNTR_SUSPRDY | USBD__CNTR_SUSPEN);
    dev->address = 0;
    dev->configuration = 0;
    dev->remote_wakeup_enabled = false;
    dev->pre_suspend_state = UsbdState_Default;
    dev->resume_pending = false;
    dev->lifecycle++;
    usbd__ep0_reset_transfer(dev);
    usbd__reset_data_endpoints(dev);
    dev->state = UsbdState_Default;
    USBD__REG(dev, USBD__DADDR) = USBD__DADDR_EF;
    (void)usbd__ep0_open(dev);
}

static void usbd__handle_ep0_tx(Usbd_Device *dev) {
    if (dev->ep0.stage == UsbdEp0Stage_DataIn && dev->ep0.in_at < dev->ep0.in_size) {
        (void)usbd__ep0_send_packet(dev);
        return;
    }

    if (dev->ep0.stage == UsbdEp0Stage_DataIn && dev->ep0.in_zlp) {
        dev->ep0.in_zlp = false;
        (void)usbd__ep0_send_packet(dev);
        return;
    }

    if (dev->ep0.stage == UsbdEp0Stage_StatusIn) {
        U8 type = (U8)(dev->ep0.setup.bmRequestType & USBD__REQ_TYPE_MASK);
        if (type == USBD__REQ_TYPE_STANDARD && (dev->ep0.pending_address || dev->ep0.setup.bRequest == USBD__REQ_SET_ADDRESS)) {
            usbd__set_address_now(dev, dev->ep0.pending_address);
        } else if (type == USBD__REQ_TYPE_STANDARD && dev->ep0.setup.bRequest == USBD__REQ_SET_CONFIGURATION) {
            usbd__reset_data_endpoints(dev);
            dev->configuration = (U8)dev->ep0.setup.wValue;
            dev->state = dev->configuration ? UsbdState_Configured : (dev->address ? UsbdState_Addressed : UsbdState_Default);
            dev->lifecycle++;
        }

        usbd__ep0_reset_transfer(dev);
        (void)usbd__ep0_rx_valid(dev);
        return;
    }

    if (dev->ep0.stage == UsbdEp0Stage_DataIn) {
        dev->ep0.stage = UsbdEp0Stage_StatusOut;
        (void)usbd__ep0_rx_valid(dev);
    } else {
        usbd__ep0_reset_transfer(dev);
        (void)usbd__ep0_rx_valid(dev);
    }
}

static void usbd__handle_ep0_data_out(Usbd_Device *dev, U16 rx_count) {
    U16 remaining = (U16)(dev->ep0.out_size - dev->ep0.out_at);
    if (rx_count > remaining ||
        !usbd_pma_read(dev, dev->ep0_rx_offset, dev->ep0.scratch + dev->ep0.out_at, rx_count)) {
        usbd__ep0_stall(dev);
        return;
    }
    dev->ep0.out_at = (U16)(dev->ep0.out_at + rx_count);
    if (dev->ep0.out_at < dev->ep0.out_size && rx_count == dev->ep0_max_packet) {
        (void)usbd__ep0_rx_valid(dev);
        return;
    }

    bool ok = dev->control_data_handler &&
              dev->control_data_handler(dev, &dev->ep0.setup, dev->ep0.scratch,
                                        dev->ep0.out_at, dev->control_context);
    if (ok) {
        (void)usbd__ep0_send_zlp(dev);
    } else {
        usbd__ep0_stall(dev);
    }
}

static void usbd__handle_ep0_out(Usbd_Device *dev, U16 rx_count) {
    if (dev->ep0.stage == UsbdEp0Stage_DataOut) {
        usbd__handle_ep0_data_out(dev, rx_count);
        return;
    }
    if (dev->ep0.stage != UsbdEp0Stage_StatusOut || rx_count != 0u) {
        usbd__ep0_stall(dev);
        return;
    }
    usbd__ep0_reset_transfer(dev);
    (void)usbd__ep0_rx_valid(dev);
}

static void usbd__handle_ep0_ctr(Usbd_Device *dev) {
    U16 ep = USBD__REG(dev, USBD__EP_REG(0u));

    if ((ep & USBD__EP_CTR_RX) != 0 && (ep & USBD__EP_SETUP) != 0) {
        U16 rx_count = usbd__ep_rx_count(dev, 0u);
        usbd__ep_clear_ctr_rx(dev, 0u);
        if ((ep & USBD__EP_CTR_TX) != 0) usbd__ep_clear_ctr_tx(dev, 0u);

        if (rx_count == 8u) {
            usbd__handle_setup(dev);
        } else {
            usbd__ep0_stall(dev);
        }
        return;
    }

    if ((ep & USBD__EP_CTR_RX) != 0) {
        U16 rx_count = usbd__ep_rx_count(dev, 0u);
        usbd__ep_clear_ctr_rx(dev, 0u);
        usbd__handle_ep0_out(dev, rx_count);
    }

    ep = USBD__REG(dev, USBD__EP_REG(0u));
    if ((ep & USBD__EP_CTR_TX) != 0) {
        usbd__ep_clear_ctr_tx(dev, 0u);
        usbd__handle_ep0_tx(dev);
    }
}

void usbd_irq(Usbd_Device *dev) {
    if (!usbd__ready(dev)) return;

    for (U32 guard = 0; guard < 32u; guard++) {
        U16 istr = USBD__REG(dev, USBD__ISTR);
        if ((istr & (USBD__ISTR_CTR | USBD__ISTR_RESET | USBD__ISTR_SUSP | USBD__ISTR_WKUP | USBD__ISTR_ERR)) == 0) break;

        if ((istr & USBD__ISTR_RESET) != 0) {
            USBD__REG(dev, USBD__ISTR) =
                (U16)~(USBD__ISTR_RESET | USBD__ISTR_SUSP | USBD__ISTR_WKUP);
            usbd__handle_reset(dev);
            continue;
        }

        if ((istr & USBD__ISTR_SUSP) != 0) {
            USBD__REG(dev, USBD__CNTR) |= USBD__CNTR_SUSPEN;
            USBD__REG(dev, USBD__ISTR) = (U16)~USBD__ISTR_SUSP;
            USBD__REG(dev, USBD__CNTR) |= USBD__CNTR_SUSPRDY;
            if (dev->state != UsbdState_Suspended) {
                dev->pre_suspend_state = dev->state;
            }
            dev->state = UsbdState_Suspended;
            dev->resume_pending = false;
            break;
        }

        if ((istr & USBD__ISTR_WKUP) != 0) {
            USBD__REG(dev, USBD__CNTR) &= ~(USBD__CNTR_SUSPRDY | USBD__CNTR_SUSPEN);
            USBD__REG(dev, USBD__ISTR) = (U16)~USBD__ISTR_WKUP;
            if (dev->state == UsbdState_Suspended) {
                dev->resume_pending = true;
            }
            continue;
        }

        if ((istr & USBD__ISTR_ERR) != 0) {
            USBD__REG(dev, USBD__ISTR) = (U16)~USBD__ISTR_ERR;
            continue;
        }

        if ((istr & USBD__ISTR_CTR) != 0) {
            if (dev->resume_pending) {
                dev->resume_pending = false;
                dev->state = dev->pre_suspend_state;
            }
            U8 ep = (U8)(istr & USBD__ISTR_EP_ID_MASK);
            if (ep == 0u) {
                usbd__handle_ep0_ctr(dev);
            } else {
                U16 ep_reg = USBD__REG(dev, USBD__EP_REG(ep));
                if ((ep_reg & USBD__EP_CTR_RX) != 0) {
                    U16 rx_count = usbd__ep_rx_count(dev, ep);
                    usbd__ep_clear_ctr_rx(dev, ep);
                    Usbd_Ep *endpoint = &dev->ep_out[ep];
                    if (endpoint->opened) {
                        usbd__ep_receive(dev, endpoint, rx_count);
                        usbd__ep_rx_arm(dev, ep);
                    }
                }
                if ((ep_reg & USBD__EP_CTR_TX) != 0) {
                    usbd__ep_clear_ctr_tx(dev, ep);
                    Usbd_Ep *endpoint = &dev->ep_in[ep];
                    if (endpoint->opened && endpoint->used != 0u) {
                        endpoint->head = usbd__slot_next(endpoint, endpoint->head);
                        endpoint->used--;
                        usbd__ep_tx_arm(dev, ep);
                    }
                }
            }
        }
    }
}

Usbd_State usbd_state(Usbd_Device const *dev) {
    return dev ? dev->state : UsbdState_Detached;
}

U32 usbd_lifecycle(Usbd_Device const *dev) {
    return dev ? dev->lifecycle : 0u;
}

void usbd_set_control_handler(Usbd_Device *dev, Usbd_ControlHandler handler, void *context) {
    if (!dev) return;
    dev->control_handler = handler;
    dev->control_context = context;
}

bool usbd_control_send(Usbd_Device *dev, void const *data, U16 size) {
    if (!dev || !data || dev->ep0.stage != UsbdEp0Stage_Idle) return false;
    if ((dev->ep0.setup.bmRequestType & 0x80u) == 0u) return false;
    if (size > sizeof(dev->ep0.scratch)) return false;

    memcpy(dev->ep0.scratch, data, size);
    return usbd__ep0_send_buffer(dev, size, dev->ep0.setup.wLength);
}

bool usbd_control_ack(Usbd_Device *dev) {
    if (!dev || dev->ep0.stage != UsbdEp0Stage_Idle) return false;
    if ((dev->ep0.setup.bmRequestType & 0x80u) != 0u || dev->ep0.setup.wLength != 0u) return false;
    return usbd__ep0_send_zlp(dev);
}

// Accepts the OUT data stage of the current request into EP0 scratch; the
// control data handler sees it once all wLength bytes (or a short packet)
// arrive.
bool usbd_control_receive(Usbd_Device *dev) {
    if (!dev || dev->ep0.stage != UsbdEp0Stage_Idle || !dev->control_data_handler) return false;
    if ((dev->ep0.setup.bmRequestType & 0x80u) != 0u || dev->ep0.setup.wLength == 0u) return false;
    if (dev->ep0.setup.wLength > sizeof(dev->ep0.scratch)) return false;

    dev->ep0.out_size = dev->ep0.setup.wLength;
    dev->ep0.out_at = 0u;
    dev->ep0.stage = UsbdEp0Stage_DataOut;
    return usbd__ep0_rx_valid(dev);
}

void usbd_set_control_data_handler(Usbd_Device *dev, Usbd_ControlDataHandler handler) {
    if (dev) dev->control_data_handler = handler;
}

#endif
