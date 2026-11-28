#ifndef CDC_UART_H
#define CDC_UART_H

// CDC-ACM function bridged to one USART. USART RX is interrupt-driven into a
// ring that the main loop drains into bulk IN packets; bulk OUT packets are
// written to TDR from the main loop as TXE allows. The line coding sets the
// USART framing; DTR/RTS and breaks are acknowledged and otherwise ignored.

#include "arch.h"
#include "core.h"
#include "usbd.h"

enum {
    CdcUartPacketSize = 64,
    CdcUartNotifySize = 16,
    CdcUartSlotCount = 4,
    CdcUartRxRingBytes = 1024,
    CdcUartDefaultBaud = 115200,
    CdcUartMinimumBaud = 1200,

    CdcRequest_SetLineCoding = 0x20,
    CdcRequest_GetLineCoding = 0x21,
    CdcRequest_SetControlLineState = 0x22,
    CdcRequest_SendBreak = 0x23,
};

static_assert((CdcUartRxRingBytes & (CdcUartRxRingBytes - 1)) == 0);

typedef struct __attribute__((packed)) {
    Usbd_InterfaceAssociationDescriptor iad;
    Usbd_InterfaceDescriptor comm_interface;
    U8 header[5];
    U8 call_management[5];
    U8 acm[4];
    U8 union_[5];
    Usbd_EndpointDescriptor notify;
    Usbd_InterfaceDescriptor data_interface;
    Usbd_EndpointDescriptor data_out;
    Usbd_EndpointDescriptor data_in;
} CdcUart_FunctionDescriptor;

static_assert(sizeof(CdcUart_FunctionDescriptor) == 66);

#define CDC_UART_FUNCTION_DESCRIPTOR(comm_if_, notify_ep_, data_ep_, string_) {          \
    .iad = {                                                                             \
        .length = sizeof(Usbd_InterfaceAssociationDescriptor),                           \
        .descriptor_type = UsbdDescriptorType_InterfaceAssociation,                      \
        .first_interface = (comm_if_),                                                   \
        .interface_count = 2,                                                            \
        .function_class = 0x02,                                                          \
        .function_subclass = 0x02,                                                       \
        .function_protocol = 0x00,                                                       \
        .function_string = (string_),                                                    \
    },                                                                                   \
    .comm_interface = {                                                                  \
        .length = sizeof(Usbd_InterfaceDescriptor),                                      \
        .descriptor_type = UsbdDescriptorType_Interface,                                 \
        .interface_number = (comm_if_),                                                  \
        .endpoint_count = 1,                                                             \
        .interface_class = 0x02,                                                         \
        .interface_subclass = 0x02,                                                      \
        .interface_protocol = 0x00,                                                      \
        .interface_string = (string_),                                                   \
    },                                                                                   \
    .header = { 5, 0x24, 0x00, 0x10, 0x01 },                                             \
    .call_management = { 5, 0x24, 0x01, 0x00, (comm_if_) + 1 },                          \
    .acm = { 4, 0x24, 0x02, 0x06 },                                                      \
    .union_ = { 5, 0x24, 0x06, (comm_if_), (comm_if_) + 1 },                             \
    .notify = {                                                                          \
        .length = sizeof(Usbd_EndpointDescriptor),                                       \
        .descriptor_type = UsbdDescriptorType_Endpoint,                                  \
        .endpoint_address = UsbdEpAddr_In(notify_ep_),                                   \
        .attributes = 0x03,                                                              \
        .max_packet_size = CdcUartNotifySize,                                            \
        .interval = 16,                                                                  \
    },                                                                                   \
    .data_interface = {                                                                  \
        .length = sizeof(Usbd_InterfaceDescriptor),                                      \
        .descriptor_type = UsbdDescriptorType_Interface,                                 \
        .interface_number = (comm_if_) + 1,                                              \
        .endpoint_count = 2,                                                             \
        .interface_class = 0x0A,                                                         \
    },                                                                                   \
    .data_out = {                                                                        \
        .length = sizeof(Usbd_EndpointDescriptor),                                       \
        .descriptor_type = UsbdDescriptorType_Endpoint,                                  \
        .endpoint_address = UsbdEpAddr_Out(data_ep_),                                    \
        .attributes = 0x02,                                                              \
        .max_packet_size = CdcUartPacketSize,                                            \
    },                                                                                   \
    .data_in = {                                                                         \
        .length = sizeof(Usbd_EndpointDescriptor),                                       \
        .descriptor_type = UsbdDescriptorType_Endpoint,                                  \
        .endpoint_address = UsbdEpAddr_In(data_ep_),                                     \
        .attributes = 0x02,                                                              \
        .max_packet_size = CdcUartPacketSize,                                            \
    },                                                                                   \
}

typedef struct {
    U32 baud;
    U8 stop_bits;  // 0 = 1, 1 = 1.5, 2 = 2
    U8 parity;     // 0 none, 1 odd, 2 even
    U8 data_bits;
} CdcUart_LineCoding;

typedef struct {
    Usbd_Device *usb;
    UPtr uart;      // USART register base
    U32 uart_hz;    // USART kernel clock
    U8 comm_interface;
    U8 notify_ep;
    U8 out_ep;
    U8 in_ep;

    CdcUart_LineCoding coding;
    bool dtr;
    bool rts;

    U32 notify_slots[USBD_SLOT_WORDS(CdcUartNotifySize)];
    U32 out_slots[USBD_SLOT_WORDS(CdcUartPacketSize) * CdcUartSlotCount];
    U32 in_slots[USBD_SLOT_WORDS(CdcUartPacketSize) * CdcUartSlotCount];

    U8 rx_ring[CdcUartRxRingBytes];
    volatile U32 rx_head;  // written by the USART IRQ
    U32 rx_tail;
    U32 rx_dropped;
    U32 rx_errors;

    U16 tx_at;
    U32 activity;  // bytes moved either way, for the activity LED
} CdcUart;

#ifndef CDC_UART__REG
#define CDC_UART__REG(base_, off_) (*(volatile U32 *)((base_) + (off_)))
#endif
#define CDC_UART__CR1  0x00u
#define CDC_UART__CR2  0x04u
#define CDC_UART__BRR  0x0Cu
#define CDC_UART__ISR  0x1Cu
#define CDC_UART__ICR  0x20u
#define CDC_UART__RDR  0x24u
#define CDC_UART__TDR  0x28u

#define CDC_UART__CR1_UE     (1u << 0)
#define CDC_UART__CR1_RE     (1u << 2)
#define CDC_UART__CR1_TE     (1u << 3)
#define CDC_UART__CR1_RXNEIE (1u << 5)
#define CDC_UART__CR1_PS     (1u << 9)
#define CDC_UART__CR1_PCE    (1u << 10)
#define CDC_UART__CR1_M0     (1u << 12)
#define CDC_UART__CR1_M1     (1u << 28)
#define CDC_UART__CR2_STOP_2 (2u << 12)
#define CDC_UART__CR2_STOP_1_5 (3u << 12)
#define CDC_UART__ISR_ERRORS 0x0Fu  // PE, FE, NE, ORE
#define CDC_UART__ISR_RXNE   (1u << 5)
#define CDC_UART__ISR_TXE    (1u << 7)

static bool cdc_uart__coding_valid(CdcUart const *c, CdcUart_LineCoding const *coding) {
    if (coding->baud < CdcUartMinimumBaud) return false;
    if (coding->baud > (c->uart_hz >> 4)) return false;  // BRR >= 16
    if (coding->stop_bits > 2u || coding->parity > 2u) return false;
    return coding->data_bits == 7u || coding->data_bits == 8u;
}

static void cdc_uart__apply(CdcUart *c) {
    UPtr u = c->uart;
    CdcUart_LineCoding const *coding = &c->coding;

    // Word length counts the parity bit: 7N -> 7, 7P/8N -> 8, 8P -> 9.
    U32 word = coding->data_bits + (coding->parity ? 1u : 0u);
    U32 cr1 = CDC_UART__CR1_RE | CDC_UART__CR1_TE | CDC_UART__CR1_RXNEIE;
    if (word == 7u) cr1 |= CDC_UART__CR1_M1;
    if (word == 9u) cr1 |= CDC_UART__CR1_M0;
    if (coding->parity) cr1 |= CDC_UART__CR1_PCE;
    if (coding->parity == 1u) cr1 |= CDC_UART__CR1_PS;

    U32 cr2 = 0u;
    if (coding->stop_bits == 1u) cr2 = CDC_UART__CR2_STOP_1_5;
    if (coding->stop_bits == 2u) cr2 = CDC_UART__CR2_STOP_2;

    CDC_UART__REG(u, CDC_UART__CR1) = 0u;
    CDC_UART__REG(u, CDC_UART__CR2) = cr2;
    CDC_UART__REG(u, CDC_UART__BRR) = udiv32_round(c->uart_hz, coding->baud);
    CDC_UART__REG(u, CDC_UART__ICR) = CDC_UART__ISR_ERRORS;
    CDC_UART__REG(u, CDC_UART__CR1) = cr1 | CDC_UART__CR1_UE;
}

static bool cdc_uart_init(CdcUart *c, Usbd_Device *usb, UPtr uart, U32 uart_hz,
                          U8 comm_interface, U8 notify_ep, U8 data_ep) {
    if (!c || !usb || !uart || !uart_hz) return false;
    *c = (CdcUart){
        .usb = usb,
        .uart = uart,
        .uart_hz = uart_hz,
        .comm_interface = comm_interface,
        .notify_ep = UsbdEpAddr_In(notify_ep),
        .out_ep = UsbdEpAddr_Out(data_ep),
        .in_ep = UsbdEpAddr_In(data_ep),
        .coding = { .baud = CdcUartDefaultBaud, .data_bits = 8u },
    };
    cdc_uart__apply(c);
    return true;
}

static bool cdc_uart_endpoints_open(CdcUart const *c) {
    return usbd_ep_opened(c->usb, c->notify_ep) &&
           usbd_ep_opened(c->usb, c->out_ep) &&
           usbd_ep_opened(c->usb, c->in_ep);
}

static bool cdc_uart__open(CdcUart *c, U8 addr, Usbd_EpType type, U16 max_packet,
                           U32 *slots, U8 slot_count) {
    if (usbd_ep_opened(c->usb, addr)) return true;
    Usbd_EpConfig config = {
        .addr = addr,
        .type = type,
        .max_packet = max_packet,
        .slots = slots,
        .slot_count = slot_count,
    };
    return usbd_ep_open(c->usb, &config);
}

// Call with interrupts masked after each SET_CONFIGURATION.
static bool cdc_uart_open_endpoints(CdcUart *c) {
    if (!cdc_uart__open(c, c->notify_ep, UsbdEpType_Interrupt, CdcUartNotifySize,
                        c->notify_slots, 1u)) return false;
    if (!cdc_uart__open(c, c->out_ep, UsbdEpType_Bulk, CdcUartPacketSize,
                        c->out_slots, CdcUartSlotCount)) return false;
    if (!cdc_uart__open(c, c->in_ep, UsbdEpType_Bulk, CdcUartPacketSize,
                        c->in_slots, CdcUartSlotCount)) return false;
    c->tx_at = 0u;
    return true;
}

static bool cdc_uart__for_me(CdcUart const *c, Usbd_SetupPacket const *setup) {
    return (setup->bmRequestType & 0x7Fu) == 0x21u &&  // class, interface
           (U8)setup->wIndex == c->comm_interface;
}

// Usbd_ControlHandler half: true when the request was answered.
static bool cdc_uart_control(CdcUart *c, Usbd_Device *dev, Usbd_SetupPacket const *setup) {
    if (!cdc_uart__for_me(c, setup)) return false;
    bool in = (setup->bmRequestType & 0x80u) != 0u;

    switch (setup->bRequest) {
    case CdcRequest_SetLineCoding:
        return !in && setup->wLength == 7u && usbd_control_receive(dev);
    case CdcRequest_GetLineCoding: {
        if (!in) return false;
        U8 reply[7];
        write_le32(reply, c->coding.baud);
        reply[4] = c->coding.stop_bits;
        reply[5] = c->coding.parity;
        reply[6] = c->coding.data_bits;
        return usbd_control_send(dev, reply, sizeof(reply));
    }
    case CdcRequest_SetControlLineState:
        if (in || setup->wLength != 0u) return false;
        c->dtr = (setup->wValue & 1u) != 0u;
        c->rts = (setup->wValue & 2u) != 0u;
        return usbd_control_ack(dev);
    case CdcRequest_SendBreak:
        return !in && setup->wLength == 0u && usbd_control_ack(dev);
    default:
        return false;
    }
}

// Usbd_ControlDataHandler half: true when the data stage was accepted.
static bool cdc_uart_control_data(CdcUart *c, Usbd_SetupPacket const *setup,
                                  U8 const *data, U16 size) {
    if (!cdc_uart__for_me(c, setup) || setup->bRequest != CdcRequest_SetLineCoding) return false;
    if (size != 7u) return false;
    CdcUart_LineCoding coding = {
        .baud = read_le32(data),
        .stop_bits = data[4],
        .parity = data[5],
        .data_bits = data[6],
    };
    if (!cdc_uart__coding_valid(c, &coding)) return false;
    c->coding = coding;
    cdc_uart__apply(c);
    return true;
}

static void cdc_uart_irq(CdcUart *c) {
    UPtr u = c->uart;
    U32 isr = CDC_UART__REG(u, CDC_UART__ISR);
    if ((isr & CDC_UART__ISR_ERRORS) != 0u) {
        CDC_UART__REG(u, CDC_UART__ICR) = isr & CDC_UART__ISR_ERRORS;
        c->rx_errors++;
    }
    while ((isr & CDC_UART__ISR_RXNE) != 0u) {
        U8 byte = (U8)CDC_UART__REG(u, CDC_UART__RDR);
        U32 head = c->rx_head;
        if (head - c->rx_tail < CdcUartRxRingBytes) {
            c->rx_ring[head & (CdcUartRxRingBytes - 1u)] = byte;
            c->rx_head = head + 1u;
        } else {
            c->rx_dropped++;
        }
        isr = CDC_UART__REG(u, CDC_UART__ISR);
    }
}

// Main-loop half; `configured` gates the bulk endpoints.
static void cdc_uart_service(CdcUart *c, bool configured) {
    Usbd_Device *usb = c->usb;
    UPtr u = c->uart;

    // USART RX -> bulk IN. Unconfigured, the ring keeps the newest backlog.
    U32 available = c->rx_head - c->rx_tail;
    if (!configured) {
        if (available > CdcUartRxRingBytes / 2u) c->rx_tail = c->rx_head - CdcUartRxRingBytes / 2u;
    } else if (available != 0u) {
        U32 lock = arch_irq_save();
        U8 *slot = usbd_ep_slot(usb, c->in_ep);
        if (slot) {
            U16 n = (U16)Min(available, (U32)CdcUartPacketSize);
            for (U16 i = 0u; i < n; i++) {
                slot[i] = c->rx_ring[(c->rx_tail + i) & (CdcUartRxRingBytes - 1u)];
            }
            if (usbd_ep_commit(usb, c->in_ep, n)) {
                c->rx_tail += n;
                c->activity += n;
            }
        }
        arch_irq_restore(lock);
    }

    // Bulk OUT -> USART TX, as far as TXE allows without waiting.
    if (!configured) return;
    U32 lock = arch_irq_save();
    U16 size = 0u;
    U8 const *packet = usbd_ep_packet(usb, c->out_ep, &size);
    if (packet) {
        while (c->tx_at < size && (CDC_UART__REG(u, CDC_UART__ISR) & CDC_UART__ISR_TXE) != 0u) {
            CDC_UART__REG(u, CDC_UART__TDR) = packet[c->tx_at++];
            c->activity++;
        }
        if (c->tx_at >= size) {
            usbd_ep_release(usb, c->out_ep);
            c->tx_at = 0u;
        }
    }
    arch_irq_restore(lock);
}

#endif
