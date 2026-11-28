// CDC-ACM console bridge and the EP0 OUT data stage, against the USB model
// and a byte-level USART model.
#include "usb_model.h"

#include <deque>

#define ARCH_H
static U32 arch_irq_save(void) {
    return 0u;
}

static void arch_irq_restore(U32 state) {
    (void)state;
}

struct UartModel {
    U32 cr1 = 0u;
    U32 cr2 = 0u;
    U32 brr = 0u;
    U32 isr_errors = 0u;
    std::deque<U8> rx;
    std::vector<U8> tx;
    U32 tx_room = 1000u;  // TDR writes accepted before TXE drops

    U32 read(U32 offset) {
        switch (offset) {
        case 0x00u: return cr1;
        case 0x04u: return cr2;
        case 0x0Cu: return brr;
        case 0x1Cu: return isr_errors | (rx.empty() ? 0u : (1u << 5)) | (tx_room ? (1u << 7) : 0u);
        case 0x24u: {
            CHECK(!rx.empty());
            U8 byte = rx.front();
            rx.pop_front();
            return byte;
        }
        default: return 0u;
        }
    }

    void write(U32 offset, U32 value) {
        switch (offset) {
        case 0x00u: cr1 = value; break;
        case 0x04u: cr2 = value; break;
        case 0x0Cu: brr = value; break;
        case 0x20u: isr_errors &= ~value; break;
        case 0x28u:
            CHECK(tx_room != 0u);
            tx.push_back((U8)value);
            tx_room--;
            break;
        default: break;
        }
    }
};

UartModel *active_uart;

struct UartRegister {
    U32 offset;
    operator U32() const { return active_uart->read(offset); }
    UartRegister &operator=(U32 value) {
        active_uart->write(offset, value);
        return *this;
    }
};

#define CDC_UART__REG(base_, off_) ((void)(base_), UartRegister{(U32)(off_)})
#include "../cdc_uart.h"

namespace {

constexpr U8 CommInterface = 0;
constexpr U8 NotifyEndpoint = 1;
constexpr U8 DataEndpoint = 2;
constexpr U8 LongVendorRequest = 0x45;

struct __attribute__((packed)) TestConfiguration {
    Usbd_ConfigurationDescriptor configuration;
    CdcUart_FunctionDescriptor console;
};

Usbd_DeviceDescriptor const device_descriptor = {
    .length = sizeof(Usbd_DeviceDescriptor),
    .descriptor_type = UsbdDescriptorType_Device,
    .usb_bcd = 0x0200,
    .device_class = 0xEF,
    .device_subclass = 0x02,
    .device_protocol = 0x01,
    .ep0_max_packet = 64,
    .vendor_id = 0x1234,
    .product_id = 0x5678,
    .configuration_count = 1,
};

TestConfiguration const configuration_descriptor = {
    .configuration = {
        .length = sizeof(Usbd_ConfigurationDescriptor),
        .descriptor_type = UsbdDescriptorType_Configuration,
        .total_length = sizeof(TestConfiguration),
        .interface_count = 2,
        .configuration_value = 1,
        .attributes = 0x80,
        .max_power_2ma = 50,
    },
    .console = CDC_UART_FUNCTION_DESCRIPTOR(CommInterface, NotifyEndpoint, DataEndpoint, 0),
};

U8 microsoft_descriptor[10];

Usbd_DescriptorSet const descriptors = {
    .device = {
        .data = reinterpret_cast<U8 const *>(&device_descriptor),
        .size = sizeof(device_descriptor),
    },
    .configuration = {
        .data = reinterpret_cast<U8 const *>(&configuration_descriptor),
        .size = sizeof(configuration_descriptor),
    },
    .microsoft_os_20 = {.data = microsoft_descriptor, .size = sizeof(microsoft_descriptor)},
    .microsoft_descriptor_index = 7,
    .language_id = 0x0409,
    .microsoft_vendor_code = 0x20,
};

CdcUart console;
std::vector<U8> long_request_data;

bool control_handler(Usbd_Device *device, Usbd_SetupPacket const *setup, void *context) {
    (void)context;
    if (cdc_uart_control(&console, device, setup)) return true;
    if (setup->bmRequestType == 0x40 && setup->bRequest == LongVendorRequest) {
        return usbd_control_receive(device);
    }
    return false;
}

bool control_data_handler(Usbd_Device *device, Usbd_SetupPacket const *setup,
                          U8 const *data, U16 size, void *context) {
    (void)device;
    (void)context;
    if (setup->bmRequestType == 0x40 && setup->bRequest == LongVendorRequest) {
        long_request_data.assign(data, data + size);
        return true;
    }
    return cdc_uart_control_data(&console, setup, data, size);
}

// Setup, OUT data packets (max 64 bytes each), then the status IN.
// Returns false when the device stalls instead of acknowledging.
bool control_write(UsbModel &model, Usbd_Device *device, U8 request_type, U8 request,
                   U16 value, U16 index, std::vector<U8> const &data) {
    setup(model, device, request_type, request, value, index, (U16)data.size());
    for (size_t at = 0; at < data.size(); at += 64u) {
        if ((model.endpoints[0] & USBD__EP_STAT_RX_MASK) != USBD__EP_STAT_RX_VALID) return false;
        U16 n = (U16)Min(data.size() - at, (size_t)64u);
        model.out(device, 0, data.data() + at, n);
    }
    if ((model.endpoints[0] & USBD__EP_STAT_TX_MASK) != USBD__EP_STAT_TX_VALID) return false;
    CHECK(model.in(device, 0).empty());
    return true;
}

std::vector<U8> line_coding(U32 baud, U8 stop, U8 parity, U8 bits) {
    std::vector<U8> data(7);
    write_le32(data.data(), baud);
    data[4] = stop;
    data[5] = parity;
    data[6] = bits;
    return data;
}

void enumerate(UsbModel &model, Usbd_Device *device) {
    model.bus_reset(device);
    control_out(model, device, 0x00, USBD__REQ_SET_ADDRESS, 3, 0);
    control_out(model, device, 0x00, USBD__REQ_SET_CONFIGURATION, 1, 0);
    CHECK(usbd_state(device) == UsbdState_Configured);
    CHECK(!cdc_uart_endpoints_open(&console));
    CHECK(cdc_uart_open_endpoints(&console));
    CHECK(cdc_uart_endpoints_open(&console));
}

void test_out_data_stage(UsbModel &model, Usbd_Device *device) {
    std::vector<U8> data(100);
    for (size_t i = 0; i < data.size(); i++) data[i] = (U8)(i * 7u);
    CHECK(control_write(model, device, 0x40, LongVendorRequest, 0, 0, data));
    CHECK(long_request_data == data);

    // Exactly one full packet, then a request too long for EP0 scratch.
    data.resize(64);
    CHECK(control_write(model, device, 0x40, LongVendorRequest, 0, 0, data));
    CHECK(long_request_data == data);
    control_expect_stall(model, device, 0x40, LongVendorRequest, 0, 0, 129);
}

void test_line_coding(UsbModel &model, Usbd_Device *device, UartModel &uart) {
    std::vector<U8> coding = control_in(model, device, 0xA1, CdcRequest_GetLineCoding, 0,
                                        CommInterface, 7);
    CHECK(coding == line_coding(115200, 0, 0, 8));
    CHECK(uart.brr == 556u);  // 64 MHz / 115200, rounded

    CHECK(control_write(model, device, 0x21, CdcRequest_SetLineCoding, 0, CommInterface,
                        line_coding(921600, 2, 2, 8)));
    coding = control_in(model, device, 0xA1, CdcRequest_GetLineCoding, 0, CommInterface, 7);
    CHECK(coding == line_coding(921600, 2, 2, 8));
    CHECK(uart.brr == 69u);
    CHECK((uart.cr1 & CDC_UART__CR1_M0) != 0u);   // 9-bit word: 8 data + parity
    CHECK((uart.cr1 & CDC_UART__CR1_PCE) != 0u);
    CHECK((uart.cr1 & CDC_UART__CR1_PS) == 0u);   // even
    CHECK((uart.cr1 & CDC_UART__CR1_UE) != 0u);
    CHECK(uart.cr2 == CDC_UART__CR2_STOP_2);

    // Rejected codings stall the status stage and leave the USART alone.
    CHECK(!control_write(model, device, 0x21, CdcRequest_SetLineCoding, 0, CommInterface,
                         line_coding(0, 0, 0, 8)));
    CHECK(!control_write(model, device, 0x21, CdcRequest_SetLineCoding, 0, CommInterface,
                         line_coding(9000000, 0, 0, 8)));
    CHECK(!control_write(model, device, 0x21, CdcRequest_SetLineCoding, 0, CommInterface,
                         line_coding(115200, 0, 0, 5)));
    CHECK(uart.brr == 69u);

    CHECK(control_write(model, device, 0x21, CdcRequest_SetLineCoding, 0, CommInterface,
                        line_coding(115200, 0, 0, 8)));
    CHECK(uart.brr == 556u && (uart.cr1 & (CDC_UART__CR1_M0 | CDC_UART__CR1_PCE)) == 0u);

    control_out(model, device, 0x21, CdcRequest_SetControlLineState, 3, CommInterface);
    CHECK(console.dtr && console.rts);
    control_out(model, device, 0x21, CdcRequest_SendBreak, 100, CommInterface);
    // Another interface's class request is not ours.
    control_expect_stall(model, device, 0xA1, CdcRequest_GetLineCoding, 0, 1, 7);
}

void test_data_paths(UsbModel &model, Usbd_Device *device, UartModel &uart) {
    // USART RX -> bulk IN, split at the 64-byte packet size.
    for (U32 i = 0; i < 100u; i++) uart.rx.push_back((U8)i);
    cdc_uart_irq(&console);
    CHECK(uart.rx.empty());
    cdc_uart_service(&console, true);
    cdc_uart_service(&console, true);
    std::vector<U8> got = model.in(device, DataEndpoint);
    std::vector<U8> rest = model.in(device, DataEndpoint);
    CHECK(got.size() == 64u && rest.size() == 36u);
    got.insert(got.end(), rest.begin(), rest.end());
    for (U32 i = 0; i < 100u; i++) CHECK(got[i] == (U8)i);

    // A full ring drops new bytes and counts them.
    for (U32 i = 0; i < CdcUartRxRingBytes + 10u; i++) uart.rx.push_back((U8)i);
    cdc_uart_irq(&console);
    CHECK(console.rx_dropped == 10u);
    while (console.rx_head != console.rx_tail) {
        cdc_uart_service(&console, true);
        (void)model.in(device, DataEndpoint);
    }

    // Line errors are cleared and counted.
    uart.isr_errors = 0x08u;
    cdc_uart_irq(&console);
    CHECK(uart.isr_errors == 0u && console.rx_errors == 1u);

    // Bulk OUT -> USART TX, resuming when TXE comes back.
    std::vector<U8> out(40);
    for (size_t i = 0; i < out.size(); i++) out[i] = (U8)(0xA0u + i);
    uart.tx.clear();
    uart.tx_room = 25u;
    model.out(device, DataEndpoint, out.data(), (U16)out.size());
    cdc_uart_service(&console, true);
    CHECK(uart.tx.size() == 25u);
    uart.tx_room = 1000u;
    cdc_uart_service(&console, true);
    CHECK(uart.tx == out);
    CHECK((model.endpoints[DataEndpoint] & USBD__EP_STAT_RX_MASK) == USBD__EP_STAT_RX_VALID);

    // Unconfigured, RX keeps only the newest half-ring of backlog.
    for (U32 i = 0; i < CdcUartRxRingBytes; i++) uart.rx.push_back((U8)i);
    cdc_uart_irq(&console);
    cdc_uart_service(&console, false);
    CHECK(console.rx_head - console.rx_tail == CdcUartRxRingBytes / 2u);
}

void run_tests(void) {
    UsbModel model;
    UartModel uart;
    active_model = &model;
    active_uart = &uart;

    Usbd_Device device{};
    Usbd_Hardware hardware = {
        .reg_base = 1,
        .pma_base = reinterpret_cast<UPtr>(model.pma),
        .pma_bytes = sizeof(model.pma),
        .endpoint_count = 8,
        .irq = 8,
        .usb_hz = 48000000,
        .flags = UsbdHardware_Pma32Bit,
    };
    CHECK(usbd_init(&device, &hardware, &descriptors));
    usbd_set_control_handler(&device, control_handler, nullptr);
    usbd_set_control_data_handler(&device, control_data_handler);
    CHECK(cdc_uart_init(&console, &device, 0x40004C00u, 64000000u, CommInterface,
                        NotifyEndpoint, DataEndpoint));
    CHECK(usbd_start(&device));
    enumerate(model, &device);

    test_out_data_stage(model, &device);
    test_line_coding(model, &device, uart);
    test_data_paths(model, &device, uart);

    puts("PASS cdc-uart: ep0 out data stage, line coding, rx/tx bridging, overflow");
}

} // namespace

int main(void) {
    run_tests();
    return 0;
}
