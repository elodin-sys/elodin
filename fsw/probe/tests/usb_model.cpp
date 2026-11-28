#include "usb_model.h"

namespace {

constexpr U8 DfuInterface = 1;
constexpr U32 DfuDetachDelayUs = 50000;
constexpr U8 EchoVendorRequest = 0x42;

struct __attribute__((packed)) TestConfiguration {
    Usbd_ConfigurationDescriptor configuration;
    Usbd_InterfaceDescriptor interface;
    Usbd_EndpointDescriptor out;
    Usbd_EndpointDescriptor in;
    Usbd_InterfaceDescriptor dfu_interface;
    UsbDfu_FunctionalDescriptor dfu_functional;
};

Usbd_DeviceDescriptor const device_descriptor = {
    .length = sizeof(Usbd_DeviceDescriptor),
    .descriptor_type = UsbdDescriptorType_Device,
    .usb_bcd = 0x0210,
    .ep0_max_packet = 64,
    .vendor_id = 0x1234,
    .product_id = 0x5678,
    .device_bcd = 0x0100,
    .manufacturer_string = 1,
    .product_string = 2,
    .serial_string = 3,
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
    .interface = {
        .length = sizeof(Usbd_InterfaceDescriptor),
        .descriptor_type = UsbdDescriptorType_Interface,
        .endpoint_count = 2,
        .interface_class = 0xFF,
        .interface_string = 4,
    },
    .out = {
        .length = sizeof(Usbd_EndpointDescriptor),
        .descriptor_type = UsbdDescriptorType_Endpoint,
        .endpoint_address = 1,
        .attributes = 2,
        .max_packet_size = 64,
    },
    .in = {
        .length = sizeof(Usbd_EndpointDescriptor),
        .descriptor_type = UsbdDescriptorType_Endpoint,
        .endpoint_address = 0x81,
        .attributes = 2,
        .max_packet_size = 64,
    },
    .dfu_interface = {
        .length = sizeof(Usbd_InterfaceDescriptor),
        .descriptor_type = UsbdDescriptorType_Interface,
        .interface_number = DfuInterface,
        .interface_class = UsbDfu_InterfaceClass,
        .interface_subclass = UsbDfu_InterfaceSubclass,
        .interface_protocol = UsbDfu_Protocol_Runtime,
    },
    .dfu_functional = {
        .length = sizeof(UsbDfu_FunctionalDescriptor),
        .descriptor_type = UsbDfu_DescriptorType_Functional,
        .attributes = UsbDfu_Attribute_WillDetach,
        .detach_timeout_ms = 1000,
        .transfer_size = 2048,
        .dfu_version = UsbDfu_Version,
    },
};

UsbDfu_Runtime dfu_runtime;
U32 vendor_requests_seen;

bool control_handler(Usbd_Device *device, Usbd_SetupPacket const *setup,
                     void *context) {
    if (usb_dfu_runtime_control(device, setup, context)) return true;
    if (setup->bmRequestType == 0xC0 && setup->bRequest == EchoVendorRequest) {
        vendor_requests_seen++;
        U8 reply[2];
        write_le16(reply, setup->wValue);
        return usbd_control_send(device, reply, sizeof(reply));
    }
    // Claims the request but never replies; usbd must stall rather than hang.
    if (setup->bmRequestType == 0xC0 && setup->bRequest == 0x43) return true;
    return false;
}

U8 const bos_descriptor[] = {5, UsbdDescriptorType_Bos, 5, 0, 0};
U8 microsoft_descriptor[80];
char const *const strings[] = {
    "Test Vendor",
    "Test Device",
    "0123456789ABCDEF",
    "Test Interface",
};

Usbd_DescriptorSet const descriptors = {
    .device = {
        .data = reinterpret_cast<U8 const *>(&device_descriptor),
        .size = sizeof(device_descriptor),
    },
    .configuration = {
        .data = reinterpret_cast<U8 const *>(&configuration_descriptor),
        .size = sizeof(configuration_descriptor),
    },
    .bos = {.data = bos_descriptor, .size = sizeof(bos_descriptor)},
    .microsoft_os_20 = {
        .data = microsoft_descriptor,
        .size = sizeof(microsoft_descriptor),
    },
    .strings = strings,
    .microsoft_descriptor_index = 7,
    .language_id = 0x0409,
    .microsoft_vendor_code = 0x20,
    .string_count = ArrayCount(strings),
};

U8 dfu_state(UsbModel &model, Usbd_Device *device) {
    std::vector<U8> state = control_in(model, device, 0xA1,
                                       UsbDfu_Request_GetState, 0,
                                       DfuInterface, 1);
    CHECK(state.size() == 1);
    return state[0];
}

void enumerate(UsbModel &model, Usbd_Device *device) {
    model.bus_reset(device);
    CHECK(usbd_state(device) == UsbdState_Default);

    // Vendor requests reach the handler before configuration; class
    // requests to an interface do not.
    std::vector<U8> echo = control_in(model, device, 0xC0,
                                      EchoVendorRequest, 0xBEEF, 0, 2);
    CHECK(echo.size() == 2 && read_le16(echo.data()) == 0xBEEF);
    control_expect_stall(model, device, 0xA1, UsbDfu_Request_GetState, 0,
                         DfuInterface, 1);
    control_expect_stall(model, device, 0x21, UsbDfu_Request_Detach, 1000,
                         DfuInterface, 0);
    CHECK(!usb_dfu_runtime_detach_pending(&dfu_runtime, 0));

    std::vector<U8> description = control_in(
        model, device, 0x80, USBD__REQ_GET_DESCRIPTOR,
        (U16)(UsbdDescriptorType_Device << 8), 0,
        sizeof(device_descriptor));
    CHECK(description.size() == sizeof(device_descriptor));
    CHECK(description[1] == UsbdDescriptorType_Device);

    std::vector<U8> configuration = control_in(
        model, device, 0x80, USBD__REQ_GET_DESCRIPTOR,
        (U16)(UsbdDescriptorType_Configuration << 8), 0,
        sizeof(configuration_descriptor));
    CHECK(configuration.size() == sizeof(configuration_descriptor));
    CHECK(read_le16(configuration.data() + 2) ==
          sizeof(configuration_descriptor));

    std::vector<U8> microsoft = control_in(
        model, device, 0xC0, descriptors.microsoft_vendor_code, 0,
        descriptors.microsoft_descriptor_index,
        sizeof(microsoft_descriptor));
    CHECK(microsoft.size() == sizeof(microsoft_descriptor));
    CHECK(memcmp(microsoft.data(), microsoft_descriptor,
                 sizeof(microsoft_descriptor)) == 0);

    control_out(model, device, 0x00, USBD__REQ_SET_ADDRESS, 7, 0);
    CHECK(usbd_state(device) == UsbdState_Addressed);

    control_out(model, device, 0x00, USBD__REQ_SET_CONFIGURATION, 1, 0);
    CHECK(usbd_state(device) == UsbdState_Configured);
}

constexpr U16 OneSlot = USBD_SLOT_WORDS(64);

void open_bulk_endpoints(Usbd_Device *device, U32 (&out_buffer)[OneSlot],
                         U32 (&in_buffer)[OneSlot]) {
    Usbd_EpConfig out = {
        .addr = 1,
        .type = UsbdEpType_Bulk,
        .max_packet = 64,
        .slots = out_buffer,
        .slot_count = 1,
    };
    Usbd_EpConfig in = {
        .addr = 0x81,
        .type = UsbdEpType_Bulk,
        .max_packet = 64,
        .slots = in_buffer,
        .slot_count = 1,
    };
    CHECK(usbd_ep_open(device, &out));
    CHECK(usbd_ep_open(device, &in));
}

void test_bulk(UsbModel &model, Usbd_Device *device) {
    U8 request[] = {0x11, 0x22, 0x33, 0x44};
    model.out(device, 1, request, sizeof(request));
    U8 received[sizeof(request)]{};
    CHECK(usbd_ep_read(device, 1, received, sizeof(received)) ==
          sizeof(received));
    CHECK(memcmp(received, request, sizeof(request)) == 0);

    U8 response[] = {0xA5, 0x5A, 0xC3};
    CHECK(usbd_ep_write(device, 0x81, response, sizeof(response)) ==
          sizeof(response));
    std::vector<U8> sent = model.in(device, 1);
    CHECK(sent.size() == sizeof(response));
    CHECK(memcmp(sent.data(), response, sizeof(response)) == 0);
}

U32 dropped_packets;

bool drop_sevens(void *context, U8 const *data, U16 size) {
    CHECK(context == &dropped_packets);
    CHECK(size != 0);
    if (data[0] != 0x07) return true;
    dropped_packets++;
    return false;
}

void test_packet_queue(UsbModel &model, Usbd_Device *device) {
    constexpr U16 Slots = 3;
    static U32 out_buffer[USBD_SLOT_WORDS(64) * Slots];
    static U32 in_buffer[USBD_SLOT_WORDS(64) * Slots];
    Usbd_EpConfig out = {
        .addr = 1,
        .type = UsbdEpType_Bulk,
        .max_packet = 64,
        .slots = out_buffer,
        .slot_count = Slots,
        .filter = drop_sevens,
        .filter_context = &dropped_packets,
    };
    Usbd_EpConfig in = {
        .addr = 0x81,
        .type = UsbdEpType_Bulk,
        .max_packet = 64,
        .slots = in_buffer,
        .slot_count = Slots,
    };
    Usbd_EpConfig no_slots = out;
    no_slots.slot_count = 0;
    CHECK(!usbd_ep_open(device, &no_slots));
    Usbd_EpConfig filtered_in = in;
    filtered_in.filter = drop_sevens;
    CHECK(!usbd_ep_open(device, &filtered_in));
    CHECK(usbd_ep_open(device, &out));
    CHECK(usbd_ep_open(device, &in));

    // Three packets of any size fill the three slots; the endpoint NAKs until
    // one is released, and the head packet is readable in place.
    std::vector<U8> a(5, 0xA1), b(64, 0xB2), c(1, 0xC3);
    model.out(device, 1, a.data(), (U16)a.size());
    model.out(device, 1, b.data(), (U16)b.size());
    model.out(device, 1, c.data(), (U16)c.size());
    CHECK((model.endpoints[1] & USBD__EP_STAT_RX_MASK) == USBD__EP_STAT_RX_NAK);

    U16 head_size = 0;
    U8 const *head = usbd_ep_packet(device, 1, &head_size);
    CHECK(head && head_size == 5 && head[0] == 0xA1);
    CHECK(usbd_ep_read_available(device, 1) == 5);

    U8 received[64]{};
    // A short read still releases the whole packet.
    CHECK(usbd_ep_read(device, 1, received, 2) == 2);
    CHECK(received[0] == 0xA1 && received[1] == 0xA1);
    CHECK((model.endpoints[1] & USBD__EP_STAT_RX_MASK) == USBD__EP_STAT_RX_VALID);
    CHECK(usbd_ep_read_available(device, 1) == 64);
    CHECK(usbd_ep_read(device, 1, received, sizeof(received)) == 64);
    CHECK(memcmp(received, b.data(), 64) == 0);
    CHECK(usbd_ep_read(device, 1, received, sizeof(received)) == 1);
    CHECK(received[0] == 0xC3);
    CHECK(usbd_ep_read_available(device, 1) == 0);
    CHECK(!usbd_ep_packet(device, 1, &head_size));
    CHECK(usbd_ep_read(device, 1, received, sizeof(received)) == 0);

    // A read must not re-arm over a packet whose interrupt has not run.
    model.out(device, 1, a.data(), (U16)a.size());
    model.out(device, 1, b.data(), (U16)b.size(), false, false);
    CHECK(usbd_ep_read(device, 1, received, sizeof(received)) == 5);
    CHECK((model.endpoints[1] & USBD__EP_STAT_RX_MASK) == USBD__EP_STAT_RX_NAK);
    CHECK(model.buffer_count(1, true) == 64);
    CHECK(usbd_ep_read_available(device, 1) == 0);
    usbd_irq(device);
    CHECK(usbd_ep_read(device, 1, received, sizeof(received)) == 64);
    CHECK(memcmp(received, b.data(), 64) == 0);
    CHECK((model.endpoints[1] & USBD__EP_STAT_RX_MASK) == USBD__EP_STAT_RX_VALID);

    // A zero-length packet and a filtered packet are both dropped.
    model.out(device, 1, nullptr, 0);
    U8 abort[] = {0x07, 0x00};
    U32 dropped = dropped_packets;
    model.out(device, 1, abort, sizeof(abort));
    CHECK(dropped_packets == dropped + 1);
    CHECK(usbd_ep_read_available(device, 1) == 0);
    CHECK((model.endpoints[1] & USBD__EP_STAT_RX_MASK) == USBD__EP_STAT_RX_VALID);

    // One IN packet per write, in order; three fill the queue.
    std::vector<U8> x(3, 0x11), y(64, 0x22), z(2, 0x33);
    U8 oversize[65]{};
    CHECK(usbd_ep_write(device, 0x81, oversize, sizeof(oversize)) == 0);
    CHECK(usbd_ep_write(device, 0x81, x.data(), 0) == 0);
    CHECK(usbd_ep_write_available(device, 0x81) == 64);
    CHECK(usbd_ep_write(device, 0x81, x.data(), (U16)x.size()) == 3);
    U8 *slot = usbd_ep_slot(device, 0x81);
    CHECK(slot);
    memcpy(slot, y.data(), y.size());
    CHECK(usbd_ep_commit(device, 0x81, 64));
    CHECK(usbd_ep_write(device, 0x81, z.data(), (U16)z.size()) == 2);
    CHECK(usbd_ep_write_available(device, 0x81) == 0);
    CHECK(!usbd_ep_slot(device, 0x81));
    CHECK(model.in(device, 1) == x);
    CHECK(usbd_ep_write_available(device, 0x81) == 64);
    CHECK(model.in(device, 1) == y);
    CHECK(model.in(device, 1) == z);
    CHECK((model.endpoints[1] & USBD__EP_STAT_TX_MASK) == USBD__EP_STAT_TX_NAK);
}

void test_suspend_resume(UsbModel &model, Usbd_Device *device) {
    U32 lifecycle = usbd_lifecycle(device);
    model.suspend(device);
    CHECK(usbd_state(device) == UsbdState_Suspended);
    CHECK(usbd_lifecycle(device) == lifecycle);
    CHECK(!dap_usb_session_invalidated(UsbdState_Configured, lifecycle,
                                       usbd_state(device),
                                       usbd_lifecycle(device), false));

    model.wakeup(device);
    U8 request[] = {0x55, 0xAA};
    model.out(device, 1, request, sizeof(request));
    CHECK(usbd_state(device) == UsbdState_Configured);
    CHECK(usbd_lifecycle(device) == lifecycle);
    CHECK(!dap_usb_session_invalidated(UsbdState_Suspended, lifecycle,
                                       usbd_state(device),
                                       usbd_lifecycle(device), false));
    U8 received[sizeof(request)]{};
    CHECK(usbd_ep_read(device, 1, received, sizeof(received)) ==
          sizeof(received));
    CHECK(memcmp(received, request, sizeof(request)) == 0);
}

void test_control_forwarding(UsbModel &model, Usbd_Device *device) {
    U32 seen = vendor_requests_seen;
    std::vector<U8> echo = control_in(model, device, 0xC0,
                                      EchoVendorRequest, 0x1234, 0, 2);
    CHECK(echo.size() == 2 && read_le16(echo.data()) == 0x1234);
    CHECK(vendor_requests_seen == seen + 1);

    // The Microsoft OS 2.0 fetch is still answered by usbd itself.
    std::vector<U8> microsoft = control_in(
        model, device, 0xC0, descriptors.microsoft_vendor_code, 0,
        descriptors.microsoft_descriptor_index, sizeof(microsoft_descriptor));
    CHECK(microsoft.size() == sizeof(microsoft_descriptor));
    CHECK(vendor_requests_seen == seen + 1);

    // Unknown vendor request, a handler that claims without replying, an OUT
    // request with a data stage, and a bad interface index all stall.
    control_expect_stall(model, device, 0xC0, 0x44, 0, 0, 2);
    control_expect_stall(model, device, 0xC0, 0x43, 0, 0, 2);
    control_expect_stall(model, device, 0x40, EchoVendorRequest, 0, 0, 4);
    control_expect_stall(model, device, 0xA1, UsbDfu_Request_GetState, 0,
                         DfuInterface + 1, 1);

    echo = control_in(model, device, 0xC0, EchoVendorRequest, 0x5678, 0, 2);
    CHECK(echo.size() == 2 && read_le16(echo.data()) == 0x5678);
}

void test_dfu_runtime(UsbModel &model, Usbd_Device *device) {
    CHECK(dfu_state(model, device) == UsbDfu_State_AppIdle);
    std::vector<U8> status = control_in(model, device, 0xA1,
                                        UsbDfu_Request_GetStatus, 0,
                                        DfuInterface, UsbDfu_GetStatusSize);
    CHECK(status.size() == UsbDfu_GetStatusSize);
    CHECK(status[0] == UsbDfu_Status_Ok);
    CHECK(status[4] == UsbDfu_State_AppIdle);

    // DFU-mode requests, requests to the wrong interface, and a DETACH with
    // the wrong direction are all refused and change nothing.
    control_expect_stall(model, device, 0x21, UsbDfu_Request_Dnload, 0,
                         DfuInterface, 0);
    control_expect_stall(model, device, 0xA1, UsbDfu_Request_Upload, 0,
                         DfuInterface, 64);
    control_expect_stall(model, device, 0x21, UsbDfu_Request_ClrStatus, 0,
                         DfuInterface, 0);
    control_expect_stall(model, device, 0x21, UsbDfu_Request_Detach, 1000,
                         0, 0);
    control_expect_stall(model, device, 0xA1, UsbDfu_Request_Detach, 1000,
                         DfuInterface, 0);
    CHECK(dfu_state(model, device) == UsbDfu_State_AppIdle);
    CHECK(!usb_dfu_runtime_detach_pending(&dfu_runtime, 0));

    // DFU_DETACH is acknowledged, moves to appDETACH, and becomes due after
    // the delay measured from the first poll rather than from the request.
    control_out(model, device, 0x21, UsbDfu_Request_Detach, 1000,
                DfuInterface);
    CHECK(dfu_state(model, device) == UsbDfu_State_AppDetach);
    status = control_in(model, device, 0xA1, UsbDfu_Request_GetStatus, 0,
                        DfuInterface, UsbDfu_GetStatusSize);
    CHECK(status.size() == UsbDfu_GetStatusSize);
    CHECK(status[4] == UsbDfu_State_AppDetach);

    U32 now = 0xFFFFF000u; // wraps during the delay
    CHECK(!usb_dfu_runtime_detach_pending(&dfu_runtime, now));
    CHECK(!usb_dfu_runtime_detach_pending(&dfu_runtime,
                                          now + DfuDetachDelayUs - 1));
    CHECK(usb_dfu_runtime_detach_pending(&dfu_runtime,
                                         now + DfuDetachDelayUs));
    CHECK(usb_dfu_runtime_detach_pending(&dfu_runtime,
                                         now + DfuDetachDelayUs + 1));

    control_out(model, device, 0x21, UsbDfu_Request_Detach, 1000,
                DfuInterface);
    CHECK(usb_dfu_runtime_detach_pending(&dfu_runtime,
                                         now + DfuDetachDelayUs + 2));

    dfu_runtime = {.interface_number = DfuInterface,
                   .detach_delay_us = DfuDetachDelayUs};
    CHECK(dfu_state(model, device) == UsbDfu_State_AppIdle);
}

void check_data_endpoints_reset(UsbModel const &model,
                                Usbd_Device const *device) {
    CHECK(!usbd_ep_opened(device, 1));
    CHECK(!usbd_ep_opened(device, 0x81));
    CHECK((model.endpoints[1] & USBD__EP_STAT_RX_MASK) ==
          USBD__EP_STAT_RX_DISABLED);
    CHECK((model.endpoints[1] & USBD__EP_STAT_TX_MASK) ==
          USBD__EP_STAT_TX_DISABLED);
    CHECK((model.endpoints[1] &
           (USBD__EP_DTOG_RX | USBD__EP_DTOG_TX)) == 0u);
}

void test_reconfiguration(UsbModel &model, Usbd_Device *device,
                          U32 (&out_buffer)[OneSlot],
                          U32 (&in_buffer)[OneSlot]) {
    model.endpoints[1] |= USBD__EP_DTOG_RX | USBD__EP_DTOG_TX;
    U32 lifecycle = usbd_lifecycle(device);
    control_out(model, device, 0x00, USBD__REQ_SET_CONFIGURATION, 0, 0);
    CHECK(usbd_state(device) == UsbdState_Addressed);
    CHECK(dap_usb_session_invalidated(UsbdState_Configured, lifecycle,
                                      usbd_state(device),
                                      usbd_lifecycle(device), false));
    check_data_endpoints_reset(model, device);

    control_out(model, device, 0x00, USBD__REQ_SET_CONFIGURATION, 1, 0);
    CHECK(usbd_state(device) == UsbdState_Configured);
    check_data_endpoints_reset(model, device);
    open_bulk_endpoints(device, out_buffer, in_buffer);

    U8 request[] = {0x33, 0xCC};
    model.out(device, 1, request, sizeof(request));
    U8 received[sizeof(request)]{};
    CHECK(usbd_ep_read(device, 1, received, sizeof(received)) ==
          sizeof(received));
    control_out(model, device, 0x00, USBD__REQ_SET_CONFIGURATION, 1, 0);
    CHECK(usbd_state(device) == UsbdState_Configured);
    check_data_endpoints_reset(model, device);
    open_bulk_endpoints(device, out_buffer, in_buffer);

    lifecycle = usbd_lifecycle(device);
    model.bus_reset(device);
    CHECK(usbd_state(device) == UsbdState_Default);
    CHECK(dap_usb_session_invalidated(UsbdState_Configured, lifecycle,
                                      usbd_state(device),
                                      usbd_lifecycle(device), false));
    check_data_endpoints_reset(model, device);
}

void run_tests(void) {
    UsbModel model;
    active_model = &model;
    for (U32 i = 0u; i < sizeof(microsoft_descriptor); i++) {
        microsoft_descriptor[i] = (U8)(i ^ 0x6Du);
    }

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
    dfu_runtime = {.interface_number = DfuInterface,
                   .detach_delay_us = DfuDetachDelayUs};
    usbd_set_control_handler(&device, control_handler, &dfu_runtime);
    CHECK(usbd_start(&device));
    enumerate(model, &device);

    U32 out_buffer[OneSlot]{};
    U32 in_buffer[OneSlot]{};
    open_bulk_endpoints(&device, out_buffer, in_buffer);
    test_bulk(model, &device);
    test_control_forwarding(model, &device);
    test_dfu_runtime(model, &device);
    test_suspend_resume(model, &device);
    test_reconfiguration(model, &device, out_buffer, in_buffer);

    enumerate(model, &device);
    test_packet_queue(model, &device);

    puts("PASS usb: enumeration, bulk, packet queues, control forwarding, "
         "dfu runtime, suspend, reconfigure, reset");
}

} // namespace

int main(void) {
    run_tests();
    return 0;
}
